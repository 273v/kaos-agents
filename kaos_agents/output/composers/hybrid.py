"""``compose_hybrid`` — narrative sections + an embedded extraction table.

The Harvey-shape M&A report: a multi-section narrative (executive
summary, per-contract review, recommendations) with an embedded
findings table. Pure composition of :func:`compose_narrative` and
:func:`compose_tabular`; no new I/O.

The composer takes:

* a memory state (carries FINDINGS Claims + ExtractionCells)
* a structure (the narrative spine)
* an optional schema (when set, an embedded table is produced and
  inserted at ``table_position`` in the section sequence)

If no schema is supplied, falls back to a pure-narrative deliverable
(equivalent to calling :func:`compose_narrative`). If the schema is
supplied but the cells walk yields nothing, the embedded table
position is skipped (no empty table).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.output.citations import aggregate_citations
from kaos_agents.output.composers.narrative import compose_narrative
from kaos_agents.output.composers.tabular import compose_tabular
from kaos_agents.output.deliverables import (
    HybridDeliverable,
    NarrativeDeliverable,
    TabularDeliverable,
)
from kaos_agents.output.walks import SpanDedupKey
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types import ZERO_USAGE, InvocationUsage

if TYPE_CHECKING:
    from kaos_llm_core.signatures.extraction import ExtractionSchema

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.output.types import SectionSpec

logger = get_logger(__name__)


async def compose_hybrid(
    memory: SessionMemory,
    structure: tuple[SectionSpec, ...],
    *,
    schema: ExtractionSchema | None = None,
    table_position: int = -1,
    table_name: str = "findings",
    model: str = DEFAULT_MODEL,
    title: str = "",
    citation_dedupe_by: SpanDedupKey = "source_uri+char_span",
) -> HybridComposeResult:
    """Compose a hybrid deliverable (narrative + optional embedded table).

    Args:
        memory: Session memory.
        structure: Narrative spine — :class:`SectionSpec` per narrative
            section.
        schema: When set, :func:`compose_tabular` runs over FINDINGS
            cells matching the schema; the resulting
            :class:`TabularDeliverable` is inserted at
            ``table_position`` in the section sequence. ``None``
            falls back to pure narrative.
        table_position: Where the table sits in the section ordering.
            ``-1`` (default) means "after every narrative section".
            ``0`` means "before the first narrative section". Other
            positive values place the table between narrative
            sections at the given index.
        table_name: Name passed to :meth:`TabularDocument.from_cells`.
        model: Producer LLM model.
        title: Top-level document title.
        citation_dedupe_by: Pass-through to citation aggregation.

    Returns:
        :class:`HybridComposeResult` carrying the deliverable + the
        narrative composition's per-section telemetry.
    """
    narrative_result = await compose_narrative(
        memory,
        structure,
        model=model,
        title=title,
        citation_dedupe_by=citation_dedupe_by,
    )
    narrative = narrative_result.deliverable

    sections: list[NarrativeDeliverable | TabularDeliverable]
    if schema is not None:
        tabular = compose_tabular(memory, schema, table_name=table_name)
        # Skip empty tables — no rows to embed.
        rows_total = sum(len(t.rows) for t in tabular.document.tables)
        if rows_total == 0:
            logger.debug("compose_hybrid: schema supplied but tabular has 0 rows — skipping embed")
            sections = [narrative]
        else:
            sections = _splice(narrative, tabular, table_position)
    else:
        sections = [narrative]

    spans = tuple(aggregate_citations(memory, dedupe_by=citation_dedupe_by))
    deliverable = HybridDeliverable(
        sections=tuple(sections),
        spans=spans,
        title=title,
    )
    return HybridComposeResult(
        deliverable=deliverable,
        narrative_total_usage=narrative_result.total_usage,
        per_section_usage=narrative_result.per_section_usage,
    )


def _splice(
    narrative: NarrativeDeliverable,
    tabular: TabularDeliverable,
    position: int,
) -> list[NarrativeDeliverable | TabularDeliverable]:
    """Insert the tabular section at ``position`` in the section ordering.

    ``position == -1`` appends at the end. ``position == 0`` prepends.
    Otherwise inserts between the n-th and (n+1)-th narrative segment;
    since we have only one NarrativeDeliverable wrapping the whole
    narrative spine, splicing into the middle isn't yet supported —
    the table goes either before or after the narrative.
    """
    if position == 0:
        return [tabular, narrative]
    return [narrative, tabular]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402 — keep result type at file-tail
from typing import Any  # noqa: E402


@dataclass(frozen=True, slots=True)
class HybridComposeResult:
    """Output of :func:`compose_hybrid`.

    Mirrors :class:`NarrativeComposeResult` but at the hybrid layer:
    deliverable + narrative telemetry. The tabular composition is
    pure (no LLM cost), so we don't track its usage separately.
    """

    deliverable: HybridDeliverable
    narrative_total_usage: InvocationUsage = field(default_factory=lambda: ZERO_USAGE)
    per_section_usage: tuple[tuple[str, InvocationUsage], ...] = ()
    extra: tuple[tuple[str, Any], ...] = ()
