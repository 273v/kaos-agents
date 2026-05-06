"""``compose_narrative`` — build a NarrativeDeliverable via per-section LLM calls.

Pure orchestration over kaos-llm-core + kaos-content + the lower-layer
walks/citations primitives:

1. For each :class:`SectionSpec` in the structure:
   a. :func:`recall` from memory (FINDINGS + ACTIONS + MESSAGES) for
      section-relevant context.
   b. ``Call(SectionWriterSignature).invoke()`` with the spec + context +
      the citation list rendered as text. Use ``.invoke()`` (not bare
      ``__call__``) so per-section cost flows into the standard
      ``Invocation`` rollup.
   c. Wrap the produced markdown body in a :class:`Heading` +
      :class:`Paragraph` block pair and accumulate.
2. After every section is written, :func:`aggregate_citations` produces
   the deliverable's authoritative span list.
3. Build a :class:`NarrativeDeliverable` via
   :func:`narrative_from_blocks`.

Cost is plumbed: every section's :class:`Invocation.usage` is folded
into the returned :class:`NarrativeComposeResult` so callers can
attribute composition cost to the agent's session ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kaos_content.model.blocks import Heading
from kaos_content.model.inlines import Text
from kaos_core.logging import get_logger
from kaos_llm_core import Call

from kaos_agents.memory.types import MemoryType
from kaos_agents.output.citations import aggregate_citations
from kaos_agents.output.deliverables import (
    NarrativeDeliverable,
    narrative_from_blocks,
    paragraph_block,
)
from kaos_agents.output.signatures import SectionWriterSignature
from kaos_agents.output.walks import SpanDedupKey
from kaos_agents.planning.recall import recall
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.usage import ZERO_USAGE, InvocationUsage

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block
    from kaos_llm_core.signatures.grounding import Span

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.output.types import SectionSpec

logger = get_logger(__name__)


# Per-section recall budget. Sized so each section's call sees a real
# slice of context but the whole deliverable composition fits in the
# 200K context budget even at 8-10 sections. 16K per section x 8
# sections = 128K, leaves ~70K for the producer's own system prompt
# + the section instruction + citation list.
_SECTION_RECALL_BUDGET_TOKENS = 16_000


@dataclass(frozen=True, slots=True)
class NarrativeComposeResult:
    """Output of :func:`compose_narrative` — deliverable + per-section telemetry.

    The deliverable carries the typed content. ``per_section_usage``
    maps section id → :class:`InvocationUsage` so callers can fold
    section cost into a session ledger or surface it via
    ``UsageObserved`` events. ``total_usage`` is the field-wise sum.
    """

    deliverable: NarrativeDeliverable
    per_section_usage: tuple[tuple[str, InvocationUsage], ...] = ()
    total_usage: InvocationUsage = field(default_factory=lambda: ZERO_USAGE)


async def compose_narrative(
    memory: SessionMemory,
    structure: tuple[SectionSpec, ...],
    *,
    model: str = DEFAULT_MODEL,
    title: str = "",
    citation_dedupe_by: SpanDedupKey = "source_uri+char_span",
) -> NarrativeComposeResult:
    """Compose a narrative deliverable section-by-section.

    Args:
        memory: Session memory; sections recall from FINDINGS / ACTIONS /
            MESSAGES per section.
        structure: Ordered tuple of :class:`SectionSpec`. Renders in
            the supplied order (ignores ``ordering`` field; callers who
            want re-sorting should sort the tuple before passing).
        model: Producer LLM model. Defaults to the agent's default.
        title: Document title; passed to :func:`narrative_from_blocks`
            as document metadata.
        citation_dedupe_by: Pass-through to
            :func:`aggregate_citations`. Default
            ``"source_uri+char_span"`` (production-correct for legal).

    Returns:
        :class:`NarrativeComposeResult` carrying the deliverable + the
        per-section usage telemetry.
    """
    spans = tuple(aggregate_citations(memory, dedupe_by=citation_dedupe_by))
    citation_text = _citations_as_text(spans)

    body_blocks: list[Block] = []
    per_section_usage: list[tuple[str, InvocationUsage]] = []
    total = ZERO_USAGE
    call = Call(SectionWriterSignature, model=model)

    for spec in structure:
        section_context = _recall_for_section(memory)
        invocation = await call.invoke(
            section_id=spec.id,
            section_title=spec.title,
            section_depth=spec.depth,
            section_instruction=spec.instruction,
            recalled_context=section_context,
            citations=citation_text,
        )
        section_body = str(invocation.output.body).strip()
        usage = InvocationUsage.from_invocation(invocation)
        per_section_usage.append((spec.id, usage))
        total = total + usage
        logger.debug(
            "compose_narrative.section: id=%s tokens=%d cost=%.6f body_chars=%d",
            spec.id,
            usage.total_tokens,
            usage.cost_usd,
            len(section_body),
        )

        # Append heading + body blocks for this section.
        body_blocks.append(Heading(depth=spec.depth, children=(Text(value=spec.title),)))
        if section_body:
            body_blocks.append(paragraph_block(section_body))
        else:
            body_blocks.append(paragraph_block("(No findings recorded for this section.)"))

    deliverable = narrative_from_blocks(
        blocks=tuple(body_blocks),
        structure=structure,
        spans=spans,
        title=title,
    )
    return NarrativeComposeResult(
        deliverable=deliverable,
        per_section_usage=tuple(per_section_usage),
        total_usage=total,
    )


def _recall_for_section(memory: SessionMemory) -> str:
    """Recall section-relevant context from FINDINGS + ACTIONS + MESSAGES.

    The default :func:`recall` is section-agnostic — it just pulls
    everything in priority order. For a per-section composer we
    accept that as a starting point; future work can route by section
    id (e.g., recall(memory, sections, query=spec.title) once recall
    grows a query-aware mode).
    """
    context = recall(
        memory,
        [MemoryType.FINDINGS, MemoryType.ACTIONS, MemoryType.MESSAGES],
        budget_tokens=_SECTION_RECALL_BUDGET_TOKENS,
        priority_order=[MemoryType.FINDINGS, MemoryType.ACTIONS, MemoryType.MESSAGES],
    )
    return context.text or "(no recalled context for this section)"


def _citations_as_text(spans: tuple[Span, ...]) -> str:
    """Render the citation list as a newline-delimited text block.

    Used as the ``citations`` input to :class:`SectionWriterSignature`.
    Format: ``- <source_uri> (page N): <quote_preview>`` per line.
    """
    if not spans:
        return "(no citations available)"
    lines: list[str] = []
    for span in spans:
        page = span.page or 0
        page_part = f" (page {page})" if page else ""
        preview = (span.quote or "")[:140].replace("\n", " ").strip()
        lines.append(f"- {span.source_uri}{page_part}: {preview}")
    return "\n".join(lines)
