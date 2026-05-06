"""Pure iterators + filters over :class:`SessionMemory`.

Single-purpose generators that hide the JSON-encoded shape of memory
items behind typed Pydantic / dataclass surfaces. Composers above
this layer never touch ``MemoryItem.metadata`` dicts directly —
they call :func:`walk_findings`, :func:`walk_cells`, etc.

Three pure functions:

* :func:`walk_findings` — iterate verified ``Claim`` instances stored
  in the FINDINGS section. Rehydrates from the dict written by
  ``patterns/research.py`` (which uses ``Claim.model_dump(mode="json")``
  per the C4 audit-fix commit).
* :func:`walk_cells` — iterate ``ExtractionCell`` instances stored
  via the extract-corpus pipeline.
* :func:`dedupe_spans` — pure deduplication over a Span sequence.
  Single-purpose; used by :mod:`citations`.

Each function is independently testable, single-purpose, and side-
effect-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Literal

from kaos_content.model.extraction import ExtractionCell
from kaos_core.logging import get_logger
from kaos_llm_core.signatures.grounding import Claim, Span

from kaos_agents.memory.types import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


SpanDedupKey = Literal["source_uri", "source_uri+char_span"]


def walk_findings(memory: SessionMemory) -> Iterator[Claim]:
    """Iterate verified :class:`Claim` instances from the FINDINGS section.

    Rehydrates from the ``metadata.claim`` dict written by
    ``patterns/research.py`` (which calls ``Claim.model_dump(mode="json")``
    after the C4 audit fix). Items missing or malformed claim metadata
    are skipped with a debug-level log; this is intentional — older
    FINDINGS items written before the model_dump migration would have
    a different shape, and we'd rather skip them than crash a
    deliverable composition.

    Order: insertion order (FIFO across the section).
    """
    if not memory.has_section(MemoryType.FINDINGS):
        return
    items = memory.get(MemoryType.FINDINGS)
    for item in items:
        claim_payload = item.metadata.get("claim") if item.metadata else None
        if not isinstance(claim_payload, dict):
            logger.debug(
                "walk_findings: skipping item id=%s — no claim payload in metadata",
                item.id,
            )
            continue
        try:
            yield Claim.model_validate(claim_payload)
        except Exception as exc:
            logger.debug(
                "walk_findings: skipping item id=%s — claim validation failed: %s",
                item.id,
                exc,
            )


def walk_cells(memory: SessionMemory) -> Iterator[ExtractionCell]:
    """Iterate :class:`ExtractionCell` instances from memory.

    Cells are stored under ``metadata["extraction_cell"]`` as the
    Pydantic ``model_dump(mode="json")`` payload. Same skip-on-malformed
    contract as :func:`walk_findings` — an absent or bad payload logs
    at debug and is omitted.

    Cells could live in FINDINGS (when produced inline by an extraction
    recipe) or in a dedicated extraction section in future. Walks
    every candidate section; emits no cell twice.
    """
    candidate_sections = (MemoryType.FINDINGS,)
    for section in candidate_sections:
        if not memory.has_section(section):
            continue
        for item in memory.get(section):
            payload = item.metadata.get("extraction_cell") if item.metadata else None
            if not isinstance(payload, dict):
                continue
            try:
                yield ExtractionCell.model_validate(payload)
            except Exception as exc:
                logger.debug(
                    "walk_cells: skipping item id=%s — cell validation failed: %s",
                    item.id,
                    exc,
                )


def dedupe_spans(
    spans: Iterable[Span],
    *,
    by: SpanDedupKey = "source_uri+char_span",
) -> list[Span]:
    """Deduplicate :class:`Span` instances, preserving first-appearance order.

    Pure function — input is consumed once, output is a fresh list.
    No memory state, no I/O.

    Args:
        spans: Source spans (any iterable).
        by: Dedup discriminator. ``"source_uri"`` collapses every span
            from the same source. ``"source_uri+char_span"`` (default)
            keeps spans that occupy distinct positions even within one
            source — the production-correct choice for legal citations
            where a single document is cited multiple times.

    Returns:
        Deduplicated list in first-appearance order.
    """
    seen: set[tuple] = set()
    out: list[Span] = []
    for span in spans:
        key: tuple = (
            (span.source_uri,) if by == "source_uri" else (span.source_uri, tuple(span.char_span))
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out
