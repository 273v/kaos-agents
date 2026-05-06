"""Query-aware context assembly.

Replaces bare ``memory.get_sections()`` with a two-path strategy:

- **Small sections** (< threshold items): FIFO ordering (unchanged behavior)
- **Large sections** (>= threshold items): BM25 search against the user's
  message to select the most relevant items within the token budget

The threshold is configurable via ``KaosAgentSettings.retrieval_threshold``
(default: 20). Below the threshold, every code path produces identical
results to the prior FIFO-only assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.search import search_memory
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.memory import MemoryItem, MemoryType

_DEFAULT_RETRIEVAL_THRESHOLD: int = KaosAgentSettings.model_fields["retrieval_threshold"].default
_DEFAULT_SEARCH_TOP_K: int = KaosAgentSettings.model_fields["retrieval_top_k"].default

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


def assemble_context(
    memory: SessionMemory,
    query: str,
    *,
    sections: list[MemoryType],
    total_budget_tokens: int,
    priority_order: list[MemoryType] | None = None,
    retrieval_threshold: int = _DEFAULT_RETRIEVAL_THRESHOLD,
    search_top_k: int = _DEFAULT_SEARCH_TOP_K,
) -> dict[MemoryType, list[MemoryItem]]:
    """Assemble context from memory sections, using BM25 for large sections.

    For each requested section:
    - If the section has fewer than ``retrieval_threshold`` items, uses
      the standard FIFO path (``memory.get()``).
    - If the section has >= ``retrieval_threshold`` items, runs BM25
      search against ``query`` to select the most relevant items.

    After per-section collection, applies the same total-budget trimming
    as ``SessionMemory.get_sections()`` (lowest-priority sections trimmed
    first, oldest-first within a section).

    Args:
        memory: The session memory to assemble context from.
        query: The user's message (used for BM25 relevance ranking).
        sections: Which memory sections to include.
        total_budget_tokens: Maximum total tokens across all sections.
        priority_order: Sections listed first are highest priority (trimmed last).
        retrieval_threshold: Section item count that triggers BM25 retrieval.
        search_top_k: Maximum items to retrieve via BM25 per section.

    Returns:
        dict mapping MemoryType to list of MemoryItem, same shape as
        ``SessionMemory.get_sections()``.
    """
    logger.debug(
        "assemble.start: query_len=%d sections=%d budget=%d threshold=%d",
        len(query),
        len(sections),
        total_budget_tokens,
        retrieval_threshold,
    )

    raw: dict[MemoryType, list[MemoryItem]] = {}
    section_tokens: dict[MemoryType, int] = {}
    bm25_sections: list[MemoryType] = []

    for mt in sections:
        if not memory.has_section(mt):
            continue

        item_count = memory.section_item_count(mt)

        if item_count >= retrieval_threshold:
            bm25_sections.append(mt)
            logger.debug(
                "assemble.section_bm25: section=%s items=%d (>= threshold %d)",
                mt.value,
                item_count,
                retrieval_threshold,
            )
        else:
            items = memory.get(mt)
            raw[mt] = items
            section_tokens[mt] = sum(item.token_count for item in items)
            logger.debug(
                "assemble.section_fifo: section=%s items=%d tokens=%d",
                mt.value,
                len(items),
                section_tokens[mt],
            )

    if bm25_sections:
        _retrieve_from_sections(memory, query, bm25_sections, search_top_k, raw, section_tokens)

    total = sum(section_tokens.values())
    if total <= total_budget_tokens:
        logger.debug(
            "assemble.complete: total_tokens=%d (within budget %d), no trimming",
            total,
            total_budget_tokens,
        )
        return raw

    logger.debug(
        "assemble.trim_needed: total_tokens=%d exceeds budget=%d by %d",
        total,
        total_budget_tokens,
        total - total_budget_tokens,
    )
    _trim_to_budget(raw, section_tokens, total_budget_tokens, total, priority_order)
    return raw


def _retrieve_from_sections(
    memory: SessionMemory,
    query: str,
    bm25_sections: list[MemoryType],
    search_top_k: int,
    raw: dict[MemoryType, list[MemoryItem]],
    section_tokens: dict[MemoryType, int],
) -> None:
    """Run BM25 search on large sections and populate raw + section_tokens."""
    results = search_memory(memory, query, sections=bm25_sections, top_k=search_top_k)

    per_section_ids: dict[MemoryType, set[str]] = {mt: set() for mt in bm25_sections}
    for r in results:
        if r.section in per_section_ids:
            per_section_ids[r.section].add(r.item_id)

    for mt in bm25_sections:
        ids = per_section_ids[mt]
        items = memory.get_by_ids(mt, ids) if ids else []
        raw[mt] = items
        section_tokens[mt] = sum(item.token_count for item in items)

    logger.debug(
        "assemble.bm25: query=%r sections=%s total_hits=%d",
        query[:50],
        [mt.value for mt in bm25_sections],
        len(results),
    )


def _trim_to_budget(
    raw: dict[MemoryType, list[MemoryItem]],
    section_tokens: dict[MemoryType, int],
    total_budget_tokens: int,
    total: int,
    priority_order: list[MemoryType] | None,
) -> None:
    """Trim sections to fit within total budget, lowest-priority first."""
    if priority_order:
        ordered = [mt for mt in priority_order if mt in raw]
        unordered = [mt for mt in raw if mt not in ordered]
        trim_order = unordered + list(reversed(ordered))
    else:
        trim_order = list(raw.keys())

    excess = total - total_budget_tokens
    for mt in trim_order:
        if excess <= 0:
            break
        items = raw[mt]
        while items and excess > 0:
            removed = items.pop(0)
            freed = removed.token_count
            section_tokens[mt] -= freed
            excess -= freed
