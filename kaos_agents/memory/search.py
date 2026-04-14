"""BM25 search over memory sections.

Indexes searchable memory sections (MESSAGES, ACTIONS, DOCUMENTS, FINDINGS)
using kaos-nlp-core's BM25 Searcher and provides ranked results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.types import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A single search result from memory.

    Attributes:
        content: The memory item's content text.
        section: Which section the item came from.
        score: BM25 relevance score.
        item_id: The MemoryItem ID.
    """

    content: str
    section: MemoryType
    score: float
    item_id: str


def search_memory(
    memory: SessionMemory,
    query: str,
    *,
    sections: list[MemoryType] | None = None,
    top_k: int = 10,
) -> list[MemorySearchResult]:
    """Search across memory sections using BM25.

    Indexes all items in the requested searchable sections, runs the
    query, and returns ranked results.

    Args:
        memory: The session memory to search.
        query: Natural-language search query.
        sections: Which sections to search. Defaults to all sections
            with ``searchable=True`` (MESSAGES, ACTIONS, DOCUMENTS, FINDINGS).
        top_k: Maximum number of results to return.

    Returns:
        Ranked list of :class:`MemorySearchResult`, highest score first.
    """
    from kaos_nlp_core.search import Searcher

    # Determine which sections to search
    if sections is None:
        sections = [
            mt
            for mt in memory.section_names
            if memory.has_section(mt)
            and hasattr(memory._sections.get(mt), "config")
            and memory._sections[mt].config.searchable
        ]

    # Collect all items into records for indexing.
    # Searcher expects integer doc IDs, so we use auto-incrementing int IDs
    # and maintain a mapping back to the original MemoryItem.
    records: list[dict[str, object]] = []
    index_to_item: dict[int, tuple[str, str, MemoryType]] = {}  # idx → (item_id, content, section)

    idx = 0
    for mt in sections:
        if not memory.has_section(mt):
            continue
        items = memory.get(mt)
        for item in items:
            records.append(
                {
                    "id": idx,
                    "text": item.content,
                }
            )
            index_to_item[idx] = (item.id, item.content, mt)
            idx += 1

    if not records:
        return []

    # Build BM25 index and search
    try:
        searcher = Searcher.from_documents(records)
        hits = searcher.search(query, top_k=top_k)
    except Exception as exc:
        logger.debug("memory search failed: %s", exc, exc_info=True)
        return []

    # Convert to MemorySearchResult
    results: list[MemorySearchResult] = []
    for hit in hits:
        hit_idx = int(hit.doc_id) if hasattr(hit, "doc_id") else 0
        item_id, content, section = index_to_item.get(hit_idx, ("", "", MemoryType.MESSAGES))
        results.append(
            MemorySearchResult(
                content=content,
                section=section,
                score=hit.score,
                item_id=item_id,
            )
        )

    logger.debug(
        "memory.search: query=%r sections=%d items=%d results=%d",
        query[:50],
        len(sections),
        len(records),
        len(results),
    )

    return results
