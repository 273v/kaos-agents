"""Corpus triage — narrow a large document corpus before planning.

When the DOCUMENTS section has many items, uses BM25 search to select
the most relevant documents for a given goal. Returns URIs and a context
summary for injection into planning prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.search import search_memory
from kaos_agents.memory.types import MemoryType
from kaos_agents.settings import KaosAgentSettings

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)

_DEFAULT_RETRIEVAL_THRESHOLD: int = KaosAgentSettings.model_fields["retrieval_threshold"].default


@dataclass(frozen=True, slots=True)
class TriageResult:
    """Result of corpus triage."""

    selected_item_ids: tuple[str, ...]
    selected_uris: tuple[str, ...]
    context_summary: str
    total_documents: int
    selected_count: int


def triage_corpus(
    memory: SessionMemory,
    goal: str,
    *,
    max_selected: int = 20,
    threshold: int = _DEFAULT_RETRIEVAL_THRESHOLD,
) -> TriageResult | None:
    """Narrow a large DOCUMENTS section to the most relevant subset.

    If the DOCUMENTS section has fewer than ``threshold`` items, returns
    ``None`` (no triage needed — all documents fit in context).

    Otherwise, runs BM25 search against ``goal`` and returns the top
    ``max_selected`` documents with a context summary suitable for
    injection into planning prompts.

    Args:
        memory: Session memory with DOCUMENTS section.
        goal: The planning goal to triage against.
        max_selected: Maximum documents to select.
        threshold: Minimum document count to trigger triage.

    Returns:
        ``TriageResult`` with selected documents, or ``None`` if no triage needed.
    """
    if not memory.has_section(MemoryType.DOCUMENTS):
        return None

    total = memory.section_item_count(MemoryType.DOCUMENTS)
    if total < threshold:
        return None

    results = search_memory(memory, goal, sections=[MemoryType.DOCUMENTS], top_k=max_selected)

    if not results:
        return None

    item_ids: list[str] = []
    uris: list[str] = []
    for r in results:
        item_ids.append(r.item_id)
        items = memory.get_by_ids(MemoryType.DOCUMENTS, {r.item_id})
        if items:
            uri = (items[0].metadata or {}).get("uri", r.item_id)
            uris.append(str(uri))

    uri_list = ", ".join(uris[:10])
    if len(uris) > 10:
        uri_list += f", ... and {len(uris) - 10} more"

    summary = (
        f"CORPUS TRIAGE: Selected {len(results)} of {total} documents "
        f"most relevant to the goal.\n"
        f"Selected documents: {uri_list}\n"
        f"Plan over these documents only. Do not attempt to process all {total}."
    )

    logger.debug(
        "triage: selected %d of %d documents for goal=%r",
        len(results),
        total,
        goal[:50],
    )

    return TriageResult(
        selected_item_ids=tuple(item_ids),
        selected_uris=tuple(uris),
        context_summary=summary,
        total_documents=total,
        selected_count=len(results),
    )
