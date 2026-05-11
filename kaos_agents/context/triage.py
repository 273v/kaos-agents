"""Corpus triage — narrow a large document corpus before planning.

When the DOCUMENTS section has many items, uses BM25 search to select
the most relevant documents for a given goal. Returns URIs and a
context summary for injection into planning prompts.

Two BM25 paths (K5):

1. **Summary-aware path** (preferred when available). When every
   DOCUMENTS item carries a ``summary_text`` field in its metadata
   (typically populated upstream by the K4 corpus-summarize tool),
   BM25 is run over those shorter, distilled texts. 50-100x faster
   on large corpora; comparable narrowing quality.
2. **Full-text path** (fallback). When any item lacks
   ``summary_text``, falls back to the existing
   :func:`search_memory` BM25 over each item's full ``content``.
   Identical behaviour to pre-K5 triage.

The choice is deterministic and zero-config: if you've pre-summarised,
you get the fast path; if not, you get the slow path. No flag, no
behavioural surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.search import search_memory
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)

_DEFAULT_RETRIEVAL_THRESHOLD: int = KaosAgentSettings.model_fields["retrieval_threshold"].default

_SUMMARY_METADATA_KEY = "summary_text"
"""Metadata key on a DOCUMENTS item carrying its pre-built summary text.

The agent (or the K4 ``kaos-content-corpus-summarize`` MCP tool when
wired into the load flow) populates this when summarizing a document
into the session. The value is the concatenated
(head_tokens + top_ngrams + bottom_ngrams) of a
:class:`~kaos_content.model.summary.DocumentSummary` — i.e. what
``kaos_content.tools._summary_search_text`` produces. We keep it as a
plain string (not a typed payload) so the memory layer doesn't need
to know about kaos-content types — round-trips through the existing
metadata dict without schema changes."""


@dataclass(frozen=True, slots=True)
class TriageResult:
    """Result of corpus triage."""

    selected_item_ids: tuple[str, ...]
    selected_uris: tuple[str, ...]
    context_summary: str
    total_documents: int
    selected_count: int
    used_summary_index: bool = False
    """True when the summary-aware BM25 path was used. Surfaces in the
    context summary so operators can verify the fast path engaged."""


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

    Two BM25 paths (see module docstring). The summary-aware path
    engages automatically when every DOCUMENTS item carries a
    ``summary_text`` in its metadata.

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

    # Try the summary-aware path first. It engages only when every
    # item carries a summary_text — partial coverage is treated as
    # "not yet ready" and we fall back to the existing path so the
    # caller never silently mixes signals from two different indexes.
    item_ids: list[str]
    uris: list[str]
    used_summary_index = False
    summary_results = _summary_aware_search(memory, goal, top_k=max_selected)
    if summary_results is not None:
        item_ids, uris = summary_results
        used_summary_index = True
    else:
        results = search_memory(memory, goal, sections=[MemoryType.DOCUMENTS], top_k=max_selected)
        if not results:
            return None
        item_ids = []
        uris = []
        for r in results:
            item_ids.append(r.item_id)
            items = memory.get_by_ids(MemoryType.DOCUMENTS, {r.item_id})
            if items:
                uri = (items[0].metadata or {}).get("uri", r.item_id)
                uris.append(str(uri))

    uri_list = ", ".join(uris[:10])
    if len(uris) > 10:
        uri_list += f", ... and {len(uris) - 10} more"

    path_label = "summary-index" if used_summary_index else "full-text"
    summary = (
        f"CORPUS TRIAGE ({path_label}): Selected {len(item_ids)} of "
        f"{total} documents most relevant to the goal.\n"
        f"Selected documents: {uri_list}\n"
        f"Plan over these documents only. Do not attempt to process all {total}."
    )

    logger.debug(
        "triage: selected %d of %d documents for goal=%r via %s",
        len(item_ids),
        total,
        goal[:50],
        path_label,
    )

    return TriageResult(
        selected_item_ids=tuple(item_ids),
        selected_uris=tuple(uris),
        context_summary=summary,
        total_documents=total,
        selected_count=len(item_ids),
        used_summary_index=used_summary_index,
    )


def _summary_aware_search(
    memory: SessionMemory,
    goal: str,
    *,
    top_k: int,
) -> tuple[list[str], list[str]] | None:
    """BM25 over ``summary_text`` metadata when every item has one.

    Returns ``(item_ids, uris)`` ordered by BM25 score, or ``None``
    when at least one item lacks a ``summary_text`` (fall back to the
    full-text path in that case).
    """
    items = memory.get(MemoryType.DOCUMENTS)
    if not items:
        return None

    # All-or-nothing: if any item lacks summary_text, fall back so the
    # caller never mixes the two BM25 corpora.
    summary_texts: list[tuple[str, str, str]] = []  # (item_id, uri, summary_text)
    for item in items:
        metadata = item.metadata or {}
        summary_text = metadata.get(_SUMMARY_METADATA_KEY)
        if not summary_text or not isinstance(summary_text, str):
            return None
        uri = str(metadata.get("uri") or item.id)
        summary_texts.append((item.id, uri, summary_text))

    if not summary_texts:
        return None

    # Build BM25 corpus + run search.
    from kaos_nlp_core.search import Searcher

    records = [{"id": i, "text": st[2]} for i, st in enumerate(summary_texts)]
    searcher = Searcher.from_documents(records)
    hits = searcher.search(goal, top_k=top_k)
    if not hits:
        return None

    item_ids: list[str] = []
    uris: list[str] = []
    for hit in hits:
        slot = int(hit.doc_id)
        item_id, uri, _ = summary_texts[slot]
        item_ids.append(item_id)
        uris.append(uri)
    return item_ids, uris
