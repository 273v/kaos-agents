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
_DEFAULT_GRAPH_CONTEXT_ENABLED: bool = KaosAgentSettings.model_fields[
    "graph_context_enabled"
].default
_DEFAULT_GRAPH_MAX_NEIGHBORS: int = KaosAgentSettings.model_fields[
    "graph_context_max_neighbors"
].default
_DEFAULT_GRAPH_MAX_FINDINGS: int = KaosAgentSettings.model_fields[
    "graph_context_max_findings"
].default

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
    graph_context_enabled: bool = _DEFAULT_GRAPH_CONTEXT_ENABLED,
    graph_max_neighbors: int = _DEFAULT_GRAPH_MAX_NEIGHBORS,
    graph_max_findings: int = _DEFAULT_GRAPH_MAX_FINDINGS,
    pin_corpus_handles: bool | None = None,
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
        pin_corpus_handles: WU-G.2 / #352 — when True (or ``None`` and
            ``memory.corpus_ever_attached`` evaluates True), the
            DOCUMENTS section is guaranteed to retain at least a
            compact handle line (one filename per attached doc) in the
            assembled context, even if the total-budget trim phase
            would otherwise drop every DOCUMENTS item. This is the
            "stable view" the agent loop needs so a follow-up like
            ``"summarize that"`` after the first turn's attached PDF
            scrolls out of MESSAGES still signals "corpus reachable
            via search_memory" to the downstream LLM.

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

    # Track 3 chunk B4 — graph-aware augmentation. After all retrieval
    # is done, walk the per-session knowledge graph 1 hop from each
    # selected FINDING to surface its provenance (cited docs, supporting
    # findings, producing tool calls). The new items land under
    # MemoryType.GRAPH so the existing budget + binding plumbing
    # handles them like any other section.
    if graph_context_enabled and MemoryType.FINDINGS in raw:
        from kaos_agents.context.graph_expand import expand_with_graph

        graph_items = expand_with_graph(
            memory,
            raw,
            max_neighbors_per_finding=graph_max_neighbors,
            max_findings=graph_max_findings,
        )
        if graph_items:
            raw[MemoryType.GRAPH] = graph_items
            section_tokens[MemoryType.GRAPH] = sum(it.token_count for it in graph_items)
            logger.debug(
                "assemble.graph_expand: %d findings → %d GRAPH items (%d tokens)",
                len(raw.get(MemoryType.FINDINGS, [])),
                len(graph_items),
                section_tokens[MemoryType.GRAPH],
            )

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
    # WU-G.2 / #352 — capture the pre-trim DOCUMENTS slot so we can
    # reconstruct a compact handle line if the trim phase wipes the
    # section. We snapshot here (not inside the trim helper) so the
    # handle reflects the documents actually present this turn, not
    # the post-trim residue.
    pre_trim_documents: list[MemoryItem] = list(raw.get(MemoryType.DOCUMENTS, []))

    _trim_to_budget(raw, section_tokens, total_budget_tokens, total, priority_order)

    # WU-G.2 / #352 — corpus-handle retention. Two conditions must be
    # true to inject the handle: (1) the session signal says the
    # corpus has been attached at some point (``memory.corpus_ever_
    # attached``, sticky once set); (2) the trim phase dropped every
    # DOCUMENTS item (otherwise the agent already has at least one
    # body the LLM can see). The handle is a single small line —
    # ``[N attached documents: file1, file2, ...]`` — that bypasses
    # the trim because it's added AFTER the trim ran. Caps at 12
    # filenames so an exceptionally large corpus doesn't blow the
    # context budget; the LLM follows up via ``search_memory`` to
    # disambiguate.
    if pin_corpus_handles is None:
        pin_corpus_handles = memory.corpus_ever_attached
    if pin_corpus_handles and pre_trim_documents and not raw.get(MemoryType.DOCUMENTS):
        handle = _build_corpus_handle_item(pre_trim_documents)
        if handle is not None:
            raw[MemoryType.DOCUMENTS] = [handle]
            logger.debug(
                "assemble.corpus_handle: documents=%d → 1 handle item (%d tokens)",
                len(pre_trim_documents),
                handle.token_count,
            )

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


# Cap on filenames we surface in the synthetic handle item. A real
# Diligence-room corpus can be hundreds of files; surfacing 12 is the
# minimum needed for the LLM to know "look, there are docs here" while
# staying well under any reasonable token budget. Past 12 the model
# follows up with ``search_memory`` to disambiguate anyway.
_MAX_HANDLE_FILENAMES: int = 12


def _build_corpus_handle_item(documents: list[MemoryItem]) -> MemoryItem | None:
    """Build the WU-G.2 / #352 stable-view handle for the DOCUMENTS section.

    Returns a single :class:`MemoryItem` whose content lists up to
    :data:`_MAX_HANDLE_FILENAMES` filenames drawn from the original
    DOCUMENTS items' metadata (``filename`` / ``uri`` / ``source_uri``
    / ``name`` / ``source``). The handle is tagged
    ``corpus_handle`` so downstream consumers can identify and
    de-duplicate it.

    Returns ``None`` when ``documents`` is empty (caller already
    guards but the helper stays defensive).
    """
    if not documents:
        return None
    names: list[str] = []
    seen: set[str] = set()
    for item in documents:
        metadata = item.metadata or {}
        anchor = (
            metadata.get("filename")
            or metadata.get("uri")
            or metadata.get("source_uri")
            or metadata.get("name")
            or metadata.get("source")
        )
        if not anchor:
            # Last-resort: take the first non-empty line of content.
            content = item.content or ""
            first = content.strip().splitlines()[:1]
            anchor = first[0] if first else ""
        if not anchor:
            continue
        if anchor in seen:
            continue
        seen.add(anchor)
        names.append(str(anchor))
        if len(names) >= _MAX_HANDLE_FILENAMES:
            break

    total = len(documents)
    if not names:
        body = (
            f"[{total} attached document(s) in session memory — "
            f"use the search_memory tool with section='documents' to "
            f"retrieve specific content.]"
        )
    else:
        suffix = ""
        if total > len(names):
            suffix = f" (+ {total - len(names)} more)"
        body = (
            f"[{total} attached document(s) in session memory: "
            f"{', '.join(names)}{suffix}. Use the search_memory tool "
            f"with section='documents' to retrieve specific content.]"
        )

    from kaos_agents.types.memory import create_item

    return create_item(
        MemoryType.DOCUMENTS,
        body,
        tags=("corpus_handle",),
        metadata={"corpus_handle": True, "total_documents": total},
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
