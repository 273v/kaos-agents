"""Apply a :class:`RetrievalPlanResult` to narrow a corpus.

Pure-mechanical strategy dispatch. No LLM calls. Composes existing
primitives:

* :func:`kaos_content.search.search_document` for TOKEN / NGRAM /
  EMBEDDING (BM25 over single-doc views; embedding sentence search).
* :func:`kaos_agents.context.triage.triage_corpus` for doc-level BM25
  narrowing (returns the top-K docs by BM25 score against a query).

Returns ``(narrowed_view, telemetry_record)``. The telemetry record
feeds the ``Span(SUBAGENT, "research.retrieval_apply")`` attributes
so operators can see what got dropped per turn.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_agents.patterns.retrieval.types import (
    RetrievalApplyResult,
    RetrievalPlanResult,
    RetrievalStrategy,
)

logger = get_logger(__name__)


def _narrow_document_from_search_results(
    document: Any,
    block_refs: set[str],
) -> Any:
    """Return a ContentDocument containing only paragraphs whose
    ``#/body/N`` JSON-pointer ref appears in ``block_refs``.

    Tiny AST walker. Stays here rather than in kaos-content because
    it's specific to the planner-applier contract (kaos-content
    searches return ``block_ref`` strings; we materialise a narrowed
    document from the set of refs that survived narrowing).
    """
    from kaos_content.model.document import ContentDocument

    if not block_refs:
        return document

    kept_blocks: list[Any] = []
    for idx, block in enumerate(document.body):
        ref = f"#/body/{idx}"
        if ref in block_refs:
            kept_blocks.append(block)

    if not kept_blocks:
        # Nothing matched — return original to avoid an empty view
        # (the grounding step refuses on empty views, which is the
        # wrong signal when the issue is narrowing-was-too-aggressive).
        logger.warning(
            "retrieval_apply: zero block_refs survived narrowing — "
            "falling back to full document (refs=%d)",
            len(block_refs),
        )
        return document

    return ContentDocument(body=tuple(kept_blocks))


async def apply_retrieval_plan(
    plan: RetrievalPlanResult,
    *,
    full_view: Any,
    full_document: Any,
    docs_items: list[Any],
    memory: Any,
    sentence_segmenter: Any,
) -> tuple[Any, RetrievalApplyResult]:
    """Narrow the corpus per ``plan``. Returns ``(narrowed_view, telemetry)``.

    Args:
        plan: The planner's typed output.
        full_view: The full :class:`DocumentView` over the merged
            corpus AST (built upstream by the VFS-byte resolver).
        full_document: The underlying ``ContentDocument`` for the
            merged corpus. Required for ``kaos_content.search``
            entry points (which take the document, not the view).
        docs_items: The raw DOCUMENTS-section items (for BM25
            doc-level triage which uses ``triage_corpus``).
        memory: The session's :class:`SessionMemory` (used by
            ``triage_corpus``).
        sentence_segmenter: Punkt tokenizer reused to build the
            narrowed view (mirrors how the caller built ``full_view``).

    Returns:
        ``(view, telemetry)`` where ``view`` is the narrowed
        :class:`DocumentView` ready for FindingsAgent, and ``telemetry``
        is the :class:`RetrievalApplyResult` for span attributes.
    """
    from kaos_content.views.document_view import DocumentView

    strategy = plan.strategy
    total = len(docs_items)

    if strategy is RetrievalStrategy.NONE:
        return full_view, RetrievalApplyResult(strategy=strategy, kept=total, dropped=0)

    if strategy in (RetrievalStrategy.TOKEN, RetrievalStrategy.NGRAM):
        # Lexical narrowing via BM25 over the probe vocabulary. Joining
        # probes as one query gets BM25 to treat each as a separate
        # term — equivalent to OR'ing single-term selectors. The query
        # field is the fallback when the probe list is empty.
        probes = plan.tokens if strategy is RetrievalStrategy.TOKEN else plan.ngrams
        query = " ".join(probes).strip()
        if not query:
            query = plan.query.strip()
        if not query:
            logger.warning(
                "retrieval_apply: %s strategy with empty probes + empty query — "
                "falling back to NONE",
                strategy.value,
            )
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="empty probes",
            )

        return _apply_search_document(
            strategy=strategy,
            query=query,
            retrieval_mode="bm25",
            full_document=full_document,
            full_view=full_view,
            top_k=plan.top_k,
            sentence_segmenter=sentence_segmenter,
            docs_total=total,
        )

    if strategy is RetrievalStrategy.BM25:
        # Doc-level BM25 narrowing against the PARSED merged corpus
        # (not the in-memory DOCUMENTS section's item.content, which
        # may be headline-only for binary docs the SPA hasn't
        # pre-extracted). Splits full_document on the filename-marker
        # paragraphs the resolver inserted, builds BM25 over each
        # doc's parsed text, ranks against the query, keeps top_k.
        from kaos_content.model.document import ContentDocument
        from kaos_nlp_core.search import Searcher

        query = plan.query.strip() or " ".join(plan.tokens) or " ".join(plan.ngrams)
        if not query:
            logger.warning("retrieval_apply: BM25 strategy with empty query — falling back to NONE")
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="empty query",
            )

        doc_chunks = _split_merged_document_by_marker(full_document)
        if not doc_chunks:
            logger.warning(
                "retrieval_apply: no filename markers found in merged doc — falling back to NONE"
            )
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="no markers in merged doc",
            )

        records = [
            {"id": idx, "text": chunk["text"]}
            for idx, chunk in enumerate(doc_chunks)
            if chunk["text"].strip()
        ]
        if not records:
            logger.warning("retrieval_apply: all parsed doc chunks empty — falling back to NONE")
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="empty parsed chunks",
            )

        searcher = Searcher.from_documents(records)
        hits = searcher.search(query, top_k=plan.top_k)
        if not hits:
            logger.debug(
                "retrieval_apply: BM25 ranked zero docs for query=%r — keeping full view",
                query[:80],
            )
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="bm25 zero hits",
            )

        selected_idxs = {int(hit.doc_id) for hit in hits}
        narrowed_blocks: list[Any] = []
        for idx, chunk in enumerate(doc_chunks):
            if idx in selected_idxs:
                narrowed_blocks.extend(chunk["blocks"])

        if not narrowed_blocks:
            logger.warning(
                "retrieval_apply: BM25 selected %d docs but no blocks "
                "materialized — falling back to NONE",
                len(selected_idxs),
            )
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="bm25 selected docs but no blocks",
            )

        narrowed_doc = ContentDocument(body=tuple(narrowed_blocks))
        narrowed_view = DocumentView(narrowed_doc, sentence_segmenter=sentence_segmenter)
        return narrowed_view, RetrievalApplyResult(
            strategy=RetrievalStrategy.BM25,
            kept=len(selected_idxs),
            dropped=max(0, len(doc_chunks) - len(selected_idxs)),
        )

    if strategy is RetrievalStrategy.EMBEDDING:
        query = plan.query.strip() or " ".join(plan.tokens) or " ".join(plan.ngrams)
        if not query:
            logger.warning(
                "retrieval_apply: EMBEDDING strategy with empty query — falling back to NONE"
            )
            return full_view, RetrievalApplyResult(
                strategy=RetrievalStrategy.NONE,
                kept=total,
                dropped=0,
                fallback_reason="empty query",
            )
        try:
            return _apply_search_document(
                strategy=strategy,
                query=query,
                retrieval_mode="embeddings",
                full_document=full_document,
                full_view=full_view,
                top_k=plan.top_k,
                sentence_segmenter=sentence_segmenter,
                docs_total=total,
            )
        except ImportError:
            # kaos-nlp-transformers not installed — degrade to BM25
            # over the same query.
            logger.debug(
                "retrieval_apply: EMBEDDING requested but kaos-nlp-transformers "
                "missing — falling back to BM25"
            )
            view, telem = _apply_search_document(
                strategy=RetrievalStrategy.BM25,
                query=query,
                retrieval_mode="bm25",
                full_document=full_document,
                full_view=full_view,
                top_k=plan.top_k,
                sentence_segmenter=sentence_segmenter,
                docs_total=total,
            )
            return view, RetrievalApplyResult(
                strategy=telem.strategy,
                kept=telem.kept,
                dropped=telem.dropped,
                fallback_reason="kaos-nlp-transformers not installed",
            )

    # Unknown strategy — degrade to NONE.
    logger.warning(
        "retrieval_apply: unhandled strategy %r — falling back to NONE",
        strategy,
    )
    return full_view, RetrievalApplyResult(
        strategy=RetrievalStrategy.NONE,
        kept=total,
        dropped=0,
        fallback_reason=f"unhandled strategy {strategy!r}",
    )


def _apply_search_document(
    *,
    strategy: RetrievalStrategy,
    query: str,
    retrieval_mode: str,
    full_document: Any,
    full_view: Any,
    top_k: int,
    sentence_segmenter: Any,
    docs_total: int,
) -> tuple[Any, RetrievalApplyResult]:
    """Run ``kaos_content.search.search_document`` and rebuild a view
    from the surviving block_refs.

    Shared by TOKEN / NGRAM / EMBEDDING paths — they only differ on
    ``query`` shape (probes joined vs free text) and ``retrieval_mode``
    (BM25 vs embeddings).
    """
    from kaos_content.search import search_document
    from kaos_content.views.document_view import DocumentView

    try:
        results = search_document(
            full_document,
            query=query,
            top_k=top_k,
            level="sentence",
            # retrieval_mode is "bm25" / "embeddings" — both valid Literal values
            # for kaos_content.search.RetrievalMode, but ty narrows from str.
            retrieval=retrieval_mode,  # ty: ignore[invalid-argument-type]
        )
    except ImportError:
        # Re-raise so the caller can decide whether to degrade (EMBEDDING
        # falls back to BM25; BM25 / TOKEN / NGRAM should never raise here).
        raise
    except Exception:
        logger.exception(
            "retrieval_apply: search_document raised on strategy=%s — falling back to NONE",
            strategy.value,
        )
        return full_view, RetrievalApplyResult(
            strategy=RetrievalStrategy.NONE,
            kept=docs_total,
            dropped=0,
            fallback_reason="search_document raised",
        )

    hits = list(results.results)
    if not hits:
        logger.debug(
            "retrieval_apply: %s yielded zero hits for query=%r — keeping full view",
            strategy.value,
            query[:80],
        )
        return full_view, RetrievalApplyResult(
            strategy=RetrievalStrategy.NONE,
            kept=docs_total,
            dropped=0,
            fallback_reason="zero search hits",
        )

    block_refs = {hit.block_ref for hit in hits if hit.block_ref}
    narrowed_doc = _narrow_document_from_search_results(full_document, block_refs)
    narrowed_view = DocumentView(narrowed_doc, sentence_segmenter=sentence_segmenter)
    kept = len(narrowed_doc.body)
    dropped = max(0, len(full_document.body) - kept)
    return narrowed_view, RetrievalApplyResult(strategy=strategy, kept=kept, dropped=dropped)


def _block_text(block: Any) -> str:
    """Best-effort concatenation of inline text from a block.

    Walks ``block.children`` looking for ``.value`` attributes; ignores
    inline images and other non-textual leaves. Used by the BM25
    chunk-builder to extract per-document text from the merged
    ContentDocument.
    """
    children = getattr(block, "children", None)
    if not children:
        return ""
    parts: list[str] = []
    for child in children:
        value = getattr(child, "value", None)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _split_merged_document_by_marker(document: Any) -> list[dict[str, Any]]:
    """Split the merged ContentDocument into per-doc chunks.

    The resolver walks the DOCUMENTS section and emits one
    ``=== {filename} ===`` marker paragraph before each document's
    body blocks. This walker re-splits on those markers, returning
    one chunk per source document with its blocks and concatenated
    text body. Used by the BM25 strategy to rank per-doc relevance
    against the parsed corpus.

    Returns a list of ``{"filename": str, "blocks": list[Block],
    "text": str}`` records. The marker paragraph IS included in
    ``blocks`` so the rebuilt narrowed ContentDocument preserves the
    filename labels that the synthesis prompt cites.
    """
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for block in document.body:
        marker = _extract_filename_marker(block)
        if marker is not None:
            if current is not None:
                chunks.append(current)
            current = {"filename": marker, "blocks": [block], "text": ""}
        elif current is not None:
            current["blocks"].append(block)
            block_text = _block_text(block)
            if block_text:
                current["text"] = (current["text"] + " " + block_text).strip()
    if current is not None:
        chunks.append(current)
    return chunks


def _extract_filename_marker(block: Any) -> str | None:
    """Return the filename if this Paragraph is one of the ``=== {filename} ===``
    markers the corpus resolver inserts between docs. Else ``None``.

    Used by BM25 triage to walk the merged corpus AST and keep only
    the body blocks belonging to selected URIs.
    """
    children = getattr(block, "children", None)
    if not children:
        return None
    text = ""
    for child in children:
        value = getattr(child, "value", None)
        if value is not None:
            text += value
    text = text.strip()
    if text.startswith("=== ") and text.endswith(" ==="):
        return text[4:-4].strip()
    return None


__all__ = [
    "apply_retrieval_plan",
]
