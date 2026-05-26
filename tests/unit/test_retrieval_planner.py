"""Unit tests for the retrieval planner + applier (Phase 1 surface).

Offline, deterministic. The LLM path
(``LLMRetrievalPlanner.plan(question, corpus_summary)``) requires
a real provider — those assertions live in
``tests/integration/test_findings_dispatch_live.py``.

This module exercises:

* :class:`RetrievalStrategy` parse + value-type construction.
* The applier (mechanical narrowing) on each strategy:
  ``NONE``, ``TOKEN``, ``NGRAM``, ``BM25``, ``EMBEDDING`` (degrades
  to BM25 without the transformers extra).
* Telemetry shape (kept / dropped / fallback_reason).
"""

from __future__ import annotations

import pytest
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.views.document_view import DocumentView
from kaos_nlp_core._defaults import get_default_punkt_tokenizer

from kaos_agents.patterns.retrieval import (
    RetrievalApplyResult,
    RetrievalPlanResult,
    RetrievalStrategy,
    apply_retrieval_plan,
)
from kaos_agents.patterns.retrieval.planner import _coerce_strategy

# ─── Fixtures ──────────────────────────────────────────────────────


def _make_text_block(text: str) -> Paragraph:
    return Paragraph(children=(Text(value=text),))


def _build_fixture_corpus() -> tuple[ContentDocument, DocumentView, list, object]:
    """Three-doc merged ContentDocument with marker paragraphs.

    Mirrors the resolver's output: ``=== filename ===`` markers
    between per-doc bodies. Used to exercise BM25 doc-level
    narrowing without invoking the resolver.
    """
    blocks = [
        _make_text_block("=== alpha.html ==="),
        _make_text_block("Alpha Corp raised $42M Series B funding from Greylock."),
        _make_text_block("Filed paperwork with Delaware secretary of state."),
        _make_text_block("=== bravo.html ==="),
        _make_text_block("Bravo Inc product launch date set for 2026-11-01."),
        _make_text_block("Engineering team completed final security review."),
        _make_text_block("=== charlie.html ==="),
        _make_text_block("Charlie LLC reports 17 enrolled customers across pilot."),
        _make_text_block("Renewal rate above industry norm at 92%."),
    ]
    document = ContentDocument(body=tuple(blocks))
    segmenter = get_default_punkt_tokenizer()
    view = DocumentView(document, sentence_segmenter=segmenter)

    # Fake docs_items shape — only the metadata the applier reads.
    class _Item:
        def __init__(self, filename: str) -> None:
            self.metadata = {
                "filename": filename,
                "uri": f"file:{filename}",
                "mime_type": "text/html",
                "vfs_path": f"sessions/test/files/{filename}",
            }
            self.content = ""
            self.id = filename

    docs_items = [_Item("alpha.html"), _Item("bravo.html"), _Item("charlie.html")]
    return document, view, docs_items, segmenter


# ─── Strategy coercion ─────────────────────────────────────────────


class TestCoerceStrategy:
    """Best-effort string-to-enum semantics."""

    def test_exact_lowercase(self) -> None:
        assert _coerce_strategy("none") is RetrievalStrategy.NONE
        assert _coerce_strategy("token") is RetrievalStrategy.TOKEN
        assert _coerce_strategy("ngram") is RetrievalStrategy.NGRAM
        assert _coerce_strategy("bm25") is RetrievalStrategy.BM25
        assert _coerce_strategy("embed") is RetrievalStrategy.EMBEDDING

    def test_uppercase_trimmed(self) -> None:
        assert _coerce_strategy("  BM25 ") is RetrievalStrategy.BM25
        assert _coerce_strategy("None") is RetrievalStrategy.NONE

    def test_unknown_falls_back_to_none(self) -> None:
        assert _coerce_strategy("vector") is RetrievalStrategy.NONE
        assert _coerce_strategy("") is RetrievalStrategy.NONE


# ─── Value types ────────────────────────────────────────────────────


class TestValueTypes:
    """Construction + default semantics."""

    def test_plan_result_defaults(self) -> None:
        plan = RetrievalPlanResult(strategy=RetrievalStrategy.NONE)
        assert plan.tokens == ()
        assert plan.ngrams == ()
        assert plan.query == ""
        assert plan.top_k == 20
        assert plan.usage is None

    def test_apply_result_defaults(self) -> None:
        result = RetrievalApplyResult(strategy=RetrievalStrategy.NONE, kept=3)
        assert result.dropped == 0
        assert result.fallback_reason == ""
        assert result.warnings == ()


# ─── Applier ────────────────────────────────────────────────────────


class TestApplier:
    """Mechanical narrowing per strategy — no LLM calls."""

    @pytest.mark.asyncio
    async def test_none_passes_view_through(self) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        plan = RetrievalPlanResult(strategy=RetrievalStrategy.NONE)

        out_view, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )

        assert out_view is view
        assert result.strategy is RetrievalStrategy.NONE
        assert result.kept == len(items)
        assert result.dropped == 0

    @pytest.mark.asyncio
    async def test_token_narrows_via_search_document(self) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        plan = RetrievalPlanResult(
            strategy=RetrievalStrategy.TOKEN,
            tokens=("Bravo", "launch"),
            top_k=5,
        )

        out_view, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )

        assert result.strategy is RetrievalStrategy.TOKEN
        # Bravo-related sentence should be in the narrowed view.
        narrowed_text = " ".join(
            child.value
            for block in out_view.document.body
            for child in block.children
            if hasattr(child, "value")
        )
        assert "Bravo" in narrowed_text or "launch" in narrowed_text.lower()

    @pytest.mark.asyncio
    async def test_bm25_narrows_by_doc_via_parsed_corpus(self) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        plan = RetrievalPlanResult(
            strategy=RetrievalStrategy.BM25,
            query="Alpha funding round Greylock",
            top_k=1,
        )

        out_view, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )

        assert result.strategy is RetrievalStrategy.BM25
        assert result.kept == 1
        assert result.dropped == 2
        # Alpha doc should be the survivor.
        narrowed_text = " ".join(
            child.value
            for block in out_view.document.body
            for child in block.children
            if hasattr(child, "value")
        )
        assert "Alpha" in narrowed_text
        assert "$42M" in narrowed_text
        assert "Bravo" not in narrowed_text
        assert "Charlie" not in narrowed_text

    @pytest.mark.asyncio
    async def test_bm25_empty_query_falls_back_to_none(self) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        plan = RetrievalPlanResult(
            strategy=RetrievalStrategy.BM25,
            query="",
            top_k=5,
        )

        out_view, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )

        assert out_view is view
        assert result.strategy is RetrievalStrategy.NONE
        assert result.fallback_reason == "empty query"

    @pytest.mark.asyncio
    async def test_ngram_with_empty_probes_falls_back_when_no_query(
        self,
    ) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        plan = RetrievalPlanResult(
            strategy=RetrievalStrategy.NGRAM,
            ngrams=(),
            query="",
            top_k=5,
        )

        _, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )

        # Empty probes + empty query → no narrowing input → fall back to NONE.
        assert result.strategy is RetrievalStrategy.NONE
        assert result.fallback_reason == "empty probes"

    @pytest.mark.asyncio
    async def test_unknown_strategy_falls_back(self) -> None:
        document, view, items, segmenter = _build_fixture_corpus()
        # _coerce_strategy would have rejected this — simulate a stale enum.
        plan = RetrievalPlanResult(strategy=RetrievalStrategy.NONE)

        _, result = await apply_retrieval_plan(
            plan,
            full_view=view,
            full_document=document,
            docs_items=items,
            memory=None,
            sentence_segmenter=segmenter,
        )
        # NONE path is the safe default.
        assert result.strategy is RetrievalStrategy.NONE
