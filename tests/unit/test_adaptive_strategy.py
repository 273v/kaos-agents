"""Tests for _assess_complexity with n_documents and intent_confidence."""

from __future__ import annotations

from kaos_agents.planning.strategies.adaptive import _assess_complexity


class TestDocumentAwareness:
    def test_large_corpus_reduces_confidence(self) -> None:
        base = _assess_complexity("find employment clauses", 5)
        with_docs = _assess_complexity("find employment clauses", 5, n_documents=500)
        assert with_docs < base
        assert with_docs < 0.6

    def test_medium_corpus_moderate_reduction(self) -> None:
        base = _assess_complexity("find clauses", 5)
        with_docs = _assess_complexity("find clauses", 5, n_documents=30)
        assert with_docs < base
        assert with_docs > _assess_complexity("find clauses", 5, n_documents=500)

    def test_small_corpus_no_change(self) -> None:
        base = _assess_complexity("find clauses", 5)
        with_zero = _assess_complexity("find clauses", 5, n_documents=0)
        assert base == with_zero

    def test_few_docs_minimal_reduction(self) -> None:
        base = _assess_complexity("find clauses", 5)
        with_5 = _assess_complexity("find clauses", 5, n_documents=5)
        assert with_5 < base
        diff = base - with_5
        assert diff <= 0.06


class TestIntentConfidence:
    def test_low_confidence_reduces_score(self) -> None:
        base = _assess_complexity("search", 5)
        with_low = _assess_complexity("search", 5, intent_confidence=0.3)
        assert with_low < base

    def test_high_confidence_no_change(self) -> None:
        base = _assess_complexity("search", 5)
        with_high = _assess_complexity("search", 5, intent_confidence=0.8)
        assert base == with_high

    def test_none_confidence_no_change(self) -> None:
        base = _assess_complexity("search", 5)
        with_none = _assess_complexity("search", 5, intent_confidence=None)
        assert base == with_none


class TestCombined:
    def test_large_corpus_and_low_confidence(self) -> None:
        score = _assess_complexity("find clauses", 5, n_documents=200, intent_confidence=0.3)
        assert score < 0.5

    def test_score_clamped_to_minimum(self) -> None:
        score = _assess_complexity(
            "step by step comprehensive analysis",
            20,
            n_documents=1000,
            intent_confidence=0.1,
        )
        assert score >= 0.1
