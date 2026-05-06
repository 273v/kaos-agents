"""Tests for kaos_agents.context.retrieval — adaptive multi-round retrieval."""

from __future__ import annotations

from kaos_agents.context.retrieval import (
    AdaptiveRetrievalResult,
    _expand_with_prf,
    _merge_results,
    adaptive_retrieve,
)
from kaos_agents.memory.search import MemorySearchResult
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


class TestMergeResults:
    def test_new_items_added(self) -> None:
        seen: dict[str, MemorySearchResult] = {}
        results = [
            MemorySearchResult(content="a", section=MemoryType.DOCUMENTS, score=1.0, item_id="1"),
            MemorySearchResult(content="b", section=MemoryType.DOCUMENTS, score=0.5, item_id="2"),
        ]
        new = _merge_results(seen, results)
        assert new == 2
        assert len(seen) == 2

    def test_higher_score_replaces(self) -> None:
        seen: dict[str, MemorySearchResult] = {
            "1": MemorySearchResult(
                content="a", section=MemoryType.DOCUMENTS, score=0.5, item_id="1"
            ),
        }
        results = [
            MemorySearchResult(content="a", section=MemoryType.DOCUMENTS, score=0.9, item_id="1"),
        ]
        new = _merge_results(seen, results)
        assert new == 0
        assert seen["1"].score == 0.9

    def test_lower_score_does_not_replace(self) -> None:
        seen: dict[str, MemorySearchResult] = {
            "1": MemorySearchResult(
                content="a", section=MemoryType.DOCUMENTS, score=0.9, item_id="1"
            ),
        }
        results = [
            MemorySearchResult(content="a", section=MemoryType.DOCUMENTS, score=0.3, item_id="1"),
        ]
        new = _merge_results(seen, results)
        assert new == 0
        assert seen["1"].score == 0.9

    def test_empty_results(self) -> None:
        seen: dict[str, MemorySearchResult] = {}
        new = _merge_results(seen, [])
        assert new == 0


class TestExpandWithPRF:
    def test_extracts_terms_from_results(self) -> None:
        results = [
            MemorySearchResult(
                content="This employment agreement contains noncompetition provisions",
                section=MemoryType.DOCUMENTS,
                score=1.0,
                item_id="1",
            ),
            MemorySearchResult(
                content="The restrictive covenant shall apply for two years",
                section=MemoryType.DOCUMENTS,
                score=0.8,
                item_id="2",
            ),
        ]
        expanded = _expand_with_prf("non-compete", results)
        assert len(expanded) > len("non-compete")
        assert "non-compete" in expanded

    def test_empty_results_returns_empty(self) -> None:
        expanded = _expand_with_prf("query", [])
        assert expanded == ""

    def test_excludes_stopwords(self) -> None:
        results = [
            MemorySearchResult(
                content="the the the shall shall shall agreement",
                section=MemoryType.DOCUMENTS,
                score=1.0,
                item_id="1",
            ),
        ]
        expanded = _expand_with_prf("test", results)
        assert "shall" not in expanded
        assert "agreement" in expanded


class TestAdaptiveRetrieveSmallCorpus:
    def test_small_corpus_uses_bm25(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(
                MemoryType.DOCUMENTS,
                f"employment agreement clause {i} about salary",
                metadata={"uri": f"doc:{i}"},
            )
        result = adaptive_retrieve(memory, "salary", max_rounds=1, top_k=10)
        assert isinstance(result, AdaptiveRetrievalResult)
        assert result.total_rounds >= 1
        assert any(r.strategy == "bm25_raw" for r in result.rounds)

    def test_max_rounds_1_skips_expansion(self) -> None:
        memory = SessionMemory("test")
        for i in range(30):
            memory.add(MemoryType.DOCUMENTS, f"document about topic {i}")
        result = adaptive_retrieve(memory, "topic", max_rounds=1, top_k=10)
        strategies = {r.strategy for r in result.rounds}
        assert "bm25_raw" in strategies
        assert "bm25_lexicon" not in strategies
        assert "bm25_hyde" not in strategies

    def test_results_sorted_by_score(self) -> None:
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(MemoryType.DOCUMENTS, f"lease agreement clause {i} about rent")
        result = adaptive_retrieve(memory, "rent", max_rounds=1, top_k=10)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)


class TestAdaptiveRetrieveLargeCorpus:
    def test_lexicon_round_fires_above_threshold(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"employment agreement clause {i}",
                metadata={"uri": f"doc:emp-{i}"},
            )
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"lease provision {i}",
                metadata={"uri": f"doc:lease-{i}"},
            )
        result = adaptive_retrieve(memory, "employment salary", max_rounds=2, top_k=20)
        assert result.unique_items > 0
        assert result.total_rounds >= 1

    def test_prf_adds_documents(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"noncompetition restrictive covenant agreement section {i}",
            )
        result = adaptive_retrieve(memory, "non-compete", max_rounds=2, top_k=20)
        has_prf = any(r.strategy == "bm25_prf" for r in result.rounds)
        assert has_prf
