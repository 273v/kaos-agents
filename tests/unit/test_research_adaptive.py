"""Tests for ResearchAgent BM25 corpus building."""

from __future__ import annotations

from kaos_agents.memory.session import SessionMemory
from kaos_agents.patterns.research import _build_corpus_bm25
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.memory import MemoryType


class TestBuildCorpusBm25:
    def test_small_corpus_returns_all(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document about topic {i}",
                metadata={"uri": f"doc:{i}"},
            )
        settings = KaosAgentSettings()
        corpus = _build_corpus_bm25(memory, "any query", settings)
        assert len(corpus) == 5

    def test_large_corpus_uses_bm25_retrieval(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"employment agreement clause {i} about salary and benefits",
                metadata={"uri": f"doc:emp-{i}"},
            )
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"real estate lease provision {i} about rent and premises",
                metadata={"uri": f"doc:lease-{i}"},
            )
        settings = KaosAgentSettings()
        corpus = _build_corpus_bm25(memory, "employment salary", settings)
        assert len(corpus) > 0
        assert len(corpus) < 100

    def test_uris_preserved(self) -> None:
        memory = SessionMemory("test")
        for i in range(30):
            memory.add(
                MemoryType.DOCUMENTS,
                f"contract clause {i}",
                metadata={"uri": f"doc:contract-{i}"},
            )
        settings = KaosAgentSettings()
        corpus = _build_corpus_bm25(memory, "contract", settings)
        for uri in corpus:
            assert uri.startswith("doc:") or uri.startswith("mem:")

    def test_no_documents_returns_empty(self) -> None:
        memory = SessionMemory("test")
        memory.add(MemoryType.MESSAGES, "hello")
        settings = KaosAgentSettings()
        corpus = _build_corpus_bm25(memory, "query", settings)
        assert len(corpus) == 0
