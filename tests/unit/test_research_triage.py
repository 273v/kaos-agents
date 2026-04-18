"""Tests for ResearchAgent corpus triage via _build_corpus_triaged."""

from __future__ import annotations

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.types import MemoryType
from kaos_agents.patterns.research import _build_corpus_triaged


class TestBuildCorpusTriaged:
    def test_small_corpus_returns_all(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document about topic {i}",
                metadata={"uri": f"doc:{i}"},
            )
        corpus = _build_corpus_triaged(memory, "any query", threshold=20)
        assert len(corpus) == 5

    def test_large_corpus_returns_subset(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"employment agreement clause {i} about salary and benefits",
                metadata={"uri": f"doc:employment-{i}"},
            )
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"real estate lease provision {i} about rent and premises",
                metadata={"uri": f"doc:lease-{i}"},
            )

        corpus = _build_corpus_triaged(memory, "employment salary benefits", threshold=20)
        assert len(corpus) < 100
        assert len(corpus) > 0

        employment_count = sum(1 for uri in corpus if "employment" in uri)
        assert employment_count > 0

    def test_uris_preserved(self) -> None:
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"contract clause {i} about indemnification",
                metadata={"uri": f"doc:contract-{i}"},
            )
        corpus = _build_corpus_triaged(memory, "indemnification", threshold=20)
        for uri in corpus:
            assert uri.startswith("doc:contract-")

    def test_no_documents_returns_empty(self) -> None:
        memory = SessionMemory("test")
        memory.add(MemoryType.MESSAGES, "hello")
        corpus = _build_corpus_triaged(memory, "query", threshold=20)
        assert len(corpus) == 0

    def test_threshold_respected(self) -> None:
        memory = SessionMemory("test")
        for i in range(15):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i}",
                metadata={"uri": f"doc:{i}"},
            )
        corpus_low = _build_corpus_triaged(memory, "query", threshold=10)
        corpus_high = _build_corpus_triaged(memory, "query", threshold=20)
        assert len(corpus_high) == 15
        assert len(corpus_low) <= 15
