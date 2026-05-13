"""Tests for kaos_agents.context.triage — corpus triage before planning."""

from __future__ import annotations

from kaos_agents.context.triage import triage_corpus
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


class TestTriageSmallCorpus:
    def test_below_threshold_returns_none(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(MemoryType.DOCUMENTS, f"doc {i}")
        result = triage_corpus(memory, "any goal", threshold=20)
        assert result is None

    def test_no_documents_section_returns_none(self) -> None:
        memory = SessionMemory("test")
        memory.add(MemoryType.MESSAGES, "hello")
        result = triage_corpus(memory, "any goal")
        assert result is None


class TestTriageLargeCorpus:
    def test_selects_relevant_subset(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"employment agreement clause {i}",
                metadata={"uri": f"doc:employment-{i}"},
            )
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"real estate lease provision {i}",
                metadata={"uri": f"doc:lease-{i}"},
            )

        result = triage_corpus(memory, "employment benefits", threshold=20, max_selected=10)
        assert result is not None
        assert result.total_documents == 100
        assert result.selected_count <= 10
        assert result.selected_count > 0

    def test_context_summary_mentions_counts(self) -> None:
        memory = SessionMemory("test")
        for i in range(30):
            memory.add(MemoryType.DOCUMENTS, f"contract about employment benefits section {i}")
        result = triage_corpus(memory, "employment", threshold=20)
        assert result is not None
        assert "30" in result.context_summary
        assert "Selected" in result.context_summary

    def test_max_selected_respected(self) -> None:
        memory = SessionMemory("test")
        for i in range(100):
            memory.add(MemoryType.DOCUMENTS, f"document {i}")
        result = triage_corpus(memory, "document", threshold=20, max_selected=5)
        assert result is not None
        assert result.selected_count <= 5

    def test_uris_extracted(self) -> None:
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document about topic {i}",
                metadata={"uri": f"doc:{i}"},
            )
        result = triage_corpus(memory, "topic 10", threshold=20)
        assert result is not None
        assert len(result.selected_uris) > 0

    def test_triage_result_is_frozen(self) -> None:
        import pytest

        memory = SessionMemory("test")
        for i in range(25):
            memory.add(MemoryType.DOCUMENTS, f"lease agreement clause about property {i}")
        result = triage_corpus(memory, "lease property", threshold=20)
        assert result is not None
        with pytest.raises(AttributeError):
            result.__setattr__("total_documents", 999)
