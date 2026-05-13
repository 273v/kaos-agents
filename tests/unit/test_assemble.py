"""Tests for kaos_agents.context.assemble — query-aware context assembly."""

from __future__ import annotations

from kaos_agents.context.assemble import assemble_context
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


def _make_memory_with_docs(n_docs: int, *, topic_prefix: str = "doc") -> SessionMemory:
    """Create a SessionMemory with n_docs DOCUMENTS items."""
    memory = SessionMemory("test-session")
    for i in range(n_docs):
        memory.add(MemoryType.DOCUMENTS, f"{topic_prefix}-{i}: content about topic {i}")
    return memory


class TestSmallCorpusFIFO:
    def test_below_threshold_uses_fifo(self) -> None:
        memory = _make_memory_with_docs(5)
        result = assemble_context(
            memory,
            "any query",
            sections=[MemoryType.DOCUMENTS],
            total_budget_tokens=100_000,
            retrieval_threshold=20,
        )
        assert MemoryType.DOCUMENTS in result
        assert len(result[MemoryType.DOCUMENTS]) == 5

    def test_at_threshold_uses_bm25(self) -> None:
        memory = _make_memory_with_docs(20)
        result = assemble_context(
            memory,
            "topic 5",
            sections=[MemoryType.DOCUMENTS],
            total_budget_tokens=100_000,
            retrieval_threshold=20,
        )
        assert MemoryType.DOCUMENTS in result
        assert len(result[MemoryType.DOCUMENTS]) <= 20

    def test_messages_always_fifo(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(MemoryType.MESSAGES, f"message {i}")
        result = assemble_context(
            memory,
            "any query",
            sections=[MemoryType.MESSAGES],
            total_budget_tokens=100_000,
            retrieval_threshold=20,
        )
        assert len(result[MemoryType.MESSAGES]) == 5


class TestLargeCorpusBM25:
    def test_large_corpus_selects_relevant(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(MemoryType.DOCUMENTS, f"document about real estate lease terms {i}")
        for i in range(50):
            memory.add(MemoryType.DOCUMENTS, f"document about employment benefits {i}")

        result = assemble_context(
            memory,
            "employment agreements",
            sections=[MemoryType.DOCUMENTS],
            total_budget_tokens=100_000,
            retrieval_threshold=20,
            search_top_k=10,
        )
        docs = result[MemoryType.DOCUMENTS]
        assert len(docs) > 0
        employment_count = sum(1 for d in docs if "employment" in d.content)
        assert employment_count > 0

    def test_budget_respected(self) -> None:
        memory = _make_memory_with_docs(100)
        result = assemble_context(
            memory,
            "topic 50",
            sections=[MemoryType.DOCUMENTS],
            total_budget_tokens=50,
            retrieval_threshold=20,
        )
        total_tokens = sum(item.token_count for items in result.values() for item in items)
        assert total_tokens <= 50

    def test_mixed_sections_small_and_large(self) -> None:
        memory = SessionMemory("test")
        for i in range(3):
            memory.add(MemoryType.MESSAGES, f"message {i}")
        for i in range(50):
            memory.add(MemoryType.DOCUMENTS, f"doc about topic {i}")

        result = assemble_context(
            memory,
            "topic 25",
            sections=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            total_budget_tokens=100_000,
            retrieval_threshold=20,
        )
        assert MemoryType.MESSAGES in result
        assert len(result[MemoryType.MESSAGES]) == 3
        assert MemoryType.DOCUMENTS in result


class TestPriorityTrimming:
    def test_low_priority_trimmed_first(self) -> None:
        memory = SessionMemory("test")
        for i in range(5):
            memory.add(MemoryType.MESSAGES, f"msg {i} " * 50)
        for i in range(5):
            memory.add(MemoryType.ACTIONS, f"action {i} " * 50)

        result = assemble_context(
            memory,
            "query",
            sections=[MemoryType.MESSAGES, MemoryType.ACTIONS],
            total_budget_tokens=200,
            priority_order=[MemoryType.MESSAGES, MemoryType.ACTIONS],
            retrieval_threshold=100,
        )
        msg_count = len(result.get(MemoryType.MESSAGES, []))
        action_count = len(result.get(MemoryType.ACTIONS, []))
        assert msg_count >= action_count


class TestGetByIds:
    def test_get_by_ids_returns_matching(self) -> None:
        memory = SessionMemory("test")
        items = []
        for i in range(5):
            item = memory.add(MemoryType.MESSAGES, f"message {i}")
            items.append(item)

        target_ids = {items[1].id, items[3].id}
        result = memory.get_by_ids(MemoryType.MESSAGES, target_ids)
        assert len(result) == 2
        result_ids = {r.id for r in result}
        assert result_ids == target_ids

    def test_get_by_ids_preserves_order(self) -> None:
        memory = SessionMemory("test")
        items = []
        for i in range(5):
            items.append(memory.add(MemoryType.MESSAGES, f"message {i}"))

        target_ids = {items[3].id, items[1].id}
        result = memory.get_by_ids(MemoryType.MESSAGES, target_ids)
        assert result[0].id == items[1].id
        assert result[1].id == items[3].id

    def test_get_by_ids_empty_set(self) -> None:
        memory = SessionMemory("test")
        memory.add(MemoryType.MESSAGES, "hello")
        result = memory.get_by_ids(MemoryType.MESSAGES, set())
        assert result == []
