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


class TestCorpusHandleRetention:
    """WU-G.2 / #352 — assemble_context must retain a stable DOCUMENTS
    handle across turns when ``memory.corpus_ever_attached`` fires.
    """

    def test_handle_injected_when_trim_drops_documents(self) -> None:
        """A small DOCUMENTS section trimmed to zero by the budget pass
        must surface a synthetic handle line so the downstream LLM still
        knows the corpus is reachable via ``search_memory``."""
        memory = SessionMemory("test-handle")
        memory.mark_corpus_attached()
        # Two small DOCUMENTS items + a flood of MESSAGES so the
        # priority trim drops DOCUMENTS first.
        memory.add(
            MemoryType.DOCUMENTS,
            "EMNA NDA boilerplate content " * 10,
            metadata={"uri": "file:EMNA-NDA.docx"},
        )
        memory.add(
            MemoryType.DOCUMENTS,
            "Acme NDA boilerplate content " * 10,
            metadata={"uri": "file:Acme-NDA.docx"},
        )
        for i in range(20):
            memory.add(MemoryType.MESSAGES, f"message {i} " * 30)

        result = assemble_context(
            memory,
            "summarize that",
            sections=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            total_budget_tokens=80,
            priority_order=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            retrieval_threshold=100,
        )
        # DOCUMENTS must be present with the synthetic handle item.
        docs = result.get(MemoryType.DOCUMENTS, [])
        assert len(docs) == 1, f"expected exactly 1 handle item, got {len(docs)}"
        handle = docs[0]
        assert handle.metadata.get("corpus_handle") is True
        assert "EMNA-NDA.docx" in handle.content or "Acme-NDA.docx" in handle.content
        assert "search_memory" in handle.content

    def test_handle_skipped_when_flag_not_set(self) -> None:
        """Sessions that never had a corpus attached must NOT see a
        synthetic handle — the prior behaviour stays unchanged for the
        no-corpus path."""
        memory = SessionMemory("test-no-handle")
        # Note: NOT calling mark_corpus_attached().
        memory.add(MemoryType.DOCUMENTS, "x" * 200, metadata={"uri": "file:x.txt"})
        for i in range(20):
            memory.add(MemoryType.MESSAGES, f"message {i} " * 30)

        result = assemble_context(
            memory,
            "anything",
            sections=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            total_budget_tokens=80,
            priority_order=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            retrieval_threshold=100,
        )
        # DOCUMENTS slot is either absent or empty — no handle injected.
        docs = result.get(MemoryType.DOCUMENTS, [])
        assert all(not (item.metadata or {}).get("corpus_handle") for item in docs), (
            "handle must not fire when corpus_ever_attached is False"
        )

    def test_handle_skipped_when_documents_survive_trim(self) -> None:
        """When the trim phase leaves at least one DOCUMENTS body in
        the assembled context, the handle is NOT injected — the agent
        already has a body to ground on."""
        memory = SessionMemory("test-partial-trim")
        memory.mark_corpus_attached()
        memory.add(MemoryType.DOCUMENTS, "tiny doc", metadata={"uri": "file:t.txt"})
        memory.add(MemoryType.MESSAGES, "tiny msg")

        result = assemble_context(
            memory,
            "anything",
            sections=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            total_budget_tokens=100_000,
            retrieval_threshold=100,
        )
        docs = result.get(MemoryType.DOCUMENTS, [])
        assert len(docs) == 1
        assert not docs[0].metadata.get("corpus_handle"), (
            "handle must not replace a surviving DOCUMENTS body"
        )

    def test_handle_caps_at_twelve_filenames(self) -> None:
        """A large corpus collapses to 12 filenames + ``(+ N more)``
        suffix in the handle so the line stays small."""
        memory = SessionMemory("test-cap")
        memory.mark_corpus_attached()
        for i in range(30):
            memory.add(
                MemoryType.DOCUMENTS,
                f"doc {i} body " * 50,
                metadata={"uri": f"file:doc-{i:02d}.txt"},
            )
        for i in range(20):
            memory.add(MemoryType.MESSAGES, f"message {i} " * 30)

        result = assemble_context(
            memory,
            "summarize",
            sections=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            total_budget_tokens=80,
            priority_order=[MemoryType.MESSAGES, MemoryType.DOCUMENTS],
            retrieval_threshold=1000,  # force FIFO path
        )
        docs = result.get(MemoryType.DOCUMENTS, [])
        assert len(docs) == 1
        handle = docs[0]
        assert handle.metadata.get("corpus_handle") is True
        assert "+ 18 more" in handle.content or "(+ 18 more)" in handle.content

    def test_corpus_ever_attached_persists_through_dict_round_trip(self) -> None:
        """The sticky flag must round-trip through ``to_dict`` /
        ``from_dict`` — otherwise the SPA backend's per-call hydration
        defeats the retention."""
        memory = SessionMemory("test-roundtrip")
        memory.mark_corpus_attached()
        memory.add(MemoryType.DOCUMENTS, "x", metadata={"uri": "file:x"})
        data = memory.to_dict()
        assert data["corpus_ever_attached"] is True

        restored = SessionMemory.from_dict(data)
        assert restored.corpus_ever_attached is True

    def test_corpus_ever_attached_defaults_false_pre_a19_snapshot(self) -> None:
        """A snapshot without the ``corpus_ever_attached`` key (i.e.
        pre-0.1.0a19) loads with the flag defaulting to False so the
        next classifying turn sets it from live state."""
        memory = SessionMemory("test-old-snapshot")
        memory.add(MemoryType.DOCUMENTS, "x")
        data = memory.to_dict()
        # Strip the new key to simulate an older snapshot.
        data.pop("corpus_ever_attached", None)
        restored = SessionMemory.from_dict(data)
        assert restored.corpus_ever_attached is False


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
