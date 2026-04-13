"""Tests for SessionMemory — the top-level memory container."""

from __future__ import annotations

import pytest

from kaos_agents.errors import MemoryBudgetExceededError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.types import MemoryType


class TestSessionMemoryBasic:
    def test_create_with_defaults(self):
        mem = SessionMemory("test")
        assert mem.session_id == "test"
        assert mem.turn_count == 0
        assert mem.total_tokens == 0
        assert len(mem.section_names) == 13  # DEFAULT_SECTIONS

    def test_add_and_read(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "Hello world")
        assert mem.section_token_count(MemoryType.MESSAGES) > 0
        assert mem.section_item_count(MemoryType.MESSAGES) == 1

    def test_add_multiple_sections(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "user message")
        mem.add(MemoryType.ACTIONS, "tool result")
        mem.add(MemoryType.FINDINGS, "key finding")
        assert mem.total_tokens > 0
        assert mem.section_item_count(MemoryType.MESSAGES) == 1
        assert mem.section_item_count(MemoryType.ACTIONS) == 1
        assert mem.section_item_count(MemoryType.FINDINGS) == 1

    def test_explicit_section_rejects_writes(self):
        mem = SessionMemory("test")
        with pytest.raises(MemoryBudgetExceededError, match="explicit-only"):
            mem.add(MemoryType.ROLE, "should fail")

    def test_unknown_section_raises(self):
        mem = SessionMemory("test")
        with pytest.raises(KeyError, match="not configured"):
            mem.add(MemoryType.LAST_USER_MESSAGE, "should fail")

    def test_has_section(self):
        mem = SessionMemory("test")
        assert mem.has_section(MemoryType.MESSAGES)
        assert not mem.has_section(MemoryType.LAST_USER_MESSAGE)


class TestContextAssembly:
    def test_get_sections_returns_requested(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "msg 1")
        mem.add(MemoryType.MESSAGES, "msg 2")
        mem.add(MemoryType.ACTIONS, "action 1")

        result = mem.get_sections(
            [MemoryType.MESSAGES, MemoryType.ACTIONS],
            total_budget_tokens=10000,
        )
        assert MemoryType.MESSAGES in result
        assert MemoryType.ACTIONS in result
        assert len(result[MemoryType.MESSAGES]) == 2
        assert len(result[MemoryType.ACTIONS]) == 1

    def test_get_sections_respects_total_budget(self):
        mem = SessionMemory("test")
        # Fill messages with enough content to exceed a small budget
        for i in range(20):
            mem.add(MemoryType.MESSAGES, f"Message number {i} with some content to use tokens")

        result = mem.get_sections(
            [MemoryType.MESSAGES],
            total_budget_tokens=50,
        )
        total = sum(item.token_count for item in result[MemoryType.MESSAGES])
        assert total <= 50

    def test_get_sections_priority_trims_lowest_first(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "important message " * 10)
        mem.add(MemoryType.ACTIONS, "less important action " * 10)

        # Request both with a tight budget, messages first in priority
        result = mem.get_sections(
            [MemoryType.MESSAGES, MemoryType.ACTIONS],
            total_budget_tokens=30,
            priority_order=[MemoryType.MESSAGES, MemoryType.ACTIONS],
        )

        # Messages (higher priority) should have more tokens than actions
        msg_tokens = sum(i.token_count for i in result.get(MemoryType.MESSAGES, []))
        act_tokens = sum(i.token_count for i in result.get(MemoryType.ACTIONS, []))
        assert msg_tokens >= act_tokens

    def test_get_sections_skips_unconfigured(self):
        mem = SessionMemory("test")
        result = mem.get_sections(
            [MemoryType.MESSAGES, MemoryType.LAST_USER_MESSAGE],  # LAST_USER_MESSAGE not configured
            total_budget_tokens=10000,
        )
        assert MemoryType.LAST_USER_MESSAGE not in result


class TestTurnLifecycle:
    def test_begin_turn_clears_ephemeral(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.WORKING, "scratch note")
        mem.add(MemoryType.PLANNING_CONTEXT, "plan context")
        assert mem.section_item_count(MemoryType.WORKING) == 1

        mem.begin_turn()
        assert mem.section_item_count(MemoryType.WORKING) == 0
        assert mem.section_item_count(MemoryType.PLANNING_CONTEXT) == 0

    def test_begin_turn_preserves_persistent(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "user message")
        mem.begin_turn()
        assert mem.section_item_count(MemoryType.MESSAGES) == 1

    def test_end_turn_increments_counter(self):
        mem = SessionMemory("test")
        assert mem.turn_count == 0
        mem.end_turn()
        assert mem.turn_count == 1
        mem.end_turn()
        assert mem.turn_count == 2


class TestSerialization:
    def test_round_trip(self):
        mem = SessionMemory("test-123")
        mem.add(MemoryType.MESSAGES, "hello")
        mem.add(MemoryType.MESSAGES, "world")
        mem.add(MemoryType.ACTIONS, "action result")
        mem.add(MemoryType.FINDINGS, "key finding")
        mem.end_turn()

        data = mem.to_dict()
        restored = SessionMemory.from_dict(data)

        assert restored.session_id == "test-123"
        assert restored.turn_count == 1
        assert restored.total_tokens == mem.total_tokens
        assert restored.section_item_count(MemoryType.MESSAGES) == 2
        assert restored.section_item_count(MemoryType.ACTIONS) == 1
        assert restored.section_item_count(MemoryType.FINDINGS) == 1

    def test_round_trip_empty_memory(self):
        mem = SessionMemory("empty")
        data = mem.to_dict()
        restored = SessionMemory.from_dict(data)
        assert restored.session_id == "empty"
        assert restored.total_tokens == 0

    def test_skips_empty_sections_in_serialization(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "only messages")
        data = mem.to_dict()
        # Only MESSAGES should be in sections (the rest are empty)
        assert "messages" in data["sections"]
        assert "actions" not in data["sections"]

    def test_unknown_section_in_data_is_skipped(self):
        data = {
            "session_id": "test",
            "turn_count": 0,
            "sections": {
                "unknown_section_type": {"items": []},
                "messages": {"items": []},
            },
        }
        # Should not raise — just skip the unknown section
        restored = SessionMemory.from_dict(data)
        assert restored.session_id == "test"


class TestEvictionUnderPressure:
    def test_messages_evict_when_over_budget(self):
        mem = SessionMemory("test")
        # Default MESSAGES budget is 3000 tokens
        # Add enough to exceed
        for i in range(200):
            mem.add(MemoryType.MESSAGES, f"Message {i}: " + "x" * 100)

        # Should have evicted to stay under budget
        assert mem.section_token_count(MemoryType.MESSAGES) <= 3000

    def test_ten_turn_conversation(self):
        """Simulate 10 turns of conversation."""
        mem = SessionMemory("test")
        for turn in range(10):
            mem.begin_turn()
            mem.add(MemoryType.MESSAGES, f"User turn {turn}: What about topic {turn}?")
            mem.add(MemoryType.WORKING, f"Thinking about topic {turn}...")
            mem.add(MemoryType.ACTIONS, f"Called tool for topic {turn} → result data")
            mem.add(MemoryType.MESSAGES, f"Assistant turn {turn}: Here's what I found...")
            mem.add(MemoryType.FINDINGS, f"Finding {turn}: Key insight about topic {turn}")
            mem.end_turn()

        # Working memory has 1 item (from the last turn — cleared at begin_turn, not end_turn)
        # After begin_turn of a hypothetical turn 11, it would be cleared
        mem.begin_turn()
        assert mem.section_item_count(MemoryType.WORKING) == 0
        # Messages should be within budget
        assert mem.section_token_count(MemoryType.MESSAGES) <= 3000
        # Findings should all be retained (NONE eviction, 1500 budget is generous)
        assert mem.section_item_count(MemoryType.FINDINGS) == 10
        # Turn counter should be 10
        assert mem.turn_count == 10
