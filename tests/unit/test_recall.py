"""Tests for the Recall planning primitive."""

from __future__ import annotations

from kaos_agents.memory.session import SessionMemory
from kaos_agents.planning.recall import Context, recall
from kaos_agents.types.memory import MemoryType


class TestRecall:
    def test_empty_memory(self):
        mem = SessionMemory("test")
        ctx = recall(mem, [MemoryType.MESSAGES])
        assert ctx.text == ""
        assert ctx.items == ()
        assert ctx.token_count == 0

    def test_single_section(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "user: Hello")
        mem.add(MemoryType.MESSAGES, "assistant: Hi there")

        ctx = recall(mem, [MemoryType.MESSAGES])
        assert "Hello" in ctx.text
        assert "Hi there" in ctx.text
        assert len(ctx.items) == 2
        assert "messages" in ctx.sections_used
        assert ctx.token_count > 0

    def test_multiple_sections(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "user: What about X?")
        mem.add(MemoryType.ACTIONS, "Called tool-a → result")
        mem.add(MemoryType.FINDINGS, "Key finding: X is true")

        ctx = recall(mem, [MemoryType.MESSAGES, MemoryType.ACTIONS, MemoryType.FINDINGS])
        assert "MESSAGES" in ctx.text
        assert "ACTIONS" in ctx.text
        assert "FINDINGS" in ctx.text
        assert len(ctx.sections_used) == 3

    def test_budget_limits_output(self):
        mem = SessionMemory("test")
        for i in range(50):
            mem.add(MemoryType.MESSAGES, f"Message {i}: " + "x" * 100)

        # With a tight budget, not all messages should fit
        ctx = recall(mem, [MemoryType.MESSAGES], budget_tokens=50)
        assert ctx.token_count <= 50

    def test_priority_order(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "important " * 10)  # ~3 tokens
        mem.add(MemoryType.ACTIONS, "action " * 10)  # ~3 tokens

        # With generous budget both sections should appear
        ctx = recall(
            mem,
            [MemoryType.MESSAGES, MemoryType.ACTIONS],
            budget_tokens=200,
            priority_order=[MemoryType.MESSAGES, MemoryType.ACTIONS],
        )
        assert "messages" in ctx.sections_used
        assert "actions" in ctx.sections_used

    def test_context_is_frozen(self):
        ctx = Context(text="test", items=("a",), sections_used=("messages",), token_count=1)
        assert ctx.text == "test"

    def test_skips_unconfigured_sections(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "hello")
        # LAST_USER_MESSAGE is not in default profile
        ctx = recall(mem, [MemoryType.MESSAGES, MemoryType.LAST_USER_MESSAGE])
        # Should not crash — just skips unconfigured sections
        assert "messages" in ctx.sections_used
