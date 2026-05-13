"""Unit tests for memory summarization."""

from __future__ import annotations

import pytest

from kaos_agents.memory.sections import Section
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.summarize import _fallback_summarize
from kaos_agents.types.memory import (
    EvictionPolicy,
    MemoryType,
    PersistenceMode,
    SectionConfig,
    SummarizationPolicy,
    create_item,
)


@pytest.mark.unit
class TestFallbackSummarize:
    """Test the mechanical fallback summarizer (no LLM)."""

    def test_empty_items(self):
        result = _fallback_summarize([], 200)
        assert "[Summary" in result

    def test_single_item(self):
        items = [create_item(MemoryType.MESSAGES, "Hello world")]
        result = _fallback_summarize(items, 200)
        assert "Hello world" in result

    def test_multiple_items(self):
        items = [
            create_item(MemoryType.MESSAGES, "First message"),
            create_item(MemoryType.MESSAGES, "Second message"),
            create_item(MemoryType.MESSAGES, "Third message"),
        ]
        result = _fallback_summarize(items, 1000)
        assert "First message" in result
        assert "Second message" in result
        assert "Third message" in result

    def test_truncation_at_target(self):
        items = [
            create_item(MemoryType.MESSAGES, "A" * 200),
            create_item(MemoryType.MESSAGES, "B" * 200),
        ]
        result = _fallback_summarize(items, 50)
        # Should truncate and include "..."
        assert "..." in result

    def test_long_content_truncated_per_line(self):
        items = [create_item(MemoryType.MESSAGES, "X" * 200)]
        result = _fallback_summarize(items, 1000)
        # Long lines are truncated to 100 chars
        assert "..." in result


@pytest.mark.unit
class TestSectionSummarizationProperties:
    """Test Section's summarization-related properties."""

    def test_needs_summarization_on_overflow(self):
        config = SectionConfig(
            memory_type=MemoryType.MESSAGES,
            budget_tokens=100,
            eviction_policy=EvictionPolicy.FIFO,
            summarization_policy=SummarizationPolicy.ON_OVERFLOW,
        )
        section = Section(config)
        assert section.needs_summarization is True
        assert section.summarization_policy == SummarizationPolicy.ON_OVERFLOW

    def test_needs_summarization_on_turn(self):
        config = SectionConfig(
            memory_type=MemoryType.REFLECTION,
            budget_tokens=100,
            eviction_policy=EvictionPolicy.FIFO,
            summarization_policy=SummarizationPolicy.ON_TURN,
        )
        section = Section(config)
        assert section.needs_summarization is True

    def test_no_summarization_never(self):
        config = SectionConfig(
            memory_type=MemoryType.ROLE,
            budget_tokens=100,
            eviction_policy=EvictionPolicy.NONE,
            summarization_policy=SummarizationPolicy.NEVER,
        )
        section = Section(config)
        assert section.needs_summarization is False

    def test_no_summarization_manual(self):
        config = SectionConfig(
            memory_type=MemoryType.FINDINGS,
            budget_tokens=100,
            eviction_policy=EvictionPolicy.NONE,
            summarization_policy=SummarizationPolicy.MANUAL,
        )
        section = Section(config)
        assert section.needs_summarization is False

    def test_collect_all_items(self):
        config = SectionConfig(
            memory_type=MemoryType.MESSAGES,
            budget_tokens=10_000,
            eviction_policy=EvictionPolicy.FIFO,
        )
        section = Section(config)
        section.add("first")
        section.add("second")
        section.add("third")
        items = section.collect_all_items()
        assert len(items) == 3
        assert items[0].content == "first"
        assert items[2].content == "third"


@pytest.mark.unit
class TestSessionSummarizeTurn:
    """Test SessionMemory.summarize_turn() with fallback (no LLM)."""

    async def test_summarize_turn_on_turn_policy(self):
        """ON_TURN sections should be summarized even when not over budget."""
        sections = (
            SectionConfig(
                memory_type=MemoryType.REFLECTION,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.ON_TURN,
                persistence_mode=PersistenceMode.STREAMING,
            ),
        )
        memory = SessionMemory("test-summarize-turn", sections=sections)
        memory.add(MemoryType.REFLECTION, "Plan failed for goal X")
        memory.add(MemoryType.REFLECTION, "Retry succeeded with different approach")

        # summarize_turn will try LLM, fail (no API key in unit tests), and use fallback
        n = await memory.summarize_turn(model="nonexistent:model")
        assert n == 1
        # After summarization, section should have 1 item (the summary)
        assert memory.section_item_count(MemoryType.REFLECTION) == 1
        items = memory.get_recent(MemoryType.REFLECTION, 5)
        assert "[Summary" in items[0].content

    async def test_summarize_turn_skip_single_item(self):
        """Sections with only 1 item should not be summarized."""
        sections = (
            SectionConfig(
                memory_type=MemoryType.REFLECTION,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.ON_TURN,
                persistence_mode=PersistenceMode.STREAMING,
            ),
        )
        memory = SessionMemory("test-skip-single", sections=sections)
        memory.add(MemoryType.REFLECTION, "Only one item")

        n = await memory.summarize_turn()
        assert n == 0  # Not summarized — only 1 item

    async def test_summarize_turn_on_overflow_not_over_budget(self):
        """ON_OVERFLOW sections should NOT be summarized when under budget."""
        sections = (
            SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.ON_OVERFLOW,
                persistence_mode=PersistenceMode.STREAMING,
            ),
        )
        memory = SessionMemory("test-no-overflow", sections=sections)
        memory.add(MemoryType.MESSAGES, "Message 1")
        memory.add(MemoryType.MESSAGES, "Message 2")

        n = await memory.summarize_turn()
        assert n == 0  # Not over budget, no summarization

    async def test_summarize_turn_never_policy(self):
        """NEVER sections should never be summarized."""
        sections = (
            SectionConfig(
                memory_type=MemoryType.ROLE,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.NONE,
                summarization_policy=SummarizationPolicy.NEVER,
                persistence_mode=PersistenceMode.SNAPSHOT,
                explicit_only=True,
            ),
        )
        memory = SessionMemory("test-never", sections=sections)
        # Can't add to explicit sections, so nothing to summarize
        n = await memory.summarize_turn()
        assert n == 0
