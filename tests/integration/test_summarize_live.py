"""Live integration tests for memory summarization with real LLM.

Tests that the LLM-powered summarizer correctly compresses memory sections.
Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.summarize import summarize_items
from kaos_agents.memory.types import (
    EvictionPolicy,
    MemoryType,
    PersistenceMode,
    SectionConfig,
    SummarizationPolicy,
    create_item,
)

MODEL = "anthropic:claude-haiku-4-5"


@pytest.mark.live
class TestSummarizeItemsLive:
    """Test LLM-powered summarization with real API calls."""

    async def test_summarize_conversation(self):
        """Summarize a conversation into a compact summary."""
        items = [
            create_item(MemoryType.MESSAGES, "user: What is the Federal Register?"),
            create_item(
                MemoryType.MESSAGES,
                "assistant: The Federal Register is the daily journal of the US government.",
            ),
            create_item(MemoryType.MESSAGES, "user: How often is it published?"),
            create_item(
                MemoryType.MESSAGES,
                "assistant: It is published every business day by the Office "
                "of the Federal Register.",
            ),
            create_item(MemoryType.MESSAGES, "user: Can I search it programmatically?"),
            create_item(
                MemoryType.MESSAGES,
                "assistant: Yes, the Federal Register has a public API at federalregister.gov/api.",
            ),
        ]

        summary = await summarize_items(
            items,
            MemoryType.MESSAGES,
            model=MODEL,
            target_tokens=100,
        )

        assert summary, "Summary should not be empty"
        assert len(summary) < sum(len(i.content) for i in items), "Summary should be shorter"
        # Should preserve key facts
        summary_lower = summary.lower()
        assert any(
            w in summary_lower for w in ("federal register", "government", "api", "published")
        ), f"Summary should preserve key facts: {summary[:200]}"

    async def test_summarize_tool_calls(self):
        """Summarize a series of tool calls."""
        items = [
            create_item(
                MemoryType.ACTIONS, "Tool: kaos-source-fr-search(query='EPA') → 10 results"
            ),
            create_item(
                MemoryType.ACTIONS,
                "Tool: kaos-source-fr-get-document(doc_number='2024-01234') "
                "→ EPA Rule on Emissions",
            ),
            create_item(
                MemoryType.ACTIONS,
                "Tool: kaos-web-get-markdown(url='https://example.com') → Page content extracted",
            ),
        ]

        summary = await summarize_items(
            items,
            MemoryType.ACTIONS,
            model=MODEL,
            target_tokens=80,
        )

        assert summary, "Summary should not be empty"
        summary_lower = summary.lower()
        assert any(w in summary_lower for w in ("epa", "search", "tool", "document", "emission")), (
            f"Summary should mention key actions: {summary[:200]}"
        )


@pytest.mark.live
class TestSummarizeTurnLive:
    """Test summarize_turn() with real LLM in context of a session."""

    async def test_on_turn_summarization(self):
        """ON_TURN sections should be summarized at end of turn."""
        sections = (
            SectionConfig(
                memory_type=MemoryType.REFLECTION,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.ON_TURN,
                persistence_mode=PersistenceMode.STREAMING,
            ),
            SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=10_000,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.NEVER,
                persistence_mode=PersistenceMode.STREAMING,
            ),
        )
        memory = SessionMemory("test-live-summarize", sections=sections)
        memory.add(MemoryType.REFLECTION, "Plan failed: tool kaos-source-fr-search returned empty")
        memory.add(MemoryType.REFLECTION, "Retry with broader query succeeded after 2 attempts")
        memory.add(MemoryType.MESSAGES, "user: Search for EPA rules")
        memory.add(MemoryType.MESSAGES, "assistant: Found 10 results")

        n = await memory.summarize_turn(model=MODEL)
        assert n == 1  # Only REFLECTION should be summarized (ON_TURN)

        # Reflection should now have 1 summarized item
        assert memory.section_item_count(MemoryType.REFLECTION) == 1
        items = memory.get_recent(MemoryType.REFLECTION, 5)
        assert "[Summary" in items[0].content

        # Messages should be unchanged (NEVER policy)
        assert memory.section_item_count(MemoryType.MESSAGES) == 2
