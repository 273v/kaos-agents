"""Regression tests for WS-0.7 — memory eviction + summarize reachability.

Pre-fix bugs:

1. ``SessionMemory.get_sections()`` trim loop did ``items.pop()`` which
   removed items from the END of an oldest-first list — discarding the
   MOST RECENT conversation context first. Almost always wrong; trimming
   should drop oldest context and keep the newest on-topic items.

2. ``Section.add()`` evicted down to budget for every policy, so
   ``summarize_turn()`` never saw ``is_over_budget=True`` for
   ``ON_OVERFLOW`` / ``AUTO`` summarization — both paths were
   unreachable in practice.

Post-fix:

- ``get_sections()`` trim pops from the front (oldest-first).
- ``Section.add()`` skips eviction when ``summarization_policy in
  (ON_OVERFLOW, AUTO)``, letting the section transiently exceed budget
  so summarization can observe + compress it.
"""

from __future__ import annotations

import pytest

from kaos_agents.memory.sections import Section
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.types import (
    EvictionPolicy,
    MemoryType,
    SectionConfig,
    SummarizationPolicy,
)


@pytest.mark.unit
class TestGetSectionsTrimOldestFirst:
    """The assemble-time trim loop must preserve recency — the WS-0.7
    headline regression was that it discarded newest first."""

    def test_trim_removes_oldest_context_first(self) -> None:
        """Given a tight total_budget, the newest messages survive the
        trim. Oldest are discarded."""
        mem = SessionMemory(
            session_id="ws07-trim",
            sections=(
                SectionConfig(
                    memory_type=MemoryType.MESSAGES,
                    budget_tokens=10_000,  # large per-section budget
                    eviction_policy=EvictionPolicy.FIFO,
                ),
            ),
        )
        # Add 5 items tagged with a unique index so we can parse which
        # ones survive the trim. Padding ensures each has non-trivial tokens.
        for i in range(5):
            mem.add(
                MemoryType.MESSAGES,
                f"idx={i} " + ("pad " * 20),
            )

        # Tight total budget that forces the assemble trim to drop most items.
        section = mem._sections[MemoryType.MESSAGES]
        total_all = sum(item.token_count for item in section.items())
        tight_budget = total_all // 3  # room for roughly the last 1-2

        result = mem.get_sections([MemoryType.MESSAGES], total_budget_tokens=tight_budget)
        items_kept = result[MemoryType.MESSAGES]

        # The items that survive must be the LATER ones (oldest-first eviction).
        contents = [item.content for item in items_kept]
        assert contents, "trim loop produced empty result — over-trimmed"

        # Post-fix: trim pops from the front (oldest-first), so the
        # kept items are the TAIL of the stream — the newest messages.
        # Pre-fix: trim popped from the end (newest), so the kept items
        # would be the head — the oldest messages, index 0 first.
        # Parse the ``idx=N`` marker we packed into each message.
        indices = [int(c.split()[0].split("=")[1]) for c in contents]
        assert indices[0] > 0, (
            f"Trim kept the OLDEST item (indices start at {indices[0]}) — "
            f"WS-0.7 regression: items.pop() used instead of items.pop(0). "
            f"Contents: {contents}"
        )


@pytest.mark.unit
class TestAddDeferredForSummarization:
    """Sections with ON_OVERFLOW / AUTO summarization must allow
    transient over-budget state so ``summarize_turn`` can trigger."""

    def test_on_overflow_section_exceeds_budget_without_eviction(self) -> None:
        section = Section(
            config=SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=50,  # very small budget
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.ON_OVERFLOW,
            )
        )
        # Add items until well past the budget.
        for i in range(10):
            section.add(f"item number {i} with some filler text " * 5)

        # Section must be OVER budget (transient state) so summarize_turn
        # can observe and compress.
        assert section.is_over_budget, (
            f"ON_OVERFLOW section did not exceed budget "
            f"({section._token_count}/{section._config.budget_tokens}) — "
            "WS-0.7 regression: add() kept evicting, so summarize_turn "
            "never fires."
        )
        assert section.item_count == 10, (
            f"add() evicted items from ON_OVERFLOW section; expected 10 "
            f"items, got {section.item_count}. Eviction must be deferred "
            "to summarize_turn for this policy."
        )

    def test_auto_section_also_defers_eviction(self) -> None:
        section = Section(
            config=SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=50,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.AUTO,
            )
        )
        for i in range(8):
            section.add(f"auto-section item {i} filler " * 5)

        assert section.is_over_budget
        assert section.item_count == 8

    def test_fifo_section_still_evicts(self) -> None:
        """Sections WITHOUT summarization policy continue to evict
        at add() time — only ON_OVERFLOW / AUTO defer."""
        section = Section(
            config=SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=50,
                eviction_policy=EvictionPolicy.FIFO,
                summarization_policy=SummarizationPolicy.NEVER,
            )
        )
        for i in range(10):
            section.add(f"filler item {i} text " * 5)

        # Without a summarization policy, add() evicts to stay under budget.
        assert not section.is_over_budget, (
            "NEVER-summarization section must stay within budget via FIFO eviction at add() time."
        )
        assert section.item_count < 10

    def test_refuse_policy_still_raises(self) -> None:
        """REFUSE eviction behavior must survive the WS-0.7 change —
        only applies when summarization policy is NOT ON_OVERFLOW/AUTO."""
        from kaos_agents.errors import MemoryBudgetExceededError

        section = Section(
            config=SectionConfig(
                memory_type=MemoryType.MESSAGES,
                budget_tokens=20,
                eviction_policy=EvictionPolicy.REFUSE,
                summarization_policy=SummarizationPolicy.NEVER,
            )
        )
        section.add("first item " * 3)
        with pytest.raises(MemoryBudgetExceededError):
            section.add("second item that forces over budget " * 5)
