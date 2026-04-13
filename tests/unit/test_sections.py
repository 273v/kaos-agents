"""Tests for Section — budgeted, eviction-aware container."""

from __future__ import annotations

import pytest

from kaos_agents.errors import MemoryBudgetExceededError
from kaos_agents.memory.sections import Section
from kaos_agents.memory.types import EvictionPolicy, MemoryType, SectionConfig


def _config(
    budget: int = 100,
    eviction: EvictionPolicy = EvictionPolicy.FIFO,
    explicit: bool = False,
    max_items: int | None = None,
) -> SectionConfig:
    return SectionConfig(
        memory_type=MemoryType.MESSAGES,
        budget_tokens=budget,
        eviction_policy=eviction,
        explicit_only=explicit,
        max_items=max_items,
    )


class TestSectionAdd:
    def test_add_returns_item(self):
        sec = Section(_config())
        item = sec.add("Hello world")
        assert item.content == "Hello world"
        assert item.token_count > 0

    def test_token_count_tracks(self):
        sec = Section(_config())
        sec.add("Hello world")
        assert sec.token_count > 0

    def test_item_count_tracks(self):
        sec = Section(_config())
        sec.add("one")
        sec.add("two")
        assert sec.item_count == 2

    def test_explicit_only_rejects_writes(self):
        sec = Section(_config(explicit=True))
        with pytest.raises(MemoryBudgetExceededError, match="explicit-only"):
            sec.add("should fail")


class TestSectionBudget:
    def test_is_over_budget_false_when_empty(self):
        sec = Section(_config(budget=100))
        assert not sec.is_over_budget

    def test_unlimited_budget_never_over(self):
        sec = Section(_config(budget=0))
        sec.add("x" * 1000)
        assert not sec.is_over_budget

    def test_available_tokens(self):
        sec = Section(_config(budget=100))
        avail_before = sec.available_tokens
        sec.add("hello")
        assert sec.available_tokens < avail_before

    def test_unlimited_available_is_negative_one(self):
        sec = Section(_config(budget=0))
        assert sec.available_tokens == -1


class TestFIFOEviction:
    def test_fifo_evicts_oldest(self):
        # Budget 10 tokens. Each "x"*40 = 10 tokens. Third add must evict first.
        sec = Section(_config(budget=15, eviction=EvictionPolicy.FIFO))
        sec.add("a" * 40)  # 10 tokens
        sec.add("b" * 20)  # 5 tokens — total 15, at budget
        sec.add("c" * 40)  # 10 tokens — must evict "a" (oldest) to fit
        items = sec.get()
        texts = [i.content for i in items]
        assert "a" * 40 not in texts
        assert "c" * 40 in texts

    def test_fifo_frees_enough_space(self):
        sec = Section(_config(budget=30, eviction=EvictionPolicy.FIFO))
        for i in range(20):
            sec.add(f"item-{i}-" + "x" * 40)  # each ~12 tokens
        # Should have evicted enough to stay under budget
        assert sec.token_count <= 30


class TestLIFOEviction:
    def test_lifo_evicts_newest(self):
        sec = Section(_config(budget=20, eviction=EvictionPolicy.LIFO))
        sec.add("keep me")
        sec.add("also keep me")
        # Adding a big item should evict the most recent (second)
        sec.add("big item that needs room" * 2)
        items = sec.get()
        texts = [i.content for i in items]
        assert "keep me" in texts


class TestREFUSEEviction:
    def test_refuse_raises_on_overflow(self):
        sec = Section(_config(budget=10, eviction=EvictionPolicy.REFUSE))
        sec.add("fill")
        with pytest.raises(MemoryBudgetExceededError, match="REFUSE"):
            sec.add("this is a much longer string that will exceed the budget")


class TestPRIORITYEviction:
    def test_evicts_lowest_priority(self):
        # Budget 15 tokens. Two items fill it. Third must evict lowest priority.
        sec = Section(_config(budget=15, eviction=EvictionPolicy.PRIORITY))
        sec.add("x" * 24, priority=1)  # 6 tokens, low priority
        sec.add("y" * 24, priority=10)  # 6 tokens, high priority — total 12
        sec.add("z" * 24, priority=5)  # 6 tokens — total 18, must evict low-priority "x"
        items = sec.get()
        contents = [i.content for i in items]
        assert "y" * 24 in contents  # high priority kept
        assert "x" * 24 not in contents  # low priority evicted


class TestMaxItems:
    def test_max_items_enforced(self):
        sec = Section(_config(budget=1000, max_items=3))
        sec.add("one")
        sec.add("two")
        sec.add("three")
        sec.add("four")  # Should evict oldest
        assert sec.item_count == 3
        items = sec.get()
        texts = [i.content for i in items]
        assert "one" not in texts
        assert "four" in texts


class TestSectionGet:
    def test_get_returns_insertion_order(self):
        sec = Section(_config())
        sec.add("first")
        sec.add("second")
        sec.add("third")
        items = sec.get()
        assert [i.content for i in items] == ["first", "second", "third"]

    def test_get_with_max_tokens(self):
        sec = Section(_config())
        sec.add("a" * 40)  # ~10 tokens
        sec.add("b" * 40)  # ~10 tokens
        sec.add("c" * 40)  # ~10 tokens
        items = sec.get(max_tokens=15)
        assert len(items) == 1  # Only first fits

    def test_get_recent(self):
        sec = Section(_config())
        sec.add("first")
        sec.add("second")
        sec.add("third")
        recent = sec.get_recent(2)
        assert [i.content for i in recent] == ["third", "second"]


class TestSectionClear:
    def test_clear_resets_tokens(self):
        sec = Section(_config())
        sec.add("hello")
        sec.add("world")
        freed = sec.clear()
        assert freed > 0
        assert sec.token_count == 0
        assert sec.item_count == 0


class TestSectionSerialization:
    def test_round_trip(self):
        config = _config()
        sec = Section(config)
        sec.add("item one", priority=5, tags=("tag1",))
        sec.add("item two", metadata={"key": "value"})

        data = sec.to_dict()
        restored = Section.from_dict(data, config)

        assert restored.item_count == 2
        assert restored.token_count == sec.token_count

        items = restored.get()
        assert items[0].content == "item one"
        assert items[0].priority == 5
        assert items[0].tags == ("tag1",)
        assert items[1].metadata == {"key": "value"}
