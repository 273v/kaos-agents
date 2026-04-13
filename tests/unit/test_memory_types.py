"""Tests for memory type system: enums, items, configs, token estimation."""

from __future__ import annotations

import pytest

from kaos_agents.memory.types import (
    DEFAULT_SECTIONS,
    EvictionPolicy,
    MemoryItem,
    MemoryType,
    PersistenceMode,
    SectionConfig,
    SummarizationPolicy,
    create_item,
    estimate_tokens,
)


class TestMemoryType:
    def test_all_types_are_strings(self):
        for mt in MemoryType:
            assert isinstance(mt.value, str)

    def test_unique_values(self):
        values = [mt.value for mt in MemoryType]
        assert len(values) == len(set(values))

    def test_explicit_sections(self):
        explicit = {MemoryType.ROLE, MemoryType.PLAYBOOKS, MemoryType.PLAN_EXAMPLES}
        for mt in explicit:
            assert mt in MemoryType

    def test_ephemeral_sections(self):
        ephemeral = {MemoryType.WORKING, MemoryType.PLANNING_CONTEXT}
        for mt in ephemeral:
            assert mt in MemoryType


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "hello" = 5 chars / 4.0 = 1.25 → 1
        assert estimate_tokens("hello") == 1

    def test_longer_string(self):
        text = "The quick brown fox jumps over the lazy dog."
        expected = max(1, int(len(text) / 4.0))
        assert estimate_tokens(text) == expected

    def test_custom_ratio(self):
        text = "hello world"  # 11 chars
        assert estimate_tokens(text, chars_per_token=2.0) == 5

    def test_minimum_is_one(self):
        assert estimate_tokens("a") == 1


class TestMemoryItem:
    def test_make_id_is_deterministic(self):
        id1 = MemoryItem.make_id("content", MemoryType.MESSAGES, 1000.0)
        id2 = MemoryItem.make_id("content", MemoryType.MESSAGES, 1000.0)
        assert id1 == id2

    def test_make_id_differs_by_content(self):
        id1 = MemoryItem.make_id("content1", MemoryType.MESSAGES, 1000.0)
        id2 = MemoryItem.make_id("content2", MemoryType.MESSAGES, 1000.0)
        assert id1 != id2

    def test_make_id_differs_by_section(self):
        id1 = MemoryItem.make_id("content", MemoryType.MESSAGES, 1000.0)
        id2 = MemoryItem.make_id("content", MemoryType.ACTIONS, 1000.0)
        assert id1 != id2

    def test_make_id_is_16_chars(self):
        id1 = MemoryItem.make_id("content", MemoryType.MESSAGES, 1000.0)
        assert len(id1) == 16

    def test_frozen(self):
        item = create_item(MemoryType.MESSAGES, "hello")
        with pytest.raises(AttributeError):
            item.content = "world"  # ty: ignore[invalid-assignment]


class TestCreateItem:
    def test_creates_valid_item(self):
        item = create_item(MemoryType.MESSAGES, "Hello world")
        assert item.section == MemoryType.MESSAGES
        assert item.content == "Hello world"
        assert item.token_count > 0
        assert item.added_at > 0
        assert len(item.id) == 16

    def test_priority_and_tags(self):
        item = create_item(
            MemoryType.FINDINGS,
            "Important finding",
            priority=10,
            tags=("legal", "contract"),
        )
        assert item.priority == 10
        assert item.tags == ("legal", "contract")

    def test_metadata(self):
        item = create_item(
            MemoryType.ACTIONS,
            "action result",
            metadata={"tool": "kaos-pdf-extract-parse"},
        )
        assert item.metadata["tool"] == "kaos-pdf-extract-parse"


class TestSectionConfig:
    def test_frozen(self):
        config = SectionConfig(
            memory_type=MemoryType.MESSAGES,
            budget_tokens=3000,
        )
        with pytest.raises(AttributeError):
            config.budget_tokens = 5000  # ty: ignore[invalid-assignment]

    def test_defaults(self):
        config = SectionConfig(
            memory_type=MemoryType.MESSAGES,
            budget_tokens=3000,
        )
        assert config.eviction_policy == EvictionPolicy.FIFO
        assert config.summarization_policy == SummarizationPolicy.NEVER
        assert config.persistence_mode == PersistenceMode.SNAPSHOT
        assert config.explicit_only is False
        assert config.searchable is False
        assert config.max_items is None


class TestDefaultSections:
    def test_has_13_sections(self):
        assert len(DEFAULT_SECTIONS) == 13

    def test_all_types_unique(self):
        types = [s.memory_type for s in DEFAULT_SECTIONS]
        assert len(types) == len(set(types))

    def test_explicit_sections_are_marked(self):
        explicit = {MemoryType.ROLE, MemoryType.PLAYBOOKS, MemoryType.PLAN_EXAMPLES}
        for config in DEFAULT_SECTIONS:
            if config.memory_type in explicit:
                assert config.explicit_only is True
            else:
                assert config.explicit_only is False

    def test_audit_is_unlimited(self):
        audit = next(c for c in DEFAULT_SECTIONS if c.memory_type == MemoryType.AUDIT)
        assert audit.budget_tokens == 0  # unlimited

    def test_messages_budget(self):
        messages = next(c for c in DEFAULT_SECTIONS if c.memory_type == MemoryType.MESSAGES)
        assert messages.budget_tokens == 3000

    def test_working_is_ephemeral(self):
        working = next(c for c in DEFAULT_SECTIONS if c.memory_type == MemoryType.WORKING)
        assert working.persistence_mode == PersistenceMode.NONE
