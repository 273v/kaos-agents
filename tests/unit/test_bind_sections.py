"""Tests for :func:`kaos_agents.context.sections_to_prompt.bind_sections`.

Track 3 chunk A3 — confirms:
- Empty memory sections render the placeholder
- One-section / multi-section binding works
- ``most_recent`` truncates correctly
- ``separator`` and ``item_prefix`` customization
- Output dict keyed by field_name (not section name) — ready for
  Signature.invoke() unpacking
- Sections not in the mapping are absent from output
"""

from __future__ import annotations

import pytest

from kaos_agents.context.sections_to_prompt import bind_sections
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types import MemoryType


@pytest.fixture
def memory() -> SessionMemory:
    """A fresh in-memory SessionMemory pre-populated with a few items."""
    mem = SessionMemory(session_id="test-session")
    mem.add(MemoryType.MESSAGES, "user: Hello agent")
    mem.add(MemoryType.MESSAGES, "assistant: Hi! How can I help?")
    mem.add(MemoryType.MESSAGES, "user: What's new?")
    mem.add(MemoryType.FINDINGS, "EPA enforcement actions Q1 2026: 3 cases")
    mem.add(MemoryType.FINDINGS, "Federal Register notice 2026-001 cited")
    return mem


@pytest.mark.unit
class TestBindSections:
    def test_single_section(self, memory: SessionMemory) -> None:
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.MESSAGES: "history"},
        )
        assert "history" in bindings
        assert "user: Hello agent" in bindings["history"]
        assert "user: What's new?" in bindings["history"]

    def test_multiple_sections(self, memory: SessionMemory) -> None:
        bindings = bind_sections(
            memory,
            section_to_field={
                MemoryType.MESSAGES: "history",
                MemoryType.FINDINGS: "facts",
            },
        )
        assert set(bindings) == {"history", "facts"}
        assert "EPA enforcement" in bindings["facts"]
        assert "Federal Register" in bindings["facts"]
        assert "history" in bindings  # MESSAGES rendered

    def test_empty_section_uses_placeholder(self, memory: SessionMemory) -> None:
        # ACTIONS is configured but empty in our fixture
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.ACTIONS: "actions"},
        )
        assert bindings == {"actions": "(empty)"}

    def test_custom_empty_placeholder(self, memory: SessionMemory) -> None:
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.ACTIONS: "actions"},
            empty_placeholder="",
        )
        assert bindings == {"actions": ""}

    def test_separator_customization(self, memory: SessionMemory) -> None:
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.FINDINGS: "facts"},
            separator=" || ",
        )
        assert " || " in bindings["facts"]

    def test_item_prefix(self, memory: SessionMemory) -> None:
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.FINDINGS: "facts"},
            item_prefix="- ",
        )
        # Both findings should be prefixed with bullet
        assert bindings["facts"].count("- ") == 2

    def test_most_recent_truncates(self, memory: SessionMemory) -> None:
        # MESSAGES has 3 items; ask for the last 1
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.MESSAGES: "history"},
            most_recent=1,
        )
        # Only the latest message survives
        assert "user: What's new?" in bindings["history"]
        assert "Hello agent" not in bindings["history"]

    def test_most_recent_zero_returns_empty_placeholder(self, memory: SessionMemory) -> None:
        """most_recent=0 means "no items" → empty placeholder fires."""
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.MESSAGES: "history"},
            most_recent=0,
        )
        assert bindings == {"history": "(empty)"}

    def test_section_not_in_mapping_is_absent(self, memory: SessionMemory) -> None:
        """The output dict only contains the requested fields."""
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.MESSAGES: "history"},
        )
        assert "facts" not in bindings
        # FINDINGS is populated in the fixture but not requested — absent
        assert MemoryType.FINDINGS.value not in bindings

    def test_field_name_not_section_name(self, memory: SessionMemory) -> None:
        """Output keys are caller-chosen field names, not section enum values."""
        bindings = bind_sections(
            memory,
            section_to_field={MemoryType.MESSAGES: "conversation_history"},
        )
        assert "conversation_history" in bindings
        assert "messages" not in bindings  # the MemoryType value, not used

    def test_empty_mapping_returns_empty_dict(self, memory: SessionMemory) -> None:
        assert bind_sections(memory, section_to_field={}) == {}
