"""SessionMemory — section-based context management for the agent runtime.

The central memory abstraction. Holds a dict of Section instances, provides
typed add/get operations, enforces per-section and total token budgets,
and supports serialization for VFS persistence.

The agent loop uses SessionMemory to:
1. Add user messages, tool results, findings, and reflections
2. Assemble context for LLM calls (get_sections with budget)
3. Clear ephemeral sections between turns
4. Snapshot/restore for persistence across MCP calls
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_agents.errors import SectionNotConfiguredError
from kaos_agents.memory.sections import Section
from kaos_agents.memory.types import (
    DEFAULT_SECTIONS,
    MemoryItem,
    MemoryType,
    SectionConfig,
    SummarizationPolicy,
)

logger = get_logger(__name__)


class SessionMemory:
    """Section-based memory for a single agent session.

    Holds one Section per configured MemoryType. Sections have independent
    token budgets and eviction policies. The memory itself is mutable —
    it's the state object the agent reads and writes on every turn.

    Thread-safety: not thread-safe. Designed for single-threaded async use
    within one agent turn.
    """

    __slots__ = ("_chars_per_token", "_sections", "_session_id", "_turn_count")

    def __init__(
        self,
        session_id: str,
        *,
        sections: tuple[SectionConfig, ...] = DEFAULT_SECTIONS,
        chars_per_token: float = 4.0,
    ) -> None:
        self._session_id = session_id
        self._chars_per_token = chars_per_token
        self._turn_count = 0
        self._sections: dict[MemoryType, Section] = {}
        for config in sections:
            self._sections[config.memory_type] = Section(config)

    # -- Properties ----------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def total_tokens(self) -> int:
        """Total tokens across all sections."""
        return sum(s.token_count for s in self._sections.values())

    @property
    def section_names(self) -> list[MemoryType]:
        """All configured section types."""
        return list(self._sections.keys())

    # -- Core operations -----------------------------------------------------

    def add(
        self,
        section: MemoryType,
        content: str,
        *,
        priority: int = 0,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Add content to a section.

        Raises KeyError if the section is not configured.
        Raises MemoryBudgetExceededError if the section is explicit-only or REFUSE and full.
        """
        sec = self._get_section(section)
        return sec.add(
            content,
            chars_per_token=self._chars_per_token,
            priority=priority,
            tags=tags,
            metadata=metadata,
        )

    def get(self, section: MemoryType, *, max_tokens: int | None = None) -> list[MemoryItem]:
        """Get items from a section, up to max_tokens."""
        sec = self._get_section(section)
        return sec.get(max_tokens=max_tokens)

    def get_recent(self, section: MemoryType, n: int) -> list[MemoryItem]:
        """Get the N most recent items from a section (newest first)."""
        sec = self._get_section(section)
        return sec.get_recent(n)

    def section_token_count(self, section: MemoryType) -> int:
        """Token count for a single section."""
        return self._get_section(section).token_count

    def section_item_count(self, section: MemoryType) -> int:
        """Item count for a single section."""
        return self._get_section(section).item_count

    def has_section(self, section: MemoryType) -> bool:
        """Check if a section type is configured."""
        return section in self._sections

    # -- Context assembly ----------------------------------------------------

    def get_sections(
        self,
        requested: list[MemoryType],
        total_budget_tokens: int,
        *,
        priority_order: list[MemoryType] | None = None,
    ) -> dict[MemoryType, list[MemoryItem]]:
        """Assemble context from requested sections within a total budget.

        Each section contributes up to its own budget_tokens. If the total
        exceeds total_budget_tokens, sections are trimmed in reverse priority
        order (lowest-priority sections trimmed first).

        Args:
            requested: Which sections to include.
            total_budget_tokens: Maximum total tokens across all sections.
            priority_order: Sections listed first are highest priority (trimmed last).
                           Defaults to the order in `requested`.

        Returns:
            Dict mapping MemoryType -> list of MemoryItems for each requested section.
        """
        if priority_order is None:
            priority_order = requested

        # Phase 1: gather each section's items within its own budget
        raw: dict[MemoryType, list[MemoryItem]] = {}
        section_tokens: dict[MemoryType, int] = {}

        for mt in requested:
            if mt not in self._sections:
                continue
            sec = self._sections[mt]
            items = sec.get(max_tokens=sec.budget_tokens if sec.budget_tokens > 0 else None)
            raw[mt] = items
            section_tokens[mt] = sum(item.token_count for item in items)

        # Phase 2: check total budget
        total = sum(section_tokens.values())
        if total <= total_budget_tokens:
            return raw

        # Phase 3: trim from lowest-priority sections
        # Build trim order: sections NOT in priority_order first, then reversed priority_order
        ordered = [mt for mt in priority_order if mt in raw]
        unordered = [mt for mt in raw if mt not in ordered]
        trim_order = unordered + list(reversed(ordered))

        excess = total - total_budget_tokens
        for mt in trim_order:
            if excess <= 0:
                break
            items = raw[mt]
            while items and excess > 0:
                removed = items.pop()  # Remove from end (most recent)
                freed = removed.token_count
                section_tokens[mt] -= freed
                excess -= freed

        logger.debug(
            "memory.assemble: requested=%d sections, budget=%d, actual=%d tokens",
            len(requested),
            total_budget_tokens,
            sum(section_tokens.values()),
        )
        return raw

    # -- Turn lifecycle ------------------------------------------------------

    def begin_turn(self) -> None:
        """Called at the start of each agent turn.

        Clears ephemeral sections (WORKING, PLANNING_CONTEXT).
        """
        for mt in (MemoryType.WORKING, MemoryType.PLANNING_CONTEXT):
            if mt in self._sections:
                freed = self._sections[mt].clear()
                if freed > 0:
                    logger.debug("memory.begin_turn: cleared %s (%d tokens)", mt.value, freed)

    def end_turn(self) -> None:
        """Called at the end of each agent turn.

        Increments turn counter. For async summarization at end of turn,
        call ``await summarize_turn()`` before ``end_turn()``.
        """
        self._turn_count += 1
        logger.debug(
            "memory.end_turn: turn=%d total_tokens=%d",
            self._turn_count,
            self.total_tokens,
        )

    async def summarize_turn(
        self,
        *,
        model: str = "anthropic:claude-haiku-4-5",
    ) -> int:
        """Summarize sections with ON_TURN or ON_OVERFLOW policy at end of turn.

        ON_TURN sections are always summarized. ON_OVERFLOW sections are
        summarized only if they're over budget.

        Args:
            model: LLM model for summarization.

        Returns:
            Number of sections summarized.
        """
        from kaos_agents.memory.summarize import summarize_items

        summarized = 0
        for mt, section in self._sections.items():
            if not section.needs_summarization:
                continue
            if section.item_count < 2:
                continue  # Nothing meaningful to summarize

            should_summarize = False
            if section.summarization_policy == SummarizationPolicy.ON_TURN:
                should_summarize = True
            elif section.summarization_policy in (
                SummarizationPolicy.ON_OVERFLOW,
                SummarizationPolicy.AUTO,
            ):
                should_summarize = section.is_over_budget

            if not should_summarize:
                continue

            items = section.collect_all_items()
            target_tokens = (
                max(50, section.budget_tokens // 3) if section.budget_tokens > 0 else 200
            )

            summary_text = await summarize_items(
                items,
                mt,
                model=model,
                target_tokens=target_tokens,
                chars_per_token=self._chars_per_token,
            )

            if summary_text:
                # Replace section contents with the summary
                section.clear()
                section.add(
                    f"[Summary of {len(items)} items] {summary_text}",
                    chars_per_token=self._chars_per_token,
                    metadata={"summarized_from": len(items)},
                )
                summarized += 1
                logger.debug(
                    "memory.summarize_turn: section=%s items=%d → summary (%d chars)",
                    mt.value,
                    len(items),
                    len(summary_text),
                )

        return summarized

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize full memory state for persistence."""
        return {
            "session_id": self._session_id,
            "turn_count": self._turn_count,
            "chars_per_token": self._chars_per_token,
            "sections": {
                mt.value: section.to_dict()
                for mt, section in self._sections.items()
                if section.item_count > 0  # Skip empty sections
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        sections: tuple[SectionConfig, ...] = DEFAULT_SECTIONS,
    ) -> SessionMemory:
        """Restore memory from persisted data."""
        memory = cls(
            session_id=data["session_id"],
            sections=sections,
            chars_per_token=data.get("chars_per_token", 4.0),
        )
        memory._turn_count = data.get("turn_count", 0)

        # Build config lookup
        config_map = {c.memory_type: c for c in sections}

        for mt_value, section_data in data.get("sections", {}).items():
            try:
                mt = MemoryType(mt_value)
            except ValueError:
                logger.warning("memory.restore: unknown section type %s, skipping", mt_value)
                continue
            if mt in config_map:
                memory._sections[mt] = Section.from_dict(section_data, config_map[mt])

        logger.debug(
            "memory.restore: session=%s turns=%d tokens=%d",
            memory._session_id,
            memory._turn_count,
            memory.total_tokens,
        )
        return memory

    # -- Internals -----------------------------------------------------------

    def _get_section(self, section: MemoryType) -> Section:
        """Get a section by type, raising SectionNotConfiguredError if not configured."""
        try:
            return self._sections[section]
        except KeyError:
            raise SectionNotConfiguredError(
                f"Section {section.value} is not configured in this memory profile. "
                f"Available sections: {[mt.value for mt in self._sections]}. "
                f"Add the section to the profile or use a different MemoryType.",
            ) from None

    def __repr__(self) -> str:
        return (
            f"SessionMemory(session={self._session_id!r}, "
            f"turn={self._turn_count}, "
            f"sections={len(self._sections)}, "
            f"tokens={self.total_tokens})"
        )
