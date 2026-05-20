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
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.memory import (
    DEFAULT_SECTIONS,
    MemoryItem,
    MemoryType,
    SectionConfig,
    SummarizationPolicy,
    create_item,
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

    __slots__ = (
        "_chars_per_token",
        "_corpus_ever_attached",
        "_graph",
        "_sections",
        "_session_id",
        "_turn_count",
    )

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
        # ``_sections`` is internal — public callers should read it via
        # :attr:`sections` (KC17-P2-4). It stays underscored because the
        # ``__slots__`` layout is implementation detail and the API needs
        # a frozen, defensively-copied view.
        self._sections: dict[MemoryType, Section] = {}
        for config in sections:
            self._sections[config.memory_type] = Section(config)
        # Track 3 chunk B1: per-session knowledge graph (RDF triple store).
        # Lazy — constructed on first access via :attr:`graph` so sessions
        # that don't use the graph (chat / direct-respond) don't pay the
        # PyO3 import cost or hold an empty Graph instance.
        self._graph: Any | None = None
        # WU-G.2 / #352 — sticky flag set by the agent loop the first
        # turn the IntentSignature's ``corpus_attached`` input goes True
        # (i.e. ``SessionMemory.DOCUMENTS`` was non-empty when the
        # classifier ran). Persisted across turns so a follow-up like
        # "summarize that" still sees the corpus context in the
        # assembled prompt even when the DOCUMENTS bodies would
        # otherwise be trimmed out by the total-budget pass.
        # Defaults to False; ``mark_corpus_attached()`` flips it.
        self._corpus_ever_attached: bool = False

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

    @property
    def sections(self) -> tuple[MemoryType, ...]:
        """Public read-only view of the configured section types.

        KC17-P2-4: replaces the public API's pre-KC17 reach-into
        ``memory._sections`` (a leading-underscore, ``__slots__``-private
        attribute). Returns a defensively-copied tuple so the underlying
        section table cannot be mutated through this accessor; iteration
        order matches the section configuration order. Use
        :attr:`section_names` when you need a ``list[MemoryType]``.

        Public callers (HTTP API, MCP tools, downstream agents) should
        prefer ``memory.sections`` over ``memory._sections``. The private
        attribute is reserved for internal mutation in this module.
        """
        return tuple(self._sections.keys())

    @property
    def graph(self) -> Any:
        """The session's knowledge graph (kaos_graph.Graph instance).

        Lazy-constructed on first access; ``directed=True, multi=True`` —
        directed because PROV-O / CiTO predicates are directional, multi
        because a finding can cite the same source via multiple
        predicates (e.g. both ``cito:cites`` and ``cito:supports``).

        Track 3 chunk B1 — backbone for the per-session RDF triple
        store. Triples are added by chunk-B2's emit_from_event hook
        and persisted as Turtle by SessionStore (chunk B1 closeout).
        """
        if self._graph is None:
            from kaos_graph import Graph  # lazy import — avoids PyO3 cost when unused

            self._graph = Graph(directed=True, multi=True, name=self._session_id)
        return self._graph

    @graph.setter
    def graph(self, value: Any) -> None:
        """Replace the session graph wholesale (used during VFS hydration)."""
        self._graph = value

    @property
    def corpus_ever_attached(self) -> bool:
        """Whether any prior turn in this session had a corpus attached.

        WU-G.2 / #352. Set to True (and persisted) the first turn the
        agent loop observes a non-empty ``MemoryType.DOCUMENTS`` section
        on entry to the classifier — i.e. the same condition that
        drives ``IntentSignature.corpus_attached=True``. Once flipped
        True for a session it stays True for the session's lifetime.

        The flag exists so a turn that arrives AFTER the documents got
        trimmed from the per-turn assembled context (because total
        budget filled with MESSAGES / ACTIONS / FINDINGS) can still
        signal "the corpus is reachable via search_memory" to the
        downstream LLM. See
        :func:`kaos_agents.context.assemble.assemble_context` for the
        retention semantics.
        """
        return self._corpus_ever_attached

    def mark_corpus_attached(self) -> None:
        """Sticky setter for :attr:`corpus_ever_attached`.

        Idempotent. Called by the agent loop (or any other caller that
        observes a non-empty ``MemoryType.DOCUMENTS`` section on
        classification) so the flag survives across turns even if the
        next user message arrives with no fresh attachments.
        """
        self._corpus_ever_attached = True

    # -- Explicit section loading ----------------------------------------------

    def load_explicit(
        self,
        section: MemoryType,
        items: list[str],
    ) -> int:
        """Load content into an explicit (immutable) memory section.

        Explicit sections (ROLE, PLAYBOOKS, PLAN_EXAMPLES) are ``explicit_only=True``
        and cannot be written to via :meth:`add` during the session. This method
        populates them at session creation time, bypassing the explicit-only check.

        Use this to load system instructions, workflow playbooks, and plan examples
        from configuration files.

        Args:
            section: The explicit section to populate.
            items: List of content strings to load.

        Returns:
            Number of items loaded.

        Example::

            memory = SessionMemory("session-1")
            memory.load_explicit(MemoryType.ROLE, [
                "You are a research assistant."
            ])
            memory.load_explicit(MemoryType.PLAYBOOKS, [
                "Data analysis: 1) Parse dataset 2) Compute metrics 3) Visualize",
                "Document extraction: 1) Parse document 2) Extract fields 3) Verify",
            ])
        """
        sec = self._get_section(section)
        count = 0
        for content in items:
            item = create_item(
                section,
                content,
                chars_per_token=self._chars_per_token,
            )
            sec.add_item(item)
            count += 1
        logger.debug(
            "memory.load_explicit: section=%s items=%d tokens=%d",
            section.value,
            count,
            sec.token_count,
        )
        return count

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

    def get_by_ids(self, section: MemoryType, item_ids: set[str]) -> list[MemoryItem]:
        """Retrieve specific items by ID from a section, preserving insertion order."""
        return self._get_section(section).get_by_ids(item_ids)

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
            # WS-0.7: ``sec.get()`` returns items oldest-first, so
            # ``items.pop()`` (previously used) removed the NEWEST first —
            # discarding the most recent conversation context, which is
            # almost always wrong. Pop from the front (index 0) to evict
            # oldest-first, which matches the default FIFO expectation
            # for trimming excess context. Sections with more specific
            # eviction policies (LRU / LFU / PRIORITY) enforce them at
            # Section.add() time; this trim loop is a post-hoc assembly
            # trim that should preserve recency.
            while items and excess > 0:
                removed = items.pop(0)  # Remove from front (oldest)
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
        model: str = DEFAULT_MODEL,
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
            # WU-G.2 / #352 — sticky corpus flag persists with the
            # session snapshot so a multi-day-old "summarize that"
            # follow-up still recognises the corpus context.
            "corpus_ever_attached": self._corpus_ever_attached,
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
        # WU-G.2 / #352 — rehydrate the sticky corpus flag. Older
        # snapshots (pre-0.1.0a19) don't have the key; default to
        # False and let the next classifying turn re-set it.
        memory._corpus_ever_attached = bool(data.get("corpus_ever_attached", False))

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
