"""SessionToolSet — declarative per-session tool-allow-list.

Track 4 chunk T4-5 introduced the value type. PRD
`kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4 (PR 2)
adds:

- :data:`DEFAULT_ALLOWED_GROUPS` — the 7-group "research" preset
  every fresh session starts with (web + browser + documents +
  citations + vfs + forensics + retrieval). Excludes ``netinfra``
  (default-off — sessions that want it opt in), ``authoring``
  (writers — opt-in for drafting workflows), ``programs`` (the
  full kaos-llm-core typed-program / optimizer / codec / batch
  surface — opt-in for power users), and ``agents`` (the
  self-recursive dispatch tools — opt-in *and* still blocked by
  :data:`DEFAULT_DENIED_TOOLS`).
- :data:`DEFAULT_DENIED_TOOLS` — the 4 self-recursive kaos-agents
  tools (``kaos-agent-chat``, ``kaos-agent-plan``,
  ``kaos-agent-findings``, ``kaos-agent-corpus-filter``). Registered
  in the runtime so power-user topologies can wire them as
  delegate sub-agents, but denied by default at the ceiling so
  accidental opt-in doesn't trigger infinite recursion.
- :attr:`SessionToolSet.auto_narrow` — the per-turn planner toggle.
  When ``True`` (default), the chat router runs the
  :class:`TurnToolPolicy` Program before each tool-able turn to
  narrow the ceiling to just the groups this message actually
  needs (saves token cost + reduces hallucination risk). When
  ``False``, the full ceiling passes to ReAct unchanged.
- :meth:`SessionToolSet.default` — a `default()` classmethod that
  returns the canonical "fresh session" config (the
  :data:`DEFAULT_ALLOWED_GROUPS` ceiling + :data:`DEFAULT_DENIED_TOOLS`
  + ``auto_narrow=True``). Use this instead of constructing a
  bare :class:`SessionToolSet` for new sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default 7-group "research" ceiling for a fresh session.
# Per PRD §2.1 round-1 decision #1: web is default-on so the 80% case
# (legal research that needs to fetch / search) Just Works without a
# preset click. Excludes:
#   - netinfra (DNS / WHOIS / cert intel) — diligence / abuse-investigation
#     workflows; default off, opt-in per session
#   - authoring (DOCX / PPTX / XLSX / PDF writers) — drafting workflows;
#     default off, opt-in per session
#   - programs (kaos-llm-core typed programs + alpha-* extractors) —
#     power-user surface; default off
#   - agents (self-recursive chat/plan/findings/corpus-filter) — multi-
#     agent topologies; default off + always-denied via DEFAULT_DENIED_TOOLS
DEFAULT_ALLOWED_GROUPS: frozenset[str] = frozenset(
    {
        "web",
        "browser",
        "documents",
        "citations",
        "vfs",
        "forensics",
        "retrieval",
    }
)


# The 4 self-recursive kaos-agents tools. Always denied by default so
# a session that accidentally opts into the ``agents`` group does not
# trigger a runaway sub-agent loop. A power-user topology that needs
# agent-as-tool dispatch explicitly clears this denial per-session by
# constructing a SessionToolSet with a narrower denied_tools.
DEFAULT_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "kaos-agent-chat",
        "kaos-agent-plan",
        "kaos-agent-findings",
        "kaos-agent-corpus-filter",
    }
)


@dataclass(frozen=True, slots=True)
class SessionToolSet:
    """Per-session tool allow / deny configuration.

    Resolution semantics (apply in order):

    1. If ``allowed_tools`` is non-empty, only tools whose name is in
       that set pass the allow check.
    2. If ``allowed_groups`` is non-empty (and the tool didn't pass
       step 1's allow), the tool passes if any of its groups is in
       this set. (Tools in no group fail this check.)
    3. If both ``allowed_tools`` and ``allowed_groups`` are empty,
       the default is **allow** (no restriction yet).
    4. ``denied_tools`` always wins — a tool present here is blocked
       regardless of any allow list.

    All collections are frozensets so the value type is hashable and
    comparison-safe.

    Attributes:
        allowed_tools: Explicit allow-list of tool names. Empty
            means "allow any tool that survives the rest of the
            checks".
        allowed_groups: Allow-list of group names; resolved through
            the :class:`ToolGroupRegistry` at filter time. Empty
            means "no group restriction".
        denied_tools: Explicit deny-list. Overrides every allow.
        auto_narrow: When ``True``, the chat router runs the
            per-turn :class:`TurnToolPolicy` Program against this
            ceiling before each tool-able turn. When ``False``, the
            full ceiling passes through unchanged. Default ``True``
            — narrowing saves token cost + reduces hallucination
            from over-broad catalogues.
    """

    allowed_tools: frozenset[str] = frozenset()
    allowed_groups: frozenset[str] = frozenset()
    denied_tools: frozenset[str] = frozenset()
    auto_narrow: bool = True

    @property
    def is_unrestricted(self) -> bool:
        """True when no allow / deny list constrains the catalogue."""
        return not self.allowed_tools and not self.allowed_groups and not self.denied_tools

    @classmethod
    def unrestricted(cls) -> SessionToolSet:
        """Identity-stable empty config (allow-all, no denies)."""
        return _UNRESTRICTED

    @classmethod
    def default(cls) -> SessionToolSet:
        """Canonical "fresh session" config.

        Returns a SessionToolSet with:

        - ``allowed_groups`` = :data:`DEFAULT_ALLOWED_GROUPS` (the
          7-group "research" preset)
        - ``denied_tools`` = :data:`DEFAULT_DENIED_TOOLS` (the 4
          self-recursive kaos-agents tools)
        - ``auto_narrow`` = ``True``

        Use this instead of ``SessionToolSet()`` when creating a new
        session — the bare constructor returns the "unrestricted"
        config (allow-all), which surfaces every registered tool
        including the recursive ones.
        """
        return _DEFAULT


_UNRESTRICTED = SessionToolSet()

_DEFAULT = SessionToolSet(
    allowed_groups=DEFAULT_ALLOWED_GROUPS,
    denied_tools=DEFAULT_DENIED_TOOLS,
    auto_narrow=True,
)


__all__ = [
    "DEFAULT_ALLOWED_GROUPS",
    "DEFAULT_DENIED_TOOLS",
    "SessionToolSet",
]
