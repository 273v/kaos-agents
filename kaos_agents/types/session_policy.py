"""SessionPolicy — two-tier ceiling + loop config for the AgenticLoop.

Per `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.1. Extends the existing :class:`SessionToolSet` with:

- **`soft_ceiling`** — the maximum the chat router may auto-elevate
  ``allowed_groups`` toward without explicit user consent. Persona presets
  set this:

    - ``"research"`` → all 9 visible groups except ``authoring``
    - ``"drafting"`` → research + ``authoring``
    - ``"forensics"`` → ``{forensics, vfs}`` only (tight)

- **`elevation_policy`** — three-tier permission model mirroring Claude
  Code's permission modes (``green-auto`` / ``yellow-confirm`` /
  ``red-blocked``). The agentic loop consults this map when the per-turn
  planner reports ``dropped_groups``: green elevates silently with an
  audit event, yellow pauses for an inline approval card, red is hard
  refused with a clear explanation.

- **Loop config** — ``auto_loop`` toggle + ``max_loop_iterations`` +
  ``max_loop_cost_usd`` + ``max_loop_wall_clock_seconds``. Three
  independent limiters (per Pydantic AI's documented best practice — see
  ``docs/internal/agentic-loop-auto-elevation-plan.md`` §7) so one
  pathological dimension (cost runaway, infinite plan-replan, or LLM
  hang) can't escape its peers.

The pre-existing :class:`SessionToolSet` is **not** replaced — it remains
the wire-stable ceiling-enforcement primitive. :class:`SessionPolicy`
wraps it with the auto-elevation + loop config the chat router needs,
and :meth:`SessionPolicy.to_session_tool_set` produces the
``SessionToolSet`` shape callers downstream of the ceiling still want.

Backward compatibility: every new field has a sensible default, so a
caller that constructs ``SessionPolicy(allowed_groups=...)`` without
specifying the elevation / loop knobs gets the same behavior as the
current ``SessionToolSet`` plus auto-elevation enabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kaos_agents.types.session_tool_set import (
    DEFAULT_ALLOWED_GROUPS,
    DEFAULT_DENIED_TOOLS,
    SessionToolSet,
)

# Three-tier elevation taxonomy. Each group maps to one of these tiers;
# the chat router picks the right behavior when the planner reports
# ``dropped_groups``.
#
#   - ``"green-auto"``  → the chat router auto-elevates, emits a
#     ``ToolPolicyElevated`` event, the turn proceeds without pausing.
#     Use for: safe read-only groups the user almost always wants
#     (``web``, ``documents``, ``citations``, ``retrieval``, ``vfs``).
#
#   - ``"yellow-confirm"`` → the chat router pauses, emits a
#     ``CapabilityRequested`` event, the SPA renders an inline approval
#     card. The agent receives the user's decision on the next turn and
#     resumes from where it paused. Use for: groups with meaningful side
#     effects or resource cost (``browser`` — Chromium spin-up,
#     ``authoring`` — file writes, ``netinfra`` — egress).
#
#   - ``"red-blocked"`` → the chat router refuses elevation, the agent
#     receives a clear explanation it can surface to the user. Use for:
#     groups the session has explicitly opted out of via SettingsSheet,
#     or groups inappropriate to the persona (``programs`` and ``agents``
#     in every default persona).
ElevationTier = Literal["green-auto", "yellow-confirm", "red-blocked"]


# Default tier mapping. Per the design doc §3.1 + the round-2 PRD
# competitive analysis: mirrors Claude Code's permission modes.
DEFAULT_ELEVATION_POLICY: dict[str, ElevationTier] = {
    # Safe read-only — auto-elevate within the soft ceiling without
    # pausing the turn. These are the groups a "Research" persona
    # uses 80% of the time.
    "web": "green-auto",
    "documents": "green-auto",
    "citations": "green-auto",
    "retrieval": "green-auto",
    "vfs": "green-auto",
    "forensics": "green-auto",
    # Meaningful side effects / resource cost — pause for inline
    # approval. These cost real money (Chromium RAM, file writes,
    # egress traffic that may show up in network logs).
    "browser": "yellow-confirm",
    "authoring": "yellow-confirm",
    "netinfra": "yellow-confirm",
    # Hard-blocked groups. ``agents`` is denied by tool name via
    # ``DEFAULT_DENIED_TOOLS`` even when the group is in the ceiling;
    # ``programs`` is the kaos-llm-core typed-program / alpha-* surface
    # which is power-user only.
    "programs": "red-blocked",
    "agents": "red-blocked",
}


# Three-corner "Research" persona — the 80% default. Soft ceiling
# includes every visible group except authoring (writers) and the two
# always-denied groups (programs, agents). Initial ``allowed_groups`` is
# the same set; auto-elevation is a no-op when ceiling == soft_ceiling.
RESEARCH_SOFT_CEILING: frozenset[str] = frozenset(
    {
        "web",
        "browser",
        "netinfra",
        "documents",
        "citations",
        "vfs",
        "forensics",
        "retrieval",
    }
)


# "Drafting" persona — research + ``authoring`` (DOCX / PPTX / XLSX /
# PDF writers). Auto-elevation can pull in ``authoring`` when the
# planner detects "draft / write / redline" intent.
DRAFTING_SOFT_CEILING: frozenset[str] = RESEARCH_SOFT_CEILING | {"authoring"}


# "Forensics" persona — tight ceiling, no auto-elevation possible
# outside the local-byte-processing surface. Matches the
# Relativity aiR for Review / DISCO Cecilia pattern: forensics workflows
# stay in lane; surprise web egress is a confidentiality bug.
FORENSICS_SOFT_CEILING: frozenset[str] = frozenset({"forensics", "vfs"})


# Default loop budget — per the round-2 PRD §7 competitive analysis
# (Pydantic AI's usage_limits + Harvey Deep Research budgets). These are
# the **maxima** the loop respects; a turn that resolves in one
# iteration costs maybe $0.005 (Haiku planner + Haiku critic +
# Haiku/Sonnet worker) — the budget exists to cap pathological loops,
# not to constrain typical use.
DEFAULT_MAX_LOOP_ITERATIONS: int = 3
DEFAULT_MAX_LOOP_COST_USD: float = 0.25
DEFAULT_MAX_LOOP_WALL_CLOCK_SECONDS: float = 60.0

# Plan §Issue 9 / B1.7 — per-tool-call intra-iteration cost cap.
# Defense-in-depth alongside the loop-level ``max_loop_cost_usd``:
# the loop cap catches "many cheap calls accumulating"; this cap
# catches "one runaway call" (e.g. a kaos-llm-core ReAct with a
# misconfigured model that bills $5 in one shot). Defaults are
# permissive — the loop cap is the primary control — but operators
# running budget-sensitive workloads can tighten this to a hard
# per-invocation ceiling. ``0.0`` disables (the historic behavior).
DEFAULT_MAX_PER_TOOL_COST_USD: float = 0.0

# Plan §Issue 8 / B1.4 — tool-result staleness gate.
# A long deal-room session may surface a cached web fetch / FR rule
# / EDGAR filing from Day 1 on Day 3, even though the source could
# have changed in between. ``tool_staleness_ttl_seconds`` is the
# wall-clock TTL after which a tool result is flagged as
# potentially stale and the context-assembly layer tags it with
# ``needs_reverification=True`` in the next thinking note. ``0.0``
# (default) disables — preserves historic behavior. Operators
# running multi-day deal rooms tighten this to e.g. 3600.0 (1 hr)
# for the financial-filing workflow or 86400.0 (1 day) for the
# slower contract-review path.
DEFAULT_TOOL_STALENESS_TTL_SECONDS: float = 0.0


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """Per-session tool-policy configuration: two-tier ceiling + loop config.

    Attributes:
        allowed_groups: Current working set of allowed tool groups. The
            chat router may auto-elevate this toward ``soft_ceiling``
            when the per-turn planner reports ``dropped_groups``
            intersecting the soft ceiling. Starts equal to the persona's
            initial set (typically the entire ``soft_ceiling``) but can
            narrow if the user manually deselects groups in the
            SettingsSheet.
        soft_ceiling: Maximum the chat router may elevate
            ``allowed_groups`` to **without explicit user consent**.
            Set at session creation by the persona chip (research /
            drafting / forensics) and immutable for the session.
        denied_tools: Explicit deny list. Wins over every allow rule.
            Inherits ``DEFAULT_DENIED_TOOLS`` (the 4 self-recursive
            kaos-agents tools).
        elevation_policy: Per-group tier mapping for the three-tier
            elevation taxonomy. Defaults to
            :data:`DEFAULT_ELEVATION_POLICY` which mirrors Claude Code's
            permission modes.
        auto_narrow: Per-turn :class:`TurnToolPolicy` planner toggle.
            When True, the chat router runs the planner to narrow the
            ceiling to the groups this message actually needs.
        auto_elevate: Master toggle for auto-elevation. When True, the
            chat router consults ``elevation_policy`` to decide what
            happens when the planner wants groups outside
            ``allowed_groups``. When False, the chat router runs the
            turn with only ``allowed_groups`` (no elevation) — the
            pre-loop behavior.
        auto_loop: Enable the multi-iteration AgenticLoop with goal
            checking between iterations. When False, the chat router
            runs a single ReAct turn (the pre-loop behavior).
        max_loop_iterations: Hard cap on plan → execute → check cycles
            per turn. Defaults to 3.
        max_loop_cost_usd: Hard cap on cumulative LLM cost per turn
            (planner + critic + worker + any goal-check calls).
            Defaults to $0.25.
        max_loop_wall_clock_seconds: Hard cap on wall-clock time per
            turn. Defaults to 60 seconds. Defense in depth against an
            LLM that hangs (which the cost cap alone wouldn't catch).
    """

    allowed_groups: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_GROUPS)
    soft_ceiling: frozenset[str] = field(default_factory=lambda: RESEARCH_SOFT_CEILING)
    denied_tools: frozenset[str] = field(default_factory=lambda: DEFAULT_DENIED_TOOLS)
    elevation_policy: dict[str, ElevationTier] = field(
        default_factory=lambda: dict(DEFAULT_ELEVATION_POLICY)
    )
    auto_narrow: bool = True
    auto_elevate: bool = True
    auto_loop: bool = True
    max_loop_iterations: int = DEFAULT_MAX_LOOP_ITERATIONS
    max_loop_cost_usd: float = DEFAULT_MAX_LOOP_COST_USD
    max_loop_wall_clock_seconds: float = DEFAULT_MAX_LOOP_WALL_CLOCK_SECONDS
    # Plan §Issue 9 / B1.7 — per-tool-call intra-iteration cost cap.
    # A single tool invocation that bills above this threshold trips a
    # ``BudgetExceeded`` event and short-circuits the loop, even if
    # the loop-level ``max_loop_cost_usd`` has headroom. ``0.0``
    # (default) disables the cap — preserves historic behavior; the
    # loop cap remains the primary control. Operators tighten this
    # to e.g. ``0.05`` to catch a misconfigured-model runaway.
    max_per_tool_cost_usd: float = DEFAULT_MAX_PER_TOOL_COST_USD
    # Plan §Issue 8 / B1.4 — tool-result staleness gate. ``0.0``
    # (default) disables — preserves historic behavior. Operators
    # running multi-day deal-room sessions set this to e.g. 3600.0
    # (1 hr) so the planner re-fetches stale FR / EDGAR results.
    tool_staleness_ttl_seconds: float = DEFAULT_TOOL_STALENESS_TTL_SECONDS

    # ─── Construction helpers ────────────────────────────────────────

    @classmethod
    def for_persona(cls, persona: str) -> SessionPolicy:
        """Build a SessionPolicy for a named persona preset.

        Args:
            persona: ``"research"`` | ``"drafting"`` | ``"forensics"``.

        Returns:
            A SessionPolicy whose ``allowed_groups`` starts equal to the
            persona's ``soft_ceiling``, with ``auto_elevate`` +
            ``auto_loop`` both enabled.

        Raises:
            ValueError: For unknown persona names.
        """
        if persona == "research":
            sc = RESEARCH_SOFT_CEILING
        elif persona == "drafting":
            sc = DRAFTING_SOFT_CEILING
        elif persona == "forensics":
            sc = FORENSICS_SOFT_CEILING
        else:
            raise ValueError(
                f"Unknown persona {persona!r}; expected one of 'research', 'drafting', 'forensics'."
            )
        return cls(
            allowed_groups=sc,
            soft_ceiling=sc,
            denied_tools=DEFAULT_DENIED_TOOLS,
        )

    @classmethod
    def default(cls) -> SessionPolicy:
        """Canonical fresh-session config — the Research persona."""
        return cls.for_persona("research")

    # ─── Immutable updates ───────────────────────────────────────────

    def with_added_groups(self, groups: frozenset[str] | set[str]) -> SessionPolicy:
        """Return a new SessionPolicy with ``groups`` added to ``allowed_groups``.

        Only groups already in ``soft_ceiling`` can be added —
        attempting to add a group outside the soft ceiling is a no-op
        (the caller should have caught this via :meth:`tier_for`).

        Used by the chat router to implement green-auto elevation
        without mutating the existing policy in place.
        """
        addable = frozenset(groups) & self.soft_ceiling
        if not addable:
            return self
        return self._with_replacements(allowed_groups=self.allowed_groups | addable)

    def with_removed_groups(self, groups: frozenset[str] | set[str]) -> SessionPolicy:
        """Return a new SessionPolicy with ``groups`` removed from ``allowed_groups``.

        Used when the user explicitly deselects a group via the
        SettingsSheet — the user's intent is "narrow this from what the
        persona allowed."
        """
        removed = self.allowed_groups - frozenset(groups)
        if removed == self.allowed_groups:
            return self
        return self._with_replacements(allowed_groups=removed)

    def _with_replacements(self, **changes) -> SessionPolicy:
        """Internal helper for frozen-dataclass field replacement."""
        from dataclasses import replace

        return replace(self, **changes)

    # ─── Tier lookup ─────────────────────────────────────────────────

    def tier_for(self, group: str) -> ElevationTier:
        """Return the elevation tier for ``group``.

        Defaults to ``"red-blocked"`` for unknown groups — safer than
        defaulting to ``green-auto`` for a group nobody has decided
        about.
        """
        return self.elevation_policy.get(group, "red-blocked")

    def can_auto_elevate(self, group: str) -> bool:
        """True iff ``group`` is in the soft ceiling AND green-auto tier."""
        return group in self.soft_ceiling and self.tier_for(group) == "green-auto"

    def needs_confirmation(self, group: str) -> bool:
        """True iff ``group`` is in the soft ceiling AND yellow-confirm tier."""
        return group in self.soft_ceiling and self.tier_for(group) == "yellow-confirm"

    def is_blocked(self, group: str) -> bool:
        """True iff ``group`` is outside the soft ceiling OR red-blocked."""
        return group not in self.soft_ceiling or self.tier_for(group) == "red-blocked"

    # ─── Conversion to SessionToolSet ────────────────────────────────

    def to_session_tool_set(self) -> SessionToolSet:
        """Adapter: produce the SessionToolSet shape ``filter_tools`` consumes.

        The wider-context SessionPolicy is for the chat router + the
        AgenticLoop orchestrator. Downstream ceiling enforcement (the
        proxy filter, the ReAct catalogue) operates on
        :class:`SessionToolSet`, which is wire-stable and predates the
        loop work. This adapter is the boundary.
        """
        return SessionToolSet(
            allowed_groups=self.allowed_groups,
            denied_tools=self.denied_tools,
            auto_narrow=self.auto_narrow,
        )


__all__ = [
    "DEFAULT_ELEVATION_POLICY",
    "DEFAULT_MAX_LOOP_COST_USD",
    "DEFAULT_MAX_LOOP_ITERATIONS",
    "DEFAULT_MAX_LOOP_WALL_CLOCK_SECONDS",
    "DEFAULT_MAX_PER_TOOL_COST_USD",
    "DRAFTING_SOFT_CEILING",
    "FORENSICS_SOFT_CEILING",
    "RESEARCH_SOFT_CEILING",
    "ElevationTier",
    "SessionPolicy",
]
