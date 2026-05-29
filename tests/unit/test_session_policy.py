"""Tests for SessionPolicy + three-tier elevation taxonomy.

PRD `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.1. Pins:

- Three persona presets (research / drafting / forensics) produce the
  documented soft ceilings.
- ``with_added_groups`` is immutable + only adds groups within the
  soft ceiling.
- ``tier_for`` returns the right tier per the default elevation policy.
- ``to_session_tool_set`` conversion preserves ceiling + denied + auto_narrow.
"""

from __future__ import annotations

import pytest

from kaos_agents.types.session_policy import (
    DEFAULT_ELEVATION_POLICY,
    DEFAULT_MAX_LOOP_COST_USD,
    DEFAULT_MAX_LOOP_ITERATIONS,
    DRAFTING_SOFT_CEILING,
    FORENSICS_SOFT_CEILING,
    PERSONA_SOFT_CEILINGS,
    RESEARCH_SOFT_CEILING,
    SessionPolicy,
)
from kaos_agents.types.session_tool_set import (
    DEFAULT_ALLOWED_GROUPS,
    DEFAULT_DENIED_TOOLS,
)

# ─── Persona presets ─────────────────────────────────────────────────


class TestPersonaPresets:
    """The three soft ceilings the SettingsSheet exposes."""

    def test_research_soft_ceiling_is_eight_groups(self) -> None:
        """Research = every default-on group except authoring."""
        assert (
            frozenset(
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
            == RESEARCH_SOFT_CEILING
        )

    def test_drafting_adds_authoring_to_research(self) -> None:
        assert RESEARCH_SOFT_CEILING | {"authoring"} == DRAFTING_SOFT_CEILING

    def test_forensics_is_tight(self) -> None:
        """Per round-2 decision: forensics persona is tight (no auto-elevation
        to web / browser / netinfra). Matches Relativity aiR / DISCO Cecilia
        — forensics workflows stay in lane.
        """
        assert frozenset({"forensics", "vfs"}) == FORENSICS_SOFT_CEILING

    def test_neither_persona_includes_programs_or_agents(self) -> None:
        """programs + agents are always red-blocked. No persona can reach
        them via auto-elevation."""
        for ceiling in (
            RESEARCH_SOFT_CEILING,
            DRAFTING_SOFT_CEILING,
            FORENSICS_SOFT_CEILING,
        ):
            assert "programs" not in ceiling
            assert "agents" not in ceiling


class TestForPersona:
    """SessionPolicy.for_persona builds a complete config from a name."""

    def test_research_persona_starts_with_full_soft_ceiling(self) -> None:
        policy = SessionPolicy.for_persona("research")
        assert policy.allowed_groups == RESEARCH_SOFT_CEILING
        assert policy.soft_ceiling == RESEARCH_SOFT_CEILING
        assert policy.denied_tools == DEFAULT_DENIED_TOOLS
        assert policy.auto_elevate is True
        assert policy.auto_loop is True
        assert policy.auto_narrow is True

    def test_drafting_persona_includes_authoring(self) -> None:
        policy = SessionPolicy.for_persona("drafting")
        assert "authoring" in policy.allowed_groups
        assert "authoring" in policy.soft_ceiling

    def test_forensics_persona_is_tight(self) -> None:
        policy = SessionPolicy.for_persona("forensics")
        assert policy.allowed_groups == FORENSICS_SOFT_CEILING
        assert policy.soft_ceiling == FORENSICS_SOFT_CEILING
        # The forensics persona CANNOT auto-elevate web — it's outside
        # the soft ceiling.
        assert "web" not in policy.soft_ceiling
        assert policy.is_blocked("web") is True

    def test_unknown_persona_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown persona"):
            SessionPolicy.for_persona("unicorn")

    def test_default_is_research(self) -> None:
        assert SessionPolicy.default() == SessionPolicy.for_persona("research")

    def test_registry_is_source_of_truth(self) -> None:
        """Every key in PERSONA_SOFT_CEILINGS builds a policy whose
        soft_ceiling is that registry value — so adding a persona is a
        single registry entry with no parallel branch to maintain."""
        assert set(PERSONA_SOFT_CEILINGS) == {"research", "drafting", "forensics"}
        for name, ceiling in PERSONA_SOFT_CEILINGS.items():
            policy = SessionPolicy.for_persona(name)
            assert policy.soft_ceiling == ceiling
            assert policy.allowed_groups == ceiling

    def test_unknown_persona_error_enumerates_registry(self) -> None:
        """The error lists the valid personas from the registry itself,
        so the message can't drift from the supported set."""
        with pytest.raises(ValueError, match="Unknown persona") as exc:
            SessionPolicy.for_persona("unicorn")
        message = str(exc.value)
        for name in PERSONA_SOFT_CEILINGS:
            assert repr(name) in message


# ─── Elevation tier lookup ───────────────────────────────────────────


class TestElevationTiers:
    """The three-tier permission model: green-auto / yellow-confirm / red-blocked."""

    def test_default_policy_covers_eleven_groups(self) -> None:
        """Every group in the 11-group taxonomy has a documented tier."""
        assert set(DEFAULT_ELEVATION_POLICY) == {
            "web",
            "browser",
            "netinfra",
            "documents",
            "citations",
            "vfs",
            "forensics",
            "retrieval",
            "authoring",
            "programs",
            "agents",
        }

    @pytest.mark.parametrize(
        ("group", "expected_tier"),
        [
            # green-auto — safe read-only
            ("web", "green-auto"),
            ("documents", "green-auto"),
            ("citations", "green-auto"),
            ("retrieval", "green-auto"),
            ("vfs", "green-auto"),
            ("forensics", "green-auto"),
            # yellow-confirm — meaningful side effects / cost
            ("browser", "yellow-confirm"),
            ("authoring", "yellow-confirm"),
            ("netinfra", "yellow-confirm"),
            # red-blocked — never auto-elevate
            ("programs", "red-blocked"),
            ("agents", "red-blocked"),
        ],
    )
    def test_default_tiers_per_group(self, group: str, expected_tier: str) -> None:
        policy = SessionPolicy.default()
        assert policy.tier_for(group) == expected_tier

    def test_unknown_group_defaults_to_red_blocked(self) -> None:
        """Defaulting to red-blocked is safer than green-auto for groups
        nobody has decided about."""
        policy = SessionPolicy.default()
        assert policy.tier_for("totally-new-group") == "red-blocked"

    def test_can_auto_elevate_research_persona(self) -> None:
        """Research persona can auto-elevate every green-auto group."""
        policy = SessionPolicy.for_persona("research").with_removed_groups({"web", "browser"})
        assert policy.can_auto_elevate("web") is True
        # Browser is yellow-confirm, not green-auto, so can_auto_elevate is False
        # even though it's in the soft ceiling.
        assert policy.can_auto_elevate("browser") is False
        assert policy.needs_confirmation("browser") is True

    def test_forensics_persona_cannot_elevate_web(self) -> None:
        """Forensics persona's soft ceiling excludes web — auto-elevation
        is impossible even though web is green-auto in the default
        elevation policy."""
        policy = SessionPolicy.for_persona("forensics")
        assert policy.can_auto_elevate("web") is False
        assert policy.is_blocked("web") is True

    def test_programs_always_red(self) -> None:
        """Even if a power user includes ``programs`` in their soft
        ceiling, the elevation policy marks it red-blocked. Belt +
        suspenders."""
        # Construct a hypothetical wider soft ceiling.
        policy = SessionPolicy(
            allowed_groups=frozenset({"documents"}),
            soft_ceiling=frozenset({"documents", "programs"}),
            denied_tools=DEFAULT_DENIED_TOOLS,
        )
        assert policy.can_auto_elevate("programs") is False
        assert policy.is_blocked("programs") is True


# ─── Immutable updates ───────────────────────────────────────────────


class TestImmutableUpdates:
    """`with_added_groups` and `with_removed_groups` return new instances
    and only act within the soft ceiling."""

    def test_with_added_groups_returns_new_policy(self) -> None:
        original = SessionPolicy.for_persona("research").with_removed_groups({"web"})
        elevated = original.with_added_groups({"web"})
        assert "web" in elevated.allowed_groups
        # Original is unchanged.
        assert "web" not in original.allowed_groups
        assert original is not elevated

    def test_with_added_groups_filters_outside_soft_ceiling(self) -> None:
        """Adding a group outside the soft ceiling is silently dropped —
        the caller should have checked ``is_blocked`` first."""
        original = SessionPolicy.for_persona("forensics")
        # ``web`` is NOT in forensics' soft ceiling.
        new_policy = original.with_added_groups({"web"})
        assert "web" not in new_policy.allowed_groups
        # Returns the unchanged instance (no-op).
        assert new_policy.allowed_groups == original.allowed_groups

    def test_with_added_groups_no_op_returns_self_equivalent(self) -> None:
        """Adding groups already in ``allowed_groups`` is a no-op."""
        original = SessionPolicy.for_persona("research")
        result = original.with_added_groups({"web"})
        assert result.allowed_groups == original.allowed_groups

    def test_with_removed_groups_narrows_allowed(self) -> None:
        original = SessionPolicy.for_persona("research")
        narrowed = original.with_removed_groups({"web", "browser"})
        assert "web" not in narrowed.allowed_groups
        assert "browser" not in narrowed.allowed_groups
        # Soft ceiling is NOT narrowed — only the current working set.
        assert "web" in narrowed.soft_ceiling
        assert "browser" in narrowed.soft_ceiling

    def test_frozen_dataclass_blocks_direct_mutation(self) -> None:
        import dataclasses

        policy = SessionPolicy.default()
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            policy.allowed_groups = frozenset()  # ty: ignore[invalid-assignment]


# ─── Loop config defaults ────────────────────────────────────────────


class TestLoopConfigDefaults:
    """Loop budgets — three independent limiters."""

    def test_default_max_iterations_is_three(self) -> None:
        assert DEFAULT_MAX_LOOP_ITERATIONS == 3
        assert SessionPolicy.default().max_loop_iterations == 3

    def test_default_max_cost_usd_is_a_quarter(self) -> None:
        assert pytest.approx(0.25) == DEFAULT_MAX_LOOP_COST_USD
        assert SessionPolicy.default().max_loop_cost_usd == pytest.approx(0.25)

    def test_default_wall_clock_is_60_seconds(self) -> None:
        assert SessionPolicy.default().max_loop_wall_clock_seconds == pytest.approx(60.0)

    def test_auto_loop_defaults_true(self) -> None:
        """The default session enables the AgenticLoop. Pre-loop
        behavior is single-turn ReAct, available via auto_loop=False."""
        assert SessionPolicy.default().auto_loop is True

    def test_auto_elevate_defaults_true(self) -> None:
        assert SessionPolicy.default().auto_elevate is True


# ─── SessionToolSet adapter ──────────────────────────────────────────


class TestToSessionToolSet:
    """Conversion preserves ceiling + denied + auto_narrow."""

    def test_round_trip_preserves_ceiling(self) -> None:
        policy = SessionPolicy.for_persona("research")
        ts = policy.to_session_tool_set()
        assert ts.allowed_groups == policy.allowed_groups
        assert ts.denied_tools == policy.denied_tools
        assert ts.auto_narrow == policy.auto_narrow

    def test_round_trip_after_elevation(self) -> None:
        """Elevation only updates SessionPolicy; the derived SessionToolSet
        reflects the new ceiling on subsequent reads."""
        before = SessionPolicy.for_persona("research").with_removed_groups({"web"})
        ts_before = before.to_session_tool_set()
        assert "web" not in ts_before.allowed_groups

        after = before.with_added_groups({"web"})
        ts_after = after.to_session_tool_set()
        assert "web" in ts_after.allowed_groups

    def test_default_session_tool_set_default_allowed_groups(self) -> None:
        """SessionPolicy.default()'s to_session_tool_set() matches the
        same default ceiling SessionToolSet.default() defines — keeps
        the two values in sync."""
        policy = SessionPolicy.default()
        ts = policy.to_session_tool_set()
        # Research persona ⊇ DEFAULT_ALLOWED_GROUPS (the latter is the
        # default 7-group SessionToolSet ceiling; the policy adds
        # ``browser`` on top of that for the persona).
        for group in DEFAULT_ALLOWED_GROUPS:
            assert group in ts.allowed_groups
