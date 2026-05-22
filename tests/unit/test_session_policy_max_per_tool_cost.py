"""Unit tests for SessionPolicy.max_per_tool_cost_usd (plan §Issue 9 / B1.7).

The loop-level ``max_loop_cost_usd`` cap catches "many cheap calls
accumulating." This field catches "one runaway call" — a single tool
invocation that bills above a per-call threshold trips a budget
event even if the loop-level cap has headroom. Default 0.0 (disabled)
preserves historic behavior; operators tighten it to e.g. 0.05 to
defend against misconfigured-model runaway.
"""

from __future__ import annotations

import pytest

from kaos_agents.types.session_policy import (
    DEFAULT_MAX_PER_TOOL_COST_USD,
    SessionPolicy,
)


@pytest.mark.unit
def test_default_value_is_zero_disabled() -> None:
    """``0.0`` is the explicit "disabled" sentinel — historic behavior."""
    assert DEFAULT_MAX_PER_TOOL_COST_USD == 0.0
    policy = SessionPolicy()
    assert policy.max_per_tool_cost_usd == 0.0


@pytest.mark.unit
def test_explicit_cap_round_trips_through_dataclass() -> None:
    """Operators can tighten the cap; the frozen dataclass holds the value."""
    policy = SessionPolicy(max_per_tool_cost_usd=0.05)
    assert policy.max_per_tool_cost_usd == 0.05


@pytest.mark.unit
def test_per_tool_cap_is_independent_of_loop_cap() -> None:
    """The two caps coexist — neither implies nor constrains the other.
    A policy can have a tight per-tool cap with a generous loop cap, or
    vice versa, depending on the failure mode the operator is guarding.
    """
    tight_per_tool = SessionPolicy(
        max_per_tool_cost_usd=0.01,
        max_loop_cost_usd=1.00,
    )
    assert tight_per_tool.max_per_tool_cost_usd == 0.01
    assert tight_per_tool.max_loop_cost_usd == 1.00


@pytest.mark.unit
def test_per_tool_cap_via_persona_helper() -> None:
    """``SessionPolicy.for_persona`` preserves the default (disabled).
    Personas can opt into tighter caps in a future iteration; this
    test pins the current "persona doesn't override" behavior so a
    future change is forced to be explicit."""
    policy = SessionPolicy.for_persona("research")
    assert policy.max_per_tool_cost_usd == 0.0


# ── Enforcement helper (plan §Issue 9 / B1.7 — agentic_loop.py path) ──


@pytest.mark.unit
def test_per_tool_budget_helper_disabled_at_zero() -> None:
    """``_per_tool_budget_exceeded`` returns False for any cost when
    the cap is 0.0 (disabled). Pins the historic behavior — turns
    that bill $5 in one shot still complete when the cap is off."""
    from kaos_agents.patterns.agentic_loop import _per_tool_budget_exceeded

    policy_disabled = SessionPolicy(max_per_tool_cost_usd=0.0)
    assert _per_tool_budget_exceeded(5.00, policy_disabled) is False
    assert _per_tool_budget_exceeded(0.0001, policy_disabled) is False


@pytest.mark.unit
def test_per_tool_budget_helper_fires_above_cap() -> None:
    """When the cap is set, a single cost delta above the threshold
    trips. The ``>`` comparison is intentional — a delta exactly at
    the cap does NOT trip (gives the operator a clean boundary)."""
    from kaos_agents.patterns.agentic_loop import _per_tool_budget_exceeded

    policy = SessionPolicy(max_per_tool_cost_usd=0.05)
    assert _per_tool_budget_exceeded(0.06, policy) is True
    assert _per_tool_budget_exceeded(0.05, policy) is False  # boundary
    assert _per_tool_budget_exceeded(0.01, policy) is False


@pytest.mark.unit
def test_per_tool_budget_helper_negative_cap_treated_as_disabled() -> None:
    """Defensive: a negative cap is treated as disabled, not as
    "trip on every call." Pydantic would normally reject negatives
    via validation, but the helper handles malformed inputs safely.
    """
    from kaos_agents.patterns.agentic_loop import _per_tool_budget_exceeded

    policy = SessionPolicy(max_per_tool_cost_usd=-1.0)
    assert _per_tool_budget_exceeded(0.5, policy) is False
