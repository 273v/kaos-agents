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
