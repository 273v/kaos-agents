"""Phase 0.A — unit tests for kaos_agents.core.plan."""

from __future__ import annotations

import dataclasses

import pytest

from kaos_agents.core.plan import TurnPlan
from kaos_agents.events.emitter import EventEmitter


def _make_emitter() -> EventEmitter:
    return EventEmitter(session_id="s1", run_id="r1")


def test_construction_with_required_fields() -> None:
    emitter = _make_emitter()
    plan = TurnPlan(
        session_id="s1",
        run_id="r1",
        turn_number=0,
        trigger="hello world",
        emitter=emitter,
    )
    assert plan.session_id == "s1"
    assert plan.run_id == "r1"
    assert plan.turn_number == 0
    assert plan.trigger == "hello world"
    assert plan.emitter is emitter
    assert plan.parent_span_id is None
    assert plan.intent is None
    assert plan.memory is None
    assert plan.working_memory == {}
    assert plan.perceiver is None
    assert plan.actor is None
    assert plan.planner is None
    assert plan.termination_judge is None
    assert plan.escalation_policy is None
    assert plan.permission_policy is None
    assert plan.seed_events == ()


def test_default_factories_produce_independent_dicts() -> None:
    """working_memory must default to a fresh dict per instance,
    not a shared singleton (the @dataclass(field) discipline)."""
    emitter = _make_emitter()
    a = TurnPlan(session_id="s", run_id="r", turn_number=0, trigger=None, emitter=emitter)
    b = TurnPlan(session_id="s", run_id="r", turn_number=0, trigger=None, emitter=emitter)
    assert a.working_memory is not b.working_memory
    assert a.working_memory == {} == b.working_memory


def test_frozen_blocks_assignment() -> None:
    emitter = _make_emitter()
    plan = TurnPlan(
        session_id="s1",
        run_id="r1",
        turn_number=0,
        trigger=None,
        emitter=emitter,
    )
    # Use setattr() so static type checkers don't flag the
    # intentionally-illegal assignment we're trying to provoke.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, "session_id", "mutated")  # noqa: B010 — defeats ty static check on frozen dataclass


def test_frozen_blocks_field_replacement() -> None:
    emitter = _make_emitter()
    plan = TurnPlan(
        session_id="s1",
        run_id="r1",
        turn_number=0,
        trigger=None,
        emitter=emitter,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, "parent_span_id", "deadbeef")  # noqa: B010 — defeats ty static check on frozen dataclass


def test_supports_dataclass_replace() -> None:
    """Frozen dataclasses still expose dataclasses.replace()."""
    emitter = _make_emitter()
    plan = TurnPlan(
        session_id="s1",
        run_id="r1",
        turn_number=0,
        trigger=None,
        emitter=emitter,
    )
    bumped = dataclasses.replace(plan, turn_number=1)
    assert bumped.turn_number == 1
    assert plan.turn_number == 0
