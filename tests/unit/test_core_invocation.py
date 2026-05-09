"""Phase 0.A — unit tests for kaos_agents.core.invocation."""

from __future__ import annotations

import asyncio
import time

import pytest

from kaos_agents.base.event import KaosEvent
from kaos_agents.core.invocation import (
    TurnInvocation,
    _active_turn_var,
    current_turn,
)
from kaos_agents.events.lifecycle import UsageObserved
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage


def _make_event(seq: int = 0) -> KaosEvent:
    """Cheap concrete KaosEvent for stream-mutation tests."""
    return UsageObserved(
        timestamp=time.monotonic(),
        sequence=seq,
        session_id="s1",
        run_id="r1",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=0.001,
        source="unit-test",
    )


# --- Construction ---------------------------------------------------------


def test_construction_defaults() -> None:
    inv = TurnInvocation()
    assert inv.id  # auto hex id
    assert len(inv.id) == 16
    assert inv.session_id == ""
    assert inv.run_id == ""
    assert inv.turn_number == 0
    assert inv.agent_envelope_hash == ""
    assert inv.trigger is None
    assert inv.intent is None
    assert inv.plan is None
    assert inv.output == ""
    assert inv.tool_executions == ()
    assert inv.events == ()
    assert inv.usage is ZERO_USAGE  # singleton identity
    assert inv.cost_usd == 0.0
    assert inv.children == ()
    assert inv.escalations == ()
    assert inv.error is None
    assert inv.extras == {}
    assert inv.started_at is not None
    assert inv.finished_at is None


def test_construction_with_overrides() -> None:
    inv = TurnInvocation(
        session_id="abc",
        run_id="run_01",
        turn_number=3,
        agent_envelope_hash="sha256:deadbeef",
    )
    assert inv.session_id == "abc"
    assert inv.run_id == "run_01"
    assert inv.turn_number == 3
    assert inv.agent_envelope_hash == "sha256:deadbeef"


def test_default_id_is_unique() -> None:
    invs = [TurnInvocation() for _ in range(100)]
    ids = {inv.id for inv in invs}
    assert len(ids) == 100


# --- Mutation -------------------------------------------------------------


def test_add_event_mutates_in_place() -> None:
    inv = TurnInvocation()
    e1 = _make_event(seq=0)
    e2 = _make_event(seq=1)
    inv.add_event(e1)
    inv.add_event(e2)
    assert inv.events == (e1, e2)


def test_add_child_accumulates_usage_and_cost() -> None:
    parent = TurnInvocation(
        usage=InvocationUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01),
        cost_usd=0.01,
    )
    child = TurnInvocation(
        usage=InvocationUsage(input_tokens=20, output_tokens=8, total_tokens=28, cost_usd=0.02),
        cost_usd=0.02,
    )
    parent.add_child(child)
    assert parent.children == (child,)
    assert parent.usage.input_tokens == 30
    assert parent.usage.output_tokens == 13
    assert parent.usage.total_tokens == 43
    assert parent.usage.cost_usd == pytest.approx(0.03)
    assert parent.cost_usd == pytest.approx(0.03)


def test_add_child_chains_multiple_children() -> None:
    parent = TurnInvocation()
    c1 = TurnInvocation(
        usage=InvocationUsage(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.01),
        cost_usd=0.01,
    )
    c2 = TurnInvocation(
        usage=InvocationUsage(input_tokens=2, output_tokens=3, total_tokens=5, cost_usd=0.02),
        cost_usd=0.02,
    )
    parent.add_child(c1)
    parent.add_child(c2)
    assert parent.children == (c1, c2)
    assert parent.usage.input_tokens == 3
    assert parent.usage.output_tokens == 4
    assert parent.usage.total_tokens == 7
    assert parent.cost_usd == pytest.approx(0.03)


# --- Finalization ---------------------------------------------------------


def test_finalize_sets_output_and_finished_at() -> None:
    inv = TurnInvocation()
    assert not inv.is_complete
    inv.finalize(output="done")
    assert inv.output == "done"
    assert inv.finished_at is not None
    assert inv.is_complete
    assert not inv.is_error


def test_finalize_sets_error() -> None:
    inv = TurnInvocation()
    err = RuntimeError("kaboom")
    inv.finalize(error=err)
    assert inv.error is err
    assert inv.is_error
    assert inv.is_complete
    assert inv.finished_at is not None


def test_finalize_is_idempotent() -> None:
    inv = TurnInvocation()
    inv.finalize(output="first")
    first_finished = inv.finished_at
    # Second call must NOT mutate any field.
    inv.finalize(output="second", error=RuntimeError("ignored"))
    assert inv.output == "first"
    assert inv.error is None
    assert inv.finished_at is first_finished


def test_is_complete_and_is_error_reflect_state() -> None:
    inv = TurnInvocation()
    assert not inv.is_complete
    assert not inv.is_error

    err_inv = TurnInvocation(error=RuntimeError("oops"))
    assert err_inv.is_error
    # is_complete only flips on finalize().
    assert not err_inv.is_complete
    err_inv.finalize(error=RuntimeError("oops"))
    assert err_inv.is_complete


# --- current_turn / _active_turn_var --------------------------------------


def test_current_turn_returns_none_outside_context() -> None:
    assert current_turn() is None


def test_current_turn_returns_set_value() -> None:
    inv = TurnInvocation(session_id="ctx-test")
    token = _active_turn_var.set(inv)
    try:
        assert current_turn() is inv
    finally:
        _active_turn_var.reset(token)
    assert current_turn() is None


# --- ContextVar isolation across asyncio tasks ----------------------------


async def test_context_var_isolated_across_asyncio_tasks() -> None:
    """Each asyncio.create_task() copies the current context, so two
    sibling tasks see independent values for _active_turn_var."""
    inv_a = TurnInvocation(session_id="a")
    inv_b = TurnInvocation(session_id="b")

    seen_in_a: list[TurnInvocation | None] = []
    seen_in_b: list[TurnInvocation | None] = []
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def task_a() -> None:
        _active_turn_var.set(inv_a)
        seen_in_a.append(current_turn())
        started.set()
        await proceed.wait()
        seen_in_a.append(current_turn())

    async def task_b() -> None:
        await started.wait()
        _active_turn_var.set(inv_b)
        seen_in_b.append(current_turn())
        proceed.set()
        # Yield once so task_a's second observation is well-ordered.
        await asyncio.sleep(0)
        seen_in_b.append(current_turn())

    a = asyncio.create_task(task_a())
    b = asyncio.create_task(task_b())
    await asyncio.gather(a, b)

    # Each task only ever saw its own invocation (or None initially).
    assert seen_in_a == [inv_a, inv_a]
    assert seen_in_b == [inv_b, inv_b]
    # And the calling test context is untouched.
    assert current_turn() is None
