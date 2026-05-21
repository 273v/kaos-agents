"""R1.3 — per-iteration tool-call cap (reliability roadmap #563).

The audit anchor is Agent 1's Sonnet P5 case which ran 32 tool calls in
iteration 1 and burned $0.67 before ``cost_exceeded`` fired
mid-synthesis. The orchestrator's per-iteration tool-call cap returns
control after N calls so M2 / circuit-breaker / budget-cap layers can
intervene BEFORE the cost-storm completes.

Default cap is 10. ``max_tool_calls_per_iteration=0`` disables the cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.policy import LoopTerminated
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.patterns.agentic_loop import (
    WorkerResult,
    run_agentic_turn,
)
from kaos_agents.planning.goal_check import GoalCheckNeedsMoreWork, GoalCheckOutcome
from kaos_agents.planning.policy import TurnToolPolicy
from kaos_agents.types.session_policy import SessionPolicy

pytestmark = pytest.mark.unit


# ── Stubs (narrowed copies from test_agentic_loop_circuit_breaker) ──


@dataclass
class _StubPlan:
    kept: set[str]
    dropped: set[str]
    rationale: str = "test"
    cost_usd: float = 0.0001

    def as_turn_tool_policy(self) -> TurnToolPolicy:
        return TurnToolPolicy(
            kept_groups=frozenset(self.kept),
            dropped_groups=frozenset(self.dropped),
            rationale=self.rationale,
            confidence=0.9,
            fell_back_to_ceiling=False,
            cost_usd=self.cost_usd,
            latency_ms=10.0,
        )


def _plan_stub(*plans: _StubPlan):
    plans_iter = iter([p.as_turn_tool_policy() for p in plans])

    async def _impl(**_kwargs: Any) -> TurnToolPolicy:
        try:
            return next(plans_iter)
        except StopIteration:
            return TurnToolPolicy(
                kept_groups=frozenset(),
                dropped_groups=frozenset(),
                rationale="fallback",
                confidence=0.5,
                fell_back_to_ceiling=True,
                cost_usd=0.0,
                latency_ms=0.0,
            )

    return _impl


def _check_stub(*outcomes: GoalCheckOutcome):
    outcomes_iter = iter(outcomes)
    last = outcomes[-1]

    async def _impl(**_kwargs: Any) -> GoalCheckOutcome:
        nonlocal last
        try:
            last = next(outcomes_iter)
            return last
        except StopIteration:
            return last

    return _impl


def _worker_stub(*results: WorkerResult):
    assert results
    results_iter = iter(results)
    last = results[-1]

    async def _impl(**_kwargs: Any) -> WorkerResult:
        nonlocal last
        try:
            last = next(results_iter)
            return last
        except StopIteration:
            return last

    return _impl


def _make_tool_complete_span(tool_name: str, *, idx: int = 0) -> Span:
    return Span(
        timestamp=0.0,
        sequence=0,
        session_id="s1",
        run_id="r1",
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.COMPLETE,
        span_id=f"span-{idx}",
        name=f"tool.{tool_name}",
        attributes={
            "tool_name": tool_name,
            "call_id": f"c{idx}",
            "is_error": False,
            # Substantive result so the circuit breaker doesn't trip;
            # we're testing the tool-call-cap path here.
            "result_summary": f"Found result {idx}: detailed content here.",
        },
    )


async def _collect(gen) -> list[Any]:
    out: list[Any] = []
    async for ev in gen:
        out.append(ev)
    return out


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_cap_fires_at_default_threshold() -> None:
    """Default cap=10: 15 calls in one iteration → tool_call_cap_exceeded."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    spans = [_make_tool_complete_span("kaos-web-search", idx=i) for i in range(15)]
    worker = _worker_stub(
        WorkerResult(
            text="Synthesized a long answer from many tool calls.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"r{i}"} for i in range(15)
            ],
            cost_usd=0.05,
            latency_ms=2000.0,
            events=list(spans),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="keep digging",
                confidence=0.5,
                rationale="not done",
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="run lots of searches",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
            )
        )

    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated) == 1
    assert terminated[0].reason == "tool_call_cap_exceeded", (
        f"expected tool_call_cap_exceeded; got {terminated[0].reason}"
    )


@pytest.mark.asyncio
async def test_tool_call_cap_at_threshold_minus_one_does_not_fire() -> None:
    """Default cap=10: 9 calls is under the cap → other terminator fires."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    spans = [_make_tool_complete_span("kaos-web-search", idx=i) for i in range(9)]
    worker = _worker_stub(
        WorkerResult(
            text="Found enough — synthesizing.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"r{i}"} for i in range(9)
            ],
            cost_usd=0.005,
            latency_ms=500.0,
            events=list(spans),
        )
    )

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.8, rationale="answered"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="moderate query",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
            )
        )

    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated) == 1
    # Cap not exceeded → GoalCheckSatisfied terminates normally.
    assert terminated[0].reason == "satisfied"


@pytest.mark.asyncio
async def test_tool_call_cap_custom_threshold() -> None:
    """Custom cap=5: 6 calls trips the cap."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    spans = [_make_tool_complete_span("kaos-web-search", idx=i) for i in range(6)]
    worker = _worker_stub(
        WorkerResult(
            text="Synthesized answer.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"r{i}"} for i in range(6)
            ],
            cost_usd=0.01,
            latency_ms=500.0,
            events=list(spans),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="keep digging",
                confidence=0.5,
                rationale="not done",
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="query",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                max_tool_calls_per_iteration=5,
            )
        )

    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated) == 1
    assert terminated[0].reason == "tool_call_cap_exceeded"


@pytest.mark.asyncio
async def test_tool_call_cap_disabled_when_zero() -> None:
    """``max_tool_calls_per_iteration=0`` disables the cap entirely."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    spans = [_make_tool_complete_span("kaos-web-search", idx=i) for i in range(30)]
    worker = _worker_stub(
        WorkerResult(
            text="Synthesized from 30 tool calls.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"r{i}"} for i in range(30)
            ],
            cost_usd=0.1,
            latency_ms=5000.0,
            events=list(spans),
        )
    )

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.7, rationale="answered"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="query",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                max_tool_calls_per_iteration=0,
            )
        )

    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated) == 1
    assert terminated[0].reason == "satisfied", (
        "with cap=0, even 30 tool calls should not trip — caller opted out"
    )
