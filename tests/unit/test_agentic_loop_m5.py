"""M5 conversation-history grounding critic — AgenticLoop integration.

* When ``m5_history_model`` is None (default), the satisfied GoalCheck
  terminates in one iteration (baseline preserved — no behavior change
  for callers that don't opt in).
* When ``m5_history_model`` is set AND GoalCheck returns satisfied but
  M5 returns ``fabricated_history`` at/above the confidence floor, the
  loop MUST override the satisfied verdict and force a grounded re-write
  iteration with the M5 directive. This is the gate that catches a
  response confabulating its own conversation history.

Stubs mirror ``test_agentic_loop_m2`` (the critic-judge function is
patched at the ``agentic_loop`` import site).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.policy import ConsistencyChecked, GoalChecked, LoopTerminated
from kaos_agents.patterns.agentic_loop import WorkerResult, run_agentic_turn
from kaos_agents.planning.goal_check import GoalCheckOutcome, GoalCheckSatisfied
from kaos_agents.planning.judge import JudgeVerdict
from kaos_agents.planning.policy import TurnToolPolicy
from kaos_agents.types.session_policy import SessionPolicy

pytestmark = pytest.mark.unit


@dataclass
class _StubPlan:
    kept: set[str]
    dropped: set[str]

    def as_turn_tool_policy(self) -> TurnToolPolicy:
        return TurnToolPolicy(
            kept_groups=frozenset(self.kept),
            dropped_groups=frozenset(self.dropped),
            rationale="test",
            confidence=0.9,
            fell_back_to_ceiling=False,
            cost_usd=0.0001,
            latency_ms=10.0,
        )


def _plan_stub(*plans: _StubPlan):
    it = iter([p.as_turn_tool_policy() for p in plans])

    async def _impl(**_kwargs: Any) -> TurnToolPolicy:
        try:
            return next(it)
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
    it = iter(outcomes)

    async def _impl(**_kwargs: Any) -> GoalCheckOutcome:
        return next(it)

    return _impl


def _worker_stub(*results: WorkerResult):
    assert results
    it = iter(results)
    last = results[-1]

    async def _impl(**_kwargs: Any) -> WorkerResult:
        nonlocal last
        try:
            last = next(it)
            return last
        except StopIteration:
            return last

    return _impl


def _m5_stub(*verdicts: JudgeVerdict):
    it = iter(verdicts)

    async def _impl(**_kwargs: Any) -> JudgeVerdict:
        try:
            return next(it)
        except StopIteration:
            return JudgeVerdict(
                label="grounded",
                confidence=1.0,
                reasoning="default grounded",
                cost_usd=0.0001,
                latency_ms=5.0,
                fell_back=False,
            )

    return _impl


async def _collect(gen) -> list[Any]:
    out: list[Any] = []
    async for ev in gen:
        out.append(ev)
    return out


def _satisfied(iteration: int) -> GoalCheckOutcome:
    return GoalCheckOutcome(
        result=GoalCheckSatisfied(confidence=0.95, rationale="answered"),
        cost_usd=0.0001,
        latency_ms=50.0,
        iteration=iteration,
    )


@pytest.mark.asyncio
async def test_m5_inactive_satisfied_terminates_in_one_iteration() -> None:
    """Default (no m5_history_model) — satisfied terminates immediately."""
    policy = SessionPolicy.default()
    worker = _worker_stub(
        WorkerResult(text="A grounded answer.", tool_calls_made=[], cost_usd=0.001, latency_ms=10.0)
    )
    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(_StubPlan({"web"}, set())),
        ),
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=_check_stub(_satisfied(1))),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="hi",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                # m5_history_model omitted → None
            )
        )
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1


@pytest.mark.asyncio
async def test_m5_fabricated_history_overrides_satisfied() -> None:
    """GoalCheck satisfied, M5 says fabricated_history → loop forces a
    grounded re-write (iteration 2)."""
    policy = SessionPolicy.default()
    transcript = (
        "user: which has the longest term\n"
        "assistant: CC Final 2 (fifth anniversary); Acme has no fixed end date."
    )
    worker = _worker_stub(
        # Iter 1: confabulates history (claims it discussed something never said).
        WorkerResult(
            text="My last reply introduced FRCP material, which was a mistake.",
            tool_calls_made=[],
            cost_usd=0.002,
            latency_ms=100.0,
        ),
        # Iter 2: grounded correction after the M5 directive.
        WorkerResult(
            text=(
                "I did not mention FRCP. Re-reading my prior reply: I called a fixed "
                "5-year term 'longest' while noting Acme is indefinite — the indefinite "
                "term is the real outlier."
            ),
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=80.0,
        ),
    )
    check = _check_stub(_satisfied(1), _satisfied(2))
    m5 = _m5_stub(
        JudgeVerdict(
            label="fabricated_history",
            confidence=0.95,
            reasoning="claims it introduced FRCP material; transcript has no FRCP",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
        JudgeVerdict(
            label="grounded",
            confidence=0.9,
            reasoning="now grounded in the actual transcript",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
    )
    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(_StubPlan({"web"}, set()), _StubPlan({"web"}, set())),
        ),
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=check),
        patch("kaos_agents.patterns.agentic_loop.judge_history_grounding", new=m5),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="do you hear what you just said?",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                recent_turns=transcript,
                m5_history_model="anthropic:claude-haiku-4-5",
            )
        )

    # M5 forced a second iteration.
    goal = [e for e in events if isinstance(e, GoalChecked)]
    assert len(goal) == 2, f"expected 2 goal_checks (M5 override forced iter 2), got {len(goal)}"
    # An M5 critic event fired and recorded the override.
    m5_events = [e for e in events if isinstance(e, ConsistencyChecked) and e.overrode_satisfied]
    assert m5_events, "expected an overriding critic event from M5"
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 2


@pytest.mark.asyncio
async def test_m5_grounded_does_not_override() -> None:
    """A grounded M5 verdict leaves the satisfied terminator alone."""
    policy = SessionPolicy.default()
    worker = _worker_stub(
        WorkerResult(
            text="As I said earlier, CC Final 2 has the fixed 5-year term.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=10.0,
        )
    )
    m5 = _m5_stub(
        JudgeVerdict(
            label="grounded",
            confidence=0.9,
            reasoning="faithful to the transcript",
            cost_usd=0.0003,
            latency_ms=50.0,
            fell_back=False,
        )
    )
    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(_StubPlan({"web"}, set())),
        ),
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=_check_stub(_satisfied(1))),
        patch("kaos_agents.patterns.agentic_loop.judge_history_grounding", new=m5),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="remind me which is longest",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                recent_turns="user: which is longest\nassistant: CC Final 2, fixed 5-year term.",
                m5_history_model="anthropic:claude-haiku-4-5",
            )
        )
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1
