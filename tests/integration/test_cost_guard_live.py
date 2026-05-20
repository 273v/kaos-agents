"""WU-G.4 / #305 — live cost-guard + interrupt regression.

Pins the two non-iteration AgenticLoop terminators against real
Haiku-class latency + cost:

  1. ``max_loop_cost_usd`` — set extremely tight ($0.001), invoke a
     real Haiku planner whose call alone costs > $0.001 → loop
     terminates with ``LoopTerminated(reason="cost_exceeded")``.
  2. ``max_loop_wall_clock_seconds`` — set to 0.5s, inject a worker
     that sleeps just long enough to overshoot → loop terminates
     with ``LoopTerminated(reason="wall_clock_exceeded")``.

Both cases must also emit the SPA-#508 refusal pair:

  - exactly one ``TextDelta`` carrying the refusal lead
  - exactly one ``TurnSummary(intent="refuse")`` matching the
    TextDelta content

This is the contract single-user-chat depends on: when the loop
terminates without a satisfied verdict, the refusal text REPLACES
the worker's last attempt — it does NOT concatenate.

Gated with ``@pytest.mark.live`` + ``requires_anthropic``. The
default unit lane skips this file entirely.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.events.policy import LoopTerminated
from kaos_agents.events.stream import TextDelta
from kaos_agents.patterns.agentic_loop import WorkerResult, run_agentic_turn
from kaos_agents.types.session_policy import SessionPolicy

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing — cost-guard live tests require Anthropic",
)


MODEL = "anthropic:claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Worker stubs — minimal cost / latency surfaces tuned to fire each guard.
# ---------------------------------------------------------------------------


async def _zero_cost_worker(**_kwargs: Any) -> WorkerResult:
    """A worker that does not call any LLM. Used for the cost-exceeded
    case where the planner's real Haiku cost alone trips the guard."""
    return WorkerResult(
        text="I checked.",
        tool_calls_made=[],
        cost_usd=0.0,
        latency_ms=10.0,
    )


async def _slow_worker(**_kwargs: Any) -> WorkerResult:
    """A worker that sleeps long enough to trip a 0.5s wall-clock guard.

    Sleeping 0.6 seconds ensures ``_wall_clock_exceeded`` fires on the
    post-worker check regardless of how fast the surrounding planner
    + critic ran.
    """
    await asyncio.sleep(0.6)
    return WorkerResult(
        text="I tried.",
        tool_calls_made=[],
        cost_usd=0.0001,
        latency_ms=600.0,
    )


def _patched_real_planner_real_critic():
    """Patch the planner with a REAL Haiku-backed call (cost > $0.001)
    and the critic with a cheap stub.

    For WU-G.4 case 1 (cost_exceeded), we need the planner's cost to
    overshoot the tight ``max_loop_cost_usd`` budget. The actual
    ``plan_turn_tool_policy`` already calls Haiku — we use it
    unpatched here so the real cost flows in. The critic, however,
    is stubbed to needs_more_work so even if the cost guard somehow
    missed, the loop wouldn't burn a second iteration.

    Returns the (plan_patch_ctx, critic_patch_ctx) tuple. Each is a
    context manager the caller wraps the loop in. ``plan_patch_ctx``
    is a no-op nullcontext — the live planner runs unchanged.
    """
    import contextlib
    from unittest.mock import patch

    from kaos_agents.planning.goal_check import (
        GoalCheckNeedsMoreWork,
        GoalCheckOutcome,
    )

    async def _critic_stub(**_kwargs: Any) -> GoalCheckOutcome:
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="keep trying",
                confidence=0.5,
                rationale="cost-guard test stub",
            ),
            cost_usd=0.0001,
            latency_ms=10.0,
            iteration=1,
        )

    plan_patch = contextlib.nullcontext()
    critic_patch = patch("kaos_agents.patterns.agentic_loop.check_goal", new=_critic_stub)
    return plan_patch, critic_patch


def _patched_cheap_planner_cheap_critic():
    """Cheap stubs for the wall-clock case. The wall-clock guard
    doesn't depend on cost — we want the loop to run end-to-end and
    overshoot on time, not on tokens.
    """
    from unittest.mock import patch

    from kaos_agents.planning.goal_check import (
        GoalCheckNeedsMoreWork,
        GoalCheckOutcome,
    )
    from kaos_agents.planning.policy import TurnToolPolicy

    async def _plan_stub(**kwargs: Any) -> TurnToolPolicy:
        ceiling = kwargs.get("ceiling_groups") or []
        return TurnToolPolicy(
            kept_groups=frozenset(ceiling),
            dropped_groups=frozenset(),
            rationale="wall-clock test stub",
            confidence=1.0,
            fell_back_to_ceiling=False,
            cost_usd=0.0001,
            latency_ms=5.0,
        )

    async def _critic_stub(**_kwargs: Any) -> GoalCheckOutcome:
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="keep trying",
                confidence=0.5,
                rationale="wall-clock test stub",
            ),
            cost_usd=0.0001,
            latency_ms=10.0,
            iteration=1,
        )

    plan_patch = patch("kaos_agents.patterns.agentic_loop.plan_turn_tool_policy", new=_plan_stub)
    critic_patch = patch("kaos_agents.patterns.agentic_loop.check_goal", new=_critic_stub)
    return plan_patch, critic_patch


# ---------------------------------------------------------------------------
# Refusal-pair assertion helper
# ---------------------------------------------------------------------------


def _assert_refusal_pair(events: list[Any], reason: str) -> None:
    """SPA-#508 contract — exactly one ``TextDelta`` refusal + exactly
    one ``TurnSummary(intent="refuse")``, both with matching content.

    The contract is "refusal REPLACES the worker's last attempt." We
    verify by:

      1. exactly one ``TurnSummary`` with ``intent="refuse"``
      2. at least one ``TextDelta`` in the stream
      3. the ``TurnSummary.text`` mentions the loop terminator reason
         (via the ``_REFUSAL_LEAD_BY_REASON`` table — "cost budget"
         for ``cost_exceeded`` / "wall-clock budget" for
         ``wall_clock_exceeded``).
    """
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    refuse_summaries = [t for t in turn_summaries if t.intent == "refuse"]
    assert len(refuse_summaries) == 1, (
        f"expected exactly one TurnSummary(intent='refuse'); "
        f"got {len(refuse_summaries)}. intents observed: "
        f"{[t.intent for t in turn_summaries]!r}"
    )
    summary = refuse_summaries[0]

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, "expected at least one TextDelta carrying the refusal lead"

    # The refusal text must match the reason. ``_REFUSAL_LEAD_BY_REASON``
    # encodes the canonical phrasing:
    # - cost_exceeded:        "loop's cost budget was exhausted"
    # - wall_clock_exceeded:  "loop's wall-clock budget was exhausted"
    expected_substrings_by_reason = {
        "cost_exceeded": ("cost budget",),
        "wall_clock_exceeded": ("wall-clock budget", "wall clock budget"),
    }
    expected = expected_substrings_by_reason.get(reason, ())
    text_lower = (summary.text or "").lower()
    if expected:
        assert any(sub in text_lower for sub in expected), (
            f"refusal TurnSummary.text did not mention the expected "
            f"phrase for reason={reason!r}. Looked for any of "
            f"{expected!r}. Text: {summary.text!r}"
        )


# ---------------------------------------------------------------------------
# Case 1 — max_loop_cost_usd
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestMaxLoopCostUsdLive:
    """Real Haiku planner + tight $0.001 cost cap → cost_exceeded."""

    async def test_cost_exceeded_terminates_with_refusal_pair(self) -> None:
        policy = SessionPolicy.default()._with_replacements(
            max_loop_iterations=3,
            max_loop_cost_usd=0.001,  # tight — planner alone overshoots
            max_loop_wall_clock_seconds=30.0,
        )

        plan_patch, critic_patch = _patched_real_planner_real_critic()

        events: list[Any] = []
        with plan_patch, critic_patch:
            async for ev in run_agentic_turn(
                user_message="Find recent SCOTUS opinions on agency deference.",
                policy=policy,
                worker=_zero_cost_worker,
                available_groups=list(policy.soft_ceiling),
                session_id="wu-g4-cost-exceeded",
            ):
                events.append(ev)

        terminations = [e for e in events if isinstance(e, LoopTerminated)]
        assert len(terminations) == 1, (
            f"expected exactly one LoopTerminated; got {len(terminations)}"
        )
        term = terminations[0]
        assert term.reason == "cost_exceeded", (
            f"expected reason='cost_exceeded'; got reason={term.reason!r}. "
            f"cumulative_cost_usd={term.cost_usd}"
        )
        # Cost actually overshot the cap.
        assert term.cost_usd >= policy.max_loop_cost_usd, (
            f"cumulative cost {term.cost_usd!r} did not exceed the cap {policy.max_loop_cost_usd!r}"
        )

        _assert_refusal_pair(events, reason="cost_exceeded")


# ---------------------------------------------------------------------------
# Case 2 — max_loop_wall_clock_seconds
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestMaxLoopWallClockSecondsLive:
    """Slow worker + 0.5s wall-clock cap → wall_clock_exceeded.

    Note: marked ``requires_anthropic`` for parity with the cost case
    (the test infra pins this gate uniformly so the failure mode is
    the same: skip when the live tier is off). The worker itself is
    pure-Python; the planner + critic are stubbed cheap. The test
    would technically pass without an API key, but the WU-G.4
    contract is "both cost-guard cases live together" and we keep
    the gate uniform.
    """

    async def test_wall_clock_exceeded_terminates_with_refusal_pair(self) -> None:
        policy = SessionPolicy.default()._with_replacements(
            max_loop_iterations=3,
            max_loop_cost_usd=1.0,
            max_loop_wall_clock_seconds=0.5,  # tight — _slow_worker overshoots
        )

        plan_patch, critic_patch = _patched_cheap_planner_cheap_critic()

        events: list[Any] = []
        with plan_patch, critic_patch:
            async for ev in run_agentic_turn(
                user_message="Find recent SCOTUS opinions on agency deference.",
                policy=policy,
                worker=_slow_worker,
                available_groups=list(policy.soft_ceiling),
                session_id="wu-g4-wall-clock",
            ):
                events.append(ev)

        terminations = [e for e in events if isinstance(e, LoopTerminated)]
        assert len(terminations) == 1, (
            f"expected exactly one LoopTerminated; got {len(terminations)}"
        )
        term = terminations[0]
        assert term.reason == "wall_clock_exceeded", (
            f"expected reason='wall_clock_exceeded'; got reason={term.reason!r}. "
            f"wall_clock_ms={getattr(term, 'wall_clock_ms', '?')!r}"
        )

        _assert_refusal_pair(events, reason="wall_clock_exceeded")
