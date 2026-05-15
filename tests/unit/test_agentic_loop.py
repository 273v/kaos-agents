"""Tests for `kaos_agents.patterns.agentic_loop.run_agentic_turn`.

PRD `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.3. Pins the orchestrator contract:

  - Single iteration happy path: planner picks groups in ceiling →
    worker runs once → Critic returns Satisfied → loop terminates
    with ``LoopTerminated(reason="satisfied")``.
  - Auto-elevation: planner reports ``dropped_groups`` ∩ soft_ceiling
    (green-auto) → ``ToolPolicyElevated`` event fires → worker runs
    with the elevated ceiling.
  - Yellow-confirm: planner reports a yellow-confirm dropped group →
    ``CapabilityRequested`` event fires; loop continues with the
    narrower ceiling (the user's decision lands on a future turn).
  - Goal-check ``needs_more_work`` → loop replans with the
    ``next_action`` threaded as a thinking-block.
  - ``insufficient_evidence`` → terminal, render as refusal.
  - ``max_iterations`` cap: 3 iterations of ``needs_more_work`` →
    ``LoopTerminated(reason="max_iterations")``.
  - ``max_loop_cost_usd`` cap: cumulative cost exceeds budget →
    terminate with ``cost_exceeded`` mid-loop.
  - Stuck detection: identical text + no new tool calls → terminate
    with ``stuck_no_progress``.
  - User interrupt (asyncio cancel): terminate with
    ``user_interrupt`` and re-raise ``CancelledError``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.policy import (
    CapabilityRequested,
    GoalChecked,
    LoopTerminated,
    ToolPolicyElevated,
)
from kaos_agents.patterns.agentic_loop import (
    WorkerResult,
    run_agentic_turn,
)
from kaos_agents.planning.goal_check import (
    GoalCheckInsufficientEvidence,
    GoalCheckNeedsMoreWork,
    GoalCheckOutcome,
    GoalCheckSatisfied,
)
from kaos_agents.planning.policy import TurnToolPolicy
from kaos_agents.types.session_policy import SessionPolicy

pytestmark = pytest.mark.unit


# ─── Stubs ───────────────────────────────────────────────────────────


@dataclass
class _StubPlan:
    """Patchable TurnToolPolicy."""

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
    """Patch ``plan_turn_tool_policy`` to return a sequence of plans."""
    plans_iter = iter([p.as_turn_tool_policy() for p in plans])

    async def _impl(**_kwargs: Any) -> TurnToolPolicy:
        try:
            return next(plans_iter)
        except StopIteration:
            # Fall back to a "no-op" plan when the loop overshoots —
            # tests should explicitly cover the iteration count.
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
    """Patch ``check_goal`` to return a sequence of outcomes."""
    outcomes_iter = iter(outcomes)

    async def _impl(**_kwargs: Any) -> GoalCheckOutcome:
        try:
            return next(outcomes_iter)
        except StopIteration:
            # Default to needs_more_work so tests don't accidentally
            # leak into the unbounded path.
            return GoalCheckOutcome(
                result=GoalCheckNeedsMoreWork(
                    next_action="continue",
                    confidence=0.5,
                    rationale="fallback stub",
                ),
                cost_usd=0.0001,
                latency_ms=10.0,
                iteration=99,
            )

    return _impl


def _worker_stub(*results: WorkerResult):
    """Patchable worker — returns a sequence of WorkerResults."""
    results_iter = iter(results)
    last_used = [results[-1] if results else None]

    async def _impl(**_kwargs: Any) -> WorkerResult:
        try:
            r = next(results_iter)
            last_used[0] = r
            return r
        except StopIteration:
            return last_used[0]  # repeat last on overshoot

    return _impl


async def _collect(gen) -> list[Any]:
    events: list[Any] = []
    async for ev in gen:
        events.append(ev)
    return events


# ─── 1. Single-iteration happy path ──────────────────────────────────


@pytest.mark.asyncio
async def test_satisfied_on_first_iteration_terminates_loop() -> None:
    """The 80% happy path: planner picks groups in ceiling, worker
    runs once, Critic returns Satisfied, loop terminates."""
    policy = SessionPolicy.default()
    plan = _StubPlan(
        kept={"web", "documents"},
        dropped=set(),
        rationale="legal research task",
    )
    worker = _worker_stub(
        WorkerResult(
            text="Found SCOTUS opinion: Loper Bright v. Raimondo (2024).",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": "5 hits"}
            ],
            cost_usd=0.01,
            latency_ms=300.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="grounded"),
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
                user_message="Find recent SCOTUS on agency deference.",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    # No elevation needed (planner stayed within ceiling).
    assert not any(isinstance(e, ToolPolicyElevated) for e in events)
    assert not any(isinstance(e, CapabilityRequested) for e in events)
    # Exactly one GoalChecked, terminal-Satisfied.
    goal = [e for e in events if isinstance(e, GoalChecked)]
    assert len(goal) == 1
    assert goal[0].kind == "satisfied"
    # Exactly one LoopTerminated with reason=satisfied.
    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(term) == 1
    assert term[0].reason == "satisfied"
    assert term[0].iterations_used == 1
    assert term[0].elevations_used == 0


# ─── 2. Green-auto elevation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_green_auto_elevation_emits_event_and_proceeds() -> None:
    """Planner wants `web`; current ceiling lacks it; soft ceiling has
    it; tier is green-auto → silent elevation, ToolPolicyElevated event,
    worker runs with elevated set."""
    # Start with a narrowed ceiling (user clicked off "web" in SettingsSheet).
    policy = SessionPolicy.for_persona("research").with_removed_groups({"web"})
    assert "web" not in policy.allowed_groups
    assert "web" in policy.soft_ceiling
    assert policy.can_auto_elevate("web") is True

    plan = _StubPlan(
        kept={"documents"},
        dropped={"web"},
        rationale="user asked to search live",
    )
    worker = _worker_stub(
        WorkerResult(
            text="Found relevant opinions via web search.",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": ""}
            ],
            cost_usd=0.005,
            latency_ms=200.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="ok"),
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
                user_message="Search SCOTUS for X.",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    elevated = [e for e in events if isinstance(e, ToolPolicyElevated)]
    assert len(elevated) == 1
    assert "web" in elevated[0].elevated_groups
    assert "web" not in elevated[0].previous_allowed

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].elevations_used == 1


# ─── 3. Yellow-confirm capability request ────────────────────────────


@pytest.mark.asyncio
async def test_yellow_confirm_emits_capability_request() -> None:
    """Planner wants `browser`; tier is yellow-confirm → emit
    CapabilityRequested but DON'T auto-elevate. Worker runs with the
    narrower ceiling; Critic may mark needs_more_work / insufficient
    accordingly."""
    policy = SessionPolicy.for_persona("research").with_removed_groups({"browser"})
    assert "browser" not in policy.allowed_groups
    assert "browser" in policy.soft_ceiling
    assert policy.needs_confirmation("browser") is True

    plan = _StubPlan(
        kept={"documents"},
        dropped={"browser"},
        rationale="page is JS-rendered, need Chromium",
    )
    worker = _worker_stub(
        WorkerResult(
            text="I tried to fetch but couldn't render the JS page.",
            tool_calls_made=[],
            cost_usd=0.003,
            latency_ms=150.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckInsufficientEvidence(
                missing="JS-rendered page; need browser tool group",
                rationale="without the browser tier, page content isn't reachable",
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
                user_message="Scrape this SPA dashboard.",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    cap_reqs = [e for e in events if isinstance(e, CapabilityRequested)]
    assert len(cap_reqs) == 1
    assert "browser" in cap_reqs[0].requested_groups
    # No auto-elevation happened.
    elevated = [e for e in events if isinstance(e, ToolPolicyElevated)]
    assert elevated == []
    # Terminal verdict was insufficient_evidence; loop ended cleanly.
    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "insufficient_evidence"


# ─── 4. Replan on needs_more_work + iteration cap ────────────────────


@pytest.mark.asyncio
async def test_needs_more_work_replans_then_terminates_on_max_iterations() -> None:
    """Three consecutive needs_more_work outcomes → hit max_iterations
    (the default cap of 3)."""
    policy = SessionPolicy.default()  # max_loop_iterations=3
    plan_each = _StubPlan(kept={"web", "documents"}, dropped=set())
    worker = _worker_stub(
        # Three distinct responses so stuck-detection doesn't fire.
        WorkerResult(
            text="Partial answer 1 — found one result.",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": ""}
            ],
            cost_usd=0.005,
            latency_ms=100.0,
        ),
        WorkerResult(
            text="Partial answer 2 — found two more results.",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": ""},
                {"name": "kaos-citations-cl-search", "is_error": False, "summary_excerpt": ""},
            ],
            cost_usd=0.005,
            latency_ms=100.0,
        ),
        WorkerResult(
            text="Partial answer 3 — still digging.",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": ""},
                {"name": "kaos-citations-cl-search", "is_error": False, "summary_excerpt": ""},
                {"name": "kaos-source-ecfr-content", "is_error": False, "summary_excerpt": ""},
            ],
            cost_usd=0.005,
            latency_ms=100.0,
        ),
    )

    def nmw(i: int) -> GoalCheckOutcome:
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action=f"keep digging at step {i}",
                confidence=0.5,
                rationale=f"more work needed at step {i}",
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=i,
        )

    check = _check_stub(nmw(1), nmw(2), nmw(3))

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan_each, plan_each, plan_each),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="Find every SCOTUS opinion on X.",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    goal_events = [e for e in events if isinstance(e, GoalChecked)]
    assert len(goal_events) == 3
    assert all(e.kind == "needs_more_work" for e in goal_events)

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "max_iterations"
    assert term[0].iterations_used == 3


# ─── 5. Cost guard ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_exceeded_terminates_mid_loop() -> None:
    """Worker's per-iteration cost balloons past max_loop_cost_usd →
    terminate with reason=cost_exceeded after the offending iteration."""
    # Tight budget to make the test fast + deterministic.
    policy = SessionPolicy.default()
    from dataclasses import replace

    policy = replace(policy, max_loop_cost_usd=0.05)

    plan = _StubPlan(kept={"web"}, dropped=set())
    # Worker burns $0.10 on iteration 1 — over the $0.05 cap.
    worker = _worker_stub(
        WorkerResult(
            text="Mid-progress, expensive call.",
            tool_calls_made=[
                {"name": "kaos-source-fr-search", "is_error": False, "summary_excerpt": ""}
            ],
            cost_usd=0.10,
            latency_ms=100.0,
        ),
    )
    # Provide a needs_more_work so the loop would otherwise continue.
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(next_action="continue", confidence=0.5, rationale="more"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan, plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="x",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "cost_exceeded"
    # Loop ran exactly 1 iteration before tripping the guard.
    assert term[0].iterations_used == 1
    assert term[0].cost_usd >= 0.05


# ─── 6. Stuck detection ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stuck_no_progress_terminates_when_identical_text_and_no_new_tools() -> None:
    """Two iterations with identical text + the same tool-call count →
    terminate with reason=stuck_no_progress without burning budget on
    a third iteration."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web", "documents"}, dropped=set())
    identical_text = "I tried and got the same result again."
    identical_tools = [
        {"name": "kaos-source-fr-search", "is_error": True, "summary_excerpt": "timeout"}
    ]
    worker = _worker_stub(
        WorkerResult(
            text=identical_text,
            tool_calls_made=identical_tools,
            cost_usd=0.005,
            latency_ms=100.0,
        ),
        WorkerResult(
            text=identical_text,
            tool_calls_made=identical_tools,
            cost_usd=0.005,
            latency_ms=100.0,
        ),
    )
    nmw = GoalCheckOutcome(
        result=GoalCheckNeedsMoreWork(next_action="try again", confidence=0.5, rationale="..."),
        cost_usd=0.0001,
        latency_ms=50.0,
        iteration=1,
    )
    check = _check_stub(nmw, nmw)

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="x",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "stuck_no_progress"
    assert term[0].iterations_used == 2


# ─── 7. User interrupt ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_interrupt_terminates_and_reraises_cancelled() -> None:
    """asyncio cancel between iterations → emit
    LoopTerminated(reason=user_interrupt) and re-raise CancelledError."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    async def _slow_worker(**_kwargs: Any) -> WorkerResult:
        # Yield to the loop so the cancel can fire.
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError

    plan_iter = _plan_stub(plan)
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=1.0, rationale="never reached"),
            cost_usd=0.0,
            latency_ms=0.0,
            iteration=1,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=plan_iter,
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
    ):
        events: list[Any] = []
        with pytest.raises(asyncio.CancelledError):
            async for ev in run_agentic_turn(
                user_message="x",
                policy=policy,
                worker=_slow_worker,
                available_groups=list(policy.soft_ceiling),
            ):
                events.append(ev)

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(term) == 1
    assert term[0].reason == "user_interrupt"


# ─── 8. Worker events are forwarded verbatim ─────────────────────────


@pytest.mark.asyncio
async def test_worker_events_passthrough() -> None:
    """The orchestrator yields worker events verbatim so the SPA's SSE
    stream gets text deltas / tool spans without rewriting."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    # Use simple sentinel objects standing in for KaosEvent — the
    # orchestrator forwards by reference, doesn't inspect type.
    sentinel_text = {"type": "text_delta", "delta": "hello"}
    sentinel_tool = {"type": "tool_call_args_delta", "delta": "{"}

    worker = _worker_stub(
        WorkerResult(
            text="hello",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=10.0,
            events=[sentinel_text, sentinel_tool],
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=10.0,
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
                user_message="hi",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    # Sentinels appear in the output, in the order the worker emitted
    # them, BEFORE the orchestrator's GoalChecked / LoopTerminated.
    assert sentinel_text in events
    assert sentinel_tool in events
    idx_text = events.index(sentinel_text)
    idx_tool = events.index(sentinel_tool)
    idx_goal = next(i for i, e in enumerate(events) if isinstance(e, GoalChecked))
    assert idx_text < idx_tool < idx_goal
