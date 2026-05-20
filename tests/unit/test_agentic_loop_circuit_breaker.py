"""Unit tests for the loop-level circuit-breaker terminator (#506-followup).

When the worker returns N consecutive ``Span(TOOL_CALL, COMPLETE)``
events that all carry ``is_error=True`` OR whose ``result_summary``
matches :func:`is_uninformative_result`, the AgenticLoop MUST:

1. Emit a :class:`CircuitBreakerTripped` event carrying the per-tool
   diagnostic (tool name, failure count, threshold).
2. Emit a clean refusal pair (TextDelta + TurnSummary(intent="refuse"))
   via ``_emit_failure_refusal``.
3. Emit a :class:`LoopTerminated` with
   ``reason="circuit_breaker_tripped"``.

The empirical anchor is session ``01KS2DEBYT341F1F16B3BRQRV0``:
12 consecutive ``kaos-web-search`` calls returned ``is_error=False``
with body ``"No results found for: ..."``. Pre-#506 the loop ran
out of iteration budget; post-fix it must trip + refuse cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.events.policy import CircuitBreakerTripped, LoopTerminated
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.events.stream import TextDelta
from kaos_agents.patterns.agentic_loop import (
    WorkerResult,
    run_agentic_turn,
)
from kaos_agents.planning.goal_check import GoalCheckNeedsMoreWork, GoalCheckOutcome
from kaos_agents.planning.policy import TurnToolPolicy
from kaos_agents.types.session_policy import SessionPolicy

pytestmark = pytest.mark.unit


# ── Stubs (narrowed copies from test_agentic_loop_m2) ────────────────


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


def _make_tool_complete_span(
    tool_name: str,
    *,
    is_error: bool = False,
    result_summary: str = "",
) -> Span:
    return Span(
        timestamp=0.0,
        sequence=0,
        session_id="s1",
        run_id="r1",
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.COMPLETE,
        span_id="span-1",
        name=f"tool.{tool_name}",
        attributes={
            "tool_name": tool_name,
            "call_id": "c1",
            "is_error": is_error,
            "result_summary": result_summary,
        },
    )


async def _collect(gen) -> list[Any]:
    out: list[Any] = []
    async for ev in gen:
        out.append(ev)
    return out


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_consecutive_uninformative_results() -> None:
    """Session DEB replay: 5 consecutive zero-result kaos-web-search
    calls in ONE worker iteration → CircuitBreakerTripped + clean
    refusal + LoopTerminated."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    five_zero_result_spans = [
        _make_tool_complete_span(
            "kaos-web-search",
            is_error=False,
            result_summary=f"No results found for: query {i}",
        )
        for i in range(5)
    ]

    worker = _worker_stub(
        WorkerResult(
            text="I tried multiple searches but nothing worked.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"No results found {i}"}
                for i in range(5)
            ],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(five_zero_result_spans),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="try a different search strategy",
                confidence=0.4,
                rationale="searches kept returning empty",
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
                user_message="What is the current Fed funds rate?",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=5,
            )
        )

    breaker_events = [e for e in events if isinstance(e, CircuitBreakerTripped)]
    assert len(breaker_events) == 1, "exactly one CircuitBreakerTripped event"
    assert breaker_events[0].tool_name == "kaos-web-search"
    assert breaker_events[0].consecutive_failures == 5
    assert breaker_events[0].failure_threshold == 5

    terminated_events = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated_events) == 1
    assert terminated_events[0].reason == "circuit_breaker_tripped"

    # The refusal pair (TextDelta + TurnSummary) must come BEFORE
    # LoopTerminated and AFTER the CircuitBreakerTripped event.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    assert any("circuit breaker" in td.content.lower() for td in text_deltas), (
        "refusal text should mention circuit breaker"
    )
    refuse_summaries = [ts for ts in turn_summaries if ts.intent == "refuse"]
    assert len(refuse_summaries) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_does_not_trip_on_informative_results() -> None:
    """5 web searches that ALL return real results must NOT trip the
    breaker — the loop terminates via the normal satisfied path."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    five_informative_spans = [
        _make_tool_complete_span(
            "kaos-web-search",
            is_error=False,
            result_summary=(
                f"Found 18 matches for query {i} on federalreserve.gov (2026 FOMC calendar)"
            ),
        )
        for i in range(5)
    ]

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    worker = _worker_stub(
        WorkerResult(
            text="The current target range is 4.25-4.50%.",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": f"Found 18 matches {i}"}
                for i in range(5)
            ],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(five_informative_spans),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(
                confidence=0.95,
                rationale="answered with citation",
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
                user_message="What is the current Fed funds rate?",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=5,
            )
        )

    assert not any(isinstance(e, CircuitBreakerTripped) for e in events)
    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated) == 1
    assert terminated[0].reason == "satisfied"


@pytest.mark.asyncio
async def test_circuit_breaker_resets_counter_on_informative_then_failures() -> None:
    """4 zero-result + 1 informative + 4 zero-result MUST NOT trip
    (no run of 5 consecutive failures)."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    interleaved = [
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 1"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 2"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 3"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 4"),
        # Reset:
        _make_tool_complete_span(
            "kaos-web-search",
            result_summary="Found 18 matches for query reset",
        ),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 5"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 6"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 7"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found 8"),
    ]

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    worker = _worker_stub(
        WorkerResult(
            text="I got mixed results but found the answer.",
            tool_calls_made=[{"tool_name": "kaos-web-search"} for _ in interleaved],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(interleaved),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="answered"),
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
                user_message="Mixed query",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=5,
            )
        )

    assert not any(isinstance(e, CircuitBreakerTripped) for e in events)


@pytest.mark.asyncio
async def test_circuit_breaker_disabled_when_threshold_zero() -> None:
    """``circuit_breaker_threshold=0`` MUST disable the loop-level
    breaker entirely — 25 zero-result calls run without tripping."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    twenty_five_zero = [
        _make_tool_complete_span(
            "kaos-web-search",
            result_summary=f"No results found {i}",
        )
        for i in range(25)
    ]

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    worker = _worker_stub(
        WorkerResult(
            text="No results across all attempts.",
            tool_calls_made=[{"tool_name": "kaos-web-search"} for _ in range(25)],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(twenty_five_zero),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.5, rationale="meh"),
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
                user_message="Will not find anything",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=0,  # disabled
            )
        )

    assert not any(isinstance(e, CircuitBreakerTripped) for e in events)
    terminated = [e for e in events if isinstance(e, LoopTerminated)]
    assert terminated[0].reason == "satisfied"


@pytest.mark.asyncio
async def test_circuit_breaker_event_carries_diagnostic_fields() -> None:
    """The emitted ``CircuitBreakerTripped`` must carry tool_name,
    consecutive_failures, failure_threshold, and uninformative_counted
    so downstream SPA consumers can render a precise banner."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    three_failed = [
        _make_tool_complete_span(
            "kaos-content-search-document",
            is_error=True,
            result_summary='{"error": "index not found"}',
        )
        for _ in range(3)
    ]

    worker = _worker_stub(
        WorkerResult(
            text="Couldn't search the documents.",
            tool_calls_made=[{"tool_name": "kaos-content-search-document"} for _ in range(3)],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(three_failed),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="retry with different query",
                confidence=0.4,
                rationale="all attempts errored",
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
                user_message="Search the documents",
                policy=policy,
                worker=worker,
                available_groups=["documents"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=3,
            )
        )

    breaker_events = [e for e in events if isinstance(e, CircuitBreakerTripped)]
    assert len(breaker_events) == 1
    bt = breaker_events[0]
    assert bt.tool_name == "kaos-content-search-document"
    assert bt.consecutive_failures == 3
    assert bt.failure_threshold == 3
    assert bt.uninformative_counted is True


@pytest.mark.asyncio
async def test_circuit_breaker_per_tool_isolation() -> None:
    """3 failures on tool A + 3 failures on tool B with threshold=5
    MUST NOT trip — counters are per-tool."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web", "documents"}, dropped=set())

    mixed = [
        _make_tool_complete_span("kaos-web-search", result_summary="No results found A1"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found A2"),
        _make_tool_complete_span("kaos-web-search", result_summary="No results found A3"),
        _make_tool_complete_span(
            "kaos-content-search-document",
            result_summary='{"results": [], "total_matches": 0}',
        ),
        _make_tool_complete_span(
            "kaos-content-search-document",
            result_summary='{"results": [], "total_matches": 0}',
        ),
        _make_tool_complete_span(
            "kaos-content-search-document",
            result_summary='{"results": [], "total_matches": 0}',
        ),
    ]

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    worker = _worker_stub(
        WorkerResult(
            text="Tried both tools, couldn't find anything definitive.",
            tool_calls_made=[{"tool_name": "kaos-web-search"} for _ in mixed],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(mixed),
        )
    )

    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.6, rationale="answered"),
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
                user_message="multi-tool query",
                policy=policy,
                worker=worker,
                available_groups=["web", "documents"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=5,
            )
        )

    assert not any(isinstance(e, CircuitBreakerTripped) for e in events)
