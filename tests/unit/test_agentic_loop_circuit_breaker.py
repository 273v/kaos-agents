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


def _make_sse_dict_tool_complete(
    tool_name: str,
    *,
    is_error: bool = False,
    result_summary: str = "",
) -> dict[str, str]:
    """Build the SSE-record shape the SPA's worker forwards.

    R1.1: ``app/services/agentic_worker.py:158-171`` builds events of
    this shape directly from the upstream wire — ``{"event": "<type>",
    "data": "<json-string>"}`` where the JSON payload includes the
    serialized ``Span`` discriminator + fields. The orchestrator's
    circuit-breaker observer must accept this shape (not just typed
    ``Span`` objects) or the SPA-mode breaker is dead code.
    """
    import json as _json

    payload = {
        "type": "span",
        "subject": "tool_call",
        "phase": "complete",
        "attributes": {
            "tool_name": tool_name,
            "call_id": "c1",
            "is_error": is_error,
            "result_summary": result_summary,
        },
    }
    return {"event": "span", "data": _json.dumps(payload)}


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

    # 0.1.2 (R0.1) contract change: the worker drafted substantive
    # text ("I tried multiple searches but nothing worked.", 47 chars)
    # AND no critic had explicitly rejected the draft (the loop hit the
    # circuit breaker before any GoalCheck verdict could land). Per R0.1
    # the loop now PRESERVES the worker draft + appends a budget footer
    # explaining the cap, rather than clobbering with a generic
    # boilerplate template. The TurnSummary's intent reflects this
    # change: ``respond_with_caveat`` instead of ``refuse``.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    # Worker draft must survive into the streamed TextDelta + final
    # TurnSummary.text.
    assert any("tried multiple searches" in td.content.lower() for td in text_deltas), (
        "worker draft text should be preserved on circuit-breaker exit (R0.1)"
    )
    # The budget footer must explain the circuit-breaker exit reason
    # so the user understands the caveat (not just the worker draft).
    # Anchor on the user-visible meaning ("a tool kept failing"), not
    # on the internal "circuit breaker" jargon that the audit's §7.3
    # plain-English rewrite replaced.
    assert any(
        "tool" in td.content.lower() and "fail" in td.content.lower() for td in text_deltas
    ), "refusal text must convey that a tool kept failing"
    # The final TurnSummary intent is ``respond_with_caveat`` for the
    # preserve-worker-draft path (was ``refuse`` pre-R0.1 — that
    # behavior is now reserved for the case where the worker text was
    # empty/trivial OR a critic rejected the draft).
    caveat_summaries = [ts for ts in turn_summaries if ts.intent == "respond_with_caveat"]
    refuse_summaries = [ts for ts in turn_summaries if ts.intent == "refuse"]
    assert len(caveat_summaries) == 1, (
        "exactly one TurnSummary with intent=respond_with_caveat when the "
        "worker had drafted substantive text and the loop exited for a "
        "non-critic reason (circuit breaker, cost cap, wall-clock cap, "
        "stuck-detection, max-iter without critic rejection)"
    )
    assert len(refuse_summaries) == 0, (
        "no refuse-intent TurnSummary: the worker's draft was not rejected "
        "by a critic, so it's not a refusal — it's a partial answer with "
        "a budget caveat"
    )


@pytest.mark.asyncio
async def test_circuit_breaker_with_empty_worker_text_uses_template() -> None:
    """0.1.2 (R0.1) sibling case to the preserve-worker-draft test
    above.

    When the worker emitted NO substantive draft text (empty or below
    the 40-char threshold), the budget-cap path falls back to the
    legacy refusal template + intent="refuse" because there's no
    worker draft worth preserving.

    Without this fallback, the SPA's #508 refusal-replace contract
    would receive a 3-char "ok" or empty string and persist nothing
    useful.
    """
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
            text="",  # Empty draft — fall through to template path.
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
    assert len(breaker_events) == 1
    terminated_events = [e for e in events if isinstance(e, LoopTerminated)]
    assert terminated_events[0].reason == "circuit_breaker_tripped"

    # Empty worker draft → fall back to legacy refusal template +
    # intent="refuse". The TextDelta must contain the boilerplate
    # "I stopped after..." pattern.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    assert any("i stopped after" in td.content.lower() for td in text_deltas), (
        "empty worker draft → legacy template fires"
    )
    refuse_summaries = [ts for ts in turn_summaries if ts.intent == "refuse"]
    assert len(refuse_summaries) == 1, "intent=refuse when there's no worker draft to preserve"


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
                # R1.3 #563: 25 calls would trip the new per-iteration
                # tool-call cap (default 10) before this test could
                # exercise the circuit-breaker=0 branch. Disable it for
                # this test so the test continues to verify what it
                # advertises (circuit-breaker=0 → no breaker trip).
                max_tool_calls_per_iteration=0,
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


# ── R1.1: SSE-dict event shape (SPA worker) ──────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_sse_dict_records() -> None:
    """R1.1 — the SPA's worker forwards raw SSE record dicts (not typed
    Spans). The orchestrator's ``_observe_for_circuit_breaker`` must
    accept both shapes.

    Pre-R1.1 the ``isinstance(event, Span)`` guard silently returned for
    every forwarded SSE dict — making the loop-level circuit breaker
    dead code in the SPA. Cost-storm sessions ran all the way to
    cost/wall-clock cap (WU-K v3 C1 with 17 tool calls, Agent 4's C7
    with 12 consecutive "No results found"). This test fixtures 5
    consecutive SSE-shaped tool-complete records and asserts the
    breaker still trips.
    """
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    sse_dict_records = [
        _make_sse_dict_tool_complete(
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
            events=list(sse_dict_records),
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
    assert len(breaker_events) == 1, (
        "circuit breaker must trip even when the worker forwards "
        "SSE-record dicts (SPA shape) rather than typed Spans"
    )
    assert breaker_events[0].tool_name == "kaos-web-search"
    assert breaker_events[0].consecutive_failures == 5

    terminated_events = [e for e in events if isinstance(e, LoopTerminated)]
    assert len(terminated_events) == 1
    assert terminated_events[0].reason == "circuit_breaker_tripped"


@pytest.mark.asyncio
async def test_circuit_breaker_handles_malformed_sse_dict() -> None:
    """R1.1 defensive — a malformed SSE record (bad JSON, missing
    fields, wrong subject) must NOT crash the observer; it should
    silently skip the record."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    malformed_records: list[dict[str, str]] = [
        {"event": "span", "data": "{not valid json"},
        {"event": "span", "data": '{"type": "span"}'},  # missing subject/phase
        {"event": "text_delta", "data": '{"content": "hi"}'},  # wrong type
        # Even one valid uninformative result should NOT trip on its own
        _make_sse_dict_tool_complete(
            "kaos-web-search",
            is_error=False,
            result_summary="No results found",
        ),
    ]

    from kaos_agents.planning.goal_check import GoalCheckSatisfied

    worker = _worker_stub(
        WorkerResult(
            text="Tried but nothing found.",
            tool_calls_made=[{"tool_name": "kaos-web-search"}],
            cost_usd=0.001,
            latency_ms=200.0,
            events=list(malformed_records),
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
                user_message="search query",
                policy=policy,
                worker=worker,
                available_groups=["web"],
                session_id="s1",
                run_id="r1",
                circuit_breaker_threshold=5,
            )
        )

    # Threshold is 5; we only emitted 1 uninformative result and 3
    # malformed events. The breaker MUST NOT trip.
    breaker_events = [e for e in events if isinstance(e, CircuitBreakerTripped)]
    assert len(breaker_events) == 0, (
        "single uninformative result + malformed events must not trip the breaker"
    )
