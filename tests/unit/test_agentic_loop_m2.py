"""Unit tests for the M2 reasoning-action-consistency wire in run_agentic_turn.

Covers the force-iteration mechanism added per
``2026-05-19-agentic-loop-honesty.md`` Stage 2:

* When ``m2_consistency_model`` is set AND GoalCheck returns
  ``satisfied`` AND the M2 critic returns a ``contradicts_*`` label,
  the loop MUST override the terminal verdict and replan with an
  M2-derived thinking_note directive.
* When M2 returns ``consistent`` (or any error/fallback), the
  satisfied terminator MUST fire as today.
* The M2 cost MUST roll into ``state.cumulative_cost_usd`` so the
  loop budget guard sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.events.policy import ConsistencyChecked, GoalChecked, LoopTerminated
from kaos_agents.events.stream import TextDelta
from kaos_agents.patterns.agentic_loop import (
    WorkerResult,
    run_agentic_turn,
)
from kaos_agents.planning.goal_check import (
    GoalCheckNeedsMoreWork,
    GoalCheckOutcome,
    GoalCheckSatisfied,
)
from kaos_agents.planning.judge import JudgeVerdict
from kaos_agents.planning.policy import TurnToolPolicy
from kaos_agents.types.session_policy import SessionPolicy

pytestmark = pytest.mark.unit


# ── Stubs (lifted from test_agentic_loop, narrowed) ─────────────────


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

    async def _impl(**_kwargs: Any) -> GoalCheckOutcome:
        return next(outcomes_iter)

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


def _m2_stub(*verdicts: JudgeVerdict):
    verdicts_iter = iter(verdicts)

    async def _impl(**_kwargs: Any) -> JudgeVerdict:
        try:
            return next(verdicts_iter)
        except StopIteration:
            return JudgeVerdict(
                label="consistent",
                confidence=1.0,
                reasoning="default consistent",
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


# ── 1. M2 inactive (default) preserves baseline ──────────────────────


@pytest.mark.asyncio
async def test_m2_inactive_satisfied_still_terminates() -> None:
    """When ``m2_consistency_model`` is None (default), the satisfied
    verdict terminates the loop on iteration 1 — no regression."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text="Branch taken: upper bound >= 5.0%. Body says 4.50%.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="good"),
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
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                # m2_consistency_model omitted — defaults to None
            )
        )

    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1


# ── 2. M2 active + contradicts_reasoning forces a re-iteration ──────


@pytest.mark.asyncio
async def test_m2_contradicts_reasoning_overrides_satisfied() -> None:
    """The load-bearing case: GoalCheck says satisfied, M2 says
    contradicts_reasoning, loop forces iteration 2 with an M2
    thinking_note directive."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    # Iter 1: the SPA bug text (headline contradicts body).
    # Iter 2: the corrected text (after replan with M2 directive).
    worker = _worker_stub(
        WorkerResult(
            text=(
                "Branch taken: upper bound >= 5.0%. "
                "The upper bound is 4.50% and does not reach 5.0%."
            ),
            tool_calls_made=[{"tool_name": "kaos-web-search", "result_summary": "4.50%"}],
            cost_usd=0.002,
            latency_ms=200.0,
        ),
        WorkerResult(
            text="Branch taken: upper bound < 5.0%. Next FOMC meeting June 16-17, 2026.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
    )
    # Both iterations: GoalCheck says satisfied (the question is well-answered
    # in terms of substance — M2 is what catches the headline error).
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="answered"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="clean"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=2,
        ),
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="contradicts_reasoning",
            confidence=0.95,
            reasoning="headline says >= 5%, body says 4.50%",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
        JudgeVerdict(
            label="consistent",
            confidence=0.9,
            reasoning="headline matches body",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="Fed funds branching question",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    # Iter 1 should have GoalChecked(satisfied) but NOT terminate.
    goal = [e for e in events if isinstance(e, GoalChecked)]
    assert len(goal) == 2, (
        f"expected 2 goal_checks (iter 1 + iter 2 after M2 override), got {len(goal)}"
    )
    # Terminal must be the iteration-2 satisfied (not iteration-1).
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 2, (
        f"M2 must have forced a 2nd iteration, got iterations_used={term.iterations_used}"
    )
    # Cost must include M2 calls (2 x $0.0003 = $0.0006 minimum).
    assert term.cost_usd >= 0.0006


# ── 2b. Confidence floor — low-confidence flags do NOT override ──────
#
# 0.1.1 fix (WU-K v2 Case E1, 2026-05-21): the M2 override path now
# requires ``verdict.confidence >= 0.85`` before flipping a satisfied
# terminator into a replan. The rubric itself says "high confidence
# (>=0.85) when the contradiction is explicit"; the loop must honor
# that threshold. Without this floor, an over-eager critic LLM can
# refuse correctly-grounded answers on weak heuristics — live E1
# repro returned ``contradicts_tool_results @ 0.92`` despite the
# response citing an entity ("Meridian Financial, LLC") that
# appeared verbatim 8x in the tool results.


@pytest.mark.asyncio
async def test_m2_low_confidence_contradicts_tool_results_does_not_override() -> None:
    """``contradicts_tool_results`` BELOW the 0.85 floor must NOT
    override a satisfied GoalCheck. The verdict is still emitted as a
    ConsistencyChecked event for observability, but the loop
    terminates on iteration 1 without forcing a replan."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text=(
                "On September 4, 2025, the SEC charged Meridian Financial, LLC "
                "with violations of the Marketing Rule."
            ),
            tool_calls_made=[
                {
                    "name": "kaos-web-search",
                    "is_error": False,
                    "summary_excerpt": "Meridian Financial, LLC ... Marketing Rule",
                }
            ],
            cost_usd=0.002,
            latency_ms=200.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="grounded answer"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )
    # Critic flags contradicts_tool_results but with confidence 0.7 —
    # below the 0.85 floor → must NOT override.
    m2 = _m2_stub(
        JudgeVerdict(
            label="contradicts_tool_results",
            confidence=0.7,
            reasoning="weak contradiction signal",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
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
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="Recent SEC enforcement action against RIA?",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    # Loop must terminate on iteration 1 — the 0.7 flag is observability,
    # not an override.
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1, (
        f"low-confidence M2 flag must NOT force a replan; "
        f"got iterations_used={term.iterations_used}"
    )
    # ConsistencyChecked still emits with the raw verdict (operator
    # surface preserved even when the override doesn't fire).
    cc = [e for e in events if isinstance(e, ConsistencyChecked)]
    assert len(cc) == 1
    assert cc[0].label == "contradicts_tool_results"
    assert cc[0].confidence == pytest.approx(0.7)
    assert cc[0].overrode_satisfied is False, (
        "below-floor confidence must record overrode_satisfied=False so "
        "downstream cost/SLO analysis can distinguish the silenced flags."
    )


class TestRemediationHintExtractor:
    """0.1.1 (#549.B) — helper unit tests for the remediation-hint
    threading logic in :func:`run_agentic_turn`.

    The helper is on the module-private surface — import via the
    `_extract_remediation_hints` / `_summarize_prior_tool_calls`
    names; these are not exported from `__all__` but are documented
    via their callers in the AgenticLoop body."""

    def test_no_hints_when_calls_empty(self) -> None:
        from kaos_agents.patterns.agentic_loop import _extract_remediation_hints

        assert _extract_remediation_hints([]) == []

    def test_no_hints_when_all_calls_succeed(self) -> None:
        from kaos_agents.patterns.agentic_loop import _extract_remediation_hints

        calls = [
            {"name": "kaos-web-search", "is_error": False, "summary_excerpt": "10 results"},
            {"name": "kaos-web-fetch-page", "is_error": False, "summary_excerpt": "blah"},
        ]
        assert _extract_remediation_hints(calls) == []

    def test_extracts_try_hint_from_errored_call(self) -> None:
        from kaos_agents.patterns.agentic_loop import _extract_remediation_hints

        calls = [
            {
                "name": "kaos-web-domain-profile",
                "is_error": True,
                "summary_excerpt": (
                    "Domain profiling failed for example.com: TaskGroup. "
                    "Try kaos-web-dns-enumerate for DNS, "
                    "kaos-web-whois-lookup for registration data."
                ),
            },
        ]
        hints = _extract_remediation_hints(calls)
        assert any("Try kaos-web-dns-enumerate" in h for h in hints)
        assert any("Try kaos-web-whois-lookup" in h for h in hints)

    def test_caps_hints_at_3(self) -> None:
        from kaos_agents.patterns.agentic_loop import _extract_remediation_hints

        calls = [
            {
                "name": "kaos-web-domain-profile",
                "is_error": True,
                "summary_excerpt": (
                    "Try kaos-tool-a. Try kaos-tool-b. Try kaos-tool-c. "
                    "Try kaos-tool-d. Try kaos-tool-e."
                ),
            },
        ]
        hints = _extract_remediation_hints(calls)
        assert len(hints) == 3


class TestPriorToolCallSummary:
    """0.1.1 (P2-B) — helper unit tests for the prior-call summary
    threading logic. The summary is appended to the next iteration's
    ``thinking_note`` so the worker LLM can avoid re-issuing near-
    duplicate queries — the loop-level mitigation for the over-
    specified-search-storm pattern observed in WU-K v2 Case C1."""

    def test_empty_summary_when_no_calls(self) -> None:
        from kaos_agents.patterns.agentic_loop import _summarize_prior_tool_calls

        assert _summarize_prior_tool_calls([]) == ""

    def test_one_line_per_call_with_is_error(self) -> None:
        from kaos_agents.patterns.agentic_loop import _summarize_prior_tool_calls

        calls = [
            {
                "name": "kaos-web-search",
                "is_error": False,
                "summary_excerpt": "10 results for site:sec.gov 2025 ...",
            },
            {
                "name": "kaos-web-fetch-page",
                "is_error": True,
                "summary_excerpt": "403 Forbidden on sec.gov/...",
            },
        ]
        summary = _summarize_prior_tool_calls(calls)
        assert "kaos-web-search" in summary and "is_error=False" in summary
        assert "kaos-web-fetch-page" in summary and "is_error=True" in summary
        # Bullet shape
        assert summary.count("\n- ") == 1  # 2 lines = 1 newline-plus-dash

    def test_truncates_long_summaries_at_120_chars(self) -> None:
        from kaos_agents.patterns.agentic_loop import _summarize_prior_tool_calls

        calls = [
            {"name": "kaos-x", "is_error": False, "summary_excerpt": "a" * 500},
        ]
        summary = _summarize_prior_tool_calls(calls)
        assert "…" in summary
        # The bullet line should be at most ~150 chars total (120 chars
        # of summary + "- kaos-x(is_error=False) — " prefix).
        assert all(len(line) <= 160 for line in summary.splitlines())

    def test_caps_at_10_calls(self) -> None:
        from kaos_agents.patterns.agentic_loop import _summarize_prior_tool_calls

        calls = [
            {"name": f"kaos-tool-{i}", "is_error": False, "summary_excerpt": "x"} for i in range(20)
        ]
        summary = _summarize_prior_tool_calls(calls)
        # 10 lines = 9 internal newlines + 1 trailing newline (no
        # trailing newline in our output) → 10 lines total.
        assert len(summary.splitlines()) == 10


@pytest.mark.asyncio
async def test_m2_at_confidence_floor_does_override() -> None:
    """At exactly ``confidence == 0.85`` (the rubric's own threshold
    for "explicit contradiction"), the override DOES fire. This
    boundary test pins the floor's >= semantics."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text="Headline asserts X but body computed not-X.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
        WorkerResult(
            text="Corrected: X is consistent throughout.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="clean"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=2,
        ),
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="contradicts_reasoning",
            confidence=0.85,  # exactly at floor
            reasoning="boundary",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
        JudgeVerdict(
            label="consistent",
            confidence=0.9,
            reasoning="ok",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="any",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.iterations_used == 2, "confidence == 0.85 is AT the floor, override must fire"


# ── 3. M2 active + consistent verdict preserves satisfied ────────────


@pytest.mark.asyncio
async def test_m2_consistent_verdict_terminates_normally() -> None:
    """When M2 returns ``consistent``, the satisfied verdict
    terminates the loop on iteration 1 with no extra iteration."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text="Branch taken: upper bound < 5.0%. The rate is 4.50%.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="consistent",
            confidence=0.9,
            reasoning="headline matches body",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
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
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1


# ── 4. M2 fell_back verdict treated conservatively (consistent) ──────


@pytest.mark.asyncio
async def test_m2_emits_consistency_checked_event_with_full_verdict() -> None:
    """Every M2 invocation must yield a ConsistencyChecked event so
    operators see the verdict without grepping memory.json."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text="Branch taken: upper bound >= 5.0%. The rate is 4.50%.",
            tool_calls_made=[{"tool_name": "kaos-web-search", "result_summary": "4.50%"}],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
        WorkerResult(
            text="Branch taken: upper bound < 5.0%. The rate is 4.50%.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="clean"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=2,
        ),
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="contradicts_reasoning",
            confidence=0.92,
            reasoning="headline contradicts body",
            cost_usd=0.0003,
            latency_ms=80.0,
            fell_back=False,
        ),
        JudgeVerdict(
            label="consistent",
            confidence=0.88,
            reasoning="headline matches body",
            cost_usd=0.0003,
            latency_ms=70.0,
            fell_back=False,
        ),
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan),
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.check_goal",
            new=check,
        ),
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    cc = [e for e in events if isinstance(e, ConsistencyChecked)]
    assert len(cc) == 2, f"expected one ConsistencyChecked per iteration, got {len(cc)}"

    # First event = the override-triggering verdict.
    assert cc[0].label == "contradicts_reasoning"
    assert cc[0].confidence == pytest.approx(0.92)
    assert cc[0].reasoning == "headline contradicts body"
    assert cc[0].iteration == 1
    assert cc[0].cost_usd == pytest.approx(0.0003)
    assert cc[0].latency_ms == pytest.approx(80.0)
    assert cc[0].fell_back is False
    assert cc[0].overrode_satisfied is True

    # Second event = the post-fix consistent verdict.
    assert cc[1].label == "consistent"
    assert cc[1].iteration == 2
    assert cc[1].overrode_satisfied is False
    assert cc[1].fell_back is False


@pytest.mark.asyncio
async def test_m2_fell_back_event_carries_signal_to_operator() -> None:
    """When the M2 critic fell back, the ConsistencyChecked event
    must surface ``fell_back=True`` AND ``overrode_satisfied=False``
    so operators can see broken-critic incidents without scraping
    log files."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(text="anything", tool_calls_made=[], cost_usd=0.001, latency_ms=100.0)
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="",
            confidence=0.0,
            reasoning="provider error: APIError",
            cost_usd=0.0,
            latency_ms=0.0,
            fell_back=True,
        )
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan),
        ),
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=check),
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    cc = next(e for e in events if isinstance(e, ConsistencyChecked))
    assert cc.fell_back is True
    assert cc.label == ""
    assert "provider error" in cc.reasoning
    assert cc.overrode_satisfied is False
    # Loop should still terminate satisfied (fell_back treated conservatively).
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"


@pytest.mark.asyncio
async def test_m2_off_emits_no_consistency_checked_event() -> None:
    """When ``m2_consistency_model`` is None (default), no
    ConsistencyChecked event must be emitted — preserves the existing
    event stream for callers that haven't opted in."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(text="any", tool_calls_made=[], cost_usd=0.001, latency_ms=100.0)
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="ok"),
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
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=check),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    assert not [e for e in events if isinstance(e, ConsistencyChecked)]


@pytest.mark.asyncio
async def test_m2_fell_back_preserves_satisfied() -> None:
    """When the M2 judge errored / disallowed-label-emitted, the
    fell_back flag MUST cause the loop to treat it as consistent —
    never loop forever on a broken critic."""
    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text="Some answer.",
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        )
    )
    m2 = _m2_stub(
        JudgeVerdict(
            label="",
            confidence=0.0,
            reasoning="provider error",
            cost_usd=0.0,
            latency_ms=0.0,
            fell_back=True,
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
        patch(
            "kaos_agents.patterns.agentic_loop.judge_reasoning_action_consistency",
            new=m2,
        ),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="anything",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
                m2_consistency_model="anthropic:claude-haiku-4-5",
            )
        )

    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "satisfied"
    assert term.iterations_used == 1


# ── 5. max_iterations refusal override (task #505) ───────────────────


@pytest.mark.asyncio
async def test_max_iterations_emits_clean_refusal_not_last_worker_text() -> None:
    """When the loop terminates ``max_iterations`` with the last
    GoalCheck verdict still ``needs_more_work``, the loop MUST NOT
    ship the worker's last attempt (the text the critic just
    rejected). It MUST emit a final TextDelta + TurnSummary carrying
    an honest refusal derived from the last critic rationale.

    Reproduces the session-DEB pattern: 3 iterations of
    needs_more_work, then max_iterations terminator, and the worker
    text was a confident-wrong hallucination.
    """
    policy = SessionPolicy.default()
    assert policy.max_loop_iterations == 3
    plan = _StubPlan(kept={"web"}, dropped=set())

    hallucinated_text = (
        "Branch taken: upper bound >= 5.0%. The Federal Reserve's current "
        "rate is 4.50%, so the upper bound is below 5.0%."
    )
    # Use 3 distinct worker outputs so the stuck-detection heuristic
    # (identical-text + zero-new-tools) doesn't terminate the loop
    # before max_iterations fires.
    worker = _worker_stub(
        WorkerResult(
            text=hallucinated_text + " [iter 1]",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": "No results found"}
            ],
            cost_usd=0.002,
            latency_ms=100.0,
        ),
        WorkerResult(
            text=hallucinated_text + " [iter 2 — retry of the same fabrication]",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": "No results found"},
                {"tool_name": "kaos-web-search", "result_summary": "No results found"},
            ],
            cost_usd=0.002,
            latency_ms=100.0,
        ),
        WorkerResult(
            text=hallucinated_text + " [iter 3 — confident-wrong about 4.25%-4.50%]",
            tool_calls_made=[
                {"tool_name": "kaos-web-search", "result_summary": "No results found"},
                {"tool_name": "kaos-web-search", "result_summary": "No results found"},
                {"tool_name": "kaos-web-search", "result_summary": "No results found"},
            ],
            cost_usd=0.002,
            latency_ms=100.0,
        ),
    )
    # All 3 iterations: critic says needs_more_work (hallucination).
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                rationale=(
                    "Agent asserted specific current Fed rates without any "
                    "successful tool call in the trace; confident "
                    "hallucination of a lookup-able public fact."
                ),
                next_action="Call the web tool now to fetch the rate from federalreserve.gov.",
                confidence=0.9,
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                rationale="Still no tool calls.",
                next_action="Use the web tool.",
                confidence=0.9,
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=2,
        ),
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                rationale="Asserted 4.25%-4.50% upper bound without tool grounding.",
                next_action="Fetch the rate before answering.",
                confidence=0.92,
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=3,
        ),
    )

    with (
        patch(
            "kaos_agents.patterns.agentic_loop.plan_turn_tool_policy",
            new=_plan_stub(plan, plan, plan),
        ),
        patch("kaos_agents.patterns.agentic_loop.check_goal", new=check),
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="rate question",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    # The loop must terminate with max_iterations.
    term = next(e for e in events if isinstance(e, LoopTerminated))
    assert term.reason == "max_iterations"
    assert term.iterations_used == 3

    # MUST emit a final TextDelta + TurnSummary with the refusal text.
    # The refusal text MUST cite the iteration count and the last
    # critic rationale, and MUST NOT echo the hallucinated worker
    # text verbatim.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    assert text_deltas, "expected at least one TextDelta for the refusal"
    final_delta = text_deltas[-1]
    # Refusal must surface the iteration count so the user knows how
    # much was tried. Anchor on the numeral itself, not on a specific
    # phrase that ages out (the audit's §7.2 plain-English rewrite
    # dropped the literal "3-iteration budget" wording).
    assert "3" in final_delta.content, (
        f"refusal text must reference iteration count, got: {final_delta.content!r}"
    )
    # Refusal must surface the critic's diagnosis. The diagnosis line
    # itself (the fixture's "Asserted 4.25%-4.50% upper bound without
    # tool grounding") is what proves it. Anchor on the structural
    # marker the template uses ("Critic's diagnosis") plus the
    # diagnosis substring — not on the legacy template's "ungrounded"
    # adjective that the audit's §7.2 plain-English rewrite dropped.
    assert "critic's diagnosis" in final_delta.content.lower(), (
        f"refusal must surface the critic's diagnosis structural marker, "
        f"got: {final_delta.content!r}"
    )
    assert "tool grounding" in final_delta.content.lower(), (
        f"refusal must echo the diagnosis text, got: {final_delta.content!r}"
    )
    # The refusal MUST NOT carry the hallucinated branch headline.
    assert "Branch taken: upper bound >= 5.0%" not in final_delta.content
    # TurnSummary must mirror the refusal text.
    assert turn_summaries, "expected a TurnSummary alongside the refusal TextDelta"
    final_summary = turn_summaries[-1]
    assert final_summary.text == final_delta.content
    assert final_summary.intent == "refuse"
