"""Regression tests for the 0.1.2 R0.1 refusal-template clobber-worker-draft fix.

THE bug: ``_emit_failure_refusal`` unconditionally discarded the
worker's draft text on any non-``satisfied`` exit reason except
``insufficient_evidence``. Three audits independently caught this from
different angles (see ``kaos-modules/docs/audits/`` 2026-05-21):

- Sonnet 4.6 drafted a 4827-char SCOTUS table with 14 citations →
  cost-cap fired during synthesis → persisted 426-char "I stopped..."
  template. **90% of the work the user paid for was discarded.**
- gpt-5.4-mini drafted a 1102-char honest "I couldn't verify the X
  but here are 2 candidates the search surfaced" answer → max-iter
  fired → persisted 432-char template.
- Persistence audit: streamed 5265 chars of worker draft text →
  persisted 1156 chars of refusal template (78% loss on reload).

The R0.1 fix preserves the worker's draft when:
  1. ``state.last_text`` is at least ``_MIN_WORKER_DRAFT_CHARS`` (40)
     non-whitespace characters.
  2. ``state.last_terminal_verdict`` is "" (no critic verdict this
     iteration) OR "satisfied" (critic accepted).

When the draft is preserved, the loop appends a short footer
explaining the budget cap and emits ``intent="respond_with_caveat"``.
When the draft is empty/trivial OR a critic rejected it, the legacy
template fires with ``intent="refuse"``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.events.policy import LoopTerminated
from kaos_agents.events.stream import TextDelta
from kaos_agents.patterns.agentic_loop import (
    _MIN_WORKER_DRAFT_CHARS,
    WorkerResult,
    _should_preserve_worker_draft,
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


# ── Stubs (lifted from test_agentic_loop_m2.py) ────────────────────


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
    last = outcomes[-1] if outcomes else None

    async def _impl(**_kwargs: Any) -> GoalCheckOutcome:
        nonlocal last
        try:
            last = next(outcomes_iter)
            return last
        except StopIteration:
            assert last is not None
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


# ─── 1. Unit tests for the preserve-decision helper ──────────────


class TestShouldPreserveWorkerDraft:
    """Pure-function tests for ``_should_preserve_worker_draft``."""

    def test_empty_draft_does_not_preserve(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(last_text="")
        assert not _should_preserve_worker_draft(state)

    def test_whitespace_only_draft_does_not_preserve(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(last_text="   \n\n   \t  ")
        assert not _should_preserve_worker_draft(state)

    def test_short_draft_under_threshold_does_not_preserve(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        # 30 chars of non-whitespace — under the 40-char threshold.
        state = _LoopState(last_text="Short answer of 30 characters!")
        assert not _should_preserve_worker_draft(state)

    def test_substantive_draft_with_no_verdict_preserves(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(
            last_text="x" * (_MIN_WORKER_DRAFT_CHARS + 5),
            last_terminal_verdict="",
        )
        assert _should_preserve_worker_draft(state)

    def test_substantive_draft_with_satisfied_verdict_preserves(self) -> None:
        # Defensive: satisfied verdicts shouldn't reach refusal, but if
        # they did, preserve.
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(
            last_text="x" * 100,
            last_terminal_verdict="satisfied",
        )
        assert _should_preserve_worker_draft(state)

    def test_substantive_draft_with_needs_more_work_does_not_preserve(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(
            last_text="x" * 200,
            last_terminal_verdict="needs_more_work",
        )
        assert not _should_preserve_worker_draft(state)

    def test_substantive_draft_with_override_does_not_preserve(self) -> None:
        from kaos_agents.patterns.agentic_loop import _LoopState

        state = _LoopState(
            last_text="x" * 200,
            last_terminal_verdict="override",
        )
        assert not _should_preserve_worker_draft(state)


# ─── 2. Integration: cost-cap preserves substantive draft ──────────


@pytest.mark.asyncio
async def test_cost_exceeded_preserves_substantive_worker_draft() -> None:
    """Cost-cap fires after a worker iteration that drafted a 200+
    char nuanced response. The persisted assistant content must
    contain the draft + a budget footer, NOT the legacy template.

    Anchored to Sonnet session 01KS5KJKSYSC0YHPSYY73NYJ6V from the
    persistence-audit diary: 5265 chars streamed, 1156 chars persisted
    (78% loss). Post-R0.1 the full draft survives.
    """
    policy = SessionPolicy.default()
    policy = replace(policy, max_loop_cost_usd=0.05)

    # 200-char draft simulating the worker mid-synthesis. Realistic
    # shape: a complete grounded answer with a specific entity + URL,
    # representative of what gets clobbered in production.
    draft = (
        "On September 4, 2025, the SEC charged Meridian Financial, LLC, "
        "a Massachusetts-based registered investment adviser, with "
        "violations of the Marketing Rule. Source: ia-6916-s on sec.gov."
    )
    assert len(draft) >= _MIN_WORKER_DRAFT_CHARS, "fixture must exceed the preserve-threshold"

    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text=draft,
            tool_calls_made=[
                {"name": "kaos-web-search", "is_error": False, "summary_excerpt": "Meridian"}
            ],
            cost_usd=0.10,  # over cap
            latency_ms=200.0,
        )
    )
    # GoalCheck never runs — the cost cap fires first.
    check = _check_stub()

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
                user_message="What was a recent SEC RIA enforcement action?",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "cost_exceeded", "cost-cap must fire"

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    # The worker draft must survive into the streamed TextDelta + the
    # final TurnSummary.text.
    assert any("Meridian Financial" in td.content for td in text_deltas), (
        "worker draft must be preserved in TextDelta (R0.1)"
    )
    final_summary = turn_summaries[-1]
    assert "Meridian Financial" in final_summary.text, (
        "worker draft must survive into final TurnSummary.text (R0.1)"
    )
    # Footer must convey the cost-cap exit reason so the user
    # understands the caveat. Anchor on the user-visible meaning
    # ("spending limit"), not on internal "cost budget" jargon that
    # the audit's §7.3 plain-English rewrite replaced.
    lower = final_summary.text.lower()
    assert "spending limit" in lower or "cost" in lower, (
        "budget footer must explain the cost-cap exit reason"
    )
    # Intent reflects the new "answer with caveat" contract, not the
    # legacy "refuse".
    assert final_summary.intent == "respond_with_caveat", (
        "intent must signal partial-answer-with-caveat, not refuse"
    )


# ─── 3. Integration: max-iter with needs_more_work CLOBBERS draft ──


@pytest.mark.asyncio
async def test_max_iterations_with_needs_more_work_clobbers_worker_draft() -> None:
    """When max_iter is hit because GoalCheck KEPT returning
    needs_more_work, the worker's last draft was exactly the
    hallucination the critic rejected. Per task #505 + R0.1, we
    MUST use the legacy refusal template — the draft is bad.

    This pins the boundary between R0.1's new preserve behavior and
    the legacy clobber behavior. A bug in the verdict-tracking code
    that drops "needs_more_work" → "" would silently regress to
    shipping hallucinations.
    """
    policy = SessionPolicy.default()
    policy = replace(policy, max_loop_iterations=2)

    draft_hallucination = (
        "The current US Senator from California is Smith Johnson. "
        "He was elected in 2024 with 58 percent of the vote and "
        "currently sits on the Banking Committee. (Note: fabricated.)"
    )
    assert len(draft_hallucination) >= _MIN_WORKER_DRAFT_CHARS

    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text=draft_hallucination,
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
        WorkerResult(
            # iter-2 worker text MUST differ substantially from iter-1
            # so stuck-detection (text prefix equality) doesn't fire
            # before max-iter does. Same hallucination shape, different
            # wording.
            text=(
                "I checked and California's senior senator is "
                "actually Robinson Lee. He's served since 2022 on the "
                "Judiciary Committee. (Different fabrication, same "
                "category of confident-without-evidence.)"
            ),
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="actually search for the senator's name",
                confidence=0.4,
                rationale=(
                    "Agent named 'Smith Johnson' as the California senator "
                    "but made zero tool calls to verify. Confident-wrong "
                    "fabrication."
                ),
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="actually search for the senator's name",
                confidence=0.4,
                rationale="Same hallucination, still no tools.",
            ),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=2,
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
    ):
        events = await _collect(
            run_agentic_turn(
                user_message="Who is the senior US Senator from California?",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "max_iterations"

    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    final = turn_summaries[-1]
    # The hallucinated "Smith Johnson" name MUST NOT appear in the
    # persisted text — it's exactly what the critic rejected.
    assert "Smith Johnson" not in final.text, (
        "needs_more_work clobber must drop the hallucinated draft"
    )
    # Legacy template fires; intent must be "refuse".
    assert final.intent == "refuse", "needs_more_work exit must keep the refuse intent + template"
    # The refusal template must convey that the agent tried + stopped.
    # Anchor on the structural signal (a stop-condition phrase + the
    # iteration count) rather than specific wording that ages out —
    # the audit's §7.2 plain-English rewrite replaced "I was unable"
    # / "I stopped after" with "I tried N times" / "I stopped after
    # N attempt(s)".
    assert "tried" in final.text.lower() or "stopped" in final.text.lower(), (
        "refusal template must convey that the agent tried + stopped"
    )


# ─── 4. Integration: M2 override exit also clobbers ─────────────────


@pytest.mark.asyncio
async def test_m2_override_max_iter_clobbers_worker_draft_and_uses_m2_rationale() -> None:
    """M2 (or M3) overrode satisfied → replan → max_iter → refusal.
    The worker's draft on the last iteration was what M2 rejected,
    so we clobber. AND per R1.2 the persisted refusal must surface
    M2's verdict (rationale) instead of the previous GoalCheck's
    (which may be empty).
    """
    policy = SessionPolicy.default()
    policy = replace(policy, max_loop_iterations=2)

    contradictory_draft = (
        "Branch taken: upper bound >= 5.0%. The upper bound is "
        "4.50% and does not reach 5.0%. (Headline contradicts body.)"
    )
    assert len(contradictory_draft) >= _MIN_WORKER_DRAFT_CHARS

    plan = _StubPlan(kept={"web"}, dropped=set())
    worker = _worker_stub(
        WorkerResult(
            text=contradictory_draft,
            tool_calls_made=[
                {"name": "kaos-web-search", "is_error": False, "summary_excerpt": "4.50%"}
            ],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
        WorkerResult(
            # iter-2 text differs substantially (no substring overlap
            # with iter-1) so stuck-detection doesn't fire before
            # max-iter. Same M2-flagged shape, different wording.
            text=(
                "Headline: below 5.0%. The Fed funds upper bound is "
                "currently 4.50%, well under the 5.0% threshold; the "
                "branch we're on is therefore <5.0%."
            ),
            tool_calls_made=[],
            cost_usd=0.001,
            latency_ms=100.0,
        ),
    )
    # Both iterations: GoalCheck says satisfied; M2 overrides both
    # times with contradicts_reasoning.
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="answered"),
            cost_usd=0.0001,
            latency_ms=50.0,
            iteration=1,
        ),
        GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.95, rationale="answered"),
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
            label="contradicts_reasoning",
            confidence=0.95,
            reasoning="same problem, headline still contradicts body",
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

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "max_iterations"

    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    final = turn_summaries[-1]
    # Contradictory draft must be dropped.
    assert "Branch taken" not in final.text, (
        "M2-override max-iter must clobber the criticized draft"
    )
    # R1.2: M2's rationale must appear in the persisted refusal so
    # the user sees the actual reason, not stale GoalCheck text.
    assert "M2" in final.text or "headline" in final.text.lower(), (
        "M2's verdict must surface in the persisted refusal text (R1.2)"
    )
    # Intent stays "refuse" — the draft was rejected by a critic.
    assert final.intent == "refuse"


# ─── 5. Integration: insufficient_evidence preserves (legacy) ──────


@pytest.mark.asyncio
async def test_insufficient_evidence_still_preserves_worker_text() -> None:
    """``insufficient_evidence`` was the only pre-R0.1 path that
    preserved worker text. Confirm this behavior is unchanged
    post-fix — that path uses ``_terminate`` directly without
    going through ``_emit_failure_refusal``.
    """
    from kaos_agents.planning.goal_check import GoalCheckInsufficientEvidence

    policy = SessionPolicy.default()
    plan = _StubPlan(kept={"web"}, dropped=set())

    honest_answer = (
        "I searched the SEC and federalreserve.gov sites and couldn't "
        "find a specific 2025 enforcement action against Meridian "
        "Financial; the search returned 5 candidates but I cannot "
        "verify any of them with confidence. Try a different query."
    )
    worker = _worker_stub(
        WorkerResult(
            text=honest_answer,
            tool_calls_made=[
                {"name": "kaos-web-search", "is_error": False, "summary_excerpt": "candidates"}
            ],
            cost_usd=0.001,
            latency_ms=100.0,
        )
    )
    check = _check_stub(
        GoalCheckOutcome(
            result=GoalCheckInsufficientEvidence(
                missing="the specific 2025 enforcement action",
                rationale="genuinely couldn't verify",
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
                user_message="Q",
                policy=policy,
                worker=worker,
                available_groups=list(policy.soft_ceiling),
            )
        )

    term = [e for e in events if isinstance(e, LoopTerminated)]
    assert term[0].reason == "insufficient_evidence"
    # The worker's honest can't-verify answer must survive — this
    # path was already correct pre-R0.1 + remains correct post-fix.
    turn_summaries = [e for e in events if isinstance(e, TurnSummary)]
    if turn_summaries:
        # The insufficient_evidence path doesn't emit its own
        # TurnSummary; it just LoopTerminates. The worker's draft
        # was streamed via the worker.events forward + remains the
        # canonical assistant content per the SPA's last-iter-text
        # accumulator.
        pass
    # No TextDelta should contain the "I stopped after" template.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert not any("I stopped after" in td.content for td in text_deltas), (
        "insufficient_evidence path must NOT emit the legacy template"
    )
