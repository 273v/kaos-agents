"""Unit tests for kaos_agents.termination.judge — TerminationJudge.

Each axis is exercised in isolation. The Judge is a Program; we
invoke it via ``await judge.invoke(**kwargs)`` and read the
``Decision`` off ``invocation.output`` (the canonical kaos-llm-core
runtime contract).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kaos_agents.events.lifecycle import RunError
from kaos_agents.events.research import EvidenceInsufficient, GroundingRefusalTriggered
from kaos_agents.termination.degrade import DegradationPolicy
from kaos_agents.termination.judge import TerminationJudge
from kaos_agents.termination.loop_detect import LoopDetector
from kaos_agents.termination.types import Decision, DecisionKind, SuccessCriteria
from kaos_agents.types.usage import InvocationUsage

# A long-enough partial that the default DegradationPolicy (32-char
# floor) will accept it under budget exhaustion.
LONG_PARTIAL = (
    "Partial answer: the agent collected the following facts before "
    "running out of budget — A, B, C."
)


# KaosEvent base requires timestamp/sequence/session_id/run_id. Pass
# them explicitly via these helpers so static type checkers see the
# concrete arg types — ``**dict[str, object]`` widens to ``object``
# and ``ty`` rejects the spread.


def _run_error(*, error_type: str, message: str) -> RunError:
    return RunError(
        timestamp=0.0,
        sequence=0,
        session_id="test-session",
        run_id="test-run",
        error_type=error_type,
        message=message,
    )


def _evidence_insufficient(*, reason: str) -> EvidenceInsufficient:
    return EvidenceInsufficient(
        timestamp=0.0,
        sequence=0,
        session_id="test-session",
        run_id="test-run",
        reason=reason,
    )


def _grounding_refusal(
    *, original_confidence: float, min_confidence: float, reason: str
) -> GroundingRefusalTriggered:
    return GroundingRefusalTriggered(
        timestamp=0.0,
        sequence=0,
        session_id="test-session",
        run_id="test-run",
        original_confidence=original_confidence,
        min_confidence=min_confidence,
        reason=reason,
    )


async def _decide(judge: TerminationJudge, **kwargs: object) -> Decision:
    invocation = await judge.invoke(**kwargs)
    return invocation.output  # type: ignore[no-any-return]


class TestDefaultFlow:
    @pytest.mark.asyncio
    async def test_no_partial_no_caps_returns_incomplete(self) -> None:
        judge = TerminationJudge()
        decision = await _decide(judge, iteration=1)
        assert decision.kind == DecisionKind.INCOMPLETE
        assert decision.is_complete is False
        assert decision.allows_replan is True
        assert "replan" in decision.feedback

    @pytest.mark.asyncio
    async def test_partial_text_returns_complete(self) -> None:
        judge = TerminationJudge()
        decision = await _decide(judge, iteration=1, partial_text=LONG_PARTIAL)
        assert decision.kind == DecisionKind.COMPLETE
        assert decision.is_complete is True
        assert decision.allows_replan is False


class TestBudgetAxis:
    @pytest.mark.asyncio
    async def test_cost_cap_exceeded_no_partial(self) -> None:
        judge = TerminationJudge(max_cost_usd=0.10)
        usage = InvocationUsage(cost_usd=0.20)
        decision = await _decide(judge, usage=usage, iteration=1)
        assert decision.kind == DecisionKind.BUDGET_EXCEEDED
        assert decision.is_complete is True
        assert decision.should_escalate is True
        assert "0.20" in decision.feedback or "cost" in decision.feedback

    @pytest.mark.asyncio
    async def test_cost_cap_exceeded_with_long_partial_degrades(self) -> None:
        judge = TerminationJudge(max_cost_usd=0.10)
        usage = InvocationUsage(cost_usd=0.20)
        decision = await _decide(judge, usage=usage, iteration=1, partial_text=LONG_PARTIAL)
        assert decision.kind == DecisionKind.DEGRADED
        assert decision.is_complete is True
        assert decision.partial_result == LONG_PARTIAL

    @pytest.mark.asyncio
    async def test_iteration_cap_exceeded(self) -> None:
        judge = TerminationJudge(max_iterations=5)
        decision = await _decide(judge, iteration=5)
        assert decision.kind == DecisionKind.BUDGET_EXCEEDED
        assert "iterations" in decision.feedback

    @pytest.mark.asyncio
    async def test_wall_clock_exceeded(self) -> None:
        judge = TerminationJudge(max_wall_clock_seconds=10.0)
        decision = await _decide(judge, iteration=1, wall_seconds=11.0)
        assert decision.kind == DecisionKind.BUDGET_EXCEEDED
        assert "wall clock" in decision.feedback

    @pytest.mark.asyncio
    async def test_no_caps_does_not_trip_budget(self) -> None:
        judge = TerminationJudge()
        usage = InvocationUsage(cost_usd=999.0)
        decision = await _decide(judge, usage=usage, iteration=1, partial_text=LONG_PARTIAL)
        assert decision.kind == DecisionKind.COMPLETE


class TestFailureAxis:
    @pytest.mark.asyncio
    async def test_run_error_event_returns_failure(self) -> None:
        judge = TerminationJudge()
        events = [_run_error(error_type="boom", message="something broke")]
        decision = await _decide(judge, iteration=1, events=events)
        assert decision.kind == DecisionKind.FAILURE
        assert decision.is_complete is True
        assert "something broke" in decision.feedback

    @pytest.mark.asyncio
    async def test_evidence_insufficient_triggers_escalate(self) -> None:
        judge = TerminationJudge()
        events = [_evidence_insufficient(reason="no source documents found")]
        decision = await _decide(judge, iteration=1, events=events)
        assert decision.kind == DecisionKind.FAILURE
        assert decision.should_escalate is True
        assert "no source" in decision.feedback

    @pytest.mark.asyncio
    async def test_grounding_refusal_triggers_escalate(self) -> None:
        judge = TerminationJudge()
        events = [
            _grounding_refusal(
                original_confidence=0.2,
                min_confidence=0.5,
                reason="confidence below threshold",
            )
        ]
        decision = await _decide(judge, iteration=1, events=events)
        assert decision.kind == DecisionKind.FAILURE
        assert decision.should_escalate is True

    @pytest.mark.asyncio
    async def test_failure_axis_short_circuits_quality(self) -> None:
        # Even with a partial result and no judge, a RunError takes priority.
        judge = TerminationJudge()
        events = [_run_error(error_type="x", message="bad")]
        decision = await _decide(judge, iteration=1, events=events, partial_text=LONG_PARTIAL)
        assert decision.kind == DecisionKind.FAILURE


class TestLoopAxis:
    @pytest.mark.asyncio
    async def test_loop_detected_returns_loop_kind(self) -> None:
        # Pre-warm the loop detector so the first invoke trips it.
        detector = LoopDetector(use_fuzzy=False)
        detector.observe("tool_x{arg=1}")
        judge = TerminationJudge(loop_detector=detector)
        decision = await _decide(judge, iteration=1, step_signature="tool_x{arg=1}")
        assert decision.kind == DecisionKind.LOOP_DETECTED
        assert decision.is_complete is True
        assert decision.should_escalate is True

    @pytest.mark.asyncio
    async def test_no_signature_skips_loop_axis(self) -> None:
        judge = TerminationJudge()
        decision = await _decide(judge, iteration=1, partial_text=LONG_PARTIAL)
        # Should fall through to COMPLETE, not LOOP_DETECTED.
        assert decision.kind == DecisionKind.COMPLETE


class _StubJudgeProgram:
    """Mimics kaos_llm_core.programs.judge.Judge for the quality axis.

    We only need the ``await judge.invoke(...)`` shape: returns an
    object with ``.output.score`` (a float). The real Judge is wired
    into the AgentLoop via the ``judge`` constructor kwarg, so we
    don't need to integrate-test the full LLM round-trip here.
    """

    def __init__(self, score: float = 1.0, *, raises: bool = False) -> None:
        self._score = score
        self._raises = raises

    async def invoke(self, **kwargs: object) -> object:
        if self._raises:
            raise RuntimeError("judge unavailable")
        return SimpleNamespace(output=SimpleNamespace(score=self._score))


class TestQualityAxis:
    @pytest.mark.asyncio
    async def test_quality_below_threshold_no_partial_accepted(self) -> None:
        judge = TerminationJudge(
            judge=_StubJudgeProgram(score=0.3),
            min_quality=0.7,
            degradation_policy=DegradationPolicy(min_partial_chars=10_000),
        )
        criteria = SuccessCriteria(criteria=("must be accurate",))
        decision = await _decide(
            judge,
            iteration=1,
            partial_text=LONG_PARTIAL,
            success_criteria=criteria,
        )
        # DegradationPolicy refuses (min_partial_chars=10_000 > LONG_PARTIAL)
        # so the kind stays QUALITY_FAILED + escalate.
        assert decision.kind == DecisionKind.QUALITY_FAILED
        assert decision.should_escalate is True
        assert "0.30" in decision.feedback or "0.70" in decision.feedback

    @pytest.mark.asyncio
    async def test_quality_failure_can_degrade_when_policy_allows(self) -> None:
        judge = TerminationJudge(
            judge=_StubJudgeProgram(score=0.3),
            min_quality=0.7,
            degradation_policy=DegradationPolicy(
                min_partial_chars=10, accept_on_quality_failure=True
            ),
        )
        criteria = SuccessCriteria(criteria=("must be accurate",))
        decision = await _decide(
            judge,
            iteration=1,
            partial_text=LONG_PARTIAL,
            success_criteria=criteria,
        )
        assert decision.kind == DecisionKind.DEGRADED
        assert decision.partial_result == LONG_PARTIAL

    @pytest.mark.asyncio
    async def test_quality_above_threshold_completes(self) -> None:
        judge = TerminationJudge(judge=_StubJudgeProgram(score=0.9), min_quality=0.7)
        criteria = SuccessCriteria(criteria=("must be accurate",))
        decision = await _decide(
            judge,
            iteration=1,
            partial_text=LONG_PARTIAL,
            success_criteria=criteria,
        )
        assert decision.kind == DecisionKind.COMPLETE

    @pytest.mark.asyncio
    async def test_no_criteria_skips_judge_call(self) -> None:
        # Stub raises; if the Judge is invoked, the test fails.
        judge = TerminationJudge(
            judge=_StubJudgeProgram(score=0.0, raises=True),
            min_quality=0.7,
        )
        # No criteria → quality axis returns 1.0 without invoking judge.
        decision = await _decide(
            judge,
            iteration=1,
            partial_text=LONG_PARTIAL,
            success_criteria=SuccessCriteria(),
        )
        assert decision.kind == DecisionKind.COMPLETE

    @pytest.mark.asyncio
    async def test_judge_failure_returns_neutral_score(self) -> None:
        # Neutral 0.5 is below min_quality=0.7 → QUALITY_FAILED.
        judge = TerminationJudge(
            judge=_StubJudgeProgram(score=0.0, raises=True),
            min_quality=0.7,
            degradation_policy=DegradationPolicy(min_partial_chars=10_000),
        )
        criteria = SuccessCriteria(criteria=("any",))
        decision = await _decide(
            judge,
            iteration=1,
            partial_text=LONG_PARTIAL,
            success_criteria=criteria,
        )
        assert decision.kind == DecisionKind.QUALITY_FAILED


class TestProgramSurface:
    """The Judge is a Program — exercise the canonical contract."""

    @pytest.mark.asyncio
    async def test_invoke_returns_invocation_with_decision_output(self) -> None:
        judge = TerminationJudge()
        invocation = await judge.invoke(iteration=1)
        assert isinstance(invocation.output, Decision)

    @pytest.mark.asyncio
    async def test_call_returns_bare_decision(self) -> None:
        judge = TerminationJudge()
        result = await judge(iteration=1, partial_text=LONG_PARTIAL)
        assert isinstance(result, Decision)
        assert result.kind == DecisionKind.COMPLETE

    def test_loop_detector_property(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        judge = TerminationJudge(loop_detector=detector)
        assert judge.loop_detector is detector

    def test_degradation_policy_property(self) -> None:
        policy = DegradationPolicy(min_partial_chars=99)
        judge = TerminationJudge(degradation_policy=policy)
        assert judge.degradation_policy is policy
