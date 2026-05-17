"""Unit tests for conditional plan-step execution (Wish #4 / 0.1.0a9).

Covers:

* ``Step.abort_if`` + ``Step.pivot_to`` fields with defaults.
* ``PlanGraph.add_step`` stores them as node properties.
* ``PlanGraph.get_step`` exposes them.
* ``evaluate_condition`` helper:
  - Returns ``(False, "")`` without an LLM call for empty condition
    or empty prior_outputs (Signature rule 1 / cost guard).
  - Returns ``(True, evidence)`` when stub Call returns
    ``holds=true, confidence=0.9``.
  - Returns ``(False, "")`` when ``holds=true`` but confidence is
    below the threshold (LLM hedging).
  - Returns ``(False, "")`` and logs a warning when the Call raises.
* ``StopReason.PIVOTED`` enum value exists.

End-to-end Compose flow (skip-on-abort + pivot break) is exercised
indirectly via the stubbed evaluator in
``tests/unit/planning/test_compose_conditional.py`` — kept in a
separate file because it's a larger fixture.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage

from kaos_agents.planning.evaluate_condition import (
    EvaluateConditionSignature,
    _format_prior_outputs,
    evaluate_condition,
)
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.types.plan import Step, StepType, StopReason

# ---------------------------------------------------------------------------
# Step type fields
# ---------------------------------------------------------------------------


class TestStepConditionalFields:
    def test_step_defaults_to_empty_conditions(self) -> None:
        step = Step(id="s1", step_type=StepType.TOOL, description="x")
        assert step.abort_if == ""
        assert step.pivot_to == ""

    def test_step_accepts_abort_if(self) -> None:
        step = Step(
            id="s1",
            step_type=StepType.TOOL,
            description="x",
            abort_if="rule was vacated or stayed",
        )
        assert step.abort_if == "rule was vacated or stayed"

    def test_step_accepts_pivot_to(self) -> None:
        step = Step(
            id="s1",
            step_type=StepType.TOOL,
            description="x",
            abort_if="rule was vacated",
            pivot_to="research the litigation status",
        )
        assert step.pivot_to == "research the litigation status"


class TestPlanGraphStoresAndRetrievesConditions:
    def test_add_step_persists_abort_if(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(
            Step(
                id="s1",
                step_type=StepType.TOOL,
                description="x",
                abort_if="rule was vacated",
                pivot_to="research litigation",
            )
        )
        props = graph.get_step("s1")
        assert props is not None
        assert props["abort_if"] == "rule was vacated"
        assert props["pivot_to"] == "research litigation"

    def test_step_without_conditions_returns_empty_strings(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(Step(id="s1", step_type=StepType.TOOL, description="x"))
        props = graph.get_step("s1")
        assert props is not None
        assert props["abort_if"] == ""
        assert props["pivot_to"] == ""


# ---------------------------------------------------------------------------
# StopReason.PIVOTED
# ---------------------------------------------------------------------------


class TestStopReasonPivoted:
    def test_pivoted_value(self) -> None:
        assert StopReason.PIVOTED.value == "pivoted"


# ---------------------------------------------------------------------------
# _format_prior_outputs
# ---------------------------------------------------------------------------


class TestFormatPriorOutputs:
    def test_empty_dict_returns_empty_string(self) -> None:
        assert _format_prior_outputs({}) == ""

    def test_renders_step_id_prefix(self) -> None:
        text = _format_prior_outputs({"s1": "result one", "s2": "result two"})
        assert "[s1]: result one" in text
        assert "[s2]: result two" in text

    def test_truncates_long_outputs(self) -> None:
        long_payload = "y" * 2000
        text = _format_prior_outputs({"s1": long_payload}, per_step_char_limit=500)
        assert long_payload not in text
        assert "..." in text


# ---------------------------------------------------------------------------
# evaluate_condition — gating + LLM stub
# ---------------------------------------------------------------------------


def _sig_output(
    *, holds: bool = False, confidence: float = 0.0, evidence: str = ""
) -> EvaluateConditionSignature:
    return EvaluateConditionSignature(
        condition="x",
        prior_outputs="y",
        holds=holds,
        confidence=confidence,
        evidence=evidence,
    )


def _invocation(output: Any) -> Invocation:
    return Invocation(
        client=None,
        model="anthropic:claude-haiku-4-5",
        context=None,
        output=output,
        trace=None,
        usage=TokenUsage(input_tokens=80, output_tokens=10, total_tokens=90, cost_usd=0.0005),
    )


def _patch_call(monkeypatch: pytest.MonkeyPatch, output: EvaluateConditionSignature) -> AsyncMock:
    mock_invoke = AsyncMock(return_value=_invocation(output))

    class _StubCall:
        def __init__(self, signature: Any, *, model: str) -> None:
            pass

        async def invoke(self, **kwargs: Any) -> Invocation:
            return await mock_invoke(**kwargs)

    import kaos_llm_core as llm_core_mod

    monkeypatch.setattr(llm_core_mod, "Call", _StubCall, raising=False)
    return mock_invoke


class TestEvaluateCondition:
    @pytest.mark.asyncio
    async def test_empty_condition_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _patch_call(monkeypatch, _sig_output(holds=True, confidence=1.0))
        holds, evidence = await evaluate_condition("", {"s1": "anything"})
        assert holds is False
        assert evidence == ""
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_prior_outputs_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _patch_call(monkeypatch, _sig_output(holds=True, confidence=1.0))
        holds, evidence = await evaluate_condition("rule was vacated", {})
        assert holds is False
        assert evidence == ""
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_high_confidence_holds_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_call(
            monkeypatch,
            _sig_output(
                holds=True, confidence=0.9, evidence="court vacated the rule on 2024-08-01"
            ),
        )
        holds, evidence = await evaluate_condition(
            "rule was vacated", {"s1": "earlier step output"}
        )
        assert holds is True
        assert "vacated the rule" in evidence

    @pytest.mark.asyncio
    async def test_low_confidence_holds_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # holds=True but confidence below threshold → return False (LLM hedging).
        _patch_call(
            monkeypatch,
            _sig_output(holds=True, confidence=0.4, evidence="maybe vacated, unclear"),
        )
        holds, evidence = await evaluate_condition(
            "rule was vacated",
            {"s1": "earlier step output"},
            holds_confidence_threshold=0.6,
        )
        assert holds is False
        assert evidence == ""

    @pytest.mark.asyncio
    async def test_holds_false_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_call(monkeypatch, _sig_output(holds=False, confidence=0.9))
        holds, _ = await evaluate_condition("X", {"s1": "anything"})
        assert holds is False

    @pytest.mark.asyncio
    async def test_call_exception_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Degraded LLM environment — evaluate_condition must NOT raise,
        # it must return (False, "") so the step still executes.
        class _ExplodingCall:
            def __init__(self, signature: Any, *, model: str) -> None:
                pass

            async def invoke(self, **kwargs: Any) -> Invocation:
                raise RuntimeError("LLM gateway 503")

        import kaos_llm_core as llm_core_mod

        monkeypatch.setattr(llm_core_mod, "Call", _ExplodingCall, raising=False)

        holds, evidence = await evaluate_condition("X", {"s1": "Y"})
        assert holds is False
        assert evidence == ""
