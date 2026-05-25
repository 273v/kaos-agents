"""Unit tests for ``kaos_agents.patterns.synthesize``.

The inner kaos-llm-core ``Call.invoke`` is stubbed so no live LLM call
is issued. We construct a synthetic :class:`Invocation` whose
``output`` is a ``SynthesizeFindingsSignature`` instance and patch
``Call.invoke`` on the module's lookup so the helper resolves to our
mock without instantiating a real LLM client.

Coverage:

* :func:`_format_step_results_for_prompt` rendering invariants
  (step-id prefix, truncation, missing description fallback).
* :func:`should_attempt_llm_synthesis` gating (empty / all-error /
  some-real-content).
* :func:`synthesize_findings` end-to-end wiring (kwargs threaded into
  the Call, ``(narrative, InvocationUsage)`` return shape, usage
  populated from the invocation).
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage

from kaos_agents.patterns.synthesize import (
    SynthesizeFindingsSignature,
    _format_step_results_for_prompt,
    should_attempt_llm_synthesis,
    synthesize_findings,
)
from kaos_agents.types.plan import ComposeResult, StopReason
from kaos_agents.types.usage import InvocationUsage

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _signature_output(narrative: str = "default narrative") -> SynthesizeFindingsSignature:
    """Build a SynthesizeFindingsSignature with sensible input/output defaults."""
    return SynthesizeFindingsSignature(
        goal="some user goal",
        step_results="[s1]: ok",
        stop_reason="success",
        narrative=narrative,
    )


def _invocation(output: Any, *, cost_usd: float = 0.001, total_tokens: int = 250) -> Invocation:
    usage = TokenUsage(
        input_tokens=200,
        output_tokens=50,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
    return Invocation(
        client=None,
        model="anthropic:claude-haiku-4-5",
        context=None,
        output=output,
        trace=None,
        usage=usage,
    )


def _compose_result(
    *,
    stop: StopReason = StopReason.SUCCESS,
    step_results: dict[str, Any] | None = None,
    steps_executed: int = 1,
) -> ComposeResult:
    return ComposeResult(
        plan_json="{}",
        stop_reason=stop,
        steps_executed=steps_executed,
        step_results=step_results or {},
    )


# ---------------------------------------------------------------------------
# _format_step_results_for_prompt
# ---------------------------------------------------------------------------


class TestFormatStepResultsForPrompt:
    def test_basic_render(self) -> None:
        text = _format_step_results_for_prompt(
            {"s1": "found 9 documents", "s2": "fetched 1 doc"},
        )
        assert "[s1]" in text
        assert "found 9 documents" in text
        assert "[s2]" in text
        assert "fetched 1 doc" in text

    def test_with_descriptions(self) -> None:
        text = _format_step_results_for_prompt(
            {"s1": "result"},
            step_descriptions={"s1": "Search FR for cyber rule"},
        )
        assert "[s1] Search FR for cyber rule:" in text

    def test_long_results_truncated(self) -> None:
        long_payload = "x" * 3000
        text = _format_step_results_for_prompt(
            {"s1": long_payload},
            per_step_char_limit=500,
        )
        # Truncated, with marker.
        assert long_payload not in text
        assert "..." in text

    def test_empty_dict_returns_empty_string(self) -> None:
        assert _format_step_results_for_prompt({}) == ""


# ---------------------------------------------------------------------------
# should_attempt_llm_synthesis
# ---------------------------------------------------------------------------


class TestShouldAttemptLlmSynthesis:
    def test_empty_step_results_returns_false(self) -> None:
        result = _compose_result(stop=StopReason.SUCCESS, step_results={})
        assert should_attempt_llm_synthesis(result) is False

    def test_all_error_results_returns_false(self) -> None:
        result = _compose_result(
            stop=StopReason.NEEDS_REPLAN,
            step_results={"s1": "ERROR: timed out", "s2": "ERROR: 500"},
        )
        assert should_attempt_llm_synthesis(result) is False

    def test_mixed_results_returns_true(self) -> None:
        # One real result is enough — synthesis can mention that one
        # and acknowledge the failures.
        result = _compose_result(
            stop=StopReason.NEEDS_REPLAN,
            step_results={"s1": "ERROR: 500", "s2": "Found 9 FR documents"},
        )
        assert should_attempt_llm_synthesis(result) is True

    def test_all_real_results_returns_true(self) -> None:
        result = _compose_result(
            stop=StopReason.SUCCESS,
            step_results={"s1": "ok", "s2": "ok2"},
        )
        assert should_attempt_llm_synthesis(result) is True


# ---------------------------------------------------------------------------
# synthesize_findings — end-to-end wiring
# ---------------------------------------------------------------------------


class TestSynthesizeFindings:
    @pytest.mark.asyncio
    async def test_returns_narrative_and_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch the module-level Call import inside synthesize_findings.
        # The function does ``from kaos_llm_core import Call`` lazily
        # inside its body — we replace that class with a stub whose
        # instances have a stub ``invoke`` returning our Invocation.
        captured: dict[str, Any] = {}

        class _StubCall:
            def __init__(self, signature: Any, *, model: str, examples: Any = None) -> None:
                captured["signature"] = signature
                captured["model"] = model

            async def invoke(self, **kwargs: Any) -> Invocation:
                captured["kwargs"] = kwargs
                return _invocation(
                    _signature_output(narrative="The SEC adopted EDGAR Filer Access [s1]."),
                    cost_usd=0.0042,
                    total_tokens=712,
                )

        # Patch the lazy import inside synthesize_findings — easiest is
        # to inject a fake kaos_llm_core.Call attribute that the inner
        # ``from kaos_llm_core import Call`` will resolve to.
        import kaos_llm_core as llm_core_mod

        monkeypatch.setattr(llm_core_mod, "Call", _StubCall, raising=False)

        result = _compose_result(
            stop=StopReason.NEEDS_REPLAN,
            step_results={"s1": "EDGAR Filer Access and Account Management"},
            steps_executed=1,
        )
        narrative, usage = await synthesize_findings(
            "what is the most recent SEC cyber rule?",
            result,
            model="anthropic:claude-haiku-4-5",
        )

        assert narrative == "The SEC adopted EDGAR Filer Access [s1]."
        assert isinstance(usage, InvocationUsage)
        assert usage.cost_usd == pytest.approx(0.0042)
        assert usage.total_tokens == 712

        # Verify the Call was constructed with the right signature + model
        assert captured["signature"] is SynthesizeFindingsSignature
        assert captured["model"] == "anthropic:claude-haiku-4-5"
        # And invoked with the goal / step_results / stop_reason
        kwargs = captured["kwargs"]
        assert kwargs["goal"] == "what is the most recent SEC cyber rule?"
        assert "s1" in kwargs["step_results"]
        assert "EDGAR Filer Access" in kwargs["step_results"]
        assert kwargs["stop_reason"] == "needs_replan"

    @pytest.mark.asyncio
    async def test_passes_descriptions_into_formatted_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class _StubCall:
            def __init__(self, signature: Any, *, model: str, examples: Any = None) -> None:
                pass

            async def invoke(self, **kwargs: Any) -> Invocation:
                captured["kwargs"] = kwargs
                return _invocation(_signature_output())

        import kaos_llm_core as llm_core_mod

        monkeypatch.setattr(llm_core_mod, "Call", _StubCall, raising=False)

        result = _compose_result(
            stop=StopReason.SUCCESS,
            step_results={"s1": "found docs"},
        )
        await synthesize_findings(
            "goal",
            result,
            model="x",
            step_descriptions={"s1": "Search Federal Register"},
        )

        # Description should appear in the rendered step_results.
        assert "Search Federal Register" in captured["kwargs"]["step_results"]

    @pytest.mark.asyncio
    async def test_propagates_invoke_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The caller in plan_execute.py is responsible for catching;
        # synthesize_findings must surface the error.
        class _ExplodingCall:
            def __init__(self, signature: Any, *, model: str, examples: Any = None) -> None:
                pass

            async def invoke(self, **kwargs: Any) -> Invocation:
                raise RuntimeError("the LLM gateway is on fire")

        import kaos_llm_core as llm_core_mod

        monkeypatch.setattr(llm_core_mod, "Call", _ExplodingCall, raising=False)

        result = _compose_result(
            stop=StopReason.SUCCESS,
            step_results={"s1": "ok"},
        )
        with pytest.raises(RuntimeError, match="LLM gateway is on fire"):
            await synthesize_findings("g", result, model="x")
