"""Unit tests for ``Span(SpanSubject.JUDGE, ...)`` and
``Span(SpanSubject.ROUTE, ...)`` emission added in kaos-agents 0.1.0a9.

Pre-0.1.0a9 the Evaluate primitive's LLM judge ran invisibly (only the
final ``Judgment`` appeared in the inner ``ComposeResult.traces``
``PrimitiveTrace``) and the Route decision was likewise locked inside
``PrimitiveTrace``. SSE consumers (SPA run inspector, OTel exporter)
saw no events for either, so when ``matched=False`` killed a plan there
was no way to ask "what did the judge see?" or "what triggered REPLAN
here?" without reading VFS-persisted SessionMemory.

This test exercises both new spans via the same Evaluate + Route call
sites the production code uses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage

from kaos_agents.events.emitter import EventEmitter
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.planning.evaluate import EvalSemanticSignature, evaluate_semantic
from kaos_agents.types.plan import EvalMode, Judgment, PlanBudget

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _judgment_signature_output(
    *,
    matched: bool = True,
    confidence: float = 0.8,
    reasoning: str = "looks good",
) -> EvalSemanticSignature:
    return EvalSemanticSignature(
        result_text="some result text",
        expected_description="some expectation",
        additional_context="",
        matched=matched,
        confidence=confidence,
        reasoning=reasoning,
        new_facts=[],
    )


def _invocation(output: Any) -> Invocation:
    return Invocation(
        client=None,
        model="anthropic:claude-haiku-4-5",
        context=None,
        output=output,
        trace=None,
        usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, cost_usd=0.001),
    )


def _judgment(matched: bool = True, confidence: float = 0.8) -> Judgment:
    return Judgment(
        matched=matched,
        confidence=confidence,
        reasoning="test",
        mode=EvalMode.SEMANTIC,
    )


# ---------------------------------------------------------------------------
# SpanSubject enum gains JUDGE + ROUTE values
# ---------------------------------------------------------------------------


class TestSpanSubjectExtensions:
    def test_judge_subject_present(self) -> None:
        assert SpanSubject.JUDGE.value == "judge"

    def test_route_subject_present(self) -> None:
        assert SpanSubject.ROUTE.value == "route"


# ---------------------------------------------------------------------------
# evaluate_semantic emits Span(JUDGE, ...) when emitter provided
# ---------------------------------------------------------------------------


def _patch_evaluate_call(
    monkeypatch: pytest.MonkeyPatch, output: EvalSemanticSignature
) -> AsyncMock:
    """Replace the lazy ``from kaos_llm_core import Call`` lookup inside
    evaluate_semantic with a stub whose instances return our
    pre-built Invocation."""
    mock_invoke = AsyncMock(return_value=_invocation(output))

    class _StubCall:
        def __init__(self, signature: Any, *, model: str, examples: Any = None) -> None:
            pass

        async def invoke(self, **kwargs: Any) -> Invocation:
            return await mock_invoke(**kwargs)

    import kaos_llm_core as llm_core_mod

    monkeypatch.setattr(llm_core_mod, "Call", _StubCall, raising=False)
    return mock_invoke


class TestEvaluateSemanticJudgeSpan:
    @pytest.mark.asyncio
    async def test_emitter_none_no_spans_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Backwards compat — direct unit-test callers that don't pass
        # an emitter must not see Span events accidentally.
        _patch_evaluate_call(monkeypatch, _judgment_signature_output())

        # Build an emitter just to collect events, but DON'T pass it
        # to evaluate_semantic. Instead, run inside a collector scope
        # so we can confirm nothing got pushed.
        from kaos_agents.events.collector import collect_events

        with collect_events() as collector:
            await evaluate_semantic("result", "expected", model="x", emitter=None)

        judge_spans = [
            e for e in collector.events if isinstance(e, Span) and e.subject == SpanSubject.JUDGE
        ]
        assert judge_spans == []

    @pytest.mark.asyncio
    async def test_judge_span_start_complete_pair_emitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_evaluate_call(
            monkeypatch,
            _judgment_signature_output(matched=True, confidence=0.92, reasoning="match"),
        )

        from kaos_agents.events.collector import collect_events

        with collect_events() as collector:
            emitter = EventEmitter(session_id="sess-1", run_id="run-1")
            await evaluate_semantic(
                "Found 9 FR documents",
                "the SEC cyber rule",
                model="x",
                emitter=emitter,
                step_id="s-3-abc",
            )

        judge_spans = [
            e for e in collector.events if isinstance(e, Span) and e.subject == SpanSubject.JUDGE
        ]
        # Exactly one START + one COMPLETE.
        assert len(judge_spans) == 2
        start, complete = judge_spans[0], judge_spans[1]
        assert start.phase == SpanPhase.START
        assert complete.phase == SpanPhase.COMPLETE

        # START carries the inputs the judge sees.
        assert start.attributes["step_id"] == "s-3-abc"
        assert start.attributes["expected"] == "the SEC cyber rule"
        assert "Found 9 FR documents" in start.attributes["result_preview"]

        # COMPLETE carries the judgment.
        assert complete.attributes["step_id"] == "s-3-abc"
        assert complete.attributes["matched"] is True
        assert complete.attributes["confidence"] == pytest.approx(0.92)
        assert "match" in complete.attributes["reasoning"]
        assert complete.attributes["mode"] == "semantic"

        # Span ids match (same span_id on START + COMPLETE).
        assert start.span_id == complete.span_id

    @pytest.mark.asyncio
    async def test_judge_span_error_on_call_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _ExplodingCall:
            def __init__(self, signature: Any, *, model: str, examples: Any = None) -> None:
                pass

            async def invoke(self, **kwargs: Any) -> Invocation:
                raise RuntimeError("LLM gateway down")

        import kaos_llm_core as llm_core_mod

        monkeypatch.setattr(llm_core_mod, "Call", _ExplodingCall, raising=False)

        from kaos_agents.events.collector import collect_events

        with collect_events() as collector:
            emitter = EventEmitter(session_id="s", run_id="r")
            judgment = await evaluate_semantic(
                "result",
                "expectation",
                model="x",
                emitter=emitter,
                step_id="s-1",
            )

        # evaluate_semantic returns an unverified judgment on failure
        # (matched=False, mode=STRUCTURAL).
        assert judgment.matched is False
        assert judgment.mode == EvalMode.STRUCTURAL

        # Span pair should be START + ERROR (not START + COMPLETE).
        judge_spans = [
            e for e in collector.events if isinstance(e, Span) and e.subject == SpanSubject.JUDGE
        ]
        assert len(judge_spans) == 2
        assert judge_spans[0].phase == SpanPhase.START
        assert judge_spans[1].phase == SpanPhase.ERROR
        err_msg = judge_spans[1].error_message
        assert err_msg is not None
        assert "LLM gateway down" in err_msg

    @pytest.mark.asyncio
    async def test_judge_span_truncates_large_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Span attributes are size-bounded so SSE consumers don't get
        # multi-KB tool outputs in their event stream.
        _patch_evaluate_call(monkeypatch, _judgment_signature_output())

        from kaos_agents.events.collector import collect_events

        big_result = "X" * 5000
        big_expected = "Y" * 5000

        with collect_events() as collector:
            emitter = EventEmitter(session_id="s", run_id="r")
            await evaluate_semantic(
                big_result, big_expected, model="x", emitter=emitter, step_id="s-1"
            )

        judge_spans = [
            e for e in collector.events if isinstance(e, Span) and e.subject == SpanSubject.JUDGE
        ]
        start = judge_spans[0]
        # Truncation cap is 200 chars per the spans.py docstring convention.
        assert len(start.attributes["expected"]) <= 200
        assert len(start.attributes["result_preview"]) <= 200


# ---------------------------------------------------------------------------
# Span(ROUTE) is emitted from compose.py — exercise via the compose loop
# ---------------------------------------------------------------------------


class TestComposeRouteSpan:
    """The ROUTE span is emitted inside compose.py's main loop after
    every Route call. We can't exercise that without a real plan + act
    flow, but we can confirm the Span enum gains the value and that
    compose accepts the ``emitter`` parameter."""

    def test_compose_accepts_emitter_kwarg(self) -> None:
        # Compile-time check via inspect — confirms the new parameter
        # didn't get accidentally dropped.
        import inspect

        from kaos_agents.planning.compose import compose

        sig = inspect.signature(compose)
        assert "emitter" in sig.parameters
        # And it defaults to None (backwards compat).
        assert sig.parameters["emitter"].default is None

    def test_route_function_still_works_without_emitter(self) -> None:
        # The route() function itself doesn't take an emitter — Span
        # emission lives in compose.py at the call site. This test
        # documents that contract.
        from kaos_agents.planning.route import route as route_fn

        result = route_fn(_judgment(matched=True, confidence=0.9), PlanBudget())
        assert result.decision.value == "continue"
