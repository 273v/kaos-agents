"""Tests for the Evaluate planning primitive."""

from __future__ import annotations

from kaos_agents.planning.evaluate import evaluate, evaluate_structural
from kaos_agents.planning.types import EvalMode


class TestEvaluateStructural:
    def test_error_result_detected(self):
        j = evaluate_structural("ERROR: Connection timeout")
        assert j is not None
        assert not j.matched
        assert j.confidence == 1.0
        assert j.mode == EvalMode.STRUCTURAL

    def test_json_error_detected(self):
        j = evaluate_structural('{"error": true, "message": "failed"}')
        assert j is not None
        assert not j.matched

    def test_empty_result_detected(self):
        j = evaluate_structural("")
        assert j is not None
        assert not j.matched
        assert "empty" in j.reasoning.lower()

    def test_whitespace_only_detected(self):
        j = evaluate_structural("   \n  ")
        assert j is not None
        assert not j.matched

    def test_tool_name_validation(self):
        j = evaluate_structural(
            "some result",
            tool_name="hallucinated-tool",
            available_tools={"kaos-pdf-parse", "kaos-web-fetch"},
        )
        assert j is not None
        assert not j.matched
        assert "not registered" in j.reasoning

    def test_valid_tool_passes(self):
        j = evaluate_structural(
            "some result",
            tool_name="kaos-pdf-parse",
            available_tools={"kaos-pdf-parse", "kaos-web-fetch"},
        )
        # Tool validation passes, but no expected output → default accept
        assert j is not None
        assert j.matched
        assert j.confidence == 0.7

    def test_no_expectation_accepts_result(self):
        j = evaluate_structural("valid non-empty result")
        assert j is not None
        assert j.matched
        assert j.confidence == 0.7

    def test_has_expectation_returns_none(self):
        """Structural can't verify semantic match — returns None for semantic fallback."""
        j = evaluate_structural("some text", expected="a list of CFR sections")
        assert j is None

    def test_none_result(self):
        j = evaluate_structural(None)
        assert j is not None
        assert not j.matched


class TestEvaluateCombined:
    async def test_structural_short_circuits(self):
        """If structural resolves it, no LLM call happens."""
        j = await evaluate("ERROR: timeout", expected="success")
        assert not j.matched
        assert j.mode == EvalMode.STRUCTURAL

    async def test_empty_no_expectation(self):
        j = await evaluate("some result")
        assert j.matched
        assert j.mode == EvalMode.STRUCTURAL

    async def test_none_result(self):
        j = await evaluate(None, expected="something")
        assert not j.matched
