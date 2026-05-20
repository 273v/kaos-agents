"""Unit tests for :mod:`kaos_agents.planning.m3_grounding`.

These tests cover the rubric SHAPE — the live label-emission tests
live in ``tests/integration/test_m3_grounding_live.py``.
"""

from __future__ import annotations

import inspect

import pytest

from kaos_agents.planning.judge import JudgeVerdict
from kaos_agents.planning.m3_grounding import (
    M3_ALLOWED_LABELS,
    M3_GROUNDING_RUBRIC,
    judge_grounding_fabrication,
)


class TestRubricShape:
    def test_allowed_labels_constant_is_tuple_of_str(self) -> None:
        assert isinstance(M3_ALLOWED_LABELS, tuple)
        assert all(isinstance(label, str) for label in M3_ALLOWED_LABELS)
        assert len(M3_ALLOWED_LABELS) == 3

    def test_allowed_labels_exact_set(self) -> None:
        assert set(M3_ALLOWED_LABELS) == {
            "grounded",
            "fabricated_with_admission",
            "fabricated_without_admission",
        }

    def test_rubric_mentions_every_allowed_label(self) -> None:
        for label in M3_ALLOWED_LABELS:
            assert label in M3_GROUNDING_RUBRIC

    def test_rubric_documents_three_decision_rules(self) -> None:
        text = M3_GROUNDING_RUBRIC.lower()
        assert "1." in text and "2." in text and "3." in text
        for label in M3_ALLOWED_LABELS:
            assert text.count(label) >= 1

    def test_rubric_calls_out_no_tools_edge_case(self) -> None:
        # When no tools were invoked, the rubric must explain what
        # counts as fabrication vs grounded.
        rubric_lower = M3_GROUNDING_RUBRIC.lower()
        assert "no tools" in rubric_lower or "context`` is empty" in rubric_lower


class TestHelperContract:
    def test_helper_is_async(self) -> None:
        assert inspect.iscoroutinefunction(judge_grounding_fabrication)

    def test_helper_signature(self) -> None:
        sig = inspect.signature(judge_grounding_fabrication)
        params = sig.parameters
        for name in ("response_text", "model", "tool_results_text"):
            assert name in params, f"missing kwarg: {name}"
            assert params[name].kind == inspect.Parameter.KEYWORD_ONLY

    def test_helper_tool_results_default_empty(self) -> None:
        sig = inspect.signature(judge_grounding_fabrication)
        assert sig.parameters["tool_results_text"].default == ""

    def test_helper_return_annotation(self) -> None:
        import typing

        hints = typing.get_type_hints(judge_grounding_fabrication)
        assert hints["return"] is JudgeVerdict


class TestFallbackPath:
    @pytest.mark.asyncio
    async def test_helper_propagates_fell_back_on_no_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "kaos_llm_core.programs.call", None)
        verdict = await judge_grounding_fabrication(
            response_text=(
                "I couldn't fetch the SEC filing but Item 1C discusses cybersecurity governance."
            ),
            model="anthropic:claude-haiku-4-5",
        )
        assert verdict.fell_back is True
        assert verdict.label == ""
        assert verdict.confidence == 0.0
