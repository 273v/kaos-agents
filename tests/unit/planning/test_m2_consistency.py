"""Unit tests for :mod:`kaos_agents.planning.m2_consistency`.

These tests cover the rubric SHAPE — the live label-emission tests
live in ``tests/integration/test_m2_consistency_live.py``.
"""

from __future__ import annotations

import inspect

import pytest

from kaos_agents.planning.judge import JudgeVerdict
from kaos_agents.planning.m2_consistency import (
    M2_ALLOWED_LABELS,
    M2_REASONING_ACTION_RUBRIC,
    judge_reasoning_action_consistency,
)


class TestRubricShape:
    """The rubric string must enumerate exactly the allowed labels."""

    def test_allowed_labels_constant_is_tuple_of_str(self) -> None:
        assert isinstance(M2_ALLOWED_LABELS, tuple)
        assert all(isinstance(label, str) for label in M2_ALLOWED_LABELS)
        assert len(M2_ALLOWED_LABELS) == 3

    def test_allowed_labels_exact_set(self) -> None:
        assert set(M2_ALLOWED_LABELS) == {
            "consistent",
            "contradicts_reasoning",
            "contradicts_tool_results",
        }

    def test_rubric_mentions_every_allowed_label(self) -> None:
        for label in M2_ALLOWED_LABELS:
            assert label in M2_REASONING_ACTION_RUBRIC, (
                f"Label {label!r} must appear verbatim in the rubric "
                "so the judge's docstring rule 1 (case-insensitive "
                "label allowlist) is satisfied."
            )

    def test_rubric_documents_three_decision_rules(self) -> None:
        # Each rule must be numbered and explain when to emit which label.
        text = M2_REASONING_ACTION_RUBRIC.lower()
        assert "1." in text and "2." in text and "3." in text
        # Each label must appear in proximity to its decision rule.
        for label in M2_ALLOWED_LABELS:
            assert text.count(label) >= 1

    def test_rubric_calls_out_no_tools_edge_case(self) -> None:
        # When no tools were invoked, contradicts_tool_results MUST NOT
        # fire — otherwise the critic loops on every tool-less response.
        rubric_lower = M2_REASONING_ACTION_RUBRIC.lower()
        assert "no tools" in rubric_lower or "empty" in rubric_lower


class TestHelperContract:
    """The wrapper function must proxy to ``judge_with_rubric``."""

    def test_helper_is_async(self) -> None:
        assert inspect.iscoroutinefunction(judge_reasoning_action_consistency)

    def test_helper_signature(self) -> None:
        sig = inspect.signature(judge_reasoning_action_consistency)
        params = sig.parameters
        # Keyword-only by design.
        for name in ("response_text", "model", "tool_results_text"):
            assert name in params, f"missing kwarg: {name}"
            assert params[name].kind == inspect.Parameter.KEYWORD_ONLY

    def test_helper_tool_results_default_empty(self) -> None:
        sig = inspect.signature(judge_reasoning_action_consistency)
        assert sig.parameters["tool_results_text"].default == ""

    def test_helper_return_annotation(self) -> None:
        # ``from __future__ import annotations`` defers evaluation, so
        # resolve via ``get_type_hints`` to compare against the real
        # JudgeVerdict object rather than the string ``"JudgeVerdict"``.
        import typing

        hints = typing.get_type_hints(judge_reasoning_action_consistency)
        assert hints["return"] is JudgeVerdict


class TestFallbackPath:
    """When kaos-llm-core isn't installed, ``judge_with_rubric`` returns
    a ``fell_back`` verdict synchronously. The helper inherits that
    contract — verify it propagates."""

    @pytest.mark.asyncio
    async def test_helper_propagates_fell_back_on_no_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force kaos-llm-core import to fail inside judge_with_rubric
        # by removing the cached module + patching the loader.
        import sys

        # If kaos_llm_core is already imported, hide the submodule.
        monkeypatch.setitem(sys.modules, "kaos_llm_core.programs.call", None)

        verdict = await judge_reasoning_action_consistency(
            response_text="Branch taken: upper bound >= 5.0%. It's 4.50%.",
            model="anthropic:claude-haiku-4-5",
        )
        assert verdict.fell_back is True
        assert verdict.label == ""
        assert verdict.confidence == 0.0
