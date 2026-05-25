"""Tests for the Expand planning primitive — step parsing and validation.

Also pins the ``multi_chain_n`` routing contract: when ``>= 2``, plan
generation MUST route through ``kaos_llm_core.programs.multi_chain_comparison.MultiChainComparison``
and forward the canonical ``load_examples("plan_expand")`` few-shot
pool to every producer chain (the Iter-4-14 grounded-Signature
contract). Requires kaos-llm-core >= 0.1.2 which added ``examples=``
forwarding to MCC.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.planning.expand import _parse_steps, expand, expand_from_steps
from kaos_agents.types.plan import Step, StepType


class TestParseSteps:
    def test_valid_tool_step(self):
        raw = [
            {
                "step_number": 1,
                "description": "Search eCFR",
                "tool_name": "kaos-source-ecfr-search",
                "input_description": "query=Clean Air Act",
                "expected_output": "List of CFR sections",
                "depends_on": [],
            }
        ]
        steps = _parse_steps(raw, {"kaos-source-ecfr-search": "Search eCFR"})
        assert len(steps) == 1
        assert steps[0].step_type == StepType.TOOL
        assert steps[0].tool_name == "kaos-source-ecfr-search"
        assert steps[0].expected_output == "List of CFR sections"

    def test_llm_step(self):
        raw = [
            {
                "step_number": 1,
                "description": "Summarize findings",
                "tool_name": "llm",
                "input_description": "Combine results",
                "expected_output": "Summary",
                "depends_on": [],
            }
        ]
        steps = _parse_steps(raw, {})
        assert len(steps) == 1
        assert steps[0].step_type == StepType.LLM
        assert steps[0].tool_name is None

    def test_hallucinated_tool_converted_to_llm(self):
        raw = [
            {
                "step_number": 1,
                "description": "Do something",
                "tool_name": "nonexistent-tool",
                "input_description": "",
                "expected_output": "",
                "depends_on": [],
            }
        ]
        steps = _parse_steps(raw, {"real-tool": "exists"})
        assert len(steps) == 1
        assert steps[0].step_type == StepType.LLM  # Converted
        assert steps[0].tool_name is None
        assert "not found" in steps[0].description

    def test_dependency_resolution(self):
        raw = [
            {
                "step_number": 1,
                "description": "Step A",
                "tool_name": "tool-a",
                "depends_on": [],
            },
            {
                "step_number": 2,
                "description": "Step B",
                "tool_name": "tool-b",
                "depends_on": [1],
            },
        ]
        steps = _parse_steps(raw, {"tool-a": "A", "tool-b": "B"})
        assert len(steps) == 2
        assert len(steps[1].depends_on) == 1
        # Dependency should reference step 1's ID
        assert steps[1].depends_on[0] == steps[0].id

    def test_invalid_dependency_skipped(self):
        raw = [
            {
                "step_number": 1,
                "description": "Step A",
                "tool_name": "tool-a",
                "depends_on": [99],  # Nonexistent step
            }
        ]
        steps = _parse_steps(raw, {"tool-a": "A"})
        assert len(steps) == 1
        assert steps[0].depends_on == ()  # Invalid dep skipped

    def test_empty_tool_name_is_llm(self):
        raw = [{"step_number": 1, "description": "Think", "tool_name": ""}]
        steps = _parse_steps(raw, {})
        assert steps[0].step_type == StepType.LLM

    def test_multiple_steps(self):
        raw = [
            {"step_number": 1, "description": "Search", "tool_name": "search-tool"},
            {"step_number": 2, "description": "Analyze", "tool_name": "llm", "depends_on": [1]},
            {"step_number": 3, "description": "Fetch", "tool_name": "fetch-tool"},
        ]
        steps = _parse_steps(raw, {"search-tool": "Search", "fetch-tool": "Fetch"})
        assert len(steps) == 3
        assert steps[0].step_type == StepType.TOOL
        assert steps[1].step_type == StepType.LLM
        assert steps[2].step_type == StepType.TOOL

    def test_no_available_tools_all_pass(self):
        """When no tool registry is provided, all tool names pass validation."""
        raw = [{"step_number": 1, "description": "Use tool", "tool_name": "any-tool"}]
        steps = _parse_steps(raw, {})
        assert steps[0].step_type == StepType.TOOL
        assert steps[0].tool_name == "any-tool"


class TestExpandFromSteps:
    def test_passthrough(self):
        steps = [
            Step(id="s1", step_type=StepType.TOOL, description="test", tool_name="t"),
        ]
        assert expand_from_steps(steps) is steps


class TestExpandMultiChainRouting:
    """Pins the ``multi_chain_n`` routing contract.

    The contract:
    - ``multi_chain_n=None`` or ``< 2`` → single ``Call(PlanExpandSignature, ...)`` path.
    - ``multi_chain_n >= 2`` → ``MultiChainComparison(PlanExpandSignature, n=...)``
      with the SAME ``load_examples("plan_expand")`` few-shot pool wired
      to every producer chain (Iter-4-14 grounded-Signature contract).

    These tests stub both Call and MultiChainComparison at the import
    site to record constructor args without making a live LLM call.
    """

    @pytest.mark.asyncio
    async def test_multi_chain_n_routes_through_mcc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        constructed_args: dict[str, Any] = {}

        class _StubMCC:
            def __init__(
                self,
                signature: type,
                *,
                n: int,
                producer_model: str | None = None,
                examples: list[Any] | None = None,
                **kwargs: Any,
            ) -> None:
                constructed_args["signature_name"] = signature.__name__
                constructed_args["n"] = n
                constructed_args["producer_model"] = producer_model
                constructed_args["examples"] = examples
                constructed_args["other_kwargs"] = kwargs

            async def invoke(self, **inputs: Any) -> Any:
                return SimpleNamespace(output=SimpleNamespace(steps=[]))

        monkeypatch.setattr(
            "kaos_llm_core.programs.multi_chain_comparison.MultiChainComparison",
            _StubMCC,
        )

        result = await expand(
            goal="explain rule 10b-5",
            model="anthropic:claude-haiku-4-5",
            multi_chain_n=5,
        )

        assert constructed_args["signature_name"] == "PlanExpandSignature"
        assert constructed_args["n"] == 5
        assert constructed_args["producer_model"] == "anthropic:claude-haiku-4-5"
        # The canonical plan_expand pool must reach the MCC producer
        # — same examples= contract the single-Call path uses. Without
        # this, MCC's chain samples lose Iter-4-14 calibration.
        examples = constructed_args["examples"]
        assert examples is not None, "examples= must be forwarded to MCC"
        assert len(examples) >= 1, "plan_expand pool must be non-empty"
        assert result == []  # stub returned no steps

    @pytest.mark.asyncio
    async def test_default_multi_chain_none_uses_single_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Back-compat: omitting multi_chain_n keeps the single-Call path."""
        constructed: list[str] = []

        class _StubCall:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                constructed.append("Call")

            async def invoke(self, **inputs: Any) -> Any:
                return SimpleNamespace(output=SimpleNamespace(steps=[]))

        class _StubMCC:
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                constructed.append("MCC")

            async def invoke(self, **inputs: Any) -> Any:  # pragma: no cover
                return SimpleNamespace(output=SimpleNamespace(steps=[]))

        monkeypatch.setattr("kaos_llm_core.Call", _StubCall)
        monkeypatch.setattr(
            "kaos_llm_core.programs.multi_chain_comparison.MultiChainComparison",
            _StubMCC,
        )

        await expand(goal="anything", model="x")
        assert constructed == ["Call"]

    @pytest.mark.asyncio
    async def test_multi_chain_n_below_2_uses_single_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """multi_chain_n=1 (or 0) keeps the single-Call path — MCC requires n>=2."""
        constructed: list[str] = []

        class _StubCall:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                constructed.append("Call")

            async def invoke(self, **inputs: Any) -> Any:
                return SimpleNamespace(output=SimpleNamespace(steps=[]))

        class _StubMCC:
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                constructed.append("MCC")

        monkeypatch.setattr("kaos_llm_core.Call", _StubCall)
        monkeypatch.setattr(
            "kaos_llm_core.programs.multi_chain_comparison.MultiChainComparison",
            _StubMCC,
        )

        await expand(goal="anything", model="x", multi_chain_n=1)
        assert constructed == ["Call"]
