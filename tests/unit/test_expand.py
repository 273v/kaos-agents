"""Tests for the Expand planning primitive — step parsing and validation."""

from __future__ import annotations

from kaos_agents.planning.expand import _parse_steps, expand_from_steps
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
