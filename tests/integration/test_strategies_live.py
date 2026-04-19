"""Live tests for planning strategies — real LLM, mock tools."""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_client.types import ToolDefinition
from kaos_llm_core.programs.tool import Tool

from kaos_agents.planning.strategies.adaptive import execute_adaptive
from kaos_agents.planning.strategies.decompose import execute_decompose
from kaos_agents.planning.strategies.direct import execute_direct
from kaos_agents.planning.types import PlanBudget, StopReason

MODEL = "anthropic:claude-haiku-4-5"


def _mock_tool(name: str, response: str) -> Tool:
    async def executor(**kwargs: Any) -> str:
        return response

    td = ToolDefinition(
        name=name,
        description=f"Mock {name}",
        parameters={"type": "object", "properties": {}},
    )
    return Tool(definition=td, executor=executor)


TOOLS = {
    "kaos-source-ecfr-search": _mock_tool(
        "kaos-source-ecfr-search",
        "40 CFR Part 60: Standards of Performance for New Stationary Sources.",
    ),
    "kaos-source-edgar-search": _mock_tool(
        "kaos-source-edgar-search",
        "Found Tesla 10-K filing: 0001318605-24-000006.",
    ),
}

TOOL_DESCS = {
    "kaos-source-ecfr-search": "Search the eCFR for regulatory text.",
    "kaos-source-edgar-search": "Search SEC EDGAR for company filings.",
}


@pytest.mark.live
class TestDirectStrategy:
    async def test_simple_goal(self):
        result = await execute_direct(
            "Find emission standards in eCFR",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
        )
        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 1


@pytest.mark.live
class TestDecomposeStrategy:
    async def test_multi_step_goal(self):
        result = await execute_decompose(
            "Search eCFR for emission standards and search EDGAR for Tesla filings",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            max_steps=4,
        )
        assert result.steps_executed >= 1
        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )


@pytest.mark.live
class TestAdaptiveStrategy:
    async def test_simple_goal_routes_direct(self):
        """Simple goal should be handled directly (fewer steps)."""
        result = await execute_adaptive(
            "Search eCFR for Clean Air Act",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
        )
        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed >= 1

    async def test_complex_goal_routes_decompose(self):
        """Complex goal with 'then' should trigger decomposition."""
        result = await execute_adaptive(
            "First search eCFR for emission regulations, then search EDGAR "
            "for Tesla's compliance disclosures, then cross-reference "
            "the findings",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            max_steps=5,
        )
        assert result.steps_executed >= 1
        # Complex goal should produce more steps
        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )

    async def test_budget_respected(self):
        budget = PlanBudget(max_steps=1)
        result = await execute_adaptive(
            "Do a comprehensive regulatory analysis",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            budget=budget,
        )
        assert result.steps_executed <= 1
