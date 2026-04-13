"""Live tests for rolling horizon strategy -- real LLM, mock tools."""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_client.types import ToolDefinition
from kaos_llm_core.programs.tool import Tool

from kaos_agents.planning.strategies.rolling import execute_rolling
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
class TestRollingStrategyLive:
    async def test_rolling_with_mock_tools(self):
        """Rolling horizon generates plan, executes, and can replan."""
        result = await execute_rolling(
            "Search eCFR for emission standards and then search EDGAR for Tesla filings",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            horizon=2,
            max_horizons=3,
        )
        assert result.steps_executed >= 1
        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )

    async def test_rolling_budget_respected(self):
        """Rolling strategy respects step budget."""
        budget = PlanBudget(max_steps=1)
        result = await execute_rolling(
            "Do comprehensive regulatory analysis across multiple sources",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            budget=budget,
            horizon=3,
            max_horizons=4,
        )
        assert result.steps_executed <= 1

    async def test_rolling_single_horizon(self):
        """Rolling with max_horizons=1 should behave like a single-pass plan."""
        result = await execute_rolling(
            "Search eCFR for Clean Air Act regulations",
            tools=TOOLS,
            tool_descriptions=TOOL_DESCS,
            model=MODEL,
            horizon=3,
            max_horizons=1,
        )
        assert result.steps_executed >= 1
        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )
