"""Live integration tests for Compose — real LLM + real/mock tool execution.

Tests the full pipeline: Expand generates a plan, Compose executes it.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_client.types import ToolDefinition
from kaos_llm_core.programs.tool import Tool

from kaos_agents.planning.compose import compose
from kaos_agents.planning.expand import expand
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.types.plan import PlanBudget, StopReason
from tests.integration._models import respond_model

MODEL = respond_model()


def _make_mock_tool(name: str, response: str) -> Tool:
    """Create a mock tool that returns a canned response."""

    async def executor(**kwargs: Any) -> str:
        return response

    td = ToolDefinition(
        name=name,
        description=f"Mock {name}",
        parameters={"type": "object", "properties": {}},
    )
    return Tool(definition=td, executor=executor)


MOCK_TOOLS = {
    "kaos-source-ecfr-search": _make_mock_tool(
        "kaos-source-ecfr-search",
        "Found: 40 CFR Part 60 Subpart OOOO - Standards of Performance for "
        "Crude Oil and Natural Gas Facilities. Effective date: October 15, 2012. "
        "Requires VOC emission reduction from storage vessels, pneumatic controllers, "
        "and compressors at oil and gas production sites.",
    ),
    "kaos-source-edgar-search": _make_mock_tool(
        "kaos-source-edgar-search",
        "Found 3 Tesla 10-K filings: "
        "0001318605-24-000006 (FY2023), "
        "0001318605-23-000004 (FY2022), "
        "0001318605-22-000005 (FY2021).",
    ),
    "kaos-source-fr-search": _make_mock_tool(
        "kaos-source-fr-search",
        "Found 2 Federal Register notices: "
        "EPA-HQ-OAR-2021-0317 (Proposed rule for Clean Air Act Section 111), "
        "EPA-HQ-OAR-2020-0044 (Final rule for Greenhouse Gas Reporting).",
    ),
    "kaos-llm-core-call": _make_mock_tool(
        "kaos-llm-core-call",
        "Analysis: Based on the regulatory text from 40 CFR Part 60 and Tesla's "
        "SEC disclosures, the Fremont facility appears to operate under applicable "
        "California air quality permits. No EPA enforcement actions were found.",
    ),
}

TOOL_DESCRIPTIONS = {
    "kaos-source-ecfr-search": "Search the eCFR for regulatory text.",
    "kaos-source-edgar-search": "Search SEC EDGAR for company filings.",
    "kaos-source-fr-search": "Search the Federal Register for rules and notices.",
    "kaos-llm-core-call": "Make a structured LLM call for analysis or synthesis.",
}


@pytest.mark.live
class TestComposeLiveExpand:
    """Expand with real LLM → Compose with mock tools."""

    async def test_expand_and_compose_simple_goal(self):
        """Simple goal: expand generates plan, compose executes it."""
        steps = await expand(
            "Find Clean Air Act emission standards in the eCFR",
            available_tools=TOOL_DESCRIPTIONS,
            model=MODEL,
        )
        assert len(steps) >= 1

        # Build graph
        pg = PlanGraph(name="simple-test")
        for s in steps:
            pg.add_step(s)

        # Execute — mock tools with expected outputs trigger tentative evaluation
        # (no semantic verification), so Route may decide REPLAN for low-confidence steps.
        # Both SUCCESS and FAILURE are acceptable outcomes.
        result = await compose(pg, tools=MOCK_TOOLS, parallel=True)

        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )
        assert result.steps_executed >= 1
        assert result.wall_clock_ms > 0

    async def test_expand_and_compose_multi_source(self):
        """Multi-source goal: plan should use multiple tools."""
        steps = await expand(
            "Research whether Tesla complies with EPA emission standards. "
            "Check the regulations, Tesla's SEC filings, and the Federal Register.",
            available_tools=TOOL_DESCRIPTIONS,
            max_steps=6,
            model=MODEL,
        )
        assert len(steps) >= 2

        pg = PlanGraph(name="multi-source")
        for s in steps:
            pg.add_step(s)

        # Verify graph structure
        levels = pg.get_execution_levels()
        assert len(levels) >= 1  # At least one level

        result = await compose(pg, tools=MOCK_TOOLS, parallel=True)

        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.FAILURE,
            StopReason.NEEDS_REPLAN,
        )
        assert result.steps_executed >= 1

    async def test_budget_limits_live_plan(self):
        """Budget should stop execution even with a valid plan."""
        steps = await expand(
            "Do a comprehensive analysis of Tesla's environmental compliance",
            available_tools=TOOL_DESCRIPTIONS,
            max_steps=5,
            model=MODEL,
        )

        pg = PlanGraph(name="budget-test")
        for s in steps:
            pg.add_step(s)

        # Budget allows only 2 steps
        budget = PlanBudget(max_steps=2)
        result = await compose(pg, tools=MOCK_TOOLS, budget=budget, parallel=False)

        assert result.steps_executed <= 2
        # Should either succeed in 2 steps or hit budget
        assert result.stop_reason in (
            StopReason.SUCCESS,
            StopReason.MAX_STEPS,
            StopReason.NEEDS_REPLAN,
            StopReason.FAILURE,
        )

    async def test_traces_from_live_compose(self):
        """Verify observability traces are collected."""
        steps = await expand(
            "Search eCFR for emission standards",
            available_tools=TOOL_DESCRIPTIONS,
            max_steps=3,
            model=MODEL,
        )

        pg = PlanGraph(name="trace-test")
        for s in steps:
            pg.add_step(s)

        result = await compose(pg, tools=MOCK_TOOLS, parallel=False)

        # Should have act + route traces
        assert len(result.traces) >= 2
        act_traces = [t for t in result.traces if t.primitive == "act"]
        route_traces = [t for t in result.traces if t.primitive == "route"]
        assert len(act_traces) >= 1
        assert len(route_traces) >= 1

        # Act traces should have timing
        for t in act_traces:
            assert t.wall_clock_ms >= 0
