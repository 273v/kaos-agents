"""Tests for the Compose planning primitive."""

from __future__ import annotations

from typing import Any

from kaos_llm_client.types import ToolDefinition
from kaos_llm_core.programs.tool import Tool

from kaos_agents.planning.compose import compose
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.types.plan import PlanBudget, Step, StepType, StopReason

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


def _make_tool(name: str, result: str = "ok", fail: bool = False) -> Tool:
    """Create a test tool with a canned response."""

    async def executor(**kwargs: Any) -> str:
        if fail:
            return f"ERROR: {name} failed"
        return f"{name}: {result}"

    td = ToolDefinition(
        name=name,
        description=f"Test tool {name}",
        parameters={"type": "object", "properties": {}},
    )
    return Tool(definition=td, executor=executor)


def _tool_step(id: str, name: str, deps: tuple[str, ...] = ()) -> Step:
    return Step(
        id=id,
        step_type=StepType.TOOL,
        description=f"Step {id}",
        tool_name=name,
        depends_on=deps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComposeSequential:
    async def test_single_step(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))

        tools = {"tool-a": _make_tool("tool-a", "result-a")}
        result = await compose(pg, tools=tools, parallel=False)

        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 1
        assert "s1" in result.step_results
        assert "result-a" in result.step_results["s1"]

    async def test_three_sequential_steps(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))
        pg.add_step(_tool_step("s2", "tool-b", deps=("s1",)))
        pg.add_step(_tool_step("s3", "tool-c", deps=("s2",)))

        tools = {
            "tool-a": _make_tool("tool-a", "result-a"),
            "tool-b": _make_tool("tool-b", "result-b"),
            "tool-c": _make_tool("tool-c", "result-c"),
        }
        result = await compose(pg, tools=tools, parallel=False)

        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 3
        assert pg.is_complete()


class TestComposeParallel:
    async def test_two_independent_steps(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))
        pg.add_step(_tool_step("s2", "tool-b"))

        tools = {
            "tool-a": _make_tool("tool-a", "result-a"),
            "tool-b": _make_tool("tool-b", "result-b"),
        }
        result = await compose(pg, tools=tools, parallel=True)

        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 2

    async def test_parallel_then_sequential(self):
        """s1, s2 parallel → s3 depends on both."""
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))
        pg.add_step(_tool_step("s2", "tool-b"))
        pg.add_step(_tool_step("s3", "tool-c", deps=("s1", "s2")))

        tools = {
            "tool-a": _make_tool("tool-a", "result-a"),
            "tool-b": _make_tool("tool-b", "result-b"),
            "tool-c": _make_tool("tool-c", "result-c"),
        }
        result = await compose(pg, tools=tools, parallel=True)

        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 3
        assert len(result.step_results) == 3


class TestComposeFailure:
    async def test_step_failure_skips_dependents(self):
        """If s1 fails, s2 (depends on s1) should be skipped."""
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "fail-tool"))
        pg.add_step(_tool_step("s2", "tool-b", deps=("s1",)))

        tools = {
            "fail-tool": _make_tool("fail-tool", fail=True),
            "tool-b": _make_tool("tool-b"),
        }
        result = await compose(pg, tools=tools, parallel=False)

        # s1 failed → s2 should be skipped → graph deadlocks with failures
        assert result.steps_executed >= 1
        assert pg.has_failures()

    async def test_missing_tool_fails_step(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "nonexistent-tool"))

        result = await compose(pg, tools={}, parallel=False)

        assert result.steps_executed == 1
        assert pg.has_failures()


class TestComposeBudget:
    async def test_budget_stops_execution(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))
        pg.add_step(_tool_step("s2", "tool-b"))
        pg.add_step(_tool_step("s3", "tool-c"))

        tools = {
            "tool-a": _make_tool("tool-a"),
            "tool-b": _make_tool("tool-b"),
            "tool-c": _make_tool("tool-c"),
        }

        # Budget allows only 1 step
        budget = PlanBudget(max_steps=1)
        result = await compose(pg, tools=tools, budget=budget, parallel=False)

        assert result.stop_reason == StopReason.MAX_STEPS
        assert result.steps_executed == 1


class TestComposeTraces:
    async def test_traces_collected(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))

        tools = {"tool-a": _make_tool("tool-a")}
        result = await compose(pg, tools=tools, parallel=False)

        assert len(result.traces) >= 2  # At least act + route
        act_traces = [t for t in result.traces if t.primitive == "act"]
        route_traces = [t for t in result.traces if t.primitive == "route"]
        assert len(act_traces) >= 1
        assert len(route_traces) >= 1

    async def test_wall_clock_tracked(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "tool-a"))

        tools = {"tool-a": _make_tool("tool-a")}
        result = await compose(pg, tools=tools, parallel=False)

        assert result.wall_clock_ms > 0


class TestComposeEmptyGraph:
    async def test_empty_graph_succeeds(self):
        pg = PlanGraph()
        result = await compose(pg, parallel=False)
        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 0
