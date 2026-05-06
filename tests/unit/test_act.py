"""Tests for the Act planning primitive."""

from __future__ import annotations

from typing import Any

from kaos_agents.planning.act import ActResult, act, make_trace
from kaos_agents.types.plan import StepType


class TestActTool:
    async def test_tool_success(self):
        from kaos_llm_client.types import ToolDefinition
        from kaos_llm_core.programs.tool import Tool

        async def echo(**kwargs: Any) -> str:
            return f"echo: {kwargs}"

        td = ToolDefinition(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}},
        )
        tool = Tool(definition=td, executor=echo)

        result = await act(StepType.TOOL, tool=tool, tool_args={"text": "hello"})
        assert not result.is_error
        assert "echo:" in result.output
        assert result.wall_clock_ms > 0

    async def test_tool_none_returns_error(self):
        result = await act(StepType.TOOL, tool=None)
        assert result.is_error
        assert "No tool provided" in result.output

    async def test_tool_exception_caught(self):
        from kaos_llm_client.types import ToolDefinition
        from kaos_llm_core.programs.tool import Tool

        async def fail(**kwargs: Any) -> str:
            raise RuntimeError("tool crashed")

        td = ToolDefinition(
            name="fail",
            description="fail",
            parameters={"type": "object", "properties": {}},
        )
        tool = Tool(definition=td, executor=fail)

        result = await act(StepType.TOOL, tool=tool)
        assert result.is_error
        assert "tool crashed" in result.output


class TestActSubplan:
    async def test_subplan_rejected(self):
        result = await act(StepType.SUBPLAN)
        assert result.is_error
        assert "Compose" in result.output


class TestActResult:
    def test_defaults(self):
        r = ActResult(output="hello")
        assert not r.is_error
        assert r.wall_clock_ms == 0.0
        assert r.token_count == 0
        assert r.cost_usd == 0.0


class TestMakeTrace:
    def test_trace_from_success(self):
        result = ActResult(output="ok", wall_clock_ms=15.5, token_count=100, cost_usd=0.001)
        trace = make_trace("s1", result)
        assert trace.primitive == "act"
        assert trace.step_id == "s1"
        assert trace.wall_clock_ms == 15.5
        assert trace.success

    def test_trace_from_error(self):
        result = ActResult(output="ERROR: fail", is_error=True, wall_clock_ms=5.0)
        trace = make_trace("s2", result)
        assert not trace.success
