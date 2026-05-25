"""Live integration tests for ChatAgent and PlanExecuteAgent.

Tests the actual LLM-backed paths through the agent patterns,
including tool dispatch via ReAct and plan dispatch via adaptive strategy.
Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.metadata import (
    ToolAnnotations,
    ToolCapability,
    ToolCategory,
    ToolMetadata,
)
from kaos_core.types.results import ToolResult
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.plan_execute import PlanExecuteAgent
from tests.integration._models import respond_model

MODEL = respond_model()


# ---------------------------------------------------------------------------
# Mock tools that register on KaosRuntime
# ---------------------------------------------------------------------------


class CalculatorTool(KaosTool):
    """A simple calculator tool for testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-agent-calculator",
            description="Performs basic arithmetic. Input: expression (str). Returns the result.",
            category=ToolCategory.DATA,
            capability=ToolCapability.ANALYZE,
            module_name="test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        expr = inputs.get("expression", "")
        try:
            # Safe eval for simple arithmetic
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult.create_text(f"Result: {result}")
        except Exception as exc:
            return ToolResult.create_error(
                f"Calculator error: {exc}. Provide a valid arithmetic expression like '2 + 3'."
            )


class GreetTool(KaosTool):
    """A tool that greets by name."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-agent-greet",
            description="Greets a person by name. Input: name (str). Returns a greeting.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.TRANSFORM,
            module_name="test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        name = inputs.get("name", "World")
        return ToolResult.create_text(f"Hello, {name}! Welcome to KAOS.")


def _make_runtime() -> KaosRuntime:
    """Create a runtime with test tools."""
    runtime = KaosRuntime()
    runtime.tools.register_tool(CalculatorTool())
    runtime.tools.register_tool(GreetTool())
    return runtime


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


# ---------------------------------------------------------------------------
# ChatAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestChatAgentLive:
    """Test ChatAgent with real LLM and mock tools."""

    async def test_simple_respond(self):
        """Simple greeting should be handled without tools."""
        vfs = _memory_vfs()
        agent = ChatAgent(vfs, model=MODEL)
        response = await agent.turn("Hello!", session_id="chat-simple")

        assert response.text
        assert len(response.text) > 3
        assert response.turn_number == 1

    async def test_tool_use_with_runtime(self):
        """Tool-use request with a runtime should dispatch to ReAct."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        agent = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=["test-agent-calculator", "test-agent-greet"],
        )

        response = await agent.turn(
            "Use the calculator to compute 42 * 17",
            session_id="chat-tool",
        )

        assert response.text
        assert response.turn_number == 1
        # The response should contain the calculation result
        # (either in text or via tool calls)

    async def test_multi_turn_memory(self):
        """Multiple turns should accumulate in memory."""
        vfs = _memory_vfs()

        agent1 = ChatAgent(vfs, model=MODEL)
        r1 = await agent1.turn("My name is Alice.", session_id="chat-multi")
        assert r1.turn_number == 1

        agent2 = ChatAgent(vfs, model=MODEL)
        r2 = await agent2.turn("What is my name?", session_id="chat-multi")
        assert r2.turn_number == 2
        # With memory, the agent should remember "Alice"
        assert "alice" in r2.text.lower() or "Alice" in r2.text


# ---------------------------------------------------------------------------
# PlanExecuteAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestPlanExecuteAgentLive:
    """Test PlanExecuteAgent with real LLM and mock tools."""

    async def test_simple_respond_passthrough(self):
        """Simple messages should pass through to base respond handler."""
        vfs = _memory_vfs()
        agent = PlanExecuteAgent(vfs, model=MODEL)
        response = await agent.turn("Hi there!", session_id="plan-simple")

        assert response.text
        assert response.turn_number == 1

    async def test_plan_intent_with_tools(self):
        """Complex multi-step request should trigger planning."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        agent = PlanExecuteAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=["test-agent-calculator", "test-agent-greet"],
        )

        response = await agent.turn(
            "First greet Alice, then calculate 100 / 4",
            session_id="plan-multi-step",
        )

        assert response.text
        assert response.turn_number == 1
        # Should have attempted some tool calls or planning
