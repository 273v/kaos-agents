"""Tests for the KaosTool → llm-core Tool bridge."""

from __future__ import annotations

from typing import Any

from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult

from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool


class _EchoTool(KaosTool):
    """A minimal KaosTool for testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-echo-tool",
            description="Echoes back the input text.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        text = inputs.get("text", "")
        return ToolResult.create_text(f"echo: {text}")


class _FailTool(KaosTool):
    """A tool that always fails."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-fail-tool",
            description="Always fails.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        return ToolResult.create_error("Something went wrong. Try again later.")


class TestKaosToolToLlmTool:
    async def test_wraps_tool_correctly(self):
        echo = _EchoTool()
        tool = kaos_tool_to_llm_tool(echo)

        assert tool.name == "test-echo-tool"
        assert tool.description == "Echoes back the input text."

    async def test_executor_calls_through(self):
        echo = _EchoTool()
        tool = kaos_tool_to_llm_tool(echo)

        result = await tool.invoke({"text": "hello world"})
        assert "echo: hello world" in result

    async def test_error_tool_raises_tool_reported_error(self):
        """``isError=True`` → raises ``ToolReportedError`` (closes P0-1 #437).

        ReAct's ``_invoke_one`` records non-exception returns as
        ``is_error=False`` by design (audit finding #3). Returning a
        JSON envelope on isError=True silently lost the failure flag —
        every downstream consumer (kaos-agents memory, audit CLI, UI
        chips, critics) saw "done ✓" against a body that said
        ``error: True``. The fix raises a dedicated ``ToolReportedError``
        so ReAct propagates ``is_error=True`` AND preserves the tool's
        own remediation hint as the payload.
        """
        import pytest
        from kaos_llm_core.errors import ToolReportedError

        fail = _FailTool()
        tool = kaos_tool_to_llm_tool(fail)

        with pytest.raises(ToolReportedError) as exc_info:
            await tool.invoke({})
        assert exc_info.value.payload["error"] is True
        assert "Something went wrong" in exc_info.value.payload["message"]
