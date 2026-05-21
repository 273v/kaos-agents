"""Tests for B0.2 — per-invocation tool timeout in the bridge executor.

Pre-B0.2 (broad-reliability roadmap §B0.2), the chat-pattern ReAct
dispatch had no per-tool deadline. A slow gov-source crawler (or any
tool blocked on a remote that never sent FIN) could pin an entire
turn indefinitely — the agent's wall-clock budget would expire while
the inner ``await kaos_tool.execute(...)`` was still running. Planning's
``_act_tool`` already had ``asyncio.timeout(...)`` wrapping; the chat
bridge did not, so the chat ReAct path was effectively timeout-free.

Post-B0.2, every executor produced by
:func:`kaos_tool_to_llm_tool` wraps its ``execute`` call in
``async with asyncio.timeout(effective_timeout)``. On ``TimeoutError``
the executor raises :class:`kaos_llm_core.errors.ToolReportedError`
with a structured payload — ReAct's ``_invoke_one`` records the failure
as ``is_error=True`` (preserving the inventory P0-1 #437 contract).
Without a caller override, the timeout comes from
``KaosAgentSettings().tool_timeout_seconds`` (default 120s) so chat
inherits the same bound as planning.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult
from kaos_llm_core.errors import ToolReportedError

from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool


class _SlowTool(KaosTool):
    """A tool that sleeps for ``delay_seconds`` before returning."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-slow-tool",
            description="Sleeps then returns.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult.create_text("done")


class _FastTool(KaosTool):
    """A tool that returns immediately (baseline)."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-fast-tool",
            description="Returns immediately.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        return ToolResult.create_text("immediate")


class TestBridgeTimeout:
    """Per-invocation timeout enforces the configured deadline."""

    @pytest.mark.asyncio
    async def test_fast_tool_completes_under_timeout(self) -> None:
        """A tool that returns well under the deadline succeeds."""
        tool = kaos_tool_to_llm_tool(_FastTool(), tool_timeout_seconds=5.0)
        result = await tool.invoke({})
        assert "immediate" in result

    @pytest.mark.asyncio
    async def test_slow_tool_raises_tool_reported_error_on_timeout(self) -> None:
        """A tool that exceeds the timeout raises ToolReportedError.

        Same envelope as the isError=True path so ReAct records
        ``is_error=True`` against the observation. The agent then sees
        the failure in memory and can re-plan (swap tools, narrow
        scope, etc.).
        """
        slow = _SlowTool(delay_seconds=2.0)
        tool = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=0.1)

        with pytest.raises(ToolReportedError) as excinfo:
            await tool.invoke({})

        payload = excinfo.value.payload
        assert payload["error"] is True
        assert "timed out" in payload["message"].lower()
        assert "test-slow-tool" in payload["message"]
        # The structured payload also carries the threshold for telemetry.
        assert payload["timeout_seconds"] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_timeout_error_chains_original_cause(self) -> None:
        """``__cause__`` is the underlying ``TimeoutError``.

        Loggers and audit trails that walk the exception chain can
        distinguish a timeout-triggered failure from a tool-reported
        failure by inspecting ``__cause__``.
        """
        slow = _SlowTool(delay_seconds=2.0)
        tool = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=0.05)

        with pytest.raises(ToolReportedError) as excinfo:
            await tool.invoke({})

        assert isinstance(excinfo.value.__cause__, TimeoutError)

    @pytest.mark.asyncio
    async def test_default_inherits_from_settings(self) -> None:
        """``tool_timeout_seconds=None`` reads ``KaosAgentSettings``.

        We don't want to actually wait 120s in a unit test, so we
        verify behaviour indirectly: a fast tool completes successfully
        under the default. The slow-tool exhaustion path is covered by
        the explicit-timeout test above.
        """
        tool = kaos_tool_to_llm_tool(_FastTool(), tool_timeout_seconds=None)
        result = await tool.invoke({})
        assert "immediate" in result

    @pytest.mark.asyncio
    async def test_timeout_does_not_swallow_pending_work(self) -> None:
        """The asyncio.timeout context cancels the pending sleep task —
        no leaked task survives the executor.

        Verified by observing that a second invocation of the same tool
        with a longer deadline still works (the executor is reusable).
        """
        slow = _SlowTool(delay_seconds=0.5)
        # First call: times out.
        tight = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=0.05)
        with pytest.raises(ToolReportedError):
            await tight.invoke({})

        # Second call: same underlying tool, generous deadline, succeeds.
        loose = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=5.0)
        result = await loose.invoke({})
        assert "done" in result


class TestBridgeRuntimeToolsTimeoutPropagation:
    """``bridge_runtime_tools`` threads ``tool_timeout_seconds`` into
    every bridged executor."""

    @pytest.mark.asyncio
    async def test_bridge_runtime_tools_threads_timeout(self) -> None:
        """When ``bridge_runtime_tools(tool_timeout_seconds=...)`` is
        set, every executor it produces uses that deadline.

        Verifies the plumbing end-to-end: an explicit short timeout
        causes the slow tool's bridged executor to fail with
        ToolReportedError.
        """
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.actions.tool_bridge import bridge_runtime_tools

        runtime = KaosRuntime.test_mode()
        runtime.tools.register_tool(_SlowTool(delay_seconds=2.0))

        bridged = bridge_runtime_tools(runtime, tool_timeout_seconds=0.1)
        assert len(bridged) == 1

        with pytest.raises(ToolReportedError) as excinfo:
            await bridged[0].invoke({})
        assert "timed out" in excinfo.value.payload["message"].lower()
