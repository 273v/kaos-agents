"""Regression tests for WS-0.6 — plan tool timeout flows end-to-end.

Pre-fix bug: ``KaosAgentSettings.tool_timeout_seconds`` existed;
``compose()`` accepted it; ``act()`` enforced it. But nothing threaded
it between them: ``compose()`` did not pass it into
``_execute_parallel`` / ``_execute_sequential`` / ``_execute_one``, and
``PlanExecuteAgent._run_plan`` did not forward
``self._settings.tool_timeout_seconds`` into ``execute_adaptive``. Tool
steps could hang indefinitely.

Post-fix: timeout threads through
``PlanExecuteAgent → execute_adaptive → execute_direct/decompose →
compose → _execute_parallel/_execute_sequential → _execute_one → act``.

These tests prove the thread: a plan step that calls a sleep-tool
whose duration exceeds ``tool_timeout_seconds`` receives a timeout
error, not a hanging process.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from kaos_llm_client.types import ToolDefinition
from kaos_llm_core.programs.tool import Tool

from kaos_agents.planning.compose import compose
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.types.plan import PlanBudget, Step, StepType


def _sleep_tool(duration_s: float) -> Tool:
    """A tool that sleeps ``duration_s`` seconds then returns 'done'."""

    async def executor(**kwargs: Any) -> str:
        await asyncio.sleep(duration_s)
        return "done"

    return Tool(
        definition=ToolDefinition(
            name="sleep-tool",
            description=f"Sleep for {duration_s}s.",
            parameters={"type": "object", "properties": {}},
        ),
        executor=executor,
    )


@pytest.mark.unit
class TestComposeToolTimeout:
    @pytest.mark.asyncio
    async def test_fast_tool_respects_timeout(self) -> None:
        """A 0.05s tool run with a 5s timeout — must succeed."""
        graph = PlanGraph(name="fast")
        graph.add_step(
            Step(
                id="s1",
                description="call sleep tool",
                step_type=StepType.TOOL,
                tool_name="sleep-tool",
                input_spec={},
            )
        )
        tools = {"sleep-tool": _sleep_tool(0.05)}

        result = await compose(
            graph,
            tools=tools,
            budget=PlanBudget(),
            tool_timeout_seconds=5.0,
        )
        assert "s1" in result.step_results
        step_result = result.step_results["s1"]
        assert "timed out" not in str(step_result).lower()

    @pytest.mark.asyncio
    async def test_slow_tool_times_out(self) -> None:
        """A 2s tool run with a 0.1s timeout — must raise TimeoutError
        inside act() and surface as an error result."""
        graph = PlanGraph(name="slow")
        graph.add_step(
            Step(
                id="s1",
                description="call sleep tool",
                step_type=StepType.TOOL,
                tool_name="sleep-tool",
                input_spec={},
            )
        )
        tools = {"sleep-tool": _sleep_tool(2.0)}

        import time

        from kaos_agents.types.plan import StopReason

        start = time.monotonic()
        result = await compose(
            graph,
            tools=tools,
            budget=PlanBudget(),
            tool_timeout_seconds=0.1,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.5, (
            f"Tool step took {elapsed:.2f}s — should have timed out at "
            "0.1s. If this fails, tool_timeout_seconds did not thread "
            "from compose() into act()."
        )
        # Timeout routes to a non-SUCCESS stop_reason — the primary WS-0.6
        # invariant is that the call returned quickly. The stop_reason
        # is a secondary signal that the step did not complete cleanly.
        assert result.stop_reason != StopReason.SUCCESS, (
            f"Timeout must not surface as SUCCESS; got {result.stop_reason!r}"
        )


@pytest.mark.unit
class TestPlanExecuteAgentForwardsTimeout:
    """Verify that the full wire from settings.tool_timeout_seconds
    reaches compose()'s kwarg (the plan_execute.py:168 gap)."""

    @pytest.mark.asyncio
    async def test_settings_value_reaches_adaptive(self) -> None:
        """Patch execute_adaptive and assert the PlanExecuteAgent call
        forwards the KaosAgentSettings.tool_timeout_seconds value."""
        from unittest.mock import AsyncMock, patch

        from kaos_core.types.enums import IsolationMode, StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.memory.store import SessionStore
        from kaos_agents.patterns.plan_execute import PlanExecuteAgent
        from kaos_agents.settings import KaosAgentSettings
        from kaos_agents.types import IntentResult, IntentType

        vfs = VirtualFileSystem(
            config=VFSConfig(
                default_backend=StorageBackend.MEMORY,
                isolation_mode=IsolationMode.GLOBAL,
            )
        )
        settings = KaosAgentSettings(tool_timeout_seconds=17.5)
        agent = PlanExecuteAgent(vfs, settings=settings)
        store = SessionStore(vfs)
        memory = await store.load_or_create("test-timeout-wire")

        captured_kwargs: dict[str, Any] = {}

        async def _fake_adaptive(goal: str, **kwargs: Any):
            captured_kwargs.update(kwargs)
            from kaos_agents.types.plan import ComposeResult, StopReason

            return ComposeResult(
                plan_json="{}",
                stop_reason=StopReason.SUCCESS,
                step_results={},
            )

        # execute_adaptive is imported inside PlanExecuteAgent's method,
        # not at module scope, so patch it at its source location.
        with (
            patch(
                "kaos_agents.planning.strategies.adaptive.execute_adaptive",
                side_effect=_fake_adaptive,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=IntentResult(
                    intent=IntentType.PLAN,
                    confidence=1.0,
                    reasoning="test",
                ),
            ),
        ):
            from kaos_agents.events import EventEmitter

            emitter = EventEmitter(session_id="test-timeout-wire", run_id="r1")
            events = []
            async for event in agent._handle_plan_streaming(
                "do something slow", memory, {}, emitter
            ):
                events.append(event)

        assert captured_kwargs.get("tool_timeout_seconds") == 17.5, (
            f"PlanExecuteAgent did not forward settings.tool_timeout_seconds "
            f"into execute_adaptive; got {captured_kwargs.get('tool_timeout_seconds')!r}"
        )
