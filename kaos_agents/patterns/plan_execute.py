"""PlanExecuteAgent — extends ChatAgent with planning strategies.

Adds PLAN intent handling via the adaptive strategy (ADaPT):
- Simple goals → direct execution
- Complex goals → hierarchical decomposition with parallel execution
- Failures → automatic fallback with Reflexion feedback

This is the full-featured agent pattern that uses all 7 planning primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.memory.types import MemoryType
from kaos_agents.models import ToolCallRecord
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.planning.recall import recall
from kaos_agents.planning.types import StopReason

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem
    from kaos_agents.settings import KaosAgentSettings

logger = get_logger(__name__)


class PlanExecuteAgent(ChatAgent):
    """Agent with full planning: adaptive strategy for PLAN intent.

    Extends ChatAgent (which handles RESPOND, CLARIFY, TOOL_USE) with
    plan-propose → execute → observe for complex multi-step goals.

    Usage:
        agent = PlanExecuteAgent(
            vfs=runtime.vfs,
            runtime=runtime,
            context=context,
            model="anthropic:claude-sonnet-4-6",
        )
        response = await agent.turn(
            "Research whether Tesla complies with EPA emission standards",
            session_id="abc",
        )
    """

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        model: str | None = None,
        tool_filter: list[str] | None = None,
        max_tools: int | None = None,
        max_react_iterations: int | None = None,
        max_plan_steps: int | None = None,
        settings: KaosAgentSettings | None = None,
    ) -> None:
        super().__init__(
            vfs,
            runtime=runtime,
            context=context,
            model=model,
            tool_filter=tool_filter,
            max_tools=max_tools,
            max_react_iterations=max_react_iterations,
            settings=settings,
        )
        self._max_plan_steps = max_plan_steps or self._settings.plan_max_steps

    async def _handle_plan(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle multi-step plan via adaptive strategy."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools
        from kaos_agents.planning.result_check import is_error_result
        from kaos_agents.planning.strategies.adaptive import execute_adaptive
        from kaos_agents.planning.types import PlanBudget

        # Build tool bridge
        tools_dict: dict[str, Any] = {}
        tool_descriptions: dict[str, str] = {}
        if self._runtime:
            llm_tools = bridge_runtime_tools(
                self._runtime,
                self._context,
                filter_names=self._tool_filter,
                max_tools=self._max_tools,
            )
            for t in llm_tools:
                tools_dict[t.name] = t
                tool_descriptions[t.name] = t.description or ""

        # Recall context for planning (includes prior failures)
        ctx = recall(
            memory,
            [MemoryType.MESSAGES, MemoryType.FINDINGS, MemoryType.ACTIONS, MemoryType.REFLECTION],
            budget_tokens=self._settings.default_context_budget_tokens // 2,
            priority_order=[MemoryType.MESSAGES, MemoryType.FINDINGS],
        )

        # Check for prior failures in reflection
        prior_failures = ""
        reflections = memory.get_recent(MemoryType.REFLECTION, 3)
        if reflections:
            prior_failures = "\n".join(r.content for r in reflections)

        # Execute via adaptive strategy
        budget = PlanBudget(
            max_steps=self._settings.plan_max_steps,
            max_replans=self._settings.plan_max_replans,
            max_cost_usd=self._settings.plan_max_cost_usd,
            max_wall_clock_seconds=self._settings.plan_max_wall_clock_seconds,
        )

        result = await execute_adaptive(
            message,
            tools=tools_dict,
            tool_descriptions=tool_descriptions,
            context=ctx.text,
            prior_failures=prior_failures,
            model=self._model,
            budget=budget,
            complexity_threshold=self._settings.complexity_threshold,
            simple_word_threshold=self._settings.simple_goal_word_threshold,
            max_steps=self._max_plan_steps,
            confidence_threshold=self._settings.confidence_threshold,
            deepen_threshold=self._settings.deepen_threshold,
        )

        # Build tool call records with actual tool names from the plan graph
        tool_calls: list[ToolCallRecord] = []
        # Parse plan graph to recover tool names per step
        step_tool_names: dict[str, str] = {}
        if result.plan_json and result.plan_json != "{}":
            try:
                from kaos_agents.planning.graph import PlanGraph

                pg = PlanGraph.from_json(result.plan_json)
                for sid in pg.step_ids():
                    props = pg.get_step(sid)
                    if props and props.get("tool_name"):
                        step_tool_names[sid] = props["tool_name"]
            except Exception as exc:
                logger.warning(
                    "plan_execute: failed to recover tool names from plan graph: %s",
                    exc,
                )

        for step_id, step_result in result.step_results.items():
            actual_tool = step_tool_names.get(step_id, step_id)
            tool_calls.append(
                ToolCallRecord.from_dict_args(
                    tool_name=actual_tool,
                    arguments={"step_id": step_id},
                    result_summary=str(step_result)[:200],
                    is_error=is_error_result(str(step_result)),
                )
            )

        # Synthesize response from results
        if result.stop_reason == StopReason.SUCCESS and result.step_results:
            response = _synthesize_results(result.step_results)
        elif result.stop_reason == StopReason.SUCCESS:
            response = "Plan completed but no results were produced."
        else:
            response = (
                f"Plan execution stopped: {result.stop_reason.value}. "
                f"Completed {result.steps_executed} steps."
            )
            # Record failure for Reflexion
            memory.add(
                MemoryType.REFLECTION,
                f"Plan failed ({result.stop_reason.value}) for goal: {message[:100]}. "
                f"Steps completed: {result.steps_executed}.",
            )

        logger.debug(
            "plan_execute: %d steps, %d tools, stop=%s, cost=$%.4f",
            result.steps_executed,
            len(tool_calls),
            result.stop_reason.value,
            result.total_cost_usd,
        )

        return response, tool_calls


def _synthesize_results(results: dict[str, Any]) -> str:
    """Combine step results into a response summary."""
    from kaos_agents.planning.result_check import is_error_result

    parts = []
    for step_id, result in results.items():
        result_str = str(result)
        if result_str and not is_error_result(result_str):
            # Truncate long results
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            parts.append(f"**{step_id}**: {result_str}")

    if parts:
        return "Plan completed with the following results:\n\n" + "\n\n".join(parts)
    return "Plan completed but results were empty or all errored."
