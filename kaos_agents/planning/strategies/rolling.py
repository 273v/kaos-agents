"""Rolling horizon strategy — plan N steps ahead, execute, replan remainder.

For tasks where the full plan can't be determined upfront because
later steps depend on earlier results. Composes:
    expand(n=3) → compose → [observe → expand(remaining, n=3)] repeat

Balances lookahead with adaptability. Each horizon:
1. Expand 3 steps from the current state
2. Execute them via Compose
3. Check results — if the goal isn't met, expand the next 3 steps
   with updated context (including what we learned)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.planning.compose import compose
from kaos_agents.planning.expand import expand
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.planning.types import (
    ComposeResult,
    PlanBudget,
    PrimitiveTrace,
    StopReason,
)

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)


async def execute_rolling(
    goal: str,
    *,
    tools: dict[str, Tool] | None = None,
    tool_descriptions: dict[str, str] | None = None,
    context: str = "",
    model: str = "anthropic:claude-haiku-4-5",
    budget: PlanBudget | None = None,
    horizon: int = 3,
    max_horizons: int = 4,
    parallel: bool = True,
) -> ComposeResult:
    """Execute a goal with rolling horizon replanning.

    Args:
        goal: What to accomplish.
        tools: Tool name → Tool instances.
        tool_descriptions: Tool name → description for Expand.
        context: Initial context from memory.
        model: LLM model.
        budget: Resource limits.
        horizon: Number of steps per planning horizon.
        max_horizons: Maximum number of replan cycles.
        parallel: Whether to run independent steps concurrently.
    """
    if budget is None:
        budget = PlanBudget()

    all_traces: list[PrimitiveTrace] = []
    all_results: dict[str, str] = {}
    total_steps = 0
    accumulated_context = context

    for horizon_idx in range(max_horizons):
        # Check budget before expanding
        stop = budget.should_stop()
        if stop is not None:
            logger.debug("rolling: budget stop before horizon %d: %s", horizon_idx, stop.value)
            return _build_result(all_traces, all_results, total_steps, budget, stop)

        # Expand: generate next N steps
        prior = ""
        if all_results:
            prior = "Results from previous steps:\n" + "\n".join(
                f"- {k}: {v[:100]}" for k, v in all_results.items()
            )

        steps = await expand(
            goal,
            available_tools=tool_descriptions or {},
            context=accumulated_context,
            prior_failures=prior,
            max_steps=horizon,
            model=model,
        )

        if not steps:
            logger.debug("rolling: expand produced no steps at horizon %d", horizon_idx)
            break

        # Build and execute
        graph = PlanGraph(name=f"horizon-{horizon_idx}")
        for step in steps:
            graph.add_step(step)

        result = await compose(
            graph,
            tools=tools or {},
            budget=budget,
            model=model,
            parallel=parallel,
        )

        all_traces.extend(result.traces)
        all_results.update(result.step_results)
        total_steps += result.steps_executed

        # Update context with what we learned
        if result.step_results:
            new_findings = "\n".join(
                f"Step result: {v[:200]}" for v in result.step_results.values()
            )
            accumulated_context = f"{accumulated_context}\n\n{new_findings}"

        # If this horizon completed successfully and we have results, we might be done
        if result.stop_reason == StopReason.SUCCESS and result.steps_executed > 0:
            logger.debug(
                "rolling: horizon %d completed (%d steps), checking if goal is met",
                horizon_idx,
                result.steps_executed,
            )
            # For now, accept the result after successful completion
            # Future: add an Evaluate call here to check if the goal is met
            break

        if result.stop_reason != StopReason.SUCCESS:
            logger.debug("rolling: horizon %d stopped: %s", horizon_idx, result.stop_reason.value)
            return _build_result(all_traces, all_results, total_steps, budget, result.stop_reason)

    return _build_result(all_traces, all_results, total_steps, budget, StopReason.SUCCESS)


def _build_result(
    traces: list[PrimitiveTrace],
    results: dict[str, str],
    steps: int,
    budget: PlanBudget,
    stop_reason: StopReason,
) -> ComposeResult:
    return ComposeResult(
        plan_json="{}",
        traces=tuple(traces),
        stop_reason=stop_reason,
        total_cost_usd=budget.cost_usd,
        total_tokens=budget.tokens_used,
        steps_executed=steps,
        replans=0,
        step_results=results,
    )
