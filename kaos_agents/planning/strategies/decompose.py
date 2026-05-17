"""Hierarchical decomposition strategy (HTN-style).

For complex goals that need multi-step plans. Composes:
    recall → expand(n=all) → build graph → validate → compose (parallel)

This is the full planning path: the LLM generates a complete plan,
we build a PlanGraph, validate it, and execute via Compose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.planning.compose import compose
from kaos_agents.planning.expand import expand
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.plan import ComposeResult, PlanBudget, StopReason

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)


async def execute_decompose(
    goal: str,
    *,
    tools: dict[str, Tool] | None = None,
    tool_descriptions: dict[str, str] | None = None,
    context: str = "",
    prior_failures: str = "",
    model: str = DEFAULT_MODEL,
    budget: PlanBudget | None = None,
    max_steps: int = 8,
    parallel: bool = True,
    confidence_threshold: float | None = None,
    deepen_threshold: float | None = None,
    tool_timeout_seconds: float = 60.0,
) -> ComposeResult:
    """Decompose a goal into a full plan and execute it.

    Steps:
    1. Expand: LLM generates a complete multi-step plan
    2. Build: steps are assembled into a PlanGraph
    3. Validate: cycle detection, tool name validation
    4. Compose: execute the graph (parallel by default)
    """
    if budget is None:
        budget = PlanBudget()

    # 1. Expand
    steps = await expand(
        goal,
        available_tools=tool_descriptions or {},
        context=context,
        prior_failures=prior_failures,
        max_steps=max_steps,
        model=model,
    )

    if not steps:
        return ComposeResult(
            plan_json="{}",
            stop_reason=StopReason.FAILURE,
            step_results={"error": "Expand produced no steps"},
        )

    # 2. Build graph
    graph = PlanGraph(name="decompose")
    for step in steps:
        graph.add_step(step)

    # 3. Validate
    issues = graph.validate()
    if issues:
        logger.warning("decompose: plan has validation issues: %s", issues)
        # Continue anyway — validation issues are warnings, not blockers

    logger.debug(
        "decompose: %d steps, %d levels, %d ready",
        graph.n_steps,
        len(graph.get_execution_levels()),
        len(graph.get_ready_steps()),
    )

    # 4. Compose
    #
    # Compose may return ``StopReason.NEEDS_REPLAN`` when the Route
    # signal decided REPLAN/DEEPEN mid-execution. An earlier attempt
    # at this layer (kaos-agents 2026-05-17) added a re-expand loop
    # here. That loop turned a 1-step-with-tool-call failure into a
    # 18-step-LLM-only failure because the prior_failures context fed
    # back into ``expand`` biased the model toward "we tried this,
    # let's analyse" instead of "we tried this, let's call a tool
    # differently". The right place to handle replan is the caller
    # (``plan_execute._handle_plan_streaming``) which can synthesise
    # ``result.step_results`` into a useful partial-progress report —
    # see the ``elif result.step_results`` branch in
    # ``patterns/plan_execute.py``.
    return await compose(
        graph,
        tools=tools or {},
        budget=budget,
        model=model,
        parallel=parallel,
        confidence_threshold=confidence_threshold,
        deepen_threshold=deepen_threshold,
        tool_timeout_seconds=tool_timeout_seconds,
    )
