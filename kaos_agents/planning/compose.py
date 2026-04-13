"""Compose primitive — orchestrates plan execution.

Walks the plan graph by topological level, executes ready steps via Act
(parallel with asyncio.gather), evaluates each result, routes control flow.

The inner loop:
    while graph has pending steps:
        ready = graph.get_ready_steps()
        results = await gather(*[act(step) for step in ready])
        for step, result in zip(ready, results):
            judgment = evaluate(result)
            graph.mark_complete(step, result, judgment)
            decision = route(judgment, budget)
            if decision != CONTINUE: handle it

Compose does NOT call Expand — that's the strategy layer's job.
Compose executes a plan that's already been built.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.planning.act import ActResult, act, make_trace
from kaos_agents.planning.evaluate import evaluate_structural
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.planning.route import route
from kaos_agents.planning.types import (
    ComposeResult,
    Decision,
    EvalMode,
    Judgment,
    PlanBudget,
    PrimitiveTrace,
    StepStatus,
    StepType,
    StopReason,
)

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)


async def compose(
    graph: PlanGraph,
    *,
    tools: dict[str, Tool] | None = None,
    budget: PlanBudget | None = None,
    model: str = "anthropic:claude-haiku-4-5",
    parallel: bool = True,
    confidence_threshold: float | None = None,
    deepen_threshold: float | None = None,
) -> ComposeResult:
    """Execute a plan graph.

    Walks the graph by topological level, executing ready steps in parallel
    (or sequentially if parallel=False). Each step is evaluated structurally
    after execution, and Route decides whether to continue.

    Args:
        graph: The plan graph to execute.
        tools: Dict of tool_name → Tool instances for TOOL steps.
        budget: Resource limits. Created with defaults if None.
        model: LLM model for LLM steps and semantic evaluation.
        parallel: Whether to execute independent steps concurrently.

    Returns:
        ComposeResult with traces, results, and stop reason.
    """
    if budget is None:
        budget = PlanBudget()

    tools = tools or {}
    traces: list[PrimitiveTrace] = []
    start_time = time.perf_counter()
    replan_count = 0
    final_stop = StopReason.SUCCESS

    while not graph.is_complete():
        ready = graph.get_ready_steps()

        if not ready:
            # No steps ready but graph not complete — deadlock or all remaining failed
            if graph.has_failures():
                final_stop = StopReason.FAILURE
                logger.warning("compose: deadlock — no ready steps, graph has failures")
            else:
                logger.warning("compose: deadlock — no ready steps, no failures")
                final_stop = StopReason.FAILURE
            break

        # Execute ready steps (parallel or sequential)
        if parallel and len(ready) > 1:
            step_results = await _execute_parallel(ready, graph, tools, model)
        else:
            step_results = await _execute_sequential(ready, graph, tools, model)

        # Process results
        for step_id, act_result in step_results:
            # Record trace
            traces.append(make_trace(step_id, act_result))

            # Evaluate structurally
            step_props = graph.get_step(step_id)
            expected = step_props.get("expected_output", "") if step_props else ""
            tool_name = step_props.get("tool_name") if step_props else None

            judgment = evaluate_structural(
                act_result.output,
                expected,
                tool_name=tool_name,
                available_tools=set(tools.keys()) if tools else None,
            )

            if judgment is None:
                # Structural was inconclusive — has expected output that needs
                # semantic verification, but we only do structural in Compose.
                # Mark as tentative match at low confidence so Route can decide.
                judgment = Judgment(
                    matched=True,
                    confidence=0.4,
                    reasoning="Structural check inconclusive (has expected output). "
                    "Result accepted tentatively — semantic verification not performed.",
                    mode=EvalMode.STRUCTURAL,
                )

            # Update graph
            if act_result.is_error or not judgment.matched:
                graph.mark_failed(step_id, act_result.output[:200])
            else:
                graph.mark_complete(step_id, act_result.output, judgment)

            # Track budget
            budget.record_step(
                cost_usd=act_result.cost_usd,
                tokens=act_result.token_count,
            )

            # Route (with configurable thresholds)
            route_kwargs: dict[str, Any] = {
                "replan_count": replan_count,
                "step_id": step_id,
            }
            if confidence_threshold is not None:
                route_kwargs["confidence_threshold"] = confidence_threshold
            if deepen_threshold is not None:
                route_kwargs["deepen_threshold"] = deepen_threshold
            decision = route(judgment, budget, **route_kwargs)

            traces.append(
                PrimitiveTrace(
                    primitive="route",
                    step_id=step_id,
                    success=True,
                    details={"decision": decision.decision.value, "reason": decision.reason},
                )
            )

            if decision.decision == Decision.STOP_BUDGET:
                # Skip remaining steps
                _skip_remaining(graph)
                final_stop = _stop_reason_from_budget(budget)
                break

            if decision.decision == Decision.STOP_FAILURE:
                _skip_remaining(graph)
                final_stop = StopReason.FAILURE
                break

            if decision.decision in (Decision.REPLAN, Decision.DEEPEN):
                replan_count += 1
                budget.record_replan()
                # Undo the completion — Route decided this step's result isn't good enough.
                # Mark as failed so downstream steps don't execute on bad results.
                graph.mark_failed(
                    step_id,
                    f"Route decided {decision.decision.value}: {decision.reason}",
                )
                _skip_dependents(graph, step_id)
                # Note: Compose itself doesn't re-expand. That's the strategy layer's job.
                # We just stop this graph's execution. The strategy can inspect the
                # ComposeResult and decide whether to expand a new plan.
                continue

            # CONTINUE — proceed normally
            # DEEPEN would require Expand, which is the strategy layer's job

        # Check if we hit a terminal decision
        if final_stop != StopReason.SUCCESS:
            break

    # Determine final status
    if graph.is_complete() and not graph.has_failures() and final_stop == StopReason.SUCCESS:
        final_stop = StopReason.SUCCESS
    elif graph.has_failures() and final_stop == StopReason.SUCCESS:
        # Graph completed but has failures — not a clean success
        final_stop = StopReason.FAILURE

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.debug(
        "compose: finished — %d steps, %d traces, stop=%s, %.0fms",
        budget.steps_executed,
        len(traces),
        final_stop.value,
        elapsed_ms,
    )

    return ComposeResult(
        plan_json=graph.to_json(),
        traces=tuple(traces),
        stop_reason=final_stop,
        total_cost_usd=budget.cost_usd,
        total_tokens=budget.tokens_used,
        steps_executed=budget.steps_executed,
        replans=replan_count,
        wall_clock_ms=elapsed_ms,
        step_results=graph.get_results(),
    )


async def _execute_parallel(
    step_ids: list[str],
    graph: PlanGraph,
    tools: dict[str, Any],
    model: str,
) -> list[tuple[str, ActResult]]:
    """Execute multiple steps concurrently."""
    tasks = []
    for step_id in step_ids:
        graph.mark_running(step_id)
        tasks.append(_execute_one(step_id, graph, tools, model))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for step_id, result in zip(step_ids, results, strict=True):
        if isinstance(result, BaseException):
            output.append(
                (
                    step_id,
                    ActResult(
                        output=f"ERROR: {result}",
                        is_error=True,
                    ),
                )
            )
        else:
            output.append((step_id, result))
    return output


async def _execute_sequential(
    step_ids: list[str],
    graph: PlanGraph,
    tools: dict[str, Any],
    model: str,
) -> list[tuple[str, ActResult]]:
    """Execute steps one at a time."""
    output = []
    for step_id in step_ids:
        graph.mark_running(step_id)
        try:
            result = await _execute_one(step_id, graph, tools, model)
        except Exception as exc:
            result = ActResult(output=f"ERROR: {exc}", is_error=True)
        output.append((step_id, result))
    return output


async def _execute_one(
    step_id: str,
    graph: PlanGraph,
    tools: dict[str, Any],
    model: str,
) -> ActResult:
    """Execute a single step via Act."""
    props = graph.get_step(step_id)
    if props is None:
        return ActResult(output=f"ERROR: Step {step_id} not found in graph.", is_error=True)

    step_type_str = props.get("step_type", "llm")
    tool_name = props.get("tool_name", "")
    input_spec = props.get("input_spec", {})

    try:
        step_type = StepType(step_type_str)
    except ValueError:
        step_type = StepType.LLM

    if step_type == StepType.TOOL and tool_name:
        tool = tools.get(tool_name)
        # Build args from input_spec
        tool_args = input_spec if isinstance(input_spec, dict) else {}
        return await act(step_type, tool=tool, tool_args=tool_args)

    if step_type == StepType.LLM:
        prompt = props.get("description", "")
        # Include input description if available
        if input_spec and isinstance(input_spec, dict):
            desc = input_spec.get("description", "")
            if desc:
                prompt = f"{prompt}\n\nInput: {desc}"
        return await act(step_type, llm_prompt=prompt, llm_model=model)

    return ActResult(output=f"ERROR: Unhandled step type: {step_type}", is_error=True)


def _skip_remaining(graph: PlanGraph) -> None:
    """Skip all pending steps."""
    for step_id in graph.step_ids():
        props = graph.get_step(step_id)
        if props and props.get("status") == StepStatus.PENDING.value:
            graph.mark_skipped(step_id, "Budget or failure stop")


def _skip_dependents(graph: PlanGraph, failed_step_id: str) -> None:
    """Skip steps that depend on a failed step."""
    from kaos_graph.algorithms import descendants

    try:
        deps = descendants(graph._graph, failed_step_id)
    except Exception:
        return

    for dep_id in deps:
        props = graph.get_step(dep_id)
        if props and props.get("status") == StepStatus.PENDING.value:
            graph.mark_skipped(dep_id, f"Dependency {failed_step_id} failed")


def _stop_reason_from_budget(budget: PlanBudget) -> StopReason:
    """Convert a budget stop to a StopReason."""
    reason = budget.should_stop()
    if reason is not None:
        return reason
    return StopReason.FAILURE
