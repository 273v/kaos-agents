"""Hierarchical decomposition strategy (HTN-style).

For complex goals that need multi-step plans. Composes:
    recall → expand(n=all) → build graph → validate → compose (parallel)
    → [if every step failed and budget allows: re-expand and retry]

This is the full planning path: the LLM generates a complete plan, we
build a PlanGraph, validate it, execute via Compose, and (when the
first plan fails completely with no useful tool calls) re-expand using
structured failure context and try once more.
"""

from __future__ import annotations

from dataclasses import replace
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


_FAILURE_PROMPT_HEADER = (
    "The previous plan failed at every step (no successful tool calls). "
    "Try a different approach for replan attempt {attempt} of {limit}. "
    "Prefer direct tool calls over LLM-only analysis steps; the previous "
    "attempt likely failed because its steps were too abstract for the "
    "available tools.\n\nPrior failures:\n"
)


def _format_prior_failures(
    base: str,
    *,
    graph: PlanGraph,
    attempt: int,
    limit: int,
) -> str:
    """Render structured ``prior_failures`` for the next ``expand()`` call.

    Only includes FAILED steps from the supplied graph (via
    ``PlanGraph.get_failures``) — never successful tool output, since
    feeding successful content back as "failures" biases the model
    toward LLM-only "analysis" steps (the regression that motivated
    bounding the replan loop to the "every step failed" case).

    The ``base`` string preserves whatever ``prior_failures`` the
    caller passed in (e.g., REFLECTION-section content from earlier
    turns) so the new failures stack on top rather than replace.
    """
    failure_lines: list[str] = []
    for sid, meta in graph.get_failures().items():
        tool = meta.get("tool_name") or ""
        step_type = meta.get("step_type") or ""
        result_text = str(meta.get("result") or "")[:200]
        tool_suffix = f", tool={tool}" if tool else ""
        failure_lines.append(f"- Step {sid} ({step_type}{tool_suffix}): {result_text}")

    new_section = _FAILURE_PROMPT_HEADER.format(attempt=attempt, limit=limit) + "\n".join(
        failure_lines
    )
    if base:
        return f"{base}\n\n{new_section}"
    return new_section


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
    """Decompose a goal into a full plan and execute it, with bounded replan.

    Steps per attempt:
        1. Expand: LLM generates a complete multi-step plan
        2. Build: steps are assembled into a PlanGraph
        3. Validate: cycle detection, tool name validation
        4. Compose: execute the graph (parallel by default)

    After Compose returns:
        * ``stop_reason != NEEDS_REPLAN`` → terminal, return immediately.
        * ``stop_reason == NEEDS_REPLAN`` AND any step produced a
          successful tool result in this round → terminal, return so
          the caller's response formatter can synthesise the partial
          findings (decision: don't burn the rest of the replan budget
          re-doing work that already succeeded; the failure was Route
          rejecting later steps, not the early successes).
        * ``stop_reason == NEEDS_REPLAN`` AND no step succeeded AND
          ``budget.replans < budget.max_replans`` → re-expand with
          structured ``prior_failures`` and try again.

    Budget semantics: every Compose call shares the same ``budget``
    instance, so cumulative cost / token / wall-clock limits apply
    across replan attempts. ``budget.replans`` is incremented by
    Compose itself when it returns NEEDS_REPLAN, so the loop's
    termination check against ``budget.max_replans`` is correct
    without manual increments here.

    Returns:
        A single :class:`ComposeResult` whose ``step_results`` is the
        union of every attempt's results. If the final attempt hit
        ``NEEDS_REPLAN`` with ``budget.replans >= budget.max_replans``,
        the ``stop_reason`` is promoted to ``MAX_REPLANS`` so callers
        can distinguish "ran out of replan budget" from "replanning
        was never tried".
    """
    if budget is None:
        budget = PlanBudget()

    cumulative_step_results: dict[str, object] = {}
    current_prior_failures = prior_failures or ""
    last_result: ComposeResult | None = None

    # Bound: one initial attempt + up to ``max_replans`` retries.
    for attempt in range(budget.max_replans + 1):
        # 1. Expand
        steps = await expand(
            goal,
            available_tools=tool_descriptions or {},
            context=context,
            prior_failures=current_prior_failures,
            max_steps=max_steps,
            model=model,
        )

        if not steps:
            # If we already accumulated useful work from a prior
            # attempt, surface it with FAILURE stop_reason rather than
            # discarding it.
            if cumulative_step_results:
                base = last_result or ComposeResult(plan_json="{}")
                return replace(
                    base,
                    stop_reason=StopReason.FAILURE,
                    step_results=cumulative_step_results,
                )
            return ComposeResult(
                plan_json="{}",
                stop_reason=StopReason.FAILURE,
                step_results={"error": "Expand produced no steps"},
            )

        # 2. Build graph (unique name per attempt aids debugging)
        graph_name = "decompose" if attempt == 0 else f"decompose-replan-{attempt}"
        graph = PlanGraph(name=graph_name)
        for step in steps:
            graph.add_step(step)

        # 3. Validate
        issues = graph.validate()
        if issues:
            logger.warning("decompose: plan has validation issues: %s", issues)
            # Continue anyway — validation issues are warnings, not blockers

        logger.debug(
            "decompose: attempt %d — %d steps, %d levels, %d ready",
            attempt,
            graph.n_steps,
            len(graph.get_execution_levels()),
            len(graph.get_ready_steps()),
        )

        # 4. Compose
        result = await compose(
            graph,
            tools=tools or {},
            budget=budget,
            model=model,
            parallel=parallel,
            confidence_threshold=confidence_threshold,
            deepen_threshold=deepen_threshold,
            tool_timeout_seconds=tool_timeout_seconds,
        )

        # Accumulate any successful step results from this attempt.
        cumulative_step_results.update(result.step_results)
        last_result = result

        # Terminal: any non-replan stop reason means the strategy is done.
        if result.stop_reason != StopReason.NEEDS_REPLAN:
            break

        # Soft-stop: if this attempt produced any successful tool
        # results, don't re-plan — replanning will at best repeat the
        # work and at worst (the original regression) generate
        # LLM-only steps that bias on the prior content. The caller's
        # synthesis branch will format the partials for the user.
        if result.step_results:
            logger.info(
                "decompose: NEEDS_REPLAN with %d successful step(s); "
                "stopping so synthesis can surface partials",
                len(result.step_results),
            )
            break

        # Hard-stop: budget exhausted (compose already incremented
        # ``budget.replans``). The post-loop promotion will rewrite
        # NEEDS_REPLAN → MAX_REPLANS for the caller.
        if budget.replans >= budget.max_replans:
            logger.info("decompose: hit max_replans=%d — stopping", budget.max_replans)
            break

        # Otherwise: every step failed AND we have budget. Build
        # structured prior_failures and retry.
        current_prior_failures = _format_prior_failures(
            prior_failures or "",
            graph=graph,
            attempt=attempt + 1,
            limit=budget.max_replans,
        )
        logger.info(
            "decompose: every step failed in attempt %d; retrying (replan %d of %d)",
            attempt,
            attempt + 1,
            budget.max_replans,
        )

    # last_result is always set after the loop (the only path that
    # leaves it None — empty steps on attempt 0 — returns early above).
    assert last_result is not None

    # Merge cumulative step_results into the final result (no-op when
    # the loop only ran once).
    if cumulative_step_results and cumulative_step_results != dict(last_result.step_results):
        last_result = replace(last_result, step_results=cumulative_step_results)

    # Promote NEEDS_REPLAN → MAX_REPLANS when we exhausted the replan
    # budget. The caller's response formatter renders these differently
    # ("max_replans" reads as a budget hit; "needs_replan" reads as a
    # control-flow signal the strategy hasn't wired up).
    if last_result.stop_reason == StopReason.NEEDS_REPLAN and budget.replans >= budget.max_replans:
        last_result = replace(last_result, stop_reason=StopReason.MAX_REPLANS)
        logger.info(
            "decompose: promoting stop_reason NEEDS_REPLAN → MAX_REPLANS (replans=%d/%d)",
            budget.replans,
            budget.max_replans,
        )

    return last_result
