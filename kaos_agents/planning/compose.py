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
import json
import time
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.planning.act import ActResult, act, make_trace
from kaos_agents.planning.evaluate import evaluate_semantic, evaluate_structural
from kaos_agents.planning.graph import PlanGraph
from kaos_agents.planning.route import route
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.plan import (
    ComposeResult,
    Decision,
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
    model: str = DEFAULT_MODEL,
    parallel: bool = True,
    confidence_threshold: float | None = None,
    tool_timeout_seconds: float = 60.0,
    emitter: Any = None,
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

    pivot_goal: str | None = None

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

        # Conditional execution (0.1.0a9 — Wish #4). For each ready
        # step that carries a non-empty ``abort_if`` predicate, query
        # the small condition-judge against the already-completed step
        # results. When the predicate holds:
        #
        #   * mark the step SKIPPED with the evidence stamped as its
        #     "result" (visible to the synthesiser / SPA run inspector);
        #   * if the step also carries a ``pivot_to`` follow-up goal,
        #     stash it and break out of the loop with stop_reason
        #     ``PIVOTED`` so the strategy layer can re-expand on the
        #     new objective;
        #   * always skip dependents — they were predicated on this
        #     step running too.
        #
        # ``abort_if`` is checked sequentially even when ``parallel=True``
        # because the check itself is one cheap LLM call; running them in
        # parallel would buy us microseconds at the cost of a more
        # complex skip-vs-execute reconciliation.
        prior_results = graph.get_results()
        if prior_results:
            from kaos_agents.planning.evaluate_condition import evaluate_condition

            survivors: list[str] = []
            for step_id in ready:
                step_props = graph.get_step(step_id)
                abort_if = (step_props or {}).get("abort_if") or ""
                if not abort_if:
                    survivors.append(step_id)
                    continue
                holds, evidence = await evaluate_condition(
                    abort_if,
                    prior_results,
                    model=model,
                )
                if not holds:
                    survivors.append(step_id)
                    continue
                # Mark the step skipped with the abort context as its
                # "result" so the synthesiser can render "Step X was
                # skipped because: <evidence>". ``mark_skipped`` already
                # prefixes the reason with "SKIPPED:" — we add the
                # "abort_if" qualifier so observers can distinguish
                # this from dependent-skip cases.
                graph.mark_skipped(
                    step_id,
                    f"abort_if fired: {evidence}" if evidence else "abort_if fired",
                )
                _skip_dependents(graph, step_id)
                pivot_to_goal = (step_props or {}).get("pivot_to") or ""
                if pivot_to_goal:
                    pivot_goal = pivot_to_goal
                    logger.info(
                        "compose: step %s aborted with pivot_to=%s — breaking for "
                        "PIVOTED stop_reason",
                        step_id,
                        pivot_to_goal[:80],
                    )
            ready = survivors
            # If pivot_to fired, stop the whole graph now; the strategy
            # layer will see PIVOTED and re-expand on pivot_goal.
            if pivot_goal is not None:
                _skip_remaining(graph)
                final_stop = StopReason.PIVOTED
                break
            # If every ready step was aborted (no survivors), advance
            # the loop — graph.get_ready_steps() will return the next
            # level (or trigger the deadlock branch above).
            if not ready:
                continue

        # Execute ready steps (parallel or sequential)
        if parallel and len(ready) > 1:
            step_results = await _execute_parallel(
                ready, graph, tools, model, tool_timeout_seconds=tool_timeout_seconds
            )
        else:
            step_results = await _execute_sequential(
                ready, graph, tools, model, tool_timeout_seconds=tool_timeout_seconds
            )

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
                # Structural was inconclusive — step has expected_output that
                # needs semantic verification. Call the semantic evaluator.
                # Pass the emitter (if any) so the LLM judge's inputs +
                # output land in the SSE stream as a Span(JUDGE, ...)
                # pair (Wish #7 — visible-event observability for the
                # Evaluate primitive).
                judgment = await evaluate_semantic(
                    act_result.output,
                    expected,
                    model=model,
                    emitter=emitter,
                    step_id=step_id,
                )

            # Update graph.
            #
            # Pre-0.1.0a9: ``mark_failed`` fired on either
            # ``act_result.is_error`` (hard tool error) OR
            # ``not judgment.matched`` (judge said the tool's output
            # didn't satisfy ``expected``). That conflated two very
            # different signals and meant a successful tool call whose
            # JSON the judge wasn't crisply satisfied with disappeared
            # from ``graph.get_results()`` (which only collects
            # COMPLETED steps), erasing the partial work the
            # ``patterns/plan_execute.py`` synthesiser needed to render
            # findings for the user.
            #
            # 0.1.0a9: only hard tool errors mark the step failed. A
            # successful tool with a fussy judge stays COMPLETED — the
            # judgment is stored on the node so downstream consumers
            # can still see ``matched=False``, but the result is
            # preserved in ``step_results``. The Route primitive then
            # decides what to do about the negative verdict (see the
            # confidence-gated REPLAN/CONTINUE branch in
            # ``route.py``).
            if act_result.is_error:
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
            decision = route(judgment, budget, **route_kwargs)

            traces.append(
                PrimitiveTrace(
                    primitive="route",
                    step_id=step_id,
                    success=True,
                    details={"decision": decision.decision.value, "reason": decision.reason},
                )
            )

            # Wish #8 — emit a Span(ROUTE, COMPLETE) so SSE consumers
            # see the decision + judgment that drove it. PrimitiveTrace
            # stays in ``ComposeResult.traces`` for post-hoc analysis,
            # but the Span is what the SPA's run inspector renders.
            if emitter is not None:
                from kaos_agents.events.spans import SpanSubject

                route_span_id = emitter.span_start(
                    SpanSubject.ROUTE,
                    name=f"route.{step_id}",
                    attributes={
                        "step_id": step_id,
                        "judgment_matched": bool(judgment.matched),
                        "judgment_confidence": float(judgment.confidence),
                    },
                ).span_id
                emitter.span_complete(
                    SpanSubject.ROUTE,
                    span_id=route_span_id,
                    name=f"route.{step_id}",
                    attributes={
                        "step_id": step_id,
                        "decision": decision.decision.value,
                        "reason": str(decision.reason)[:200],
                        "judgment_matched": bool(judgment.matched),
                        "judgment_confidence": float(judgment.confidence),
                        "replan_count": replan_count,
                    },
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

            if decision.decision == Decision.REPLAN:
                replan_count += 1
                budget.record_replan()
                # Mark the step as failed and skip dependents
                graph.mark_failed(
                    step_id,
                    f"Route decided {decision.decision.value}: {decision.reason}",
                )
                _skip_dependents(graph, step_id)
                _skip_remaining(graph)
                # Return NEEDS_REPLAN so the strategy layer can re-expand
                final_stop = StopReason.NEEDS_REPLAN
                break

            # CONTINUE — proceed normally

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
    *,
    tool_timeout_seconds: float = 60.0,
) -> list[tuple[str, ActResult]]:
    """Execute multiple steps concurrently."""
    tasks = []
    for step_id in step_ids:
        graph.mark_running(step_id)
        tasks.append(
            _execute_one(step_id, graph, tools, model, tool_timeout_seconds=tool_timeout_seconds)
        )
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
    *,
    tool_timeout_seconds: float = 60.0,
) -> list[tuple[str, ActResult]]:
    """Execute steps one at a time."""
    output = []
    for step_id in step_ids:
        graph.mark_running(step_id)
        try:
            result = await _execute_one(
                step_id, graph, tools, model, tool_timeout_seconds=tool_timeout_seconds
            )
        except Exception as exc:
            result = ActResult(output=f"ERROR: {exc}", is_error=True)
        output.append((step_id, result))
    return output


async def _execute_one(
    step_id: str,
    graph: PlanGraph,
    tools: dict[str, Any],
    model: str,
    *,
    tool_timeout_seconds: float = 60.0,
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
        # Build args from input_spec. The planner stores input_spec as
        # ``{"description": "..."}`` — a free-form description, not
        # structured tool arguments. So when the spec only carries a
        # description, synthesize structured args at runtime from the
        # tool's parameter schema + the description + prior step
        # outputs. Without this, plan-execute called every tool with
        # an empty args dict → error → needs_replan, even when prior
        # steps had already produced everything the tool needed
        # (workflow C symptom: step 1 + 2 succeed, step 3
        # `kaos-source-fr-get-document` errors with "document_number
        # is required" despite step 2 having extracted it).
        if isinstance(input_spec, dict) and _is_description_only(input_spec):
            tool_args = await _synthesize_tool_args(
                tool=tool,
                description=str(input_spec.get("description") or props.get("description", "")),
                prior_outputs=_collect_predecessor_results(graph, step_id),
                model=model,
            )
        else:
            tool_args = input_spec if isinstance(input_spec, dict) else {}
        return await act(
            step_type,
            tool=tool,
            tool_args=tool_args,
            tool_timeout_seconds=tool_timeout_seconds,
        )

    if step_type == StepType.LLM:
        prompt = props.get("description", "")
        # Include input description if available
        if input_spec and isinstance(input_spec, dict):
            desc = input_spec.get("description", "")
            if desc:
                prompt = f"{prompt}\n\nInput: {desc}"
        # Thread predecessor step outputs into the prompt. Without this,
        # an LLM step like "Extract document_number from the most recent
        # search result" sees only the description ("the most recent
        # search result") with no actual data — and judges "expected
        # output value, not a request for input" as the route layer
        # observed in the audit. Pulling each completed predecessor's
        # result into the prompt closes the data-flow gap.
        prior_outputs = _collect_predecessor_results(graph, step_id)
        if prior_outputs:
            prompt = f"{prompt}\n\n{prior_outputs}"
        # Surface the step's expected_output to the LLM. The evaluate
        # phase uses ``expected_output`` as the success criterion (a
        # semantic LLM judge compares the actual output against it).
        # Previously the LLM step never SAW its own success criterion,
        # so it produced "what felt right" while the judge graded
        # against a hidden target — the dominant cause of legitimate
        # plan-execute REPLAN cycles. Showing the expectation up front
        # raises step-completion rate without sacrificing the judge's
        # quality bar (the judge still runs, just with a more aligned
        # target).
        expected = props.get("expected_output", "") or ""
        if expected:
            prompt = f"{prompt}\n\nExpected output: {expected}"
        return await act(step_type, llm_prompt=prompt, llm_model=model)

    return ActResult(output=f"ERROR: Unhandled step type: {step_type}", is_error=True)


def _is_description_only(input_spec: dict[str, Any]) -> bool:
    """True when the planner's input_spec carries only a free-form description.

    The planner emits ``input_spec = {"description": "<text>"}`` for every
    step — never structured per-arg fields. Detect this shape so we know
    when to synthesize real args vs. when the caller passed structured
    args (e.g., a test or future planner that does fill input_spec
    properly).
    """
    if not input_spec:
        return False
    keys = set(input_spec.keys())
    return keys <= {"description"}


async def _synthesize_tool_args(
    *,
    tool: Any | None,
    description: str,
    prior_outputs: str,
    model: str,
) -> dict[str, Any]:
    """Use an LLM to populate a tool's args from a description + prior outputs.

    Returns ``{}`` if synthesis fails — the tool will then fail with a
    clearer "missing argument" error than the silent empty-dict path.
    Tool-side errors propagate via the standard route → REPLAN cycle.
    """
    if tool is None:
        return {}

    # Pull the tool's input schema from the kaos-llm-core Tool wrapper.
    # Different Tool implementations expose the schema under different
    # attributes — try the common ones in order, fall back to empty.
    schema: dict[str, Any] = {}
    for attr in ("input_schema", "parameters", "schema", "definition"):
        candidate = getattr(tool, attr, None)
        if candidate is None:
            continue
        # ``definition`` may be a wrapper that nests parameters/schema.
        if hasattr(candidate, "input_schema"):
            schema = candidate.input_schema or {}
        elif hasattr(candidate, "parameters"):
            schema = candidate.parameters or {}
        elif isinstance(candidate, dict):
            schema = candidate
        if schema:
            break

    if not schema:
        # No schema → can't safely synthesize. Return empty so the
        # tool's own validation surfaces the missing-arg error.
        return {}

    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call, InputField, OutputField, Signature

    class _ToolArgSynthesisSignature(Signature):
        tool_description: str = InputField(description="Description of what the tool does.")
        tool_schema: str = InputField(description="JSON schema for the tool's arguments.")
        step_description: str = InputField(description="What this step is trying to accomplish.")
        prior_outputs: str = InputField(description="Outputs from previous plan steps.")
        args_json: str = OutputField(
            description=(
                "A JSON object with the tool's arguments populated from prior "
                "outputs and the step description. Output ONLY the JSON object, "
                "no prose, no code fences, no commentary."
            )
        )

    tool_desc = str(getattr(tool, "description", "") or getattr(tool, "name", "") or "tool")
    schema_json = json.dumps(schema, default=str)[:4000]

    call = Call(_ToolArgSynthesisSignature, model=model)
    try:
        invocation = await call.invoke(
            tool_description=tool_desc,
            tool_schema=schema_json,
            step_description=description,
            prior_outputs=prior_outputs,
        )
    except Exception as exc:
        logger.warning("compose: arg synthesis failed for tool: %s", exc)
        return {}

    raw = str(invocation.output.args_json) if invocation.output else "{}"
    raw = raw.strip()
    # Strip code fences if the LLM produced them despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json\n"):
            raw = raw[len("json\n") :]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("compose: tool-arg synthesis returned non-JSON: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("compose: tool-arg synthesis returned non-dict: %r", type(parsed).__name__)
        return {}
    return parsed


def _collect_predecessor_results(
    graph: PlanGraph,
    step_id: str,
    *,
    per_predecessor_char_budget: int | None = None,
) -> str:
    """Gather completed predecessor step outputs as prompt-ready context.

    Returns an empty string when the step has no completed predecessors.
    Otherwise returns a labeled block:

        --- output of step <pred_id> ---
        <result text>

    Args:
        graph: PlanGraph holding step results.
        step_id: The current step whose predecessors we're gathering.
        per_predecessor_char_budget: When set, each predecessor's
            ``result`` text is truncated to this many characters with
            a ``\\n... (truncated <N> more chars)`` marker. **Default
            ``None`` — no truncation.** The next step's LLM sees the
            FULL predecessor output so its reasoning is grounded in
            the same text the operator can audit. Pass an explicit
            cap only when the operator has measured a specific
            context-window pressure; never as a default safety net
            (silent truncation of a JSON tail can flip a downstream
            schema decision). When the cap actually fires, a
            ``logger.warning`` is emitted so the truncation is
            auditable in the structured log.
    """
    try:
        # PlanGraph wraps a kaos-graph Graph at ``self._graph``. Use
        # predecessors() directly — the public PlanGraph API exposes
        # readiness checks but not raw predecessor lookup.
        preds = list(graph._graph.predecessors(step_id))
    except Exception as exc:
        logger.debug("compose: predecessor lookup failed for %s: %s", step_id, exc)
        return ""
    if not preds:
        return ""
    blocks: list[str] = []
    for pred_id in preds:
        pred_props = graph.get_step(pred_id)
        if not pred_props:
            continue
        if pred_props.get("status") != StepStatus.COMPLETED.value:
            continue
        result_text = str(pred_props.get("result") or "").strip()
        if not result_text:
            continue
        if (
            per_predecessor_char_budget is not None
            and len(result_text) > per_predecessor_char_budget
        ):
            dropped = len(result_text) - per_predecessor_char_budget
            logger.warning(
                "compose: predecessor result truncated step_id=%s pred_id=%s "
                "kept=%d dropped=%d budget=%d",
                step_id,
                pred_id,
                per_predecessor_char_budget,
                dropped,
                per_predecessor_char_budget,
            )
            result_text = (
                result_text[:per_predecessor_char_budget]
                + f"\n... (truncated {dropped} more chars)"
            )
        blocks.append(f"--- output of step {pred_id} ---\n{result_text}")
    if not blocks:
        return ""
    return "Outputs from prior steps:\n\n" + "\n\n".join(blocks)


def _skip_remaining(graph: PlanGraph) -> None:
    """Skip all pending steps."""
    for step_id in graph.step_ids():
        props = graph.get_step(step_id)
        if props and props.get("status") == StepStatus.PENDING.value:
            graph.mark_skipped(step_id, "Budget or failure stop")


def _skip_dependents(graph: PlanGraph, failed_step_id: str) -> None:
    """Skip steps that depend on a failed step."""
    try:
        deps = graph.descendants(failed_step_id)
    except Exception:
        logger.debug("Failed to skip dependents of %s", failed_step_id, exc_info=True)
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
