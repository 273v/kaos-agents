"""Direct execution strategy — single-step reactive (ReAct-style).

For simple goals that need one tool call or one LLM reasoning step.
No plan graph, no multi-step orchestration. Just: expand(n=1) → act → evaluate.

This is the fast path for tool_use intent. It composes:
    recall → expand(n=1) → act → evaluate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.planning.act import act
from kaos_agents.planning.evaluate import evaluate_structural
from kaos_agents.planning.expand import expand
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.plan import ComposeResult, PlanBudget, StepType, StopReason

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)


async def execute_direct(
    goal: str,
    *,
    tools: dict[str, Tool] | None = None,
    tool_descriptions: dict[str, str] | None = None,
    context: str = "",
    model: str = DEFAULT_MODEL,
    budget: PlanBudget | None = None,
    tool_timeout_seconds: float = 60.0,
    emitter: Any = None,
) -> ComposeResult:
    """Execute a goal directly — one step, no plan graph.

    Expands with n=1 (single step), executes it, evaluates the result.
    Suitable for simple tool_use or single-action goals.

    ``emitter`` is accepted for signature parity with
    :func:`execute_decompose` (so :func:`execute_adaptive` can pass
    it through uniformly) but unused here: direct-execution does only
    structural evaluation (no LLM judge to attach a Span(JUDGE, ...)
    to, and no Route call since success/failure is decided inline by
    the structural judgment).
    """
    if budget is None:
        budget = PlanBudget()

    # Expand: generate one step
    steps = await expand(
        goal,
        available_tools=tool_descriptions or {},
        context=context,
        max_steps=1,
        model=model,
    )

    if not steps:
        return ComposeResult(
            plan_json="{}",
            stop_reason=StopReason.FAILURE,
            step_results={"error": "Expand produced no steps"},
        )

    step = steps[0]

    # Act: execute the step
    tool = tools.get(step.tool_name) if tools and step.tool_name else None
    result = await act(
        step.step_type,
        tool=tool,
        tool_args=step.input_spec if step.step_type == StepType.TOOL else None,
        llm_prompt=step.description if step.step_type == StepType.LLM else None,
        llm_model=model,
        tool_timeout_seconds=tool_timeout_seconds,
    )

    budget.record_step(cost_usd=result.cost_usd, tokens=result.token_count)

    # Evaluate structurally
    judgment = evaluate_structural(result.output, step.expected_output)
    is_success = judgment is None or judgment.matched

    return ComposeResult(
        plan_json="{}",
        stop_reason=StopReason.SUCCESS if is_success else StopReason.FAILURE,
        total_cost_usd=budget.cost_usd,
        total_tokens=budget.tokens_used,
        steps_executed=1,
        wall_clock_ms=result.wall_clock_ms,
        step_results={"step-1": result.output},
    )
