"""Act primitive — execute a single action.

The only primitive with side effects. Dispatches by step type:
- TOOL: calls a KaosTool via ToolBridge
- LLM: makes an LLM call via Call

Act does NOT recurse (subplan execution is Compose's job).
Failures are returned as error results, not raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.planning.types import PrimitiveTrace, StepType

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ActResult:
    """The output of an Act primitive call."""

    output: str  # Result text (or error message)
    is_error: bool = False
    wall_clock_ms: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0


async def act(
    step_type: StepType,
    *,
    tool: Tool | None = None,
    tool_args: dict[str, Any] | None = None,
    llm_prompt: str | None = None,
    llm_model: str = "anthropic:claude-haiku-4-5",
    tool_timeout_seconds: float = 60.0,
) -> ActResult:
    """Execute a single action.

    Args:
        step_type: What kind of action (TOOL or LLM).
        tool: For TOOL type — the kaos-llm-core Tool to invoke.
        tool_args: For TOOL type — arguments to pass.
        llm_prompt: For LLM type — the prompt to send.
        llm_model: For LLM type — the model to use.

    Returns:
        ActResult with output text and metadata.
    """
    start = time.perf_counter()

    try:
        if step_type == StepType.TOOL:
            result = await _act_tool(tool, tool_args or {}, timeout_seconds=tool_timeout_seconds)
        elif step_type == StepType.LLM:
            result = await _act_llm(llm_prompt or "", llm_model)
        elif step_type == StepType.SUBPLAN:
            # Subplan execution is Compose's responsibility, not Act's
            return ActResult(
                output="ERROR: SUBPLAN steps must be handled by Compose, not Act.",
                is_error=True,
                wall_clock_ms=0.0,
            )
        else:
            return ActResult(
                output=f"ERROR: Unknown step type: {step_type}",
                is_error=True,
                wall_clock_ms=0.0,
            )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("act: %s failed: %s", step_type.value, exc)
        return ActResult(
            output=f"ERROR: {exc}",
            is_error=True,
            wall_clock_ms=elapsed_ms,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return ActResult(
        output=result.get("output", ""),
        is_error=result.get("is_error", False),
        wall_clock_ms=elapsed_ms,
        token_count=result.get("token_count", 0),
        cost_usd=result.get("cost_usd", 0.0),
    )


async def _act_tool(
    tool: Tool | None,
    args: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Execute a tool via the kaos-llm-core Tool interface."""
    import asyncio

    if tool is None:
        return {"output": "ERROR: No tool provided for TOOL step.", "is_error": True}

    from kaos_agents.planning.result_check import is_error_result

    try:
        async with asyncio.timeout(timeout_seconds):
            result = await tool.invoke(args)
    except TimeoutError:
        return {
            "output": f"ERROR: Tool '{tool.name}' timed out after {timeout_seconds}s. "
            "Try a simpler query or increase the timeout.",
            "is_error": True,
        }

    result_str = str(result)

    return {
        "output": result_str,
        "is_error": is_error_result(result_str),
    }


async def _act_llm(prompt: str, model: str) -> dict[str, Any]:
    """Execute an LLM call."""
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call, InputField, OutputField, Signature

    class LLMStep(Signature):
        """Execute a reasoning step."""

        instruction: str = InputField(description="What to do")
        response: str = OutputField(description="The result")

    call = Call(LLMStep, model=model)
    invocation = await call.invoke(instruction=prompt)

    token_count = 0
    cost_usd = 0.0
    if invocation.usage:
        token_count = invocation.usage.total_tokens or 0
        cost_usd = invocation.usage.cost_usd or 0.0

    return {
        "output": str(invocation.output.response) if invocation.output else "",
        "is_error": invocation.error is not None,
        "token_count": token_count,
        "cost_usd": cost_usd,
    }


def make_trace(
    step_id: str,
    result: ActResult,
) -> PrimitiveTrace:
    """Create an observability trace from an Act result."""
    return PrimitiveTrace(
        primitive="act",
        step_id=step_id,
        wall_clock_ms=result.wall_clock_ms,
        token_count=result.token_count,
        cost_usd=result.cost_usd,
        success=not result.is_error,
        details={"output_length": len(result.output)},
    )
