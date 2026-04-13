"""Expand primitive — generate plan steps from a goal.

Takes a goal description and produces structured Steps. The LLM generates
the plan; this primitive validates tool names against the registry and
handles recursive expansion for complex goals.

Cardinality:
- n=1: one next step (ReAct-style)
- n=all (default): complete plan
- n=k: k alternatives for branching

Depth:
- shallow: only primitive tool calls
- recursive: steps can be subgoals that get further expanded
"""

from __future__ import annotations

import uuid
from typing import Any

from kaos_core.logging import get_logger
from pydantic import BaseModel, Field

from kaos_agents.planning.types import Step, StepType

logger = get_logger(__name__)


async def expand(
    goal: str,
    *,
    available_tools: dict[str, str] | None = None,
    context: str = "",
    prior_failures: str = "",
    max_steps: int = 10,
    model: str = "anthropic:claude-haiku-4-5",
) -> list[Step]:
    """Generate plan steps for a goal via LLM.

    Args:
        goal: What to accomplish.
        available_tools: Dict of tool_name → description for valid tools.
        context: Assembled memory context (from Recall).
        prior_failures: Description of previous failed attempts (Reflexion feedback).
        max_steps: Maximum number of steps to generate.
        model: LLM model for plan generation.

    Returns:
        List of Steps with validated tool names and dependency references.
    """
    from kaos_llm_core import Call, InputField, OutputField, Signature

    tool_descriptions = ""
    if available_tools:
        tool_descriptions = "\n".join(
            f"- {name}: {desc}" for name, desc in sorted(available_tools.items())
        )
    else:
        tool_descriptions = "(no tools available — generate LLM-only steps)"

    failure_context = ""
    if prior_failures:
        failure_context = f"\n\nPREVIOUS FAILED ATTEMPTS (avoid repeating these):\n{prior_failures}"

    class PlanExpand(Signature):
        """Generate a structured plan to accomplish a goal.

        Each step should use one of the available tools or be an LLM reasoning step.
        Steps can depend on previous steps by referencing their step numbers.
        """

        goal: str = InputField(description="The goal to accomplish")
        tools: str = InputField(description="Available tools and their descriptions")
        context: str = InputField(description="Conversation context and prior knowledge")
        steps: list[dict[str, Any]] = OutputField(
            description="List of plan steps. Each step is a dict with: "
            "step_number (int, 1-indexed), "
            "description (str, what this step does), "
            "tool_name (str, name of the tool to use, or 'llm' for a reasoning step), "
            "input_description (str, what input to provide), "
            "expected_output (str, what output to expect), "
            "depends_on (list of int, step numbers this depends on, empty list for no deps)"
        )

    instructions = (
        "Generate a concrete, actionable plan using ONLY the listed tools. "
        "Each step must reference a specific tool by its exact name, or use 'llm' for reasoning steps. "
        "Keep plans focused — use the minimum number of steps needed. "
        f"Maximum {max_steps} steps."
        f"{failure_context}"
    )

    call = Call(PlanExpand, model=model, instructions=instructions)

    try:
        result = await call(
            goal=goal,
            tools=tool_descriptions,
            context=context[:3000],
        )
        raw_steps = result.steps if result.steps else []
    except Exception as exc:
        logger.warning("expand: LLM plan generation failed: %s", exc)
        return []

    # Convert and validate
    steps = _parse_steps(raw_steps, available_tools or {})

    logger.debug("expand: generated %d steps for goal: %s", len(steps), goal[:80])
    return steps


class GeneratedStep(BaseModel):
    """Pydantic model for validating LLM-generated plan steps.

    This is the schema that raw LLM output is validated against before
    being converted to Step instances. Any field the LLM omits gets
    a sensible default instead of a silent KeyError.
    """

    step_number: int = 1
    description: str = ""
    tool_name: str = "llm"
    input_description: str = ""
    expected_output: str = ""
    depends_on: list[int] = Field(default_factory=list)


def _validate_raw_steps(raw_steps: list[Any]) -> list[GeneratedStep]:
    """Validate raw LLM output into typed GeneratedStep instances.

    Malformed entries are logged and skipped instead of crashing.
    """
    validated = []
    for i, raw in enumerate(raw_steps):
        try:
            if isinstance(raw, dict):
                validated.append(GeneratedStep.model_validate(raw))
            else:
                logger.warning("expand: step %d is not a dict, skipping: %s", i, type(raw))
        except Exception as exc:
            logger.warning("expand: step %d validation failed, skipping: %s", i, exc)
    return validated


def _parse_steps(
    raw_steps: list[dict[str, Any]],
    available_tools: dict[str, str],
) -> list[Step]:
    """Parse and validate LLM-generated steps.

    Validates structure via Pydantic model, then validates tool names
    against the registry. Steps with hallucinated tool names are
    converted to LLM reasoning steps with a warning.
    """
    valid_tool_names = set(available_tools.keys())
    validated = _validate_raw_steps(raw_steps)
    steps: list[Step] = []
    id_map: dict[int, str] = {}

    for gen in validated:
        step_num = gen.step_number
        step_id = f"step-{step_num}-{uuid.uuid4().hex[:6]}"
        id_map[step_num] = step_id

        tool_name = gen.tool_name.strip()
        description = gen.description or f"Step {step_num}"
        input_desc = gen.input_description
        expected = gen.expected_output
        # Determine step type and validate tool
        if tool_name == "llm" or tool_name == "":
            step_type = StepType.LLM
            validated_tool = None
        elif valid_tool_names and tool_name not in valid_tool_names:
            # Hallucinated tool — convert to LLM step with warning
            logger.warning(
                "expand: tool '%s' not registered, converting step %d to LLM",
                tool_name,
                step_num,
            )
            step_type = StepType.LLM
            validated_tool = None
            description = f"[Tool '{tool_name}' not found] {description}"
        else:
            step_type = StepType.TOOL
            validated_tool = tool_name

        # Resolve dependencies
        deps = tuple(id_map[d] for d in gen.depends_on if isinstance(d, int) and d in id_map)

        steps.append(
            Step(
                id=step_id,
                step_type=step_type,
                description=description,
                tool_name=validated_tool,
                input_spec={"description": input_desc} if input_desc else {},
                expected_output=expected,
                depends_on=deps,
            )
        )

    return steps


def expand_from_steps(steps: list[Step]) -> list[Step]:
    """Passthrough for pre-built step lists (e.g., user-provided plans)."""
    return steps
