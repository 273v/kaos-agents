"""Adaptive strategy (ADaPT-style) — confidence-gated depth selection.

The metastrategy. Evaluates goal complexity, then routes:
- Simple goals (high confidence) → direct execution
- Complex goals (low confidence) → hierarchical decomposition

Composes: evaluate(complexity) → route(direct or decompose)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.planning.strategies.decompose import execute_decompose
from kaos_agents.planning.strategies.direct import execute_direct
from kaos_agents.planning.types import ComposeResult, PlanBudget, StopReason
from kaos_agents.settings import DEFAULT_MODEL

if TYPE_CHECKING:
    from kaos_llm_core.programs.tool import Tool

logger = get_logger(__name__)

# Patterns that suggest multi-step complexity.
# Intentionally conservative — false negatives (missing a complex goal)
# are cheaper than false positives (over-planning a simple goal).
_COMPLEX_PATTERNS = (
    "then",
    "after that",
    "first",
    "step by step",
    "compare",
    "cross-reference",
    "analyze and",
    "multiple",
    "comprehensive",
)


def _assess_complexity(
    goal: str,
    n_tools: int,
    *,
    simple_word_threshold: int = 15,
) -> float:
    """Heuristic complexity assessment. Returns confidence that direct execution suffices.

    This is a crude heuristic, not a calibrated model. It exists to avoid
    paying LLM latency tax on obviously simple goals. For ambiguous cases
    it returns ~0.5, letting the threshold parameter decide.

    Args:
        goal: The user's goal text.
        n_tools: Number of available tools.
        simple_word_threshold: Goals with fewer words are assessed as simple.
            Configurable via KaosAgentSettings.simple_goal_word_threshold.

    Returns:
        Score in [0.1, 0.95]. Higher = more likely to be simple.
    """
    words = goal.lower().split()
    n_words = len(words)
    goal_lower = goal.lower()

    score = 0.8 if n_words <= simple_word_threshold else 0.5

    # Complex language patterns reduce confidence
    for pattern in _COMPLEX_PATTERNS:
        if pattern in goal_lower:
            score -= 0.15
            break

    # Multiple sentences suggest complexity
    if goal.count(".") >= 2 or goal.count(",") >= 3:
        score -= 0.1

    # Few tools available → simpler execution
    if n_tools <= 2:
        score += 0.1

    return max(0.1, min(0.95, score))


async def execute_adaptive(
    goal: str,
    *,
    tools: dict[str, Tool] | None = None,
    tool_descriptions: dict[str, str] | None = None,
    context: str = "",
    prior_failures: str = "",
    model: str = DEFAULT_MODEL,
    budget: PlanBudget | None = None,
    complexity_threshold: float = 0.6,
    simple_word_threshold: int = 15,
    max_steps: int = 8,
    parallel: bool = True,
    confidence_threshold: float | None = None,
    deepen_threshold: float | None = None,
) -> ComposeResult:
    """Execute a goal adaptively — simple goals go direct, complex goals decompose.

    The complexity assessment is heuristic (no LLM call) to avoid paying
    latency tax on simple goals. If direct execution fails, falls back
    to decomposition automatically.
    """
    if budget is None:
        budget = PlanBudget()

    n_tools = len(tool_descriptions) if tool_descriptions else 0
    confidence = _assess_complexity(goal, n_tools, simple_word_threshold=simple_word_threshold)

    logger.debug(
        "adaptive: goal complexity=%.2f (threshold=%.2f) → %s",
        confidence,
        complexity_threshold,
        "direct" if confidence >= complexity_threshold else "decompose",
    )

    if confidence >= complexity_threshold:
        # Simple → direct execution
        result = await execute_direct(
            goal,
            tools=tools,
            tool_descriptions=tool_descriptions,
            context=context,
            model=model,
            budget=budget,
        )

        # If direct failed, fall back to decomposition
        if result.stop_reason != StopReason.SUCCESS and budget.should_stop() is None:
            logger.debug("adaptive: direct failed, falling back to decompose")
            return await execute_decompose(
                goal,
                tools=tools,
                tool_descriptions=tool_descriptions,
                context=context,
                prior_failures=prior_failures,
                model=model,
                budget=budget,
                max_steps=max_steps,
                parallel=parallel,
                confidence_threshold=confidence_threshold,
                deepen_threshold=deepen_threshold,
            )
        return result

    # Complex → decompose directly
    return await execute_decompose(
        goal,
        tools=tools,
        tool_descriptions=tool_descriptions,
        context=context,
        prior_failures=prior_failures,
        model=model,
        budget=budget,
        max_steps=max_steps,
        parallel=parallel,
        confidence_threshold=confidence_threshold,
        deepen_threshold=deepen_threshold,
    )
