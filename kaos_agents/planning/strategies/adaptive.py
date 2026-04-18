"""Adaptive strategy (ADaPT-style) — confidence-gated depth selection.

The metastrategy. Evaluates goal complexity, then routes:
- Simple goals (high confidence) → direct execution
- Complex goals (low confidence) → hierarchical decomposition

Composes: evaluate(complexity) → route(direct or decompose)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents._constants import COMPLEX_MIN_COMMAS, COMPLEX_MIN_PERIODS, FEW_TOOLS_THRESHOLD
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

# Complexity score adjustments — documented here so they can be tuned.
_PENALTY_COMPLEX_PATTERN = 0.15  # Deduction for multi-step language
_PENALTY_MULTI_SENTENCE = 0.1  # Deduction for >= 2 sentences or >= 3 commas
_BONUS_FEW_TOOLS = 0.1  # Bonus when <= 2 tools available

# Document-count thresholds and their confidence penalties
_DOC_THRESHOLDS = ((100, 0.25), (20, 0.15), (5, 0.05))
_PENALTY_LOW_INTENT_CONFIDENCE = 0.1  # Deduction when intent confidence < 0.5
_LOW_INTENT_CONFIDENCE_FLOOR = 0.5  # Below this, penalty applies

_SCORE_SIMPLE_BASE = 0.8  # Starting score for short goals
_SCORE_COMPLEX_BASE = 0.5  # Starting score for long goals
_SCORE_MIN = 0.1
_SCORE_MAX = 0.95


def _assess_complexity(
    goal: str,
    n_tools: int,
    *,
    simple_word_threshold: int = 15,
    n_documents: int = 0,
    intent_confidence: float | None = None,
) -> float:
    """Heuristic complexity assessment. Returns confidence that direct execution suffices.

    This is a crude heuristic, not a calibrated model. It exists to avoid
    paying LLM latency tax on obviously simple goals. For ambiguous cases
    it returns ~0.5, letting the threshold parameter decide.

    Args:
        goal: The user's goal text.
        n_tools: Number of available tools.
        simple_word_threshold: Goals with fewer words are assessed as simple.
        n_documents: Number of documents in the DOCUMENTS memory section.
            Large corpora strongly suggest multi-step planning is needed.
        intent_confidence: Confidence from intent classification (0-1).
            Low confidence suggests ambiguity that benefits from planning.

    Returns:
        Score in [0.1, 0.95]. Higher = more likely to be simple.
    """
    words = goal.lower().split()
    n_words = len(words)
    goal_lower = goal.lower()

    score = _SCORE_SIMPLE_BASE if n_words <= simple_word_threshold else _SCORE_COMPLEX_BASE

    # Complex language patterns reduce confidence
    for pattern in _COMPLEX_PATTERNS:
        if pattern in goal_lower:
            score -= _PENALTY_COMPLEX_PATTERN
            break

    # Multiple sentences suggest complexity
    if goal.count(".") >= COMPLEX_MIN_PERIODS or goal.count(",") >= COMPLEX_MIN_COMMAS:
        score -= _PENALTY_MULTI_SENTENCE

    # Few tools available → simpler execution
    if n_tools <= FEW_TOOLS_THRESHOLD:
        score += _BONUS_FEW_TOOLS

    # Large corpus strongly suggests multi-step planning
    for threshold, penalty in _DOC_THRESHOLDS:
        if n_documents >= threshold:
            score -= penalty
            break

    # Low intent confidence suggests ambiguity → more planning needed
    if intent_confidence is not None and intent_confidence < _LOW_INTENT_CONFIDENCE_FLOOR:
        score -= _PENALTY_LOW_INTENT_CONFIDENCE

    return max(_SCORE_MIN, min(_SCORE_MAX, score))


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
    tool_timeout_seconds: float = 60.0,
    n_documents: int = 0,
    intent_confidence: float | None = None,
) -> ComposeResult:
    """Execute a goal adaptively — simple goals go direct, complex goals decompose.

    The complexity assessment is heuristic (no LLM call) to avoid paying
    latency tax on simple goals. If direct execution fails, falls back
    to decomposition automatically.
    """
    if budget is None:
        budget = PlanBudget()

    n_tools = len(tool_descriptions) if tool_descriptions else 0
    confidence = _assess_complexity(
        goal,
        n_tools,
        simple_word_threshold=simple_word_threshold,
        n_documents=n_documents,
        intent_confidence=intent_confidence,
    )

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
            tool_timeout_seconds=tool_timeout_seconds,
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
                tool_timeout_seconds=tool_timeout_seconds,
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
        tool_timeout_seconds=tool_timeout_seconds,
    )
