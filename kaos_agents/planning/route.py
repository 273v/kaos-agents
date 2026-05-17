"""Route primitive — deterministic control flow decisions.

Pure Python logic. No LLM calls. This is the if/elif/else that decides:
continue executing? Replan? Deepen? Escalate? Stop?

Route is the only primitive that should be hand-coded logic, not
LLM-generated. It uses outputs from Evaluate but doesn't call the LLM itself.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_agents.settings import KaosAgentSettings as _Settings
from kaos_agents.types.plan import (
    Decision,
    Judgment,
    PlanBudget,
    RouteResult,
)

logger = get_logger(__name__)

# Defaults derived from settings to avoid duplication.
# In production, callers should pass settings values explicitly.
_DEFAULT_CONFIDENCE_THRESHOLD: float = _Settings.model_fields["confidence_threshold"].default


def route(
    judgment: Judgment,
    budget: PlanBudget,
    *,
    replan_count: int = 0,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    step_id: str | None = None,
) -> RouteResult:
    """Make a control flow decision based on judgment and budget.

    Decision priority (checked in order):
    1. Budget exceeded → STOP_BUDGET
    2. Max replans exceeded → STOP_FAILURE (circuit breaker)
    3. Step succeeded with high confidence → CONTINUE
    4. Step failed → REPLAN
    5. Step succeeded with low confidence → REPLAN
    6. Default → CONTINUE

    Args:
        judgment: The Evaluate result for the current step.
        budget: Current resource consumption and limits.
        replan_count: How many times we've replanned (for circuit breaker).
        confidence_threshold: Below this, consider replanning.
        step_id: Which step triggered this decision (for logging).

    Returns:
        RouteResult with decision and rationale.

    Note:
        The DEEPEN decision and its ``deepen_threshold`` argument were
        removed in kaos-agents 0.1.0a9 — both branches collapsed to the
        same ``REPLAN`` behavior in ``compose`` (see compose.py:178 on
        ``main`` pre-0.1.0a9: ``if decision.decision in (Decision.REPLAN,
        Decision.DEEPEN)``), so DEEPEN was a confusing alias rather than
        a distinct control-flow path. A future ADaPT-style implementation
        that substep-decomposes the failed step (via
        ``PlanGraph.insert_subplan``) can re-introduce the distinction
        with non-trivial semantics.
    """
    # Validate confidence is in [0, 1] — catch upstream bugs early
    if not 0.0 <= judgment.confidence <= 1.0:
        logger.warning(
            "route: judgment.confidence=%.2f out of [0,1], clamping", judgment.confidence
        )

    # 1. Budget check — always first
    stop_reason = budget.should_stop()
    if stop_reason is not None:
        result = RouteResult(
            decision=Decision.STOP_BUDGET,
            reason=f"Budget limit reached: {stop_reason.value}",
            step_id=step_id,
        )
        logger.debug("route: %s — %s", result.decision.value, result.reason)
        return result

    # 2. Circuit breaker — prevent unbounded replanning
    if replan_count >= budget.max_replans:
        result = RouteResult(
            decision=Decision.STOP_FAILURE,
            reason=f"Max replans ({budget.max_replans}) exceeded. "
            f"Plan cannot converge. Consider breaking the goal into smaller tasks.",
            step_id=step_id,
        )
        logger.debug("route: %s — %s", result.decision.value, result.reason)
        return result

    # 3. Step succeeded with good confidence → continue
    if judgment.matched and judgment.confidence >= confidence_threshold:
        result = RouteResult(
            decision=Decision.CONTINUE,
            reason=f"Step succeeded (confidence={judgment.confidence:.2f}): {judgment.reasoning}",
            step_id=step_id,
        )
        logger.debug("route: %s — %s", result.decision.value, result.reason)
        return result

    # 4. Step explicitly failed → replan
    if not judgment.matched:
        result = RouteResult(
            decision=Decision.REPLAN,
            reason=f"Step failed: {judgment.reasoning}",
            step_id=step_id,
        )
        logger.debug("route: %s — %s", result.decision.value, result.reason)
        return result

    # 5. Step succeeded but low confidence → replan
    if judgment.confidence < confidence_threshold:
        result = RouteResult(
            decision=Decision.REPLAN,
            reason=f"Low confidence ({judgment.confidence:.2f}): "
            f"{judgment.reasoning}. Replanning with updated context.",
            step_id=step_id,
        )
        logger.debug("route: %s — %s", result.decision.value, result.reason)
        return result

    # 6. Default — continue
    result = RouteResult(
        decision=Decision.CONTINUE,
        reason=f"Proceeding (confidence={judgment.confidence:.2f})",
        step_id=step_id,
    )
    logger.debug("route: %s — %s", result.decision.value, result.reason)
    return result
