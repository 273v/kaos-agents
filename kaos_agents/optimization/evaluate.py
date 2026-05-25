"""evaluate_agent — run an :class:`AgentLoop` across a labeled dataset.

The agent-layer counterpart to
:func:`kaos_llm_core.optimization.evaluation.evaluate`. Each example
becomes one full turn through the loop. Per-example metric scoring
follows the kaos-llm-core convention: ``metric(prediction,
gold) -> float in [0, 1]``.

Composes with the rest of the kaos-llm-core optimizer machinery:

* The result type is :class:`AgentEvalResult`, a typed extension of
  :class:`kaos_llm_core.optimization.evaluation.EvalResult` that adds
  agent-specific aggregates (total cost, total tokens, average
  escalations per turn).
* When called inside a
  :class:`~kaos_llm_core.optimization.trial_runner.TrialRunner` scope,
  every turn's TurnSummary is published to the trial via the Phase 5.A
  :class:`CostTrackingHook` wiring — no manual plumbing.

Example::

    from kaos_agents.optimization import evaluate_agent, AgentExample

    examples = [
        AgentExample(
            inputs={"message": "What is 2+2?"},
            outputs={"contains": "4"},
        ),
        AgentExample(
            inputs={"message": "Capital of France?"},
            outputs={"contains": "Paris"},
        ),
    ]

    def contains_metric(prediction, gold):
        return 1.0 if str(gold.get("contains", "")).lower() in (
            prediction.output or ""
        ).lower() else 0.0

    result = await evaluate_agent(
        loop=my_agent_loop,
        examples=examples,
        metric=contains_metric,
    )
    print(f"score: {result.score:.2f}  cost: ${result.total_cost_usd:.4f}")
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core.optimization.evaluation import EvalResult, ExampleResult
from kaos_llm_core.types import Example as LLMExample

from kaos_agents.core.invocation import TurnInvocation
from kaos_agents.triggers.base import Trigger

logger = get_logger(__name__)


# A metric: (TurnInvocation, gold dict) -> score in [0, 1].
AgentMetricFn = Callable[[TurnInvocation, dict[str, Any]], float]


@dataclass(frozen=True, slots=True)
class AgentExample:
    """One labeled (input, gold) pair for ``evaluate_agent``.

    ``inputs`` keys recognised by Phase 5.B:
      - ``message`` (str): becomes the trigger's payload['message'].
      - ``session_id`` (str, optional): the trigger's session/source id.
        Defaults to ``"eval_<index>"``.

    ``outputs`` is a free-form gold dict; the metric receives it as-is.
    Convention-friendly keys (``expected``, ``contains``, ``equals``,
    ``score``) but not enforced — callers control the metric.
    """

    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_example(self) -> LLMExample:
        """Project to a kaos-llm-core ``Example`` for compatibility."""
        return LLMExample(
            inputs=self.inputs,
            outputs=self.outputs,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class AgentEvalResult:
    """Aggregated evaluation result for an :func:`evaluate_agent` run.

    Mirrors :class:`kaos_llm_core.optimization.evaluation.EvalResult`
    field-for-field but adds agent-specific aggregates:

      total_cost_usd: sum of TurnInvocation.cost_usd across examples.
      total_tokens: sum of TurnInvocation.usage.total_tokens.
      total_escalations: sum of len(TurnInvocation.escalations).
    """

    score: float
    n_total: int
    n_correct: int
    n_errors: int
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_escalations: int = 0
    per_example: list[ExampleResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return 0.0 if self.n_total == 0 else self.n_correct / self.n_total

    @property
    def error_rate(self) -> float:
        return 0.0 if self.n_total == 0 else self.n_errors / self.n_total

    @property
    def average_cost_usd(self) -> float:
        return 0.0 if self.n_total == 0 else self.total_cost_usd / self.n_total

    def failures(self) -> list[ExampleResult]:
        """Examples that scored < 1.0 without raising."""
        return [r for r in self.per_example if r.score < 1.0 and r.error is None]

    def errors(self) -> list[ExampleResult]:
        """Examples that raised exceptions."""
        return [r for r in self.per_example if r.error is not None]

    def to_eval_result(self) -> EvalResult:
        """Project to the kaos-llm-core EvalResult for downstream tooling."""
        return EvalResult(
            score=self.score,
            n_total=self.n_total,
            n_correct=self.n_correct,
            n_errors=self.n_errors,
            per_example=list(self.per_example),
        )


async def evaluate_agent(
    *,
    loop: Any,
    examples: list[AgentExample],
    metric: AgentMetricFn,
    concurrency: int = 1,
    progress: bool = False,
) -> AgentEvalResult:
    """Run an :class:`AgentLoop` across labeled examples.

    Args:
        loop: A configured :class:`AgentLoop`. Pass an instance — the
            harness reuses it across examples (no re-construction per
            example, which would defeat any cached state on the loop).
        examples: List of :class:`AgentExample` rows.
        metric: A callable ``(TurnInvocation, gold dict) -> float in [0, 1]``.
            Score >= 1.0 counts as correct.
        concurrency: Max parallel example evaluations. Default 1
            (sequential — safest for shared state). Higher values trade
            isolation for throughput.
        progress: When True, log per-example progress.

    Returns:
        :class:`AgentEvalResult` with aggregate score, cost, tokens,
        escalations, and per-example results.

    Cost attribution: when called inside a
    ``TrialRunner.trial(...)`` scope, each turn's TurnSummary publishes
    cost+tokens to the active trial via the Phase 5.A
    :class:`CostTrackingHook` integration. The harness reads
    ``invocation.cost_usd`` directly so the result's ``total_cost_usd``
    is correct whether or not a trial is active.
    """
    if not examples:
        return AgentEvalResult(score=0.0, n_total=0, n_correct=0, n_errors=0)

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(idx: int, example: AgentExample) -> ExampleResult:
        async with semaphore:
            return await _evaluate_example(idx, example, loop, metric, progress)

    # A+ Theme C — partial-success eval. ``_evaluate_example`` already
    # catches BaseException and returns an ``ExampleResult(error=...)``,
    # so this gather was safe in the happy path. But ``return_exceptions=
    # False`` meant any future defensive layer (semaphore acquire,
    # metric setup, hook dispatch) that raised BEFORE the inner
    # try/except would crash the entire eval run and lose every
    # partial result. Flip to ``True`` and convert raw exceptions into
    # ``ExampleResult(error=...)`` so one bad example never costs the
    # whole batch.
    raw_results = await asyncio.gather(
        *(_run_one(i, ex) for i, ex in enumerate(examples)),
        return_exceptions=True,
    )
    per_example: list[ExampleResult] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, BaseException):
            logger.warning("eval[%d] raised outside the inner try/except: %s", i, r)
            from kaos_llm_core.types import Example as _LLMExample

            per_example.append(
                ExampleResult(
                    example=_LLMExample(
                        inputs=dict(examples[i].inputs),
                        outputs=dict(examples[i].outputs),
                    ),
                    prediction=None,
                    score=0.0,
                    error=f"runner error: {r}",
                    error_class=type(r).__name__,
                    trace=None,
                )
            )
        else:
            per_example.append(r)

    n_correct = sum(1 for r in per_example if r.score >= 1.0 and r.error is None)
    n_errors = sum(1 for r in per_example if r.error is not None)
    sum_score = sum(r.score for r in per_example if r.error is None)
    n_total = len(per_example)

    # Walk per_example.trace (we stash the TurnInvocation there) for
    # cost / token / escalation aggregates.
    total_cost = 0.0
    total_tokens = 0
    total_escalations = 0
    for r in per_example:
        invocation = r.trace
        if isinstance(invocation, TurnInvocation):
            total_cost += float(invocation.cost_usd)
            total_tokens += int(invocation.usage.total_tokens)
            total_escalations += len(invocation.escalations)

    score = sum_score / n_total if n_total > 0 else 0.0
    return AgentEvalResult(
        score=score,
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_errors,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        total_escalations=total_escalations,
        per_example=list(per_example),
    )


async def _evaluate_example(
    idx: int,
    example: AgentExample,
    loop: Any,
    metric: AgentMetricFn,
    progress: bool,
) -> ExampleResult:
    """Run one example through the loop and apply the metric."""
    message = str(example.inputs.get("message", ""))
    session_id = str(example.inputs.get("session_id", f"eval_{idx}"))
    trigger = Trigger.mcp(message=message, session_id=session_id)

    llm_example = example.to_llm_example()
    try:
        invocation = await loop.invoke(trigger=trigger)
    except BaseException as exc:
        # kaos-llm-core ExampleResult convention: error message + class.
        if progress:
            logger.warning("eval[%d] failed: %s", idx, exc)
        return ExampleResult(
            example=llm_example,
            prediction=None,
            score=0.0,
            error=str(exc),
            error_class=type(exc).__name__,
            trace=getattr(exc, "turn_invocation", None),
        )

    try:
        score = float(metric(invocation, dict(example.outputs)))
    except BaseException as exc:
        score = 0.0
        if progress:
            logger.warning("eval[%d] metric raised: %s", idx, exc)
        return ExampleResult(
            example=llm_example,
            prediction=invocation,
            score=score,
            error=f"metric error: {exc}",
            error_class=type(exc).__name__,
            trace=invocation,
        )

    if progress:
        logger.info(
            "eval[%d] score=%.2f cost=$%.4f tokens=%d",
            idx,
            score,
            invocation.cost_usd,
            invocation.usage.total_tokens,
        )

    return ExampleResult(
        example=llm_example,
        prediction=invocation,
        score=max(0.0, min(1.0, score)),
        error=None,
        trace=invocation,
    )


__all__ = [
    "AgentEvalResult",
    "AgentExample",
    "AgentMetricFn",
    "evaluate_agent",
]
