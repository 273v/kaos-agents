"""``RefineDeliverable`` — produce → critique → iterate at the deliverable layer.

The Refine pattern lifted from the Program layer (single LLM call) to
the Deliverable layer (full agent task → structured artifact). Built
on the same :class:`~kaos_llm_core.programs.loop_runner.LoopRunner`
kernel that powers ``Refine`` and ``ReAct`` — no new iteration logic,
no parallel trace tree, no parallel cost machinery.

Invariants (from `docs/design/reflexive-agent-v2.md`):

1. **Best-of-iterations selection.** The loop returns the
   highest-``weighted_pass_rate`` deliverable across all iterations.
   Iterating never makes things worse.
2. **Patience-bounded early stop.** Stops when iteration N's gain
   over the prior best is below ``min_gain_per_iter`` *and* the
   patience window has expired. Avoids paying for iterations that
   are statistically indistinguishable from noise.
3. **Reflection-driven feedback.** Failures are formatted via
   :func:`format_gap_feedback` and injected into the producer's next
   iteration via the standard REFLECTION memory channel. Producers
   that don't read REFLECTION won't benefit, but the channel is
   already wired in `plan_execute.py`.
4. **No same-family default.** The critic carries its own model
   choice; callers that want the cross-family invariant pass
   ``producer_model`` to :class:`RubricDeliverableCritic` to enable
   the warning at construction.

Producer interface
------------------
The producer is a callable: ``async def() -> Deliverable``. Most
callers will lambda-wrap an agent + composer pipeline:

    async def produce() -> Deliverable:
        await runner.turn(task, session_id=...)
        return (await compose_hybrid(memory, structure)).deliverable

That contract keeps :class:`RefineDeliverable` agnostic of how the
deliverable is built. It just iterates: produce → score → if-failed-
inject-feedback-and-loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger
from kaos_llm_core.programs.loop_runner import (
    LoopConfig,
    LoopRunner,
    StepOutcome,
    StopReason,
)

from kaos_agents.output.feedback import FeedbackStrategy, format_gap_feedback
from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.output.critic import (
        DeliverableCritic,
        DeliverableVerdict,
        RubricCriterion,
    )
    from kaos_agents.output.types import Deliverable

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Per-iteration trace: the deliverable + verdict + feedback.

    Stored in :attr:`RefineDeliverableResult.history` for audit. The
    feedback is the markdown text actually injected into REFLECTION
    for the *next* iteration; the final iteration's feedback is the
    "(no further iterations)" placeholder.
    """

    iteration: int
    deliverable: Deliverable
    verdict: DeliverableVerdict
    feedback: str = ""


@dataclass(frozen=True, slots=True)
class RefineDeliverableResult:
    """Aggregate result of :class:`RefineDeliverable.run`.

    Attributes:
        best_deliverable: Highest-scoring deliverable across all
            iterations. Never worse than the producer's first attempt.
        best_iteration: 1-indexed iteration that produced
            ``best_deliverable``. ``1`` means the first attempt won
            and the loop didn't improve on it.
        history: Per-iteration records in order.
        stop_reason: Why the loop terminated:
            ``"completed"`` (max_iterations exhausted),
            ``"early_stop"`` (patience-bounded gain below
            threshold), ``"perfect"`` (every criterion passed),
            ``"error"`` (exception in step).
        total_critic_cost_usd: Sum of all critic cost across
            iterations.
    """

    best_deliverable: Deliverable
    best_iteration: int
    history: tuple[IterationRecord, ...]
    stop_reason: str
    total_critic_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# State + step
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RefineState:
    """Per-run mutable state held by the LoopRunner.

    The runner itself is generic over the State shape (see
    :class:`LoopRunner` in kaos-llm-core). This dataclass holds
    everything one Refine run accumulates: last-iteration's verdict,
    current best, patience counter.
    """

    best_deliverable: Deliverable | None = None
    best_pass_rate: float = -1.0
    best_iteration: int = 0
    patience_remaining: int = 0
    total_cost: float = 0.0
    perfect: bool = False


# ---------------------------------------------------------------------------
# RefineDeliverable
# ---------------------------------------------------------------------------


class RefineDeliverable:
    """Refine a Deliverable against a Rubric using a Critic.

    Constructor args:

    * ``producer`` — async callable returning a :class:`Deliverable`.
      The agent's compose pipeline goes here.
    * ``critic`` — a :class:`DeliverableCritic` that scores against
      ``rubric``.
    * ``rubric`` — sequence of :class:`RubricCriterion`.
    * ``memory`` — session memory whose REFLECTION section receives
      the per-iteration gap feedback. Producers that read REFLECTION
      via the existing recall() machinery see the feedback on next
      iteration.
    * ``max_iterations`` — hard cap.
    * ``patience`` — number of iterations to wait without improvement
      before early-stop.
    * ``min_gain_per_iter`` — minimum weighted_pass_rate delta over
      best-so-far that counts as an improvement (resets patience).
    * ``feedback_strategy`` — passed to :func:`format_gap_feedback`.
    * ``min_pass_rate_for_perfect`` — if a verdict's
      ``weighted_pass_rate >= this``, the loop terminates with
      ``stop_reason="perfect"``. Default 1.0 (every criterion passed).

    The :meth:`run` entry point delegates iteration to a
    :class:`LoopRunner` — same kernel ReAct + the upstream Refine use.
    """

    def __init__(
        self,
        *,
        producer: Callable[[], Awaitable[Deliverable]],
        critic: DeliverableCritic,
        rubric: Sequence[RubricCriterion],
        memory: SessionMemory,
        max_iterations: int = 3,
        patience: int = 1,
        min_gain_per_iter: float = 0.05,
        feedback_strategy: FeedbackStrategy = "gap_list",
        min_pass_rate_for_perfect: float = 1.0,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._producer = producer
        self._critic = critic
        self._rubric = tuple(rubric)
        self._memory = memory
        self._max_iterations = max_iterations
        self._patience = max(0, patience)
        self._min_gain = max(0.0, min_gain_per_iter)
        self._feedback_strategy = feedback_strategy
        self._min_perfect = min_pass_rate_for_perfect

    async def run(self) -> RefineDeliverableResult:
        """Execute the refine loop.

        Returns the typed :class:`RefineDeliverableResult` with the
        best-of-iterations deliverable and the per-iteration history.
        """
        history: list[IterationRecord] = []

        async def step(iteration: int, state: _RefineState) -> StepOutcome[IterationRecord]:
            # 1-indexed iteration number for IterationRecord display
            n = iteration + 1
            deliverable = await self._producer()
            verdict = await self._critic.score(deliverable, self._rubric)
            state.total_cost += verdict.total_cost_usd

            # Track best-of-iterations.
            improved = verdict.weighted_pass_rate > state.best_pass_rate + self._min_gain
            if verdict.weighted_pass_rate > state.best_pass_rate:
                state.best_deliverable = deliverable
                state.best_pass_rate = verdict.weighted_pass_rate
                state.best_iteration = n

            # Patience handling.
            stop_reason: str | None = None
            if verdict.weighted_pass_rate >= self._min_perfect:
                state.perfect = True
                stop_reason = "perfect"
            elif improved:
                state.patience_remaining = self._patience
            else:
                if state.patience_remaining > 0:
                    state.patience_remaining -= 1
                else:
                    stop_reason = "early_stop"

            # Inject feedback for next iteration (skip on terminal step).
            feedback = ""
            if stop_reason is None and n < self._max_iterations:
                feedback = format_gap_feedback(verdict, strategy=self._feedback_strategy)
                self._memory.add(
                    MemoryType.REFLECTION,
                    feedback,
                    metadata={
                        "iteration": n,
                        "weighted_pass_rate": verdict.weighted_pass_rate,
                        "n_failed": verdict.n_failed,
                    },
                )

            record = IterationRecord(
                iteration=n,
                deliverable=deliverable,
                verdict=verdict,
                feedback=feedback,
            )
            history.append(record)
            logger.debug(
                "RefineDeliverable.step n=%d wpr=%.3f n_failed=%d patience=%d cost=%.4f",
                n,
                verdict.weighted_pass_rate,
                verdict.n_failed,
                state.patience_remaining,
                verdict.total_cost_usd,
            )
            return StepOutcome(record=record, stop_reason=stop_reason)

        def make_state() -> _RefineState:
            return _RefineState(patience_remaining=self._patience)

        def build_result(
            state: _RefineState, _records: list[IterationRecord]
        ) -> RefineDeliverableResult:
            best = state.best_deliverable
            if best is None:  # pragma: no cover — only on max_iterations=0 (impossible)
                raise RuntimeError("RefineDeliverable: no deliverable produced")
            stop = "perfect" if state.perfect else StopReason.COMPLETED.value
            return RefineDeliverableResult(
                best_deliverable=best,
                best_iteration=state.best_iteration,
                history=tuple(history),
                stop_reason=stop,
                total_critic_cost_usd=state.total_cost,
            )

        runner: LoopRunner[_RefineState, IterationRecord, RefineDeliverableResult] = LoopRunner(
            config=LoopConfig(max_iterations=self._max_iterations),
            make_state=make_state,
            step=step,
            build_result=build_result,
        )
        loop_result = await runner.run()

        # The LoopRunner classifies StopReason via its own taxonomy
        # ("completed", "early_stop", "error"). Promote the
        # program-specific "perfect" / "early_stop" semantics over the
        # generic "completed" when our step returned them.
        result = loop_result.result
        program_stop = loop_result.stop_reason
        if program_stop != StopReason.COMPLETED.value:
            object.__setattr__(result, "stop_reason", program_stop)
        return result


__all__ = [
    "IterationRecord",
    "RefineDeliverable",
    "RefineDeliverableResult",
]
