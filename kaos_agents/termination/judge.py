"""TerminationJudge — composes 5 axes into a single Decision.

Phase 4.B (paper §Q7): the agent's "am I done?" oracle. Five axes:

  1. **Budget** — cost / iterations / wall-clock caps.
  2. **Failure** — ``RunError`` / ``EvidenceInsufficient`` /
     ``GroundingRefusalTriggered`` events seen since the last
     checkpoint.
  3. **Loop** — :class:`~kaos_agents.termination.loop_detect.LoopDetector`
     trips on signature similarity (Resolved Decision #7).
  4. **Quality** — kaos-llm-core
     :class:`~kaos_llm_core.programs.judge.Judge` against
     ``intent.goal.success_criteria``. Optional and gated on a
     ``judge`` constructor kwarg — when ``None``, the quality axis
     is skipped.
  5. **Graceful degradation** — partial result acceptable when budget
     bound; delegated to
     :class:`~kaos_agents.termination.degrade.DegradationPolicy`.

The TerminationJudge does NOT terminate the turn — it returns a
:class:`~kaos_agents.termination.types.Decision`; the AgentLoop's
step 4 reads the Decision and acts on it. This separation lets the
judge be invoked from hooks, tests, and replan branches without any
loop-side coupling.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_core.programs.base import Program

from kaos_agents.events.lifecycle import RunError
from kaos_agents.events.research import EvidenceInsufficient, GroundingRefusalTriggered
from kaos_agents.termination.degrade import DegradationPolicy
from kaos_agents.termination.loop_detect import LoopDetector
from kaos_agents.termination.types import Decision, DecisionKind, SuccessCriteria
from kaos_agents.types.usage import InvocationUsage


class TerminationJudge(Program):
    """Composite termination decision Program.

    Constructor kwargs:

      max_cost_usd: cost cap (default ``None`` = unlimited).
      max_iterations: iteration cap (default 10).
      max_wall_clock_seconds: time cap (default ``None``).
      judge: optional kaos-llm-core ``Judge`` for quality scoring
        (``None`` → skip quality axis).
      min_quality: judge score threshold (default 0.7).
      loop_detector: optional preconfigured
        :class:`~kaos_agents.termination.loop_detect.LoopDetector`
        (default ``LoopDetector()``).
      degradation_policy: optional
        :class:`~kaos_agents.termination.degrade.DegradationPolicy`.

    The AgentLoop calls
    ``TerminationJudge.invoke(intent=..., usage=..., events=...,
    iteration=..., partial_text=..., success_criteria=...,
    step_signature=...)`` after each planner pass. Returns a
    :class:`Decision`. ``forward()`` is the typed entry point; the
    base ``Program.invoke()`` wraps it with trace + usage rollup.
    """

    def __init__(
        self,
        *,
        max_cost_usd: float | None = None,
        max_iterations: int = 10,
        max_wall_clock_seconds: float | None = None,
        judge: Any | None = None,
        min_quality: float = 0.7,
        loop_detector: LoopDetector | None = None,
        degradation_policy: DegradationPolicy | None = None,
    ) -> None:
        super().__init__()
        self._max_cost = max_cost_usd
        self._max_iter = int(max_iterations)
        self._max_wall = max_wall_clock_seconds
        self._judge = judge
        self._min_quality = float(min_quality)
        self._loop_detector = loop_detector or LoopDetector()
        self._degrade = degradation_policy or DegradationPolicy()

    @property
    def loop_detector(self) -> LoopDetector:
        return self._loop_detector

    @property
    def degradation_policy(self) -> DegradationPolicy:
        return self._degrade

    async def forward(self, **kwargs: Any) -> Decision:
        """The 5-axis evaluation.

        Axis order is intentional:

        1. Budget first — if we're out of money/time/iterations we
           don't need to ask anything else; we either degrade or
           escalate.
        2. Failure next — terminal events are short-circuit signals
           the planner should not paper over.
        3. Loop — cheap to check and a strong signal that the planner
           is stuck.
        4. Quality (optional) — only when a judge is wired AND we
           have a partial to score. Most expensive axis (LLM call).
        5. Default — complete if we have output, otherwise replan.
        """
        usage: InvocationUsage = kwargs.get("usage") or InvocationUsage()
        iteration: int = int(kwargs.get("iteration", 1))
        wall_seconds: float = float(kwargs.get("wall_seconds", 0.0))
        partial_text: str = str(kwargs.get("partial_text", "") or "")

        # 1. Budget axis -----------------------------------------------
        if self._max_cost is not None and usage.cost_usd >= self._max_cost:
            return self._maybe_degrade(
                DecisionKind.BUDGET_EXCEEDED,
                f"cost ${usage.cost_usd:.4f} >= cap ${self._max_cost:.4f}",
                partial_text,
            )
        if iteration >= self._max_iter:
            return self._maybe_degrade(
                DecisionKind.BUDGET_EXCEEDED,
                f"iterations {iteration} >= cap {self._max_iter}",
                partial_text,
            )
        if self._max_wall is not None and wall_seconds >= self._max_wall:
            return self._maybe_degrade(
                DecisionKind.BUDGET_EXCEEDED,
                f"wall clock {wall_seconds:.1f}s >= cap {self._max_wall:.1f}s",
                partial_text,
            )

        # 2. Failure axis ---------------------------------------------
        events: tuple[Any, ...] = tuple(kwargs.get("events", ()) or ())
        # Bound the scan to the most-recent 20 events to keep the cost
        # of a long-running turn's checkpoints predictable.
        for event in reversed(events[-20:]):
            if isinstance(event, RunError):
                return Decision(
                    kind=DecisionKind.FAILURE,
                    is_complete=True,
                    feedback=str(getattr(event, "message", "") or ""),
                )
            if isinstance(event, (EvidenceInsufficient, GroundingRefusalTriggered)):
                return Decision(
                    kind=DecisionKind.FAILURE,
                    is_complete=True,
                    should_escalate=True,
                    feedback=str(
                        getattr(event, "reason", "") or "evidence insufficient",
                    ),
                )

        # 3. Loop axis -------------------------------------------------
        signature = kwargs.get("step_signature")
        if signature is not None:
            loop_result = self._loop_detector.observe(str(signature))
            if loop_result.detected:
                return Decision(
                    kind=DecisionKind.LOOP_DETECTED,
                    is_complete=True,
                    should_escalate=True,
                    feedback=loop_result.reason,
                )

        # 4. Quality axis (optional) ----------------------------------
        if self._judge is not None and partial_text:
            criteria_obj = kwargs.get("success_criteria")
            criteria: SuccessCriteria | None = (
                criteria_obj if isinstance(criteria_obj, SuccessCriteria) else None
            )
            quality_score = await self._score_quality(partial_text, criteria)
            if quality_score < self._min_quality:
                return self._maybe_degrade(
                    DecisionKind.QUALITY_FAILED,
                    f"quality {quality_score:.2f} < {self._min_quality:.2f}",
                    partial_text,
                )

        # 5. Default ---------------------------------------------------
        if partial_text:
            return Decision(kind=DecisionKind.COMPLETE, is_complete=True)
        return Decision(
            kind=DecisionKind.INCOMPLETE,
            is_complete=False,
            allows_replan=True,
            feedback="no result produced; replan",
        )

    # ---- helpers ---------------------------------------------------

    def _maybe_degrade(
        self,
        kind: DecisionKind,
        reason: str,
        partial: str,
    ) -> Decision:
        outcome = self._degrade.evaluate(kind=kind.value, partial_text=partial)
        if outcome.accept:
            return Decision(
                kind=DecisionKind.DEGRADED,
                is_complete=True,
                feedback=outcome.reason,
                partial_result=outcome.partial,
            )
        return Decision(
            kind=kind,
            is_complete=True,
            should_escalate=True,
            feedback=reason,
        )

    async def _score_quality(
        self,
        text: str,
        criteria: SuccessCriteria | None,
    ) -> float:
        """Score the partial via the configured Judge.

        Returns ``1.0`` when there are no criteria to evaluate (no
        Judge call is made) and ``0.5`` (neutral) on any Judge
        failure — Phase 4.B is best-effort. The AgentLoop re-tries
        with a different planner if quality is consistently neutral,
        so a flaky Judge doesn't trap the loop.

        The Judge interface is duck-typed: we call ``await
        judge.invoke(output=text, criteria=...)`` and read the score.
        kaos-llm-core ``Judge.invoke()`` returns an ``Invocation`` whose
        ``.output`` is a ``JudgedResult`` with ``.judgment.quality_score``.
        Live test DEFECT-1 (May 2026) found we were reading
        ``invocation.output.score`` which doesn't exist on JudgedResult —
        the quality axis silently collapsed to 0.0. The fix probes the
        documented kaos-llm-core path first, then falls back to legacy
        shapes that test stubs may use (a flat ``.score`` on the output
        object).
        """
        if criteria is None or not criteria.criteria:
            return 1.0
        # Local pin so ty narrows away the Optional in the try block —
        # the caller already gated on ``self._judge is not None``.
        judge = self._judge
        if judge is None:
            return 1.0
        try:
            invocation = await judge.invoke(
                output=text,
                criteria=" / ".join(criteria.criteria),
            )
            score = _extract_quality_score(invocation)
        except Exception:
            return 0.5  # neutral on judge failure
        # Clamp to [0, 1] in case the Judge produced an out-of-range value.
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score


def _extract_quality_score(invocation: Any) -> float:
    """Pull a quality score from a Judge ``Invocation`` defensively.

    Probe order:

      1. ``invocation.output.judgment.quality_score`` — kaos-llm-core
         ``JudgedResult.judgment.quality_score`` (the documented shape).
      2. ``invocation.output.score`` — flat ``.score`` on the output
         object (used by stub Judges in unit tests).
      3. ``invocation.score`` — legacy / extra-flat shape; treat as
         a defensive fallback for callers that pre-flatten.

    Returns ``0.0`` when no probe yields a numeric value. The caller
    clamps the score to ``[0, 1]``.
    """
    if invocation is None:
        return 0.0
    output = getattr(invocation, "output", None)
    if output is not None:
        judgment = getattr(output, "judgment", None)
        if judgment is not None:
            value = getattr(judgment, "quality_score", None)
            if value is not None:
                return _coerce_score(value)
        flat = getattr(output, "score", None)
        if flat is not None:
            return _coerce_score(flat)
    extra_flat = getattr(invocation, "score", None)
    if extra_flat is not None:
        return _coerce_score(extra_flat)
    return 0.0


def _coerce_score(value: Any) -> float:
    """Best-effort conversion of a probed score to a float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["TerminationJudge"]
