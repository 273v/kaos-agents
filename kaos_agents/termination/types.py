"""Termination value types — Decision and SuccessCriteria.

A :class:`Decision` is the verdict from
:class:`~kaos_agents.termination.judge.TerminationJudge`. ``DecisionKind``
discriminates the reason; ``allows_replan`` and ``should_escalate`` tell
the AgentLoop what to do next.

Phase 4.B (paper §Q7): the TerminationJudge is the agent's "am I done?"
oracle. It composes 5 axes — budget, quality, failure, loop, graceful
degradation — and returns a single :class:`Decision` for the AgentLoop's
step 4. The Decision is intentionally cheap to construct and frozen so
hooks and wire serialisers can pass it across awaits without copying.

``SuccessCriteria`` is derived from the upstream
:class:`kaos_agents.intent.types.IntentResult` at intent-classification
time and threaded through the loop so the quality axis has typed inputs
without re-importing the IntentResult class.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@unique
class DecisionKind(StrEnum):
    """The flavor of a termination decision.

    Eight kinds; pattern-match in the AgentLoop on this discriminator
    rather than re-deriving the verdict from the boolean flags. The
    flags (``is_complete`` / ``allows_replan`` / ``should_escalate``)
    are the *action* the loop should take; ``kind`` is the *reason*.
    """

    COMPLETE = "complete"  # success criteria met
    INCOMPLETE = "incomplete"  # not done; replan if budget allows
    BUDGET_EXCEEDED = "budget_exceeded"  # any axis: cost / time / iterations
    QUALITY_FAILED = "quality_failed"  # judge < threshold
    FAILURE = "failure"  # RunError or EvidenceInsufficient
    LOOP_DETECTED = "loop_detected"  # the same step fired N times
    DEGRADED = "degraded"  # partial result acceptable
    ESCALATE = "escalate"  # outside competence


class Decision(BaseModel):
    """Verdict from :class:`~kaos_agents.termination.judge.TerminationJudge`.

    Read by AgentLoop step 4 (termination judge). When ``is_complete``
    and ``allows_replan`` and ``should_escalate`` are all False, the
    loop falls through with whatever exec_result the planner produced.

    All fields have safe defaults so tests / hooks can construct
    minimal Decisions without enumerating the full surface. Frozen so
    consumers can cache or pass across awaits without copying.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DecisionKind
    is_complete: bool = False
    allows_replan: bool = False
    should_escalate: bool = False
    feedback: str = ""  # text for replan / quality follow-up
    partial_result: str | None = None  # populated when DEGRADED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuccessCriteria(BaseModel):
    """Derived from intent.goal at intent-classification time.

    Phase 4.B accepts the bare success-criteria tuple from
    :class:`~kaos_agents.intent.types.IntentResult` and a goal
    statement. Future Phase 4+ may add structured fields (deadline,
    required_format, must_include_sources) — the shape is forward-
    compatible because ``extra="forbid"`` rejects unknown fields and
    the optional ones have safe defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_statement: str = ""
    criteria: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()  # required substrings in output
    target_format: str | None = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @classmethod
    def from_intent(cls, intent: Any) -> SuccessCriteria:
        """Construct from an :class:`IntentResult`-shaped object.

        Duck-typed — accepts anything with a ``.goal`` attribute that
        carries ``.statement`` / ``.success_criteria`` / ``.target_format``.
        Returns an empty :class:`SuccessCriteria` if the intent has no
        ``goal`` attribute (e.g. legacy intents) so the loop never
        crashes on degraded inputs.
        """
        goal = getattr(intent, "goal", None)
        if goal is None:
            return cls()
        statement = getattr(goal, "statement", "") or ""
        raw_criteria = getattr(goal, "success_criteria", ()) or ()
        target_format = getattr(goal, "target_format", None)
        return cls(
            goal_statement=str(statement),
            criteria=tuple(raw_criteria),
            target_format=target_format,
        )


__all__ = ["Decision", "DecisionKind", "SuccessCriteria"]
