"""Adaptive model escalation — pick the right model for the next step.

Today every agent run picks one model at construction and lives with
it. Real cost-aware behaviour is "use the cheap model 90% of the time;
spend on the expensive model on the hard 10%."

``ModelEscalationPolicy`` is the typed decision surface: given the
current ``CostBudget`` and a step-level signal (failure, low confidence,
invariant violation), it picks one of three actions and a target model.

The default :class:`DefaultEscalationPolicy` implements a sensible
baseline:

- Below 50% spent, on the **first** attempt: stay on the cheap model.
- On retry (or below 75% spent + failure signal): escalate to the
  mid-tier model.
- Above 75% spent: refuse further escalation regardless of signal —
  the budget is the cap and high-cost retries can blow it.

Callers can pass their own policy implementing :class:`ModelEscalationPolicy`
for domain-specific rules. The hooks fire in :meth:`BaseAgent._model_for_role`
when the agent is given a budget-aware ProviderConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from kaos_agents.types.budget import CostBudget


class StepOutcome(StrEnum):
    """The observed result of the previous step.

    The escalation policy reads this to decide whether to upgrade,
    stay, or downgrade the model for the next attempt.
    """

    INITIAL = "initial"
    """No previous step — this is the first attempt of the run."""

    SUCCESS = "success"
    """Previous step produced a usable result. Default action: stay."""

    LOW_CONFIDENCE = "low_confidence"
    """Result was usable but flagged low-confidence (e.g.
    ``answer.confidence < refusal_policy.min_confidence``). Default
    action: consider escalating."""

    REFUSAL = "refusal"
    """Step returned ``InsufficientEvidence`` /
    ``NeedsAggregation``. Default action: escalate (the cheaper
    model may have given up; the bigger model often succeeds)."""

    INVARIANT_VIOLATION = "invariant_violation"
    """A step invariant (see ``kaos_agents.runtime.invariants``)
    rejected the previous output. Default action: escalate on the
    retry, with the violation message in the prompt."""

    PROVIDER_ERROR = "provider_error"
    """The provider returned a transient error (rate limit, 500,
    timeout). Default action: stay on same model and retry."""


class EscalationAction(StrEnum):
    """The decision a policy returns for the next step.

    Distinct from ``StepOutcome`` (which describes the past) — this
    describes the policy's chosen behaviour for the future.
    """

    STAY = "stay"
    """Use the same model as the previous step."""

    UPGRADE = "upgrade"
    """Use the next-tier-up model. The Runner resolves "next tier"
    against the ProviderConfig's tier list."""

    DOWNGRADE = "downgrade"
    """Use the next-tier-down model. Useful when the budget is
    almost gone and the remaining steps can tolerate the cheaper
    model."""

    STOP = "stop"
    """Halt the run cleanly. The Runner emits ``BudgetExceeded``
    (or a comparable event) and returns. Policies use this when
    the budget is exhausted AND the step needs escalation that
    can't fit."""


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """Frozen result returned by a policy's ``choose`` call.

    Attributes:
        action: The Runner's directive for the next step.
        reason: Short human-readable explanation. Surfaces in
            ``ModelEscalated`` telemetry events and ``--explain``
            output.
    """

    action: EscalationAction
    reason: str


class ModelEscalationPolicy(Protocol):
    """The contract a policy implements.

    A policy is a pure function of (budget, outcome, attempt_number)
    — no I/O, no LLM calls. The Runner consults it before every step.
    """

    def choose(
        self,
        *,
        budget: CostBudget,
        outcome: StepOutcome,
        attempt: int,
    ) -> EscalationDecision:
        """Pick the next-step action given current state.

        Args:
            budget: Current ``CostBudget``. Read ``fraction_spent``,
                ``headroom_usd``, ``exceeded`` to reason about cost.
            outcome: What happened in the previous step.
            attempt: 1-based attempt number for the CURRENT logical
                step (i.e. 1 on the first try, 2 on the first retry).
        """
        ...


@dataclass(frozen=True, slots=True)
class DefaultEscalationPolicy:
    """The platform-default escalation policy.

    Implements a budget-aware ladder:

    - Refusal / invariant-violation → UPGRADE unless budget is
      ≥ ``high_water_mark`` spent (default 0.75) in which case STOP.
    - Low confidence → UPGRADE only if budget < 0.5 spent. Above 0.5
      we accept the low-confidence answer to preserve headroom.
    - Provider error → STAY (retry on same model; the error is
      transient, not a quality signal).
    - Success or initial → STAY.

    Attributes:
        high_water_mark: Fraction of budget consumed past which the
            policy refuses to upgrade (default 0.75). At this point
            an upgrade would risk blowing the cap on the retry.
        max_attempts_per_step: Beyond this attempt count the policy
            returns STOP regardless of outcome — prevents infinite
            retry loops on a step the model can't solve.
    """

    high_water_mark: float = 0.75
    max_attempts_per_step: int = 3
    _allowed_outcomes_for_upgrade: tuple[StepOutcome, ...] = field(
        default=(
            StepOutcome.REFUSAL,
            StepOutcome.INVARIANT_VIOLATION,
            StepOutcome.LOW_CONFIDENCE,
        ),
    )

    def choose(
        self,
        *,
        budget: CostBudget,
        outcome: StepOutcome,
        attempt: int,
    ) -> EscalationDecision:
        # Stop early when the per-step retry budget is exhausted.
        if attempt > self.max_attempts_per_step:
            return EscalationDecision(
                action=EscalationAction.STOP,
                reason=(
                    f"attempt {attempt} exceeded max_attempts_per_step="
                    f"{self.max_attempts_per_step}; halting"
                ),
            )

        # The budget cap is non-negotiable. STOP regardless of outcome.
        if budget.exceeded:
            return EscalationDecision(
                action=EscalationAction.STOP,
                reason=(f"budget exceeded ({budget.spent_usd:.4f} / {budget.total_usd:.4f} USD)"),
            )

        # Transient provider errors: retry on the same model.
        if outcome == StepOutcome.PROVIDER_ERROR:
            return EscalationDecision(
                action=EscalationAction.STAY,
                reason="provider error — retrying on same model",
            )

        # Initial / success → stay put.
        if outcome in (StepOutcome.INITIAL, StepOutcome.SUCCESS):
            return EscalationDecision(
                action=EscalationAction.STAY,
                reason="no failure signal; staying on current model",
            )

        # The remaining outcomes are quality signals. Decide based on
        # budget headroom.
        spent_fraction = budget.fraction_spent
        if spent_fraction >= self.high_water_mark:
            return EscalationDecision(
                action=EscalationAction.STOP,
                reason=(
                    f"budget at {spent_fraction:.0%} spent (≥ "
                    f"{self.high_water_mark:.0%} high-water mark); "
                    "refusing to escalate"
                ),
            )

        # Low-confidence is a softer signal than refusal — only
        # escalate when there's plenty of headroom (< 0.5 spent).
        if outcome == StepOutcome.LOW_CONFIDENCE and spent_fraction >= 0.5:
            return EscalationDecision(
                action=EscalationAction.STAY,
                reason=(
                    f"low-confidence but budget already at {spent_fraction:.0%}; "
                    "accepting answer to preserve headroom"
                ),
            )

        return EscalationDecision(
            action=EscalationAction.UPGRADE,
            reason=f"{outcome.value} at {spent_fraction:.0%} spent — escalating",
        )


__all__ = [
    "DefaultEscalationPolicy",
    "EscalationAction",
    "EscalationDecision",
    "ModelEscalationPolicy",
    "StepOutcome",
]
