"""CostBudget — typed cost ceiling + accumulated spend tracker.

A single typed value carrying both the **cap** and the **spend so far**.
Frozen, slotted, pointwise-summable like :class:`InvocationUsage`.
Replaces the scattered ``KAOS_AGENT_MAX_COST_USD`` env var + per-runner
``--max-cost`` flag handling with one composable value type.

Two use cases drive the design:

1. **Hard cost ceiling.** ``budget.exceeded`` returns True once the
   spend has crossed the cap; callers stop the run cleanly and emit
   a typed ``BudgetExceeded`` event. The existing chat-CLI ``--max-cost``
   path becomes one ``CostBudget`` construction at session start.

2. **Adaptive model escalation.** A budget-aware policy can ask
   ``budget.fraction_spent`` and ``budget.headroom_usd`` to decide
   whether the next step should use a cheaper or more expensive
   model. The platform's research-pattern default of sonnet doesn't
   scale to long sessions when 80% of the budget is gone — the
   policy can downgrade to haiku for the tail steps.

Usage::

    budget = CostBudget(total_usd=0.50)
    # ... after each LLM call:
    budget = budget.spend(invocation_usage.cost_usd)
    if budget.exceeded:
        runner.emit_budget_exceeded()
        return

Distinction from :class:`kaos_llm_core.optimization.budget.Budget`:

  This type is the **runtime per-turn** cap+accumulator threaded through
  :class:`~kaos_agents.core.invocation.Invocation` to drive per-step
  escalation decisions (model downgrade, BudgetExceeded emission). It is
  frozen, cost-only, and exposes ``fraction_spent`` / ``headroom_usd`` /
  ``exceeded`` as decision properties.

  ``kaos_llm_core.optimization.budget.Budget`` is the **optimizer-loop**
  cap: cap-only (mutability lives in a separate ``BudgetTracker``),
  multi-dimensional (max_trials, max_wall_seconds, max_cost_usd,
  max_tokens), and exposes ``BudgetTracker.exhausted() -> StopReason``
  for trial-loop termination.

  They are complementary, not duplicates. Reach for ``Budget`` when
  capping optimizer trials; reach for ``CostBudget`` when threading a
  runtime turn's cost-cap through escalation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostBudget:
    """Typed cost ceiling + accumulated spend.

    Construct with ``total_usd`` (the cap). Append spend with
    :meth:`spend` which returns a new instance — frozen, so the
    existing budget is unchanged.

    Attributes:
        total_usd: The cap. Must be positive. Use a large value
            (e.g., 1e6) for "effectively unlimited" rather than 0.
        spent_usd: Accumulated spend so far. Always ≤ ``total_usd``
            after a normal ``spend(...)`` call; a single oversized
            spend can push past the cap, in which case ``exceeded``
            becomes True and the caller should stop the run.

    Sentinels:
        ``UNLIMITED``: a budget with ``total_usd = float("inf")`` —
        ``exceeded`` is always False. For ergonomic
        "I don't want to cap" callers.
    """

    total_usd: float
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.total_usd <= 0:
            raise ValueError(
                f"CostBudget.total_usd must be positive, got {self.total_usd!r}. "
                "Use CostBudget.UNLIMITED for effectively-unbounded budgets."
            )
        if self.spent_usd < 0:
            raise ValueError(f"CostBudget.spent_usd cannot be negative, got {self.spent_usd!r}.")

    # -- Properties ----------------------------------------------------------

    @property
    def remaining_usd(self) -> float:
        """How much budget is left. Clamped to 0 once spent exceeds cap."""
        return max(0.0, self.total_usd - self.spent_usd)

    @property
    def headroom_usd(self) -> float:
        """Alias for ``remaining_usd``; the name policy implementations
        prefer when reasoning about whether a step can fit."""
        return self.remaining_usd

    @property
    def fraction_spent(self) -> float:
        """Fraction of budget consumed in ``[0, 1]``. >1.0 when overrun.

        ``inf`` budgets always return 0.0 — there's no meaningful
        fraction of infinity.
        """
        if self.total_usd == float("inf"):
            return 0.0
        return self.spent_usd / self.total_usd

    @property
    def exceeded(self) -> bool:
        """True when ``spent_usd >= total_usd``.

        ``inf`` budgets are never exceeded.
        """
        if self.total_usd == float("inf"):
            return False
        return self.spent_usd >= self.total_usd

    # -- Builders ------------------------------------------------------------

    def spend(self, amount_usd: float) -> CostBudget:
        """Return a new CostBudget with ``amount_usd`` added to spent.

        ``amount_usd`` < 0 raises ValueError — negative spend is
        meaningless (would represent a refund, which has no model
        in this type).
        """
        if amount_usd < 0:
            raise ValueError(
                f"CostBudget.spend(amount_usd) must be ≥ 0, got "
                f"{amount_usd!r}. Negative spend is undefined."
            )
        return CostBudget(
            total_usd=self.total_usd,
            spent_usd=self.spent_usd + amount_usd,
        )

    def with_cap(self, total_usd: float) -> CostBudget:
        """Return a new CostBudget with the same spend but a different cap."""
        return CostBudget(total_usd=total_usd, spent_usd=self.spent_usd)


# Sentinel for "effectively unlimited" budgets. Module-level rather
# than a class attribute so it doesn't trip ty / frozen-dataclass
# semantics. Use ``UNLIMITED_BUDGET`` when you want an explicit
# no-cap construction.
UNLIMITED_BUDGET: CostBudget = CostBudget(total_usd=float("inf"))


__all__ = ["UNLIMITED_BUDGET", "CostBudget"]
