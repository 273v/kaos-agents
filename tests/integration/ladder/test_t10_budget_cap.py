"""Tier 10: BudgetExceeded fires when a plan exceeds its cost cap.

Configure ``plan_max_cost_usd`` very low so a real plan run hits the
cap mid-execution. Verifies BudgetExceeded fires (regression for
last session's wiring).

Cost target: ~$0.05 — the run intentionally bails on cost.
"""

from __future__ import annotations

import pytest

from kaos_agents.events import BudgetExceeded, KaosEvent
from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
)

pytestmark = pytest.mark.live

TIER = 10
BUDGET_USD = 0.10


def _budget_events(events: list[KaosEvent]) -> list[BudgetExceeded]:
    return [e for e in events if isinstance(e, BudgetExceeded)]


@pytest.mark.asyncio
async def test_budget_cap_emits_budget_exceeded(ladder_runner_factory, turn_session_id):
    """A plan with a $0.001 cost cap exceeds budget → BudgetExceeded fires."""
    from kaos_agents.config import AgentPattern
    from kaos_agents.settings import KaosAgentSettings

    # Tight settings — any non-trivial plan-execute run will blow past
    # this cap on the first LLM call. plan_max_cost_usd=0.001 is well
    # below the cost of even one Sonnet call (~$0.003 minimum), so
    # BudgetExceeded MUST fire after step 1's evaluator.
    tight_settings = KaosAgentSettings(plan_max_cost_usd=0.001)

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": "You are a research assistant.",
            "model": model_for_tier(TIER),
            "pattern": AgentPattern.PLAN,
            "tools": ("kaos-source-fr-search",),
            "settings": tight_settings,
        },
        register_source=True,
    )
    msg = (
        "Use kaos-source-fr-search to find recent EPA Federal Register "
        "notices. Plan it in 3+ steps: search, extract, summarize."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    # Cost gate is a wider sanity check — the budget for the test is
    # higher than plan_max_cost_usd because the LLM call already runs
    # before BudgetExceeded fires.
    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-10")

    budgets = _budget_events(events)
    assert budgets, (
        "expected >=1 BudgetExceeded event with plan_max_cost_usd=0.001. "
        "Zero events means the BudgetExceeded wiring in plan_execute.py "
        "regressed (it was added in commit a773477)."
    )

    # The kind should be 'cost' (the dimension that exceeded).
    kinds = {b.kind for b in budgets}
    assert "cost" in kinds, f"expected BudgetExceeded(kind='cost'); got kinds={kinds}"
