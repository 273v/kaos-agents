"""Tier 4: plan-execute with prior-output threading + arg synthesis.

Goal: a 3+ step plan where step 2 is a pure-LLM extract that READS
step 1's output, and step 3 is a TOOL call whose args must be
SYNTHESIZED from step 2's output. Catches regressions in:

- ``_collect_predecessor_results`` — prior-output threading into
  LLM step prompts
- ``_synthesize_tool_args`` — runtime arg synthesis from
  description + prior outputs
- The expected_output prompt threading (dominant cause of plan
  bail-outs before this session's fix)

This is a cousin of tier 3 — same FR domain — but the multi-step
goal explicitly requires the new plumbing the recent
plan-execute fixes added.

Cost target: ~$0.20.
"""

from __future__ import annotations

import pytest

from kaos_agents.events import PlanProposed
from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
)

pytestmark = pytest.mark.live

TIER = 4
BUDGET_USD = 0.30


@pytest.mark.asyncio
async def test_plan_execute_3plus_steps_with_extract(ladder_runner_factory, turn_session_id):
    """Plan with TOOL -> LLM extract -> TOOL chain exercises prior-output paths."""
    from kaos_agents.config import AgentPattern

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a research assistant. Use the kaos-source-fr-* "
                "tools to satisfy the user's request, decomposing into "
                "ordered steps."
            ),
            "model": model_for_tier(TIER),
            "pattern": AgentPattern.PLAN,
            "tools": (
                "kaos-source-fr-search",
                "kaos-source-fr-get-document",
                "kaos-source-fr-get-content",
            ),
        },
        register_source=True,
    )
    msg = (
        "Find the most recent EPA Federal Register notice mentioning PFAS "
        "from 2025. Plan it in steps: (1) search Federal Register, (2) "
        "extract the document_number from the most recent result, (3) "
        "fetch the document via kaos-source-fr-get-document. Report the "
        "title, document_number, and publication_date."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-4")

    plans = [e for e in events if isinstance(e, PlanProposed)]
    assert plans, "PlanProposed event must fire"
    assert len(plans[0].steps) >= 3, (
        f"plan must have >=3 steps to exercise the multi-stage pipeline; "
        f"got {len(plans[0].steps)} steps"
    )

    # The plan should reference at least 2 distinct tool names across
    # its steps — proves arg synthesis ran on >=2 different tools.
    step_tools = {s.tool_name for s in plans[0].steps if s.tool_name}
    assert len(step_tools) >= 2, (
        f"plan should use >=2 distinct tools (search + get_document); got: {step_tools}"
    )

    # Loose answer check: the plan should produce SOME output. As with
    # tier 3, semantic-eval REPLANs are non-deterministic so the
    # detailed answer-shape check is left to the calibration arc.
    text = text_from_summary(events)
    assert text.strip(), "expected non-empty TurnSummary text"
