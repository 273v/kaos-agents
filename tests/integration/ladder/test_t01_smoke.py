"""Tier 1: smoke — chat with no tools.

Goal: verify the agent runs end-to-end on the simplest possible
prompt. No tools registered, no memory hydration, no external calls
beyond the LLM. Catches regressions in the most foundational layers
(Runner dispatch, EventEmitter, IntentExtractor, BaseAgent.run).

Cost target: ~$0.005 (one Anthropic call on sonnet-4-6).
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    intent_events,
    model_for_tier,
    text_from_summary,
    turn_spans,
)

pytestmark = pytest.mark.live

TIER = 1
BUDGET_USD = 0.01


@pytest.mark.asyncio
async def test_smoke_chat_no_tools(ladder_runner_factory, turn_session_id):
    """A no-tool chat agent answers a math question and emits a clean stream."""
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": "You are a helpful assistant. Answer concisely.",
            "model": model_for_tier(TIER),
        },
    )
    events = await collect_events(runner.run("What is 17 + 25?", turn_session_id))

    # Cost gate
    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-1")

    # Stream-shape regression: turn span pair + intent + text
    spans = turn_spans(events)
    assert len(spans) >= 2, f"expected >=2 turn spans (start+complete), got {len(spans)}"
    intents = intent_events(events)
    assert len(intents) >= 1, "expected at least one IntentClassified event"

    # Answer regression: must contain the numeric answer (42)
    text = text_from_summary(events)
    assert "42" in text, f"expected answer '42' in output, got: {text!r}"
