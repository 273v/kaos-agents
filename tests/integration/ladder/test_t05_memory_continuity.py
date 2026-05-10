"""Tier 5: memory continuity across two turns.

Same session_id used for two consecutive ``runner.run`` calls. Turn 1
introduces a fact; turn 2 references it. Verifies SessionMemory
hydrates on each call, the MESSAGES section accumulates, and
multi-turn memory tier works end-to-end.

Cost target: ~$0.05 (two short Anthropic calls).
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
)

pytestmark = pytest.mark.live

TIER = 5
BUDGET_USD = 0.10


@pytest.mark.asyncio
async def test_memory_continuity_two_turns(ladder_runner_factory, turn_session_id):
    """Turn 2 must reference a fact established in turn 1 via shared memory."""
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a helpful assistant. Use any context from prior "
                "turns in the conversation when answering follow-ups."
            ),
            "model": model_for_tier(TIER),
        },
    )

    # Turn 1: state a memorable fact. The session_id is shared via
    # the ``turn_session_id`` fixture; SessionStore persists/loads
    # the memory snapshot between calls.
    turn1_events = await collect_events(
        runner.run(
            "My favorite color is mauve, and the magic number is 7919. Please acknowledge.",
            turn_session_id,
        )
    )
    turn1_text = text_from_summary(turn1_events)
    assert turn1_text.strip(), "turn 1 must produce some response"

    # Turn 2: ask about turn 1's fact. The agent MUST remember it
    # via SessionMemory (no in-prompt repetition).
    turn2_events = await collect_events(runner.run("What was my magic number?", turn_session_id))

    # Cost gate covers BOTH turns.
    assert_within_budget(turn1_events + turn2_events, budget_usd=BUDGET_USD, label="tier-5")

    turn2_text = text_from_summary(turn2_events).lower()
    assert "7919" in turn2_text, (
        f"turn 2 must reference the magic number 7919 from turn 1's memory; "
        f"got: {turn2_text[:200]!r}"
    )
