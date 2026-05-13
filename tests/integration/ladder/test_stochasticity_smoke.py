"""Stochasticity gate — runs T01-smoke five times, requires >=4/5 pass.

LLMs are stochastic. Even the simplest test can occasionally trip on
a provider blip, a model-side oddity, or pure randomness in sampling.
A SINGLE green run isn't strong evidence the test is stable; this
gate runs the cheapest test (T01, ~$0.005, ~3s) five times in a row
and requires 4 of 5 to pass.

Why T01 only:
- Cheapest tier (~$0.025 total for 5 runs)
- Tests the most foundational path
- Failures here are red flags about provider reliability OR test code
- Running the whole ladder 5x would cost ~$5.50 and 9 minutes —
  excessive for what flake-gate signal needs

To extend to other tiers, copy the parameterize list. Don't run
T11/T12 5x without a deliberate cost trade-off.

Cost: ~$0.025 / 5 runs / ~20s wall-clock.
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    intent_events,
    text_from_summary,
    turn_spans,
)

pytestmark = pytest.mark.live

MODEL = "anthropic:claude-sonnet-4-6"
BUDGET_USD = 0.01


async def _run_t01_once(ladder_runner_factory, session_id: str) -> tuple[bool, str]:
    """Run a single iteration of the T01 contract. Returns (passed, reason)."""
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": "You are a helpful assistant. Answer concisely.",
            "model": MODEL,
        },
    )
    try:
        events = await collect_events(runner.run("What is 17 + 25?", session_id))
    except Exception as exc:
        return False, f"crashed: {exc!r}"

    try:
        assert_within_budget(events, budget_usd=BUDGET_USD, label="t01-stoch")
        spans = turn_spans(events)
        if len(spans) < 2:
            return False, f"insufficient turn spans: {len(spans)}"
        intents = intent_events(events)
        if len(intents) < 1:
            return False, "no IntentClassified event"
        text = text_from_summary(events)
        if "42" not in text:
            return False, f"answer missing '42': {text[:200]!r}"
    except AssertionError as exc:
        return False, f"assertion: {exc}"

    return True, "ok"


@pytest.mark.asyncio
async def test_t01_stable_under_repetition(ladder_runner_factory, turn_session_id):
    """Run T01 five times serially. Require >=4/5 pass.

    Why 4/5 not 5/5: the 5/5 bar is brittle to provider blips; a 4/5
    contract still catches consistently-broken tests while tolerating
    rare upstream hiccups (rate-limit transient, network jitter).
    A 3/5 bar would tolerate too much real flakiness.
    """
    results: list[tuple[bool, str]] = []
    for i in range(5):
        session = f"{turn_session_id}-run-{i}"
        passed, reason = await _run_t01_once(ladder_runner_factory, session)
        results.append((passed, reason))

    n_passed = sum(1 for passed, _ in results if passed)
    assert n_passed >= 4, (
        f"T01 stability gate failed: {n_passed}/5 passed. Reasons: {[r for _, r in results]!r}"
    )
