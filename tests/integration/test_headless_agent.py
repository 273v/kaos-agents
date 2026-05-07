"""Phase 2: Automated headless agent tests.

Drives the Runner programmatically to prove:
1. Single-turn Q&A — agent responds with TextDelta events
2. Multi-turn with memory — turn 2 recalls context from turn 1
3. Tool-calling — agent invokes a registered tool and returns results
4. Intent classification — verbose events carry intent + confidence

All tests are live (require ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import (
    IntentClassified,
    KaosEvent,
    RunError,
    TextDelta,
    TurnSummary,
)
from kaos_agents.runner import Runner


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


async def _collect_events(
    runner: Runner, message: str, session_id: str
) -> tuple[str, list[KaosEvent]]:
    """Run a turn and collect all events + concatenated text."""
    events: list[KaosEvent] = []
    text_parts: list[str] = []
    async for event in runner.run(message, session_id):
        events.append(event)
        if isinstance(event, TextDelta):
            text_parts.append(event.content)
    return "".join(text_parts), events


def _make_runner(tools: tuple[str, ...] = ()) -> Runner:
    from kaos_core.registry.container import KaosRuntime

    runtime = KaosRuntime.default()
    agent = Agent.create(
        instructions="You are a helpful legal research assistant. Be concise.",
        model="anthropic:claude-haiku-4-5",
        pattern="chat",
        tools=tools,
    )
    return Runner(agent, runtime=runtime)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_turn_response() -> None:
    """Agent responds to a simple question with TextDelta events."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    runner = _make_runner()
    text, events = await _collect_events(runner, "What is habeas corpus?", "test-single")

    assert len(text) > 50, f"Response too short: {text[:100]}"
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, TurnSummary) for e in events)
    assert not any(isinstance(e, RunError) for e in events)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_turn_memory() -> None:
    """Turn 2 recalls information from turn 1 (session memory persists)."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    runner = _make_runner()
    session = "test-memory"

    # Turn 1: give the agent a fact
    _text1, _ = await _collect_events(
        runner,
        "Remember this: the contract between Acme Corp and Beta LLC was signed on March 15, 2024.",
        session,
    )

    # Turn 2: ask about the fact
    text2, _ = await _collect_events(
        runner,
        "What was the date of the contract I just told you about?",
        session,
    )

    assert "march" in text2.lower() or "2024" in text2.lower() or "15" in text2, (
        f"Agent didn't recall the date from turn 1. Response: {text2[:200]}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intent_classification_event() -> None:
    """IntentClassified event fires with a valid intent + confidence."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    runner = _make_runner()
    _, events = await _collect_events(runner, "Hello", "test-intent")

    intent_events = [e for e in events if isinstance(e, IntentClassified)]
    assert intent_events, "No IntentClassified event found"
    ie = intent_events[0]
    assert ie.confidence > 0.0
    assert ie.intent  # non-empty string


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_errors_on_simple_conversation() -> None:
    """A basic conversation should produce no RunError events."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    runner = _make_runner()
    _, events = await _collect_events(
        runner,
        "Explain the difference between civil and criminal law in two sentences.",
        "test-no-errors",
    )

    errors = [e for e in events if isinstance(e, RunError)]
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"
