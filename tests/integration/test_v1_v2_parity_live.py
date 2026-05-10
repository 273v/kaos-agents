"""Phase 6.A — v1↔v2 dispatch-path parity smoke test.

Runs the SAME prompts through ``Runner.run`` with explicit
``agent_loop_version="v1"`` and ``"v2"`` and compares observable
outputs:

  - Both paths produce a non-empty answer.
  - Both paths emit a Span(TURN, START) and Span(TURN, COMPLETE).
  - Both paths charge a non-zero cost (proves DEFECT-2 fix is live —
    UsageObserved emission from ReActPlanner reaches the loop's
    usage roll-up).
  - Both paths populate :attr:`AgentResponse.intent` with a sensible
    classification.

This isn't a regression-pinning test — provider quirks make exact
equality across paths impossible. It's a parity SMOKE: prove the
v2 path delivers the same broad shape of answer as v1 so a future
default-flip is safe. Documented divergences become Phase 6.B fix
items.

Cost: ~$0.10 across the suite (one short turn x 4 paths x 2 models).

Run with::

    uv run pytest tests/integration/test_v1_v2_parity_live.py \\
        -m live -v --tb=short --no-cov -s
"""

from __future__ import annotations

import os

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import IntentClassified, Span, SpanPhase, SpanSubject
from kaos_agents.runtime.runner import Runner

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY missing"
)
requires_openai = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY missing"
)

# Models — pinned. Source of truth: kaos-llm-client/tests/integration/test_live.py.
ANTHROPIC = "anthropic:claude-haiku-4-5"
OPENAI = "openai:gpt-5.4-mini"

PROMPT = "What is the capital of France? Answer in one short sentence."


async def _collect_events(runner: Runner, message: str, session_id: str) -> list:
    events = []
    async for ev in runner.run(message, session_id):
        events.append(ev)
    return events


def _has_span(events: list, subject: SpanSubject, phase: SpanPhase) -> bool:
    return any(isinstance(e, Span) and e.subject == subject and e.phase == phase for e in events)


def _intent_event_present(events: list) -> bool:
    return any(isinstance(e, IntentClassified) for e in events)


def _summary_total_cost(events: list) -> float:
    """Sum cost from any TurnSummary in the stream (v1 path), or fall
    back to walking UsageObserved (v2 path)."""
    from kaos_agents.events import TurnSummary, UsageObserved

    for e in events:
        if isinstance(e, TurnSummary):
            return float(e.cost_usd)
    # No TurnSummary — sum UsageObserved (v2 path may emit multiple).
    total = 0.0
    for e in events:
        if isinstance(e, UsageObserved):
            total += float(getattr(e, "cost_usd", 0.0) or 0.0)
    return total


def _summary_text(events: list) -> str:
    """Find the answer text in either a TurnSummary or text deltas."""
    from kaos_agents.events import TextDelta, TurnSummary

    for e in events:
        if isinstance(e, TurnSummary) and e.text:
            return e.text
    # No summary text — concatenate TextDelta content as a fallback.
    deltas = [getattr(e, "content", "") for e in events if isinstance(e, TextDelta)]
    return "".join(deltas)


@pytest.mark.live
@requires_anthropic
class TestV1V2ParityAnthropic:
    """Side-by-side v1/v2 comparison on Claude."""

    async def test_both_paths_produce_an_answer(self) -> None:
        agent = Agent(model=ANTHROPIC)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1")
        v1_text = _summary_text(v1_events).lower()

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2")
        v2_text = _summary_text(v2_events).lower()

        # Both paths produced *something*. Don't assert exact equality —
        # provider stochasticity means the two paths may produce
        # different phrasings.
        assert v1_text, (
            f"v1 produced no answer text. Events: {[type(e).__name__ for e in v1_events]}"
        )
        assert v2_text, (
            f"v2 produced no answer text. Events: {[type(e).__name__ for e in v2_events]}"
        )

        # Both should mention Paris (the canonical answer).
        assert "paris" in v1_text, f"v1 answer missing 'paris': {v1_text!r}"
        assert "paris" in v2_text, f"v2 answer missing 'paris': {v2_text!r}"

    async def test_both_paths_emit_turn_lifecycle(self) -> None:
        agent = Agent(model=ANTHROPIC)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1-life")
        assert _has_span(v1_events, SpanSubject.TURN, SpanPhase.START), "v1 missing TURN.START"
        assert _has_span(v1_events, SpanSubject.TURN, SpanPhase.COMPLETE), (
            "v1 missing TURN.COMPLETE"
        )

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2-life")
        assert _has_span(v2_events, SpanSubject.TURN, SpanPhase.START), "v2 missing TURN.START"
        assert _has_span(v2_events, SpanSubject.TURN, SpanPhase.COMPLETE), (
            "v2 missing TURN.COMPLETE"
        )

    async def test_both_paths_classify_intent(self) -> None:
        agent = Agent(model=ANTHROPIC)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1-intent")
        assert _intent_event_present(v1_events), "v1 missing IntentClassified"

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2-intent")
        assert _intent_event_present(v2_events), "v2 missing IntentClassified"

    async def test_both_paths_charge_cost(self) -> None:
        """DEFECT-2 fix end-to-end check: v2 must charge non-zero cost.

        Pre-fix, the v2 path's UsageObserved emission from ReActPlanner
        was missing — invocation.usage.cost_usd was always 0. Post-fix,
        ReActPlanner emits UsageObserved which the loop's roll-up
        captures.
        """
        agent = Agent(model=ANTHROPIC)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1-cost")
        v1_cost = _summary_total_cost(v1_events)
        assert v1_cost > 0.0, (
            f"v1 cost is 0 — TurnSummary cost not populated. "
            f"Events: {[type(e).__name__ for e in v1_events]}"
        )

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2-cost")
        v2_cost = _summary_total_cost(v2_events)
        assert v2_cost > 0.0, (
            f"v2 cost is 0 — UsageObserved emission from ReActPlanner missing "
            f"(DEFECT-2 regression?). Events: "
            f"{[type(e).__name__ for e in v2_events]}"
        )


@pytest.mark.live
@requires_openai
class TestV1V2ParityOpenAI:
    """Side-by-side v1/v2 comparison on OpenAI."""

    async def test_both_paths_produce_an_answer(self) -> None:
        agent = Agent(model=OPENAI)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1-oai")
        v1_text = _summary_text(v1_events).lower()

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2-oai")
        v2_text = _summary_text(v2_events).lower()

        assert v1_text, (
            f"v1/openai produced no answer. Events: {[type(e).__name__ for e in v1_events]}"
        )
        assert v2_text, (
            f"v2/openai produced no answer. Events: {[type(e).__name__ for e in v2_events]}"
        )
        assert "paris" in v1_text
        assert "paris" in v2_text

    async def test_both_paths_charge_cost(self) -> None:
        agent = Agent(model=OPENAI)

        v1 = Runner(agent, agent_loop_version="v1")
        v1_events = await _collect_events(v1, PROMPT, session_id="parity-v1-oai-cost")
        assert _summary_total_cost(v1_events) > 0.0

        v2 = Runner(agent, agent_loop_version="v2")
        v2_events = await _collect_events(v2, PROMPT, session_id="parity-v2-oai-cost")
        assert _summary_total_cost(v2_events) > 0.0
