"""Extended live tests for the 0.1.0a11 PatternMismatch event-delivery fix.

These complement test_plan_intent_dispatches_with_tools.py by covering:

* Multi-model: Haiku 4.5 (the second model that exhibited the v2-matrix
  failure) — proves the dispatcher is model-agnostic, not coupled to
  any provider's intent classifier quirks.
* Negative case: trivial RESPOND-intent prompt MUST NOT fire
  PatternMismatch (so the gate is selective, not always-on).
* Multi-turn: a second turn in the same session must still dispatch
  correctly (catches per-turn state leaks).

Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import pytest
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig
from kaos_source import register_federal_register_tools

from kaos_agents.events import IntentClassified, PatternMismatch, Span, SpanPhase, SpanSubject
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings
from tests.integration._models import probe_model, respond_model

# HAIKU here is the intentional weak-model probe: this file proves
# the dispatcher fix is model-agnostic by exercising the same path
# on a weaker reasoner. SONNET is the floor-enforced respond model.
HAIKU = probe_model()
SONNET = respond_model()

PLAN_PROMPT = (
    "Search the Federal Register for the most recent SEC rule about "
    "cybersecurity disclosure from 2024, fetch its full text, then "
    "list any defined terms it introduces."
)
TRIVIAL_PROMPT = "What is 2 + 2?"
FOLLOWUP_PROMPT = "And what about cybersecurity rules from 2023?"


def _runtime() -> KaosRuntime:
    rt = KaosRuntime.test_mode()
    register_federal_register_tools(rt)
    return rt


def _vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    )


def _settings() -> KaosAgentSettings:
    return KaosAgentSettings(plan_max_cost_usd=0.50, plan_max_wall_clock_seconds=120.0)


@pytest.mark.live
@pytest.mark.asyncio
async def test_haiku_plan_intent_emits_pattern_mismatch_and_uses_tools() -> None:
    """Same shape as the Sonnet test, but on Haiku 4.5.

    Proves the dispatcher fix is model-agnostic — the per-turn intent
    classifier returns PLAN on this prompt for both providers, and both
    must redirect through ReAct rather than fabricating training-data
    answers.
    """
    agent = ChatAgent(_vfs(), runtime=_runtime(), model=HAIKU, settings=_settings())
    events = [e async for e in agent.run(PLAN_PROMPT, session_id="dispatch-haiku-plan")]

    intent_classified = [e for e in events if isinstance(e, IntentClassified)]
    assert intent_classified, "no IntentClassified event — agent stream is broken"
    assert str(intent_classified[0].intent) == "plan", (
        f"haiku didn't classify the FR-search prompt as PLAN (got "
        f"{str(intent_classified[0].intent)!r} confidence={intent_classified[0].confidence:.2f}) "
        "— if this happens we should pick a different exemplar prompt for haiku, not assume the "
        "fix is broken"
    )

    mismatches = [e for e in events if isinstance(e, PatternMismatch)]
    assert len(mismatches) == 1, (
        f"expected exactly one PatternMismatch event on Haiku, got {len(mismatches)}. "
        "The dispatcher should fire model-agnostically when ChatAgent sees PLAN intent."
    )

    tool_calls = [
        e
        for e in events
        if isinstance(e, Span)
        and e.subject == SpanSubject.TOOL_CALL
        and e.phase == SpanPhase.COMPLETE
    ]
    assert len(tool_calls) >= 1, (
        "Haiku redirect to _handle_tool_use produced zero tool calls "
        "(regression of the fix). ReAct fallback isn't engaging."
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_trivial_respond_intent_no_pattern_mismatch_sonnet() -> None:
    """Negative case — trivial arithmetic must NOT fire PatternMismatch.

    Confirms the gate is selective: only intent in {PLAN, RESEARCH}
    AND default handler triggers the redirect. A normal RESPOND
    intent on ChatAgent should be untouched.
    """
    agent = ChatAgent(_vfs(), runtime=_runtime(), model=SONNET, settings=_settings())
    events = [e async for e in agent.run(TRIVIAL_PROMPT, session_id="dispatch-trivial")]

    mismatches = [e for e in events if isinstance(e, PatternMismatch)]
    assert mismatches == [], (
        "PatternMismatch fired on a trivial RESPOND/TOOL_USE-intent prompt — the gate is "
        f"too broad. Got {len(mismatches)} mismatch events. Inspect the IntentClassified "
        "event to see what intent the classifier returned."
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_followup_turn_in_same_session_still_dispatches_correctly() -> None:
    """Multi-turn — the second turn in the same session must still
    dispatch via the redirect (catches per-turn state leaks).

    Pre-fix the silent-degradation bug would persist across turns
    because the dispatcher was stateless. The fix must be too.
    """
    agent = ChatAgent(_vfs(), runtime=_runtime(), model=SONNET, settings=_settings())

    turn1_events = [e async for e in agent.run(PLAN_PROMPT, session_id="dispatch-multi-turn")]
    turn1_mismatches = [e for e in turn1_events if isinstance(e, PatternMismatch)]
    turn1_tools = [
        e
        for e in turn1_events
        if isinstance(e, Span)
        and e.subject == SpanSubject.TOOL_CALL
        and e.phase == SpanPhase.COMPLETE
    ]
    assert len(turn1_mismatches) == 1
    assert len(turn1_tools) >= 1

    turn2_events = [e async for e in agent.run(FOLLOWUP_PROMPT, session_id="dispatch-multi-turn")]
    turn2_intent = [e for e in turn2_events if isinstance(e, IntentClassified)]
    assert turn2_intent, "no IntentClassified event on turn 2"
    turn2_mismatches = [e for e in turn2_events if isinstance(e, PatternMismatch)]
    turn2_tools = [
        e
        for e in turn2_events
        if isinstance(e, Span)
        and e.subject == SpanSubject.TOOL_CALL
        and e.phase == SpanPhase.COMPLETE
    ]
    # Either: turn 2 also classifies as PLAN → must mismatch + dispatch
    # tools, OR turn 2 classifies as RESPOND → no mismatch, no tools
    # required. State leaking would look like "turn 2 classifies as PLAN
    # but no mismatch event" (the bug we just fixed at the event-delivery
    # layer) or "turn 2 emits PatternMismatch but no tools fire" (the
    # redirect-broken case).
    if str(turn2_intent[0].intent) == "plan":
        assert len(turn2_mismatches) == 1, (
            "turn 2 classified as PLAN but no PatternMismatch fired — state leak across turns"
        )
        assert len(turn2_tools) >= 1, "turn 2 PatternMismatch fired but ReAct did not engage tools"
