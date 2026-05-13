"""Phase 3.D — AgentLoop classifier-driven planner selection.

Tests Resolved Decision #3: when ``planner=`` is not explicitly passed
to AgentLoop AND ``auto_select_planner=True`` (default), the loop
selects a Planner from ``intent.pattern`` at dispatch time:

  AgentPattern.CHAT     → ReActPlanner
  AgentPattern.PLAN     → PlanExecutePlanner
  AgentPattern.RESEARCH → HierarchicalPlanner

Explicit ``planner=`` always wins. ``auto_select_planner=False``
preserves the Phase 2.B skeleton path for tests that want to verify
loop behavior without exercising real planners.

The ReActPlanner auto-selected for CHAT would require an LLM call in
its ``execute()`` step. To keep these unit tests deterministic, the
selection-helper is exercised directly via the public method
``_select_planner_for_intent`` rather than through ``forward()``.
The full forward-with-auto-select integration test stubs the
auto-selected planner via monkeypatch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.config import AgentPattern
from kaos_agents.intent import IntentResult
from kaos_agents.intent.types import Goal
from kaos_agents.loop.agent_loop import AgentLoop
from kaos_agents.planning.hierarchical_planner import HierarchicalPlanner
from kaos_agents.planning.plan_execute_planner import PlanExecutePlanner
from kaos_agents.planning.react_planner import ReActPlanner
from kaos_agents.triggers import Trigger
from kaos_agents.types import IntentType


def _intent_for(
    pattern: AgentPattern, intent_type: IntentType = IntentType.RESPOND
) -> IntentResult:
    return IntentResult(
        goal=Goal(statement="test goal", intent_type=intent_type),
        constraints=(),
        ambiguities=(),
        requires_clarification=False,
        pattern=pattern,
        confidence=0.9,
        raw_input="test",
    )


class _StubExtractor:
    """Returns a pre-baked IntentResult; mimics IntentExtractor.invoke."""

    def __init__(self, intent: IntentResult) -> None:
        self._intent = intent

    async def invoke(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            output=self._intent,
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001
            ),
        )


def _stub_extractor(intent: IntentResult) -> Any:
    """Type-erased stub-extractor builder.

    AgentLoop's ``intent_extractor=`` is typed ``IntentExtractor | None``.
    The duck-typed ``_StubExtractor`` matches the runtime contract but
    not the static type. Returning ``Any`` here erases the mismatch in
    one place rather than ignoring it at eight call sites.
    """
    return _StubExtractor(intent)


class _RecordingPlanner:
    """Captures plan/execute calls without invoking real LLMs."""

    def __init__(self, label: str = "stub") -> None:
        self.label = label
        self.plan_calls = 0
        self.execute_calls = 0

    async def plan(self, intent: Any, memory: Any = None) -> SimpleNamespace:
        self.plan_calls += 1
        return SimpleNamespace(pattern="stub", goal=intent.goal.statement)

    async def execute(
        self, plan: Any, *, perceiver: Any = None, actor: Any = None
    ) -> SimpleNamespace:
        self.execute_calls += 1
        return SimpleNamespace(text=f"{self.label}-result", output=f"{self.label}-result")


@pytest.mark.unit
class TestSelectPlannerForIntent:
    """Direct unit-tests of the selection helper."""

    def test_chat_pattern_picks_react_planner(self) -> None:
        loop = AgentLoop(intent_extractor=_stub_extractor(_intent_for(AgentPattern.CHAT)))
        intent = _intent_for(AgentPattern.CHAT)
        planner = loop._select_planner_for_intent(intent)
        assert isinstance(planner, ReActPlanner)

    def test_plan_pattern_picks_plan_execute_planner(self) -> None:
        loop = AgentLoop(intent_extractor=_stub_extractor(_intent_for(AgentPattern.PLAN)))
        intent = _intent_for(AgentPattern.PLAN)
        planner = loop._select_planner_for_intent(intent)
        assert isinstance(planner, PlanExecutePlanner)

    def test_research_pattern_picks_hierarchical_planner(self) -> None:
        loop = AgentLoop(intent_extractor=_stub_extractor(_intent_for(AgentPattern.RESEARCH)))
        intent = _intent_for(AgentPattern.RESEARCH)
        planner = loop._select_planner_for_intent(intent)
        assert isinstance(planner, HierarchicalPlanner)

    def test_default_planner_model_threads_into_react(self) -> None:
        # ReActPlanner is the only auto-selected planner that takes a model.
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent_for(AgentPattern.CHAT)),
            default_planner_model="anthropic:claude-sonnet-4-6",
        )
        planner = loop._select_planner_for_intent(_intent_for(AgentPattern.CHAT))
        assert isinstance(planner, ReActPlanner)
        assert planner._model == "anthropic:claude-sonnet-4-6"


@pytest.mark.unit
class TestForwardAutoSelectsPlanner:
    """Integration test: forward() picks a planner and dispatches."""

    async def test_chat_intent_dispatches_through_react_planner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Replace ReActPlanner with a recording planner so we don't make an
        # LLM call. The selection helper builds the planner; if we patch the
        # class, the construction path returns our recording instance.
        recorded = _RecordingPlanner(label="react-stub")

        # Patch the loop's selector to return our recorder for CHAT.
        intent = _intent_for(AgentPattern.CHAT)

        def _fake_select(self: AgentLoop, _intent: IntentResult) -> Any:
            return recorded

        monkeypatch.setattr(AgentLoop, "_select_planner_for_intent", _fake_select)

        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        assert recorded.plan_calls == 1
        assert recorded.execute_calls == 1
        assert invocation.output == "react-stub-result"
        assert invocation.extras.get("selected_planner") == "_RecordingPlanner"
        # Skeleton phase should NOT be set when auto-select succeeds.
        assert invocation.extras.get("phase") != "skeleton"

    async def test_explicit_planner_wins_over_auto_select(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        explicit = _RecordingPlanner(label="explicit")
        auto = _RecordingPlanner(label="auto")

        def _fake_select(self: AgentLoop, _intent: IntentResult) -> Any:
            return auto

        monkeypatch.setattr(AgentLoop, "_select_planner_for_intent", _fake_select)

        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent_for(AgentPattern.CHAT)),
            planner=explicit,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        assert explicit.execute_calls == 1
        assert auto.execute_calls == 0
        assert invocation.output == "explicit-result"
        # selected_planner extras only set on auto-select branch.
        assert "selected_planner" not in invocation.extras

    async def test_auto_select_disabled_falls_back_to_skeleton(self) -> None:
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent_for(AgentPattern.CHAT)),
            auto_select_planner=False,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))
        assert invocation.output == ""
        assert invocation.extras.get("phase") == "skeleton"
        assert "selected_planner" not in invocation.extras


@pytest.mark.unit
class TestSelectPlannerForUnknownPattern:
    """When intent.pattern is unrecognised, selection returns None."""

    def test_unknown_pattern_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Construct an intent whose pattern is a known enum but mock the
        # selector to test the None path. AgentPattern is a closed enum
        # so a "real" unknown pattern can't be constructed without
        # bypassing pydantic validation. The helper's branch coverage is
        # adequately tested by the three known-pattern paths above; this
        # test pins the contract via direct method override.
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent_for(AgentPattern.CHAT)),
        )

        # Synthesize an "unknown" pattern by patching an attribute access.
        class _OutOfBandIntent:
            @property
            def pattern(self) -> str:
                return "definitely_not_a_known_pattern"

            goal = SimpleNamespace(statement="x", intent_type=IntentType.RESPOND)

        # Deliberately bypass the type guard to verify the
        # unrecognised-pattern fallback. cast() erases the static type
        # at the call site so ty doesn't complain.
        from typing import cast

        result = loop._select_planner_for_intent(cast(IntentResult, _OutOfBandIntent()))
        assert result is None
