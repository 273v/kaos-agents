"""Phase 5.D — ReActPlanner threads runner hooks into kaos-llm-core ReAct.

Closes the audit gap "two parallel observability systems that never meet."
A single :class:`KaosHook` on the Runner now observes both:

  - Outer agent events (Span(TURN/STEP/...), IntentClassified, TurnSummary)
  - Inner kaos-llm-core CallHooks / ProgramHooks events from the ReAct
    Program

via the Phase 0.C :func:`adapt_hooks` adapter wired into the planner
constructor.

These tests stub the inner ReAct so we don't make live LLM calls — the
contract under test is "did the planner construct ReAct with the
adapted CallHooks / ProgramHooks?", not "does the inner ReAct fire
hooks correctly" (the latter is owned by kaos-llm-core's own tests).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.hooks.base import KaosHook
from kaos_agents.intent.types import Goal, IntentResult
from kaos_agents.planning.react_planner import ReActPlanner
from kaos_agents.types import IntentType


def _intent() -> IntentResult:
    from kaos_agents.config import AgentPattern

    return IntentResult(
        goal=Goal(statement="test", intent_type=IntentType.RESPOND),
        constraints=(),
        ambiguities=(),
        requires_clarification=False,
        pattern=AgentPattern.CHAT,
        confidence=0.9,
        raw_input="test",
    )


@pytest.mark.unit
class TestReActPlannerHookThreading:
    """Verify hooks= constructor kwarg flows into ReAct construction."""

    async def test_no_hooks_skips_adapter(self) -> None:
        """Default ``hooks=()`` skips the adapter; ReAct gets no
        ``hooks=`` / ``program_hooks=`` kwargs (kaos-llm-core defaults
        apply)."""
        captured_kwargs: dict[str, Any] = {}

        def _capture(self_signature: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            # Return a stubbed ReAct-like object — _build_react only
            # ever returns the constructed object.
            from types import SimpleNamespace

            return SimpleNamespace(name="stub-react")

        with patch(
            "kaos_agents.planning.react_planner.ReAct",
            side_effect=_capture,
        ):
            planner = ReActPlanner()
            # Force construction by calling _build_react directly.
            planner._build_react(tools=())

        # No hooks were passed (the adapter wasn't invoked).
        assert "hooks" not in captured_kwargs
        assert "program_hooks" not in captured_kwargs

    async def test_hooks_threaded_through_adapter(self) -> None:
        """When hooks= is non-empty, ReAct is constructed with adapted
        CallHooks + ProgramHooks."""
        captured_kwargs: dict[str, Any] = {}

        def _capture(self_signature: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(name="stub-react")

        my_hook = KaosHook()
        with patch(
            "kaos_agents.planning.react_planner.ReAct",
            side_effect=_capture,
        ):
            planner = ReActPlanner(hooks=(my_hook,))
            planner._build_react(tools=())

        # CallHooks + ProgramHooks present.
        from kaos_llm_core.programs.hooks import CallHooks
        from kaos_llm_core.programs.program_hooks import ProgramHooks

        assert isinstance(captured_kwargs.get("hooks"), CallHooks)
        assert isinstance(captured_kwargs.get("program_hooks"), ProgramHooks)

    async def test_multiple_hooks_threaded_together(self) -> None:
        """Multiple KaosHooks pass through adapt_hooks as one bundle."""
        captured_kwargs: dict[str, Any] = {}

        def _capture(self_signature: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(name="stub-react")

        h1 = KaosHook()
        h2 = KaosHook()
        with patch(
            "kaos_agents.planning.react_planner.ReAct",
            side_effect=_capture,
        ):
            planner = ReActPlanner(hooks=(h1, h2))
            planner._build_react(tools=())

        # Both layers present.
        from kaos_llm_core.programs.hooks import CallHooks
        from kaos_llm_core.programs.program_hooks import ProgramHooks

        assert isinstance(captured_kwargs.get("hooks"), CallHooks)
        assert isinstance(captured_kwargs.get("program_hooks"), ProgramHooks)


@pytest.mark.unit
class TestAgentLoopHookForwarding:
    """AgentLoop's auto-selected ReActPlanner forwards self._hooks."""

    async def test_auto_selected_react_planner_carries_hooks(self) -> None:
        """When the loop auto-selects ReActPlanner, it constructs it
        with hooks=self._hooks so a Runner-level OTelHook reaches the
        inner ReAct."""
        from kaos_agents.loop.agent_loop import AgentLoop

        my_hook = KaosHook()
        loop = AgentLoop(hooks=(my_hook,))
        planner = loop._select_planner_for_intent(_intent())
        assert isinstance(planner, ReActPlanner)
        assert planner._hooks == (my_hook,), (
            "Phase 5.D: AgentLoop must thread its KaosHook tuple into "
            "the auto-selected ReActPlanner's hooks= kwarg so the inner "
            "ReAct's CallHooks/ProgramHooks observe the same hook tree "
            "as the agent layer."
        )
