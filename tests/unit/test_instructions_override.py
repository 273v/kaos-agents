"""Unit tests for :class:`BaseAgent.override_instructions`.

The override primitive replaces the legacy ``self._instructions =
augmented; try: ...; finally: self._instructions = saved`` mutation
that lived in ``ResearchAgent`` (and was flagged by Iteration 7 of the
kaos-llm-core design audit). These tests pin:

1. The base property resolves to ``self._instructions`` when no override
   is active.
2. The contextmanager pushes the override and auto-restores on exit.
3. Concurrent async tasks see independent overrides — the ContextVar
   propagates per task, not per instance, so simultaneous calls on the
   same agent instance never collide.
"""

from __future__ import annotations

import asyncio

import pytest
from kaos_core.registry.container import KaosRuntime

from kaos_agents.runtime.agent import BaseAgent


def _make_agent(default: str | None) -> BaseAgent:
    rt = KaosRuntime.test_mode()
    return BaseAgent(vfs=rt.vfs, model="anthropic:claude-haiku-4-5", instructions=default)


def test_property_returns_default_when_no_override() -> None:
    agent = _make_agent(default="be helpful")
    assert agent.instructions == "be helpful"


def test_property_returns_none_when_no_default_no_override() -> None:
    agent = _make_agent(default=None)
    assert agent.instructions is None


def test_override_pushes_and_restores() -> None:
    agent = _make_agent(default="be helpful")
    assert agent.instructions == "be helpful"
    with agent.override_instructions("ESCALATED: be exhaustive"):
        assert agent.instructions == "ESCALATED: be exhaustive"
    assert agent.instructions == "be helpful"


def test_override_restores_on_exception() -> None:
    agent = _make_agent(default="be helpful")
    with pytest.raises(RuntimeError, match="boom"), agent.override_instructions("ESCALATED"):
        assert agent.instructions == "ESCALATED"
        raise RuntimeError("boom")
    assert agent.instructions == "be helpful"


def test_classmethod_callable_from_subclass_or_instance() -> None:
    # The contextmanager is a classmethod so escalation paths can use
    # either ``self.override_instructions(...)`` or
    # ``BaseAgent.override_instructions(...)`` interchangeably.
    agent = _make_agent(default="base")
    with BaseAgent.override_instructions("via-class"):
        assert agent.instructions == "via-class"
    with agent.override_instructions("via-instance"):
        assert agent.instructions == "via-instance"
    assert agent.instructions == "base"


@pytest.mark.asyncio
async def test_concurrent_tasks_see_independent_overrides() -> None:
    """ContextVar is task-local; concurrent tasks must not see each other's overrides."""
    agent = _make_agent(default="base")
    observed: dict[str, list[str | None]] = {"a": [], "b": []}

    async def task(label: str, override_value: str, delay: float) -> None:
        with agent.override_instructions(override_value):
            await asyncio.sleep(delay)
            observed[label].append(agent.instructions)
            await asyncio.sleep(delay)
            observed[label].append(agent.instructions)

    # Two concurrent tasks with different overrides; if the override
    # leaked across tasks, observed lists would interleave.
    await asyncio.gather(
        task("a", "OVERRIDE-A", 0.01),
        task("b", "OVERRIDE-B", 0.01),
    )

    assert observed["a"] == ["OVERRIDE-A", "OVERRIDE-A"], (
        f"Task A saw {observed['a']!r} — ContextVar leaked from task B"
    )
    assert observed["b"] == ["OVERRIDE-B", "OVERRIDE-B"], (
        f"Task B saw {observed['b']!r} — ContextVar leaked from task A"
    )
    # And after both tasks return, the base value is restored.
    assert agent.instructions == "base"
