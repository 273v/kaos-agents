"""Regression tests for the AgenticLoop iteration leak (task #458).

Pre-fix bug: every iteration of an outer AgenticLoop POSTed the SAME
user message back to ``/v1/sessions/{id}/messages``. ``BaseAgent.run``
unconditionally appended ``user: <msg>`` and ``assistant: <draft>``
to ``SessionMemory.MESSAGES`` on every call, so a 3-iteration loop
left 3 user-msg entries + 3 intermediate assistant-msg entries in
the section. The next turn's context assembly then fed the agent its
own self-conversation as if it were authoritative prior turns.

Fix: ``BaseAgent.run(..., is_internal_iteration=True)`` skips both
writes. A new ``POST /v1/sessions/{id}/memory/messages/turn``
endpoint performs the canonical (user, final-assistant) write once
at loop exit. See ``docs/plans/2026-05-19-agentic-loop-honesty.md``
§3.1.a for the design.

These tests exercise the agent-layer contract directly — no FastAPI
app, no httpx — so the regression net stays fast and deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.events import KaosEvent, TextDelta
from kaos_agents.runtime.agent import BaseAgent
from kaos_agents.types import ZERO_USAGE, IntentResult, IntentType, InvocationUsage
from kaos_agents.types.memory import MemoryType


def _vfs() -> VirtualFileSystem:
    """Test-isolated in-memory VFS (matches the canonical fixture)."""
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        )
    )


def _fake_intent() -> IntentResult:
    """Minimal IntentResult so the dispatch path runs without an LLM."""
    return IntentResult(
        intent=IntentType.RESPOND,
        confidence=0.9,
        reasoning="stub",
    )


class _StubBaseAgent(BaseAgent):
    """``BaseAgent`` subclass with classifier + dispatch stubbed out.

    Bypasses the LLM so the test can focus on the memory-write contract.
    """

    def __init__(self, vfs: VirtualFileSystem, *, response: str) -> None:
        super().__init__(vfs)
        self._stub_response = response

    async def _classify(
        self,
        message: str,
        memory: Any,
        context_items: Any | None = None,
    ) -> IntentResult:
        return _fake_intent()

    async def _handle_respond(
        self,
        message: str,
        memory: Any,
        context_items: Any,
    ) -> tuple[str, list, InvocationUsage]:
        return self._stub_response, [], ZERO_USAGE


async def _drain(stream: AsyncIterator[KaosEvent]) -> list[KaosEvent]:
    return [event async for event in stream]


@pytest.mark.unit
class TestIterationLeakUserMessage:
    """``is_internal_iteration=True`` must NOT append a user-msg entry."""

    @pytest.mark.asyncio
    async def test_default_writes_user_message(self) -> None:
        """Baseline: the default code path (is_internal_iteration=False)
        still persists the user message (back-compat for single-shot
        callers)."""
        agent = _StubBaseAgent(_vfs(), response="hi there")
        await _drain(agent.run("hello", "sess-baseline"))

        memory = await agent._store.load("sess-baseline")
        msgs = memory.get_recent(MemoryType.MESSAGES, 50)
        contents = [m.content for m in msgs]
        assert any(c.startswith("user: hello") for c in contents), (
            f"Default run should persist user message — got: {contents!r}"
        )
        assert any(c.startswith("assistant: hi there") for c in contents), (
            f"Default run should persist assistant message — got: {contents!r}"
        )

    @pytest.mark.asyncio
    async def test_internal_iteration_skips_user_message(self) -> None:
        """``is_internal_iteration=True`` must NOT add ``user: <msg>``
        to ``SessionMemory.MESSAGES`` — the canonical write is performed
        once by the post-loop ``/memory/messages/turn`` endpoint."""
        agent = _StubBaseAgent(_vfs(), response="draft-1")
        await _drain(agent.run("hello", "sess-internal", is_internal_iteration=True))

        memory = await agent._store.load("sess-internal")
        msgs = memory.get_recent(MemoryType.MESSAGES, 50)
        contents = [m.content for m in msgs]
        assert not any(c.startswith("user: ") for c in contents), (
            f"Internal iteration must not persist user message — got: {contents!r}"
        )
        assert not any(c.startswith("assistant: ") for c in contents), (
            f"Internal iteration must not persist intermediate assistant — got: {contents!r}"
        )

    @pytest.mark.asyncio
    async def test_three_internal_iterations_leave_section_empty(self) -> None:
        """The regression: 3 critic-driven replays of the same user turn
        must leave ``MESSAGES`` empty (the loop will call
        ``/memory/messages/turn`` exactly once at the end).

        Pre-fix this produced 6 entries (3 user + 3 assistant)."""
        vfs = _vfs()
        agent = _StubBaseAgent(vfs, response="draft")

        for _ in range(3):
            await _drain(
                agent.run(
                    "what is the diesel rule",
                    "sess-three-iters",
                    is_internal_iteration=True,
                )
            )

        memory = await agent._store.load("sess-three-iters")
        msgs = memory.get_recent(MemoryType.MESSAGES, 50)
        assert len(msgs) == 0, (
            f"After 3 internal iterations MESSAGES should be empty "
            f"(canonical write happens post-loop) — got {len(msgs)} entry(ies): "
            f"{[m.content for m in msgs]!r}"
        )

    @pytest.mark.asyncio
    async def test_text_delta_still_streams_under_internal_iteration(self) -> None:
        """Internal iterations must still surface the assistant's draft on
        the SSE wire — the SPA worker reads it to feed the critic. Only
        the persistent MEMORY write is gated."""
        agent = _StubBaseAgent(_vfs(), response="visible draft")
        events = await _drain(agent.run("hello", "sess-stream", is_internal_iteration=True))

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        joined = "".join(td.content for td in text_deltas)
        assert "visible draft" in joined, (
            f"TextDelta stream missing under internal iteration: {joined!r}"
        )


@pytest.mark.unit
class TestIterationLeakViaRunner:
    """``Runner.run(..., is_internal_iteration=True)`` forwards the flag
    down to the internal agent. End-to-end check that the kwarg survives
    the Runner indirection."""

    @pytest.mark.asyncio
    async def test_runner_threads_flag_to_internal_agent(self) -> None:
        from kaos_agents.config import Agent
        from kaos_agents.runtime.runner import Runner

        captured: dict[str, Any] = {}

        async def _spy_run(
            self: Any,
            message: str,
            session_id: str,
            *,
            is_internal_iteration: bool = False,
        ) -> AsyncIterator[KaosEvent]:
            captured["is_internal_iteration"] = is_internal_iteration
            if False:  # make this an async generator
                yield  # pragma: no cover

        agent_config = Agent()
        runner = Runner(agent_config, vfs=_vfs())

        with patch("kaos_agents.runtime.agent.BaseAgent.run", _spy_run):
            async for _ in runner.run("hi", "sess-runner", is_internal_iteration=True):
                pass

        assert captured["is_internal_iteration"] is True

    @pytest.mark.asyncio
    async def test_runner_default_is_not_internal(self) -> None:
        from kaos_agents.config import Agent
        from kaos_agents.runtime.runner import Runner

        captured: dict[str, Any] = {}

        async def _spy_run(
            self: Any,
            message: str,
            session_id: str,
            *,
            is_internal_iteration: bool = False,
        ) -> AsyncIterator[KaosEvent]:
            captured["is_internal_iteration"] = is_internal_iteration
            if False:
                yield  # pragma: no cover

        agent_config = Agent()
        runner = Runner(agent_config, vfs=_vfs())

        with patch("kaos_agents.runtime.agent.BaseAgent.run", _spy_run):
            async for _ in runner.run("hi", "sess-runner-default"):
                pass

        assert captured["is_internal_iteration"] is False


@pytest.mark.unit
class TestMemoryTurnEndpoint:
    """The post-loop ``POST /v1/sessions/{id}/memory/messages/turn``
    endpoint writes the canonical (user, final-assistant) pair to
    ``SessionMemory.MESSAGES`` without invoking the LLM."""

    @pytest.mark.asyncio
    async def test_endpoint_writes_user_and_assistant(self) -> None:
        from httpx import ASGITransport, AsyncClient
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.api.server import create_app
        from kaos_agents.api.settings import KaosAgentsApiSettings

        settings = KaosAgentsApiSettings(api_allow_unauth_localhost=True)
        app = create_app(runtime=KaosRuntime.test_mode(), api_settings=settings)
        transport = ASGITransport(app=app, client=("127.0.0.1", 12345))

        sid = "sess-memory-turn"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/sessions/{sid}/memory/messages/turn",
                json={
                    "user_message": "what is the diesel rule",
                    "assistant_message": "The diesel rule is 40 CFR Part 80.",
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["session_id"] == sid
            assert body["appended"] == 2

            # Read MESSAGES back via the existing memory endpoint.
            read = await client.get(
                f"/v1/sessions/{sid}/memory/{MemoryType.MESSAGES.value}",
                params={"limit": 10},
            )
            assert read.status_code == 200, read.text
            items = read.json()["items"]
            contents = [item["content"] for item in items]
            assert any("user: what is the diesel rule" in c for c in contents)
            assert any("assistant: The diesel rule is 40 CFR Part 80." in c for c in contents)

    @pytest.mark.asyncio
    async def test_endpoint_skips_empty_fields(self) -> None:
        from httpx import ASGITransport, AsyncClient
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.api.server import create_app
        from kaos_agents.api.settings import KaosAgentsApiSettings

        settings = KaosAgentsApiSettings(api_allow_unauth_localhost=True)
        app = create_app(runtime=KaosRuntime.test_mode(), api_settings=settings)
        transport = ASGITransport(app=app, client=("127.0.0.1", 12345))

        sid = "sess-memory-turn-partial"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/sessions/{sid}/memory/messages/turn",
                json={"user_message": "", "assistant_message": "answer-only"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["appended"] == 1
