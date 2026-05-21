"""Tests for B0.3 — MESSAGES summarization fires in canonical-turn append.

Pre-B0.3 (broad-reliability roadmap §B0.3), the
``POST /v1/sessions/{id}/memory/messages/turn`` endpoint wrote via
``memory.add()`` only — never called ``end_turn()`` /
``summarize_turn()``. Combined with the prior fix #458 (which moved
per-iteration writes off the canonical path), this endpoint became
the SOLE persistence surface for MESSAGES. Pre-fix, a real attorney
50-turn session built up ~50k tokens of unsummarized MESSAGES by
turn 25 and the assemble-context call OOM'd the planner's prompt
budget. Symptom: planning quality silently degraded from turn 25.

Post-B0.3, ``append_memory_turn``:

1. Calls ``memory.add()`` for both user + assistant content.
2. Calls ``await memory.summarize_turn()`` (best-effort — LLM
   failures fall through to a logged warning so the canonical
   write still completes).
3. Calls ``memory.end_turn()`` to keep ``turn_count`` honest.
4. Persists via ``store.save(memory)`` (unchanged).

These tests pin the contract without requiring a live LLM —
summarize_turn either fires once (and we mock it) or is a no-op for
short sessions where no section is over budget.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from kaos_core.registry.container import KaosRuntime

from kaos_agents.api.server import create_app
from kaos_agents.api.settings import KaosAgentsApiSettings


def _make_app() -> tuple[object, ASGITransport]:
    settings = KaosAgentsApiSettings(api_allow_unauth_localhost=True)
    app = create_app(runtime=KaosRuntime.test_mode(), api_settings=settings)
    return app, ASGITransport(app=app, client=("127.0.0.1", 12345))


@pytest.mark.asyncio
async def test_turn_count_increments_per_canonical_append() -> None:
    """Every canonical-turn POST must increment ``turn_count`` so
    long-session bookkeeping stays honest. Pre-fix this never moved."""
    _app, transport = _make_app()
    sid = "sess-b03-turn-count"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(3):
            resp = await client.post(
                f"/v1/sessions/{sid}/memory/messages/turn",
                json={
                    "user_message": f"q{i}",
                    "assistant_message": f"a{i}",
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["turn_count"] == i + 1


@pytest.mark.asyncio
async def test_summarize_turn_failure_does_not_break_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM outage in ``summarize_turn`` must not abort the canonical
    write. Best-effort contract: log + continue.

    We patch ``SessionMemory.summarize_turn`` to raise so any LLM
    failure inside it is simulated without touching providers; the
    endpoint must still return 200 with ``appended=2`` and the
    MESSAGES section must still contain the appended items."""
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryType

    async def _explode(self: SessionMemory, **_kwargs: object) -> int:
        raise RuntimeError("simulated summarizer outage")

    monkeypatch.setattr(SessionMemory, "summarize_turn", _explode)

    _app, transport = _make_app()
    sid = "sess-b03-summarize-fail"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/v1/sessions/{sid}/memory/messages/turn",
            json={
                "user_message": "user q",
                "assistant_message": "assistant a",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["appended"] == 2
        assert resp.json()["turn_count"] == 1

        # MESSAGES still has both items — summarizer failure didn't
        # roll back the writes.
        read = await client.get(
            f"/v1/sessions/{sid}/memory/{MemoryType.MESSAGES.value}",
            params={"limit": 10},
        )
        items = read.json()["items"]
        contents = [item["content"] for item in items]
        assert any("user: user q" in c for c in contents)
        assert any("assistant: assistant a" in c for c in contents)


@pytest.mark.asyncio
async def test_summarize_turn_invoked_on_canonical_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical append calls ``summarize_turn`` exactly once
    per request (regardless of whether anything actually got
    summarized). This is the load-bearing wiring test for #577 B0.3."""
    from kaos_agents.memory.session import SessionMemory

    call_count = {"n": 0}

    async def _spy_summarize(self: SessionMemory, **_kwargs: object) -> int:
        call_count["n"] += 1
        return 0  # No-op return matches the "nothing to summarize" path

    monkeypatch.setattr(SessionMemory, "summarize_turn", _spy_summarize)

    _app, transport = _make_app()
    sid = "sess-b03-summarize-called"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/v1/sessions/{sid}/memory/messages/turn",
            json={"user_message": "q", "assistant_message": "a"},
        )
        assert resp.status_code == 200
    assert call_count["n"] == 1, (
        "B0.3 regression: canonical-turn append must call summarize_turn() "
        "exactly once per request. Counted "
        f"{call_count['n']} calls — pre-fix this was 0."
    )


@pytest.mark.asyncio
async def test_summarize_skipped_when_both_messages_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller posts empty user + assistant, nothing is
    appended — summarize_turn must NOT fire (no work to do, and
    incrementing turn_count would be misleading)."""
    from kaos_agents.memory.session import SessionMemory

    call_count = {"n": 0}

    async def _spy(self: SessionMemory, **_kwargs: object) -> int:
        call_count["n"] += 1
        return 0

    monkeypatch.setattr(SessionMemory, "summarize_turn", _spy)

    _app, transport = _make_app()
    sid = "sess-b03-empty"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/v1/sessions/{sid}/memory/messages/turn",
            json={"user_message": "", "assistant_message": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["appended"] == 0
        assert resp.json()["turn_count"] == 0
    assert call_count["n"] == 0
