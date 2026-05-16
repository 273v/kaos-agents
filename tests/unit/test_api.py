"""Tests for the FastAPI HTTP API (Phase 3).

Covers:
- POST /v1/sessions/{id}/messages returns SSE stream
- POST /v1/sessions/{id}/messages/json returns JSON response
- POST /v1/sessions creates a session
- GET /v1/sessions/{id} returns session state
- DELETE /v1/sessions/{id} deletes session
- 404 for nonexistent sessions

KC17-P0-3: ``create_app()`` now requires an auth source. These tests use
the localhost-dev escape hatch (``api_allow_unauth_localhost=True``) so
the existing pre-auth flows still exercise. See ``test_api_auth.py`` for
the bearer-token + tenant-scoping + CORS regression coverage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from kaos_core import KaosRuntime

from kaos_agents.api.server import create_app
from kaos_agents.api.settings import KaosAgentsApiSettings
from kaos_agents.types import IntentResult, IntentType


@pytest.fixture
def app():
    """Create a test FastAPI app with an isolated in-memory runtime.

    The disk-backed VFS is the production default for ``create_app()``;
    tests pass an explicit in-memory runtime via ``KaosRuntime.test_mode()``
    to keep the working directory free of ``.kaos-vfs/`` artifacts and
    each test isolated from prior runs.
    """
    settings = KaosAgentsApiSettings(api_allow_unauth_localhost=True)
    return create_app(runtime=KaosRuntime.test_mode(), api_settings=settings)


@pytest.fixture
async def client(app):
    """Create an async test client (binds to 127.0.0.1)."""
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.unit
class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_create_session(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/sessions", json={"session_id": "test-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-1"
        assert data["turn_count"] == 0
        assert isinstance(data["sections"], list)

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session(self, client: AsyncClient) -> None:
        # Create then delete
        await client.post("/v1/sessions", json={"session_id": "to-delete"})
        resp = await client.delete("/v1/sessions/to-delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_actually_removes_persisted_memory(self, client: AsyncClient, app) -> None:
        """KC17-P1-1 regression: DELETE removes the persisted session.

        Pre-KC17 the API called ``vfs.cleanup_context(session_id)`` only,
        which did not remove ``kaos-agents/sessions/{id}/memory.json``.
        After DELETE, ``SessionStore.exists()`` stayed True and GET
        returned 200.
        """
        from kaos_agents.memory.session import SessionMemory
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        # Seed a real persisted session (not just load_or_create which
        # leaves nothing on disk until save()).
        store = SessionStore(app.state.vfs)
        mem = SessionMemory("delete-proof")
        mem.add(MemoryType.MESSAGES, "user: please remember this")
        await store.save(mem)
        assert await store.exists("delete-proof"), "fixture precondition"

        # DELETE must actually delete.
        resp = await client.delete("/v1/sessions/delete-proof")
        assert resp.status_code == 200

        # The store must agree.
        assert not await store.exists("delete-proof"), (
            "KC17-P1-1 regression: SessionStore.exists() still True after "
            "DELETE — the persisted memory.json was not removed"
        )

        # The follow-up GET should be 404.
        get_resp = await client.get("/v1/sessions/delete-proof")
        assert get_resp.status_code == 404


@pytest.mark.unit
class TestMessageEndpoints:
    @pytest.mark.asyncio
    async def test_send_message_sse_stream(self, client: AsyncClient) -> None:
        """POST /messages returns SSE-formatted event stream."""
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Hello from agent!", []),
            ),
        ):
            resp = await client.post(
                "/v1/sessions/sse-test/messages",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        text = resp.text
        # Span boundaries are wire-typed as "span"; the typed turn-end
        # aggregate is "turn_summary".
        assert "event: span" in text
        assert "event: intent_classified" in text
        assert "event: turn_summary" in text

    @pytest.mark.asyncio
    async def test_send_message_json_via_accept_header(self, client: AsyncClient) -> None:
        """Same URL returns JSON when Accept: application/json is requested."""
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("JSON response!", []),
            ),
        ):
            resp = await client.post(
                "/v1/sessions/json-test/messages",
                json={"message": "Hello"},
                headers={"Accept": "application/json"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert data["text"] == "JSON response!"
        assert data["intent"] == "respond"

    @pytest.mark.asyncio
    async def test_approve_endpoint_404_for_unknown_run(self, client: AsyncClient) -> None:
        """POST /v1/runs/{id}/approve returns 404 if the run is not paused."""
        resp = await client.post(
            "/v1/runs/nonexistent/approve",
            json={"approved": True},
        )
        assert resp.status_code == 404
        assert "no paused run" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_approve_endpoint_streams_resume_events(self, client: AsyncClient, app) -> None:
        """POST /v1/runs/{id}/approve with a denied run streams a RunError."""
        from kaos_agents.runtime.interrupts import PendingToolCall, RunState, save_run_state

        # Persist a fake RunState so the endpoint can find it
        state = RunState(
            run_id="api_run_test",
            session_id="api_sess",
            pending_tool_call=PendingToolCall(call_id="tc", tool_name="dangerous"),
            original_message="kill",
        )
        await save_run_state(state, app.state.vfs)

        resp = await client.post(
            "/v1/runs/api_run_test/approve",
            json={"approved": False},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = resp.text
        # Denial path yields a single RunError SSE event
        assert "event: run_error" in text
        assert "approval_denied" in text

    @pytest.mark.asyncio
    async def test_memory_query_endpoint_404(self, client: AsyncClient) -> None:
        """GET /memory/{section} returns 404 with three-part message for unknown session."""
        resp = await client.get("/v1/sessions/does-not-exist/memory/messages")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "does-not-exist" in detail
        assert "verify" in detail.lower()
        assert "alternative" in detail.lower()

    @pytest.mark.asyncio
    async def test_memory_query_endpoint_400_invalid_section(
        self, client: AsyncClient, app
    ) -> None:
        """GET /memory/{bad-section} returns 400 listing valid section names.

        Seeds a session by saving memory directly, so load() succeeds and the
        400 branch (unknown section) fires.
        """
        from kaos_agents.memory.store import SessionStore

        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create("mem-badsec")
        await store.save(memory)

        resp = await client.get("/v1/sessions/mem-badsec/memory/not_a_section")
        assert resp.status_code == 400
        assert "not_a_section" in resp.json()["detail"]
        assert "messages" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_memory_query_endpoint_returns_items(self, client: AsyncClient, app) -> None:
        """GET /memory/messages returns the messages section content."""
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        # Seed a session with a message
        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create("mem-seeded")
        memory.add(MemoryType.MESSAGES, "user: hello")
        memory.add(MemoryType.MESSAGES, "assistant: hi")
        await store.save(memory)

        resp = await client.get("/v1/sessions/mem-seeded/memory/messages", params={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "mem-seeded"
        assert data["section"] == "messages"
        assert data["item_count"] == 2
        contents = [item["content"] for item in data["items"]]
        assert any("user: hello" in c for c in contents)

    @pytest.mark.asyncio
    async def test_memory_search_endpoint_404(self, client: AsyncClient) -> None:
        """GET /memory/search returns 404 for unknown session."""
        resp = await client.get("/v1/sessions/nope/memory/search", params={"query": "x"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_memory_search_endpoint_returns_results(self, client: AsyncClient, app) -> None:
        """GET /memory/search returns BM25 results from seeded session."""
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create("search-seeded")
        memory.add(MemoryType.MESSAGES, "The quick brown fox jumps")
        memory.add(MemoryType.MESSAGES, "The lazy dog sleeps")
        await store.save(memory)

        resp = await client.get(
            "/v1/sessions/search-seeded/memory/search",
            params={"query": "fox", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "search-seeded"
        assert data["query"] == "fox"
        assert data["result_count"] >= 1

    @pytest.mark.asyncio
    async def test_default_accept_is_sse(self, client: AsyncClient) -> None:
        """No Accept header defaults to SSE streaming."""
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("stream me", []),
            ),
        ):
            resp = await client.post(
                "/v1/sessions/default-accept/messages",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")


@pytest.mark.unit
class TestResolveVFS:
    """``create_app(runtime=None)`` must default to a disk-backed VFS.

    The kaos-core platform decision is "disk-first VFS" — ``KaosRuntime()``
    and ``VirtualFileSystem()`` both default to ``StorageBackend.DISK``
    rooted at ``.kaos-vfs/``. ``create_app()`` is one of the construction
    sites in the kaos-* ecosystem and must follow the same convention so
    persisted ``SessionMemory`` survives uvicorn restarts.

    Pre-fix this returned an in-memory VFS, silently losing every
    conversation on restart.
    """

    def test_default_vfs_is_disk_backed(self, monkeypatch, tmp_path):
        """No runtime → disk-backed VFS, not in-memory."""
        from kaos_core.types.enums import StorageBackend

        from kaos_agents.api.server import _resolve_vfs

        monkeypatch.chdir(tmp_path)

        vfs = _resolve_vfs(None)
        assert vfs.config.default_backend == StorageBackend.DISK

    def test_runtime_vfs_passed_through(self):
        """Explicit runtime.vfs takes precedence over the default."""
        from kaos_core.types.enums import StorageBackend

        from kaos_agents.api.server import _resolve_vfs

        rt = KaosRuntime.test_mode()  # in-memory + isolated
        vfs = _resolve_vfs(rt)
        assert vfs is rt.vfs
        assert vfs.config.default_backend == StorageBackend.MEMORY
