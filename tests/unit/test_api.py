"""Tests for the FastAPI HTTP API (Phase 3).

Covers:
- POST /v1/sessions/{id}/messages returns SSE stream
- POST /v1/sessions/{id}/messages/json returns JSON response
- POST /v1/sessions creates a session
- GET /v1/sessions/{id} returns session state
- DELETE /v1/sessions/{id} deletes session
- 404 for nonexistent sessions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from kaos_agents.api.server import create_app
from kaos_agents.types import IntentResult, IntentType


@pytest.fixture
def app():
    """Create a test FastAPI app with no runtime (tool-free)."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
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
