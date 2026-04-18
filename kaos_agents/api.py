"""FastAPI application for kaos-agents streaming API.

Exposes the Agent/Runner as an HTTP service with SSE streaming.
This module requires the ``[api]`` extra: ``pip install kaos-agents[api]``.

Endpoints:
- POST /v1/sessions/{session_id}/messages — send message, stream SSE events
- POST /v1/sessions — create a new session
- GET /v1/sessions/{session_id} — get session state
- DELETE /v1/sessions/{session_id} — delete session
- GET /v1/sessions/{session_id}/memory/{section} — read memory section
- GET /v1/sessions/{session_id}/memory/search — BM25 search memory
"""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kaos_core import KaosRuntime
from kaos_core.logging import get_logger
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.runner import Runner
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.wire import events_to_sse

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    """Request body for sending a message to an agent session."""

    message: str = Field(description="The user's message to the agent.")
    model: str | None = Field(
        default=None,
        description="LLM model override. Default: anthropic:claude-haiku-4-5.",
    )
    pattern: str = Field(
        default="chat",
        description="Agent pattern: 'chat', 'plan', or 'research'.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool name patterns (globs) to make available.",
    )


class SessionCreateRequest(BaseModel):
    """Request body for creating a session (optional metadata)."""

    session_id: str = Field(description="Unique session identifier.")


class ApprovalRequest(BaseModel):
    """Request body for approving or denying a paused run."""

    approved: bool = Field(
        description="True to continue the run, False to deny and yield a RunError."
    )


class SessionResponse(BaseModel):
    """Response body for session state."""

    session_id: str
    turn_count: int
    sections: list[str]


class MemoryItemResponse(BaseModel):
    """A single memory item."""

    content: str
    # added_at is a monotonic-ish timestamp (float seconds) from the
    # SessionMemory item. Serialized as a float, not an ISO string.
    added_at: float


class MemoryQueryResponse(BaseModel):
    """Response body for memory queries."""

    session_id: str
    section: str
    turn_count: int
    item_count: int
    items: list[MemoryItemResponse]


class MemorySearchResultResponse(BaseModel):
    """A single search result."""

    content: str
    section: str
    score: float


class MemorySearchResponse(BaseModel):
    """Response body for memory search."""

    session_id: str
    query: str
    result_count: int
    results: list[MemorySearchResultResponse]


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    runtime: KaosRuntime | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create the FastAPI application for kaos-agents.

    Args:
        runtime: KaosRuntime for tool execution. None for tool-free agents.
        cors_origins: Allowed CORS origins. Default: ["*"] (permissive for dev).

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="KAOS Agents API",
        description="Streaming agent API with SSE event delivery.",
        version="0.1.0",
    )

    # CORS for web app development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store runtime in app state for dependency injection
    app.state.runtime = runtime
    app.state.vfs = _resolve_vfs(runtime)

    # Register routes
    _register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.post("/v1/sessions/{session_id}/messages", response_model=None)
    async def send_message(
        session_id: str = Path(description="Session identifier"),
        body: MessageRequest = ...,  # type: ignore[assignment]
        accept: str = Header(default="text/event-stream"),
    ) -> StreamingResponse | JSONResponse:
        """Send a message; responds per ``Accept`` header content negotiation.

        - ``Accept: text/event-stream`` (default): streams SSE events.
        - ``Accept: application/json``: collects all events and returns
          a single JSON response with the final text, intent, tool calls,
          and token usage.

        Each SSE message has:
        - ``event:`` — event type (e.g., ``turn_start``, ``tool_call_start``)
        - ``data:`` — JSON-encoded event
        - ``id:`` — sequence number for reconnection
        """
        agent_config = Agent(
            pattern=AgentPattern(body.pattern),
            model=body.model or DEFAULT_MODEL,
            tools=tuple(body.tools),
        )
        runner = Runner(
            agent_config,
            runtime=app.state.runtime,
            vfs=app.state.vfs,
        )

        # Content negotiation: JSON-only clients take the blocking path.
        if "application/json" in accept and "text/event-stream" not in accept:
            response = await runner.turn(body.message, session_id)
            return JSONResponse(
                {
                    "text": response.text,
                    "intent": response.intent.intent.value if response.intent else "unknown",
                    "turn_number": response.turn_number,
                    "tokens_used": response.tokens_used,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "result_summary": tc.result_summary,
                            "is_error": tc.is_error,
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

        # Default: SSE streaming
        event_stream = runner.run(body.message, session_id)
        return StreamingResponse(
            events_to_sse(event_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/runs/{run_id}/approve", response_model=None)
    async def approve_run(
        run_id: str = Path(description="Run identifier from a paused ToolCallApprovalRequired"),
        body: ApprovalRequest = ...,  # type: ignore[assignment]
    ) -> StreamingResponse:
        """Approve or deny a paused run, then stream the continuation as SSE.

        Loads the persisted ``RunState`` for the given ``run_id`` from VFS,
        invokes ``Runner.resume(state, approved=body.approved)``, and
        streams the resulting events back as SSE.

        Returns 404 if no RunState exists for the run_id (the run was
        never paused, or the VFS state has expired).
        """
        from kaos_agents.interrupts import load_run_state

        try:
            state = await load_run_state(run_id, app.state.vfs)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No paused run found for run_id={run_id!r}: {exc}. "
                    "Either the run was never paused, or the VFS state has expired. "
                    "Check the run_state_ref returned with the original ToolCallApprovalRequired event."
                ),
            ) from exc

        # WS-0.3: reconstruct the Runner with the original Agent config
        # when available. Pre-WS-0.3 this fell back to ``Agent()`` defaults,
        # silently losing pattern / model / tools / instructions from the
        # paused run (so a paused plan would resume as default chat).
        # The snapshot is captured at pause time on ``state.agent_config``;
        # older RunStates without it continue to fall back to defaults.
        agent_config = state.agent_config.to_agent() if state.agent_config is not None else Agent()
        runner = Runner(
            agent_config,
            runtime=app.state.runtime,
            vfs=app.state.vfs,
        )
        event_stream = runner.resume(state, approved=body.approved)
        return StreamingResponse(
            events_to_sse(event_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/sessions", response_model=SessionResponse)
    async def create_session(
        body: SessionCreateRequest,
    ) -> SessionResponse:
        """Create a new session (initializes memory)."""
        from kaos_agents.memory.store import SessionStore

        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create(body.session_id)
        section_names = [mt.value for mt in memory._sections]
        return SessionResponse(
            session_id=body.session_id,
            turn_count=memory.turn_count,
            sections=section_names,
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str = Path(description="Session identifier"),
    ) -> SessionResponse:
        """Get session state (turn count, configured sections)."""
        from kaos_agents.memory.store import SessionStore

        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found: {exc}. "
                "Check the session_id matches a previous message call.",
            ) from exc
        section_names = [mt.value for mt in memory._sections]
        return SessionResponse(
            session_id=session_id,
            turn_count=memory.turn_count,
            sections=section_names,
        )

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(
        session_id: str = Path(description="Session identifier"),
    ) -> dict[str, str]:
        """Delete a session and its memory."""
        vfs: VirtualFileSystem = app.state.vfs
        await vfs.cleanup_context(session_id)
        return {"status": "deleted", "session_id": session_id}

    # Note: register /memory/search BEFORE /memory/{section} so FastAPI's
    # first-match routing sends "search" to the search endpoint rather than
    # treating it as a section name.
    @app.get(
        "/v1/sessions/{session_id}/memory/search",
        response_model=MemorySearchResponse,
    )
    async def search_memory_endpoint(
        session_id: str = Path(description="Session identifier"),
        query: str = Query(description="Search query"),
        top_k: int = Query(default=10, ge=1, le=50, description="Max results"),
    ) -> MemorySearchResponse:
        """BM25 search across memory sections."""
        from kaos_agents.memory.search import search_memory
        from kaos_agents.memory.store import SessionStore

        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Session '{session_id}' not found when searching memory: {exc}. "
                    "Verify the session_id matches a previous message call. "
                    "Alternative: GET /v1/sessions/{id} first to check the session exists."
                ),
            ) from exc

        results = search_memory(memory, query, top_k=top_k)
        return MemorySearchResponse(
            session_id=session_id,
            query=query,
            result_count=len(results),
            results=[
                MemorySearchResultResponse(
                    content=r.content[:200],
                    section=r.section.value,
                    score=round(r.score, 4),
                )
                for r in results
            ],
        )

    @app.get(
        "/v1/sessions/{session_id}/memory/{section}",
        response_model=MemoryQueryResponse,
    )
    async def get_memory(
        session_id: str = Path(description="Session identifier"),
        section: str = Path(description="Memory section name"),
        limit: int = Query(default=20, ge=1, le=100, description="Max items"),
    ) -> MemoryQueryResponse:
        """Read a memory section."""
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.memory.types import MemoryType

        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Session '{session_id}' not found when reading memory: {exc}. "
                    "Verify the session_id matches a previous POST to "
                    "/v1/sessions/{id}/messages. "
                    "Alternative: POST /v1/sessions first to pre-create the session."
                ),
            ) from exc

        try:
            mem_type = MemoryType(section)
        except ValueError as exc:
            valid = ", ".join(mt.value for mt in MemoryType)
            raise HTTPException(
                status_code=400,
                detail=f"Unknown section '{section}'. Valid: {valid}.",
            ) from exc

        items = memory.get_recent(mem_type, limit)
        return MemoryQueryResponse(
            session_id=session_id,
            section=section,
            turn_count=memory.turn_count,
            item_count=len(items),
            items=[
                MemoryItemResponse(content=item.content, added_at=item.added_at) for item in items
            ],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_vfs(runtime: KaosRuntime | None) -> VirtualFileSystem:
    """Get VFS from runtime, falling back to in-memory VFS."""
    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs
    config = VFSConfig(default_backend=StorageBackend.MEMORY)
    return VirtualFileSystem(config=config)
