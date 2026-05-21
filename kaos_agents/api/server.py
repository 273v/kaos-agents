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

All endpoints require auth (KC17-P0-3): either a bearer token via
``Authorization: Bearer <KAOS_AGENTS_API_TOKEN>`` OR an unauthenticated
request from 127.0.0.1 when ``KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1``.
``create_app`` refuses to start with no auth source configured.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kaos_core import KaosRuntime
from kaos_core.logging import get_logger
from kaos_core.vfs.core import VirtualFileSystem
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from kaos_agents.api.settings import (
    KaosAgentsApiSettings,
    scope_session_id,
)
from kaos_agents.api.wire import events_to_sse
from kaos_agents.config import Agent, AgentPattern
from kaos_agents.runtime.runner import Runner
from kaos_agents.settings import DEFAULT_MODEL

logger = get_logger(__name__)

_LOCALHOST_IPS = frozenset({"127.0.0.1", "::1"})

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
    max_cost_usd: float | None = Field(
        default=None,
        description=(
            "Per-plan cost ceiling in USD. Threads to KaosAgentSettings."
            "plan_max_cost_usd. When the plan exceeds this cap, the Runner "
            "emits a BudgetExceeded event and stops."
        ),
    )
    require_approval_for_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns of tool names that require explicit approval "
            "(ASK rule). When the agent tries to call a matching tool, "
            "the Runner pauses with ToolCallApprovalRequired. The "
            "non-streaming JSON path will return an empty text response "
            "(approval not yet implemented over JSON). Use the SSE path "
            "or /v1/runs/{run_id}/approve to resume."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "System-prompt override for the agent (persona, rubric, "
            "refusal policy). Mirrors the CLI's --instructions flag and "
            "the MCP tool's instructions parameter. Example: 'You are "
            "senior counsel; flag every deviation by exact filename "
            "and section.'"
        ),
    )
    is_internal_iteration: bool = Field(
        default=False,
        description=(
            "True when the caller is replaying the SAME user message under "
            "an outer multi-iteration loop (e.g. kaos-agents' AgenticLoop "
            "critic-driven replan). The agent will NOT persist the user "
            "message to SessionMemory.MESSAGES (it's already there from "
            "iteration 1), and will NOT persist the intermediate assistant "
            "response. After the loop terminates, the caller is expected to "
            "POST the canonical (user, final-assistant) pair to "
            "/v1/sessions/{id}/memory/messages/turn so memory has exactly "
            "one user entry + one assistant entry per turn. Default False "
            "keeps single-turn callers unchanged."
        ),
    )


class MemoryTurnRequest(BaseModel):
    """Request body for POST /v1/sessions/{id}/memory/messages/turn.

    Writes the canonical (user, final-assistant) pair to
    ``SessionMemory.MESSAGES`` without running the LLM. Used by outer
    multi-iteration loops (e.g. kaos-agents' AgenticLoop) that delegated
    each iteration with ``is_internal_iteration=True`` and now need to
    persist the resolved turn exactly once. See the iteration-leak fix
    in ``docs/plans/2026-05-19-agentic-loop-honesty.md`` §3.1.a.
    """

    user_message: str = Field(description="The user's message for this turn.")
    assistant_message: str = Field(description="The agent's final assistant text for this turn.")


class MemoryTurnResponse(BaseModel):
    """Response body for POST /v1/sessions/{id}/memory/messages/turn."""

    session_id: str
    turn_count: int
    appended: int = Field(
        description="Number of memory items appended (typically 2: user + assistant).",
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
    api_settings: KaosAgentsApiSettings | None = None,
) -> FastAPI:
    """Create the FastAPI application for kaos-agents.

    Args:
        runtime: KaosRuntime for tool execution. None for tool-free agents.
        cors_origins: Explicit CORS origins override. When None, falls back
            to ``api_settings.api_cors_allow_origins``. Wildcard ``*`` is
            rejected by :class:`KaosAgentsApiSettings` because it is
            incompatible with credentialed requests.
        api_settings: Auth + CORS config (KC17-P0-3). When None, loads from
            env (``KAOS_AGENTS_API_*``). At least one of ``api_token`` or
            ``api_allow_unauth_localhost`` must be set or
            :class:`InsecureApiConfigurationError` is raised.

    Returns:
        Configured FastAPI application.

    Raises:
        InsecureApiConfigurationError: when neither ``KAOS_AGENTS_API_TOKEN``
            nor ``KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1`` is configured.
            Pre-KC17 the API would happily start with no auth + wildcard
            CORS; that silent footgun is no longer permitted.
    """
    settings = api_settings if api_settings is not None else KaosAgentsApiSettings()
    if not settings.is_auth_configured():
        raise settings.insecure_startup_error()

    app = FastAPI(
        title="KAOS Agents API",
        description="Streaming agent API with SSE event delivery.",
        version="0.1.0",
    )

    # CORS — explicit allow-list only. Default empty (no cross-origin).
    effective_origins: list[str]
    if cors_origins is not None:
        # Caller override — still reject wildcards because credentials are on.
        if "*" in cors_origins:
            raise ValueError(
                "cors_origins=['*'] is incompatible with allow_credentials=True. "
                "List explicit origins instead. See KaosAgentsApiSettings docs."
            )
        effective_origins = list(cors_origins)
    else:
        effective_origins = list(settings.api_cors_allow_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=effective_origins,
        allow_credentials=bool(effective_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.api_token is None:
        logger.warning(
            "kaos-agents API: starting with no bearer token; "
            "localhost-dev mode enabled. Do NOT expose this beyond 127.0.0.1."
        )

    # Store runtime + settings in app state for dependency injection
    app.state.runtime = runtime
    app.state.vfs = _resolve_vfs(runtime)
    app.state.api_settings = settings
    app.state.tenant_id = settings.tenant_id()

    # Register routes
    _register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str | None:
    """Best-effort client IP. Honors ``X-Forwarded-For`` ONLY when the
    immediate peer is localhost — never trust a public-facing peer to
    set its own client IP. Returns the original ``request.client.host``
    otherwise.
    """
    peer = request.client.host if request.client else None
    if peer in _LOCALHOST_IPS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First entry is the original client when proxies append.
            first = forwarded.split(",")[0].strip()
            return first or peer
    return peer


async def _require_auth(request: Request) -> str | None:
    """FastAPI dependency: authenticate the request and return tenant id.

    KC17-P0-3 contract:
    - When ``api_token`` is configured: requires ``Authorization: Bearer X``
      with constant-time compare to the token. Returns the derived tenant
      id (12-char SHA-256 prefix of the token).
    - Otherwise when ``api_allow_unauth_localhost`` is configured: requires
      the request to originate from 127.0.0.1 / ::1. Emits a per-request
      warning log. Returns ``None`` (no tenant scoping).
    - Otherwise: 401 (defensive — :func:`create_app` should have refused
      to start, but if a caller monkey-patched settings, this still blocks).

    The dependency NEVER returns 403 (would leak existence of the
    endpoint to an authenticated tenant A querying tenant B's id).
    """
    settings: KaosAgentsApiSettings = request.app.state.api_settings

    if settings.api_token is not None:
        auth = request.headers.get("authorization", "")
        # Case-insensitive scheme match; the value after the space is the token.
        scheme, _, presented = auth.partition(" ")
        if scheme.lower() != "bearer" or not settings.check_token(presented):
            raise HTTPException(
                status_code=401,
                detail=(
                    "Authentication required. Send 'Authorization: Bearer "
                    "<KAOS_AGENTS_API_TOKEN>' header. "
                    "Sessions and runs are scoped per token — token A "
                    "cannot read token B's data."
                ),
            )
        return settings.tenant_id()

    if settings.api_allow_unauth_localhost:
        peer = _client_ip(request)
        if peer not in _LOCALHOST_IPS:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Localhost-dev mode: only 127.0.0.1 / ::1 origins "
                    "permitted. Either set KAOS_AGENTS_API_TOKEN to enable "
                    "bearer-token auth, or bind the API to 127.0.0.1."
                ),
            )
        logger.warning(
            "kaos-agents API: unauthenticated localhost-dev request "
            "(method=%s path=%s peer=%s). MUST NOT be exposed in production.",
            request.method,
            request.url.path,
            peer,
        )
        return None

    # Belt + suspenders — create_app should have refused to start.
    raise HTTPException(
        status_code=401,
        detail=(
            "kaos-agents API is not authenticated. Server-side misconfiguration: "
            "neither KAOS_AGENTS_API_TOKEN nor "
            "KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST is set."
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.post("/v1/sessions/{session_id}/messages", response_model=None)
    async def send_message(
        session_id: Annotated[str, Path(description="Session identifier")],
        body: Annotated[MessageRequest, Body()],
        tenant_id: Annotated[str | None, Depends(_require_auth)],
        accept: Annotated[str, Header()] = "text/event-stream",
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
        # KC17-P0-3 — tenant-scope the session id so token A cannot
        # reach token B's sessions. localhost-dev mode (tenant_id=None)
        # leaves the id unchanged.
        effective_session_id = scope_session_id(session_id, tenant_id)

        # Optional per-request settings override — currently scoped to
        # the plan-cost cap (the only field consumed by Runner today).
        agent_settings = None
        if body.max_cost_usd is not None:
            from kaos_agents.settings import KaosAgentSettings

            agent_settings = KaosAgentSettings(plan_max_cost_usd=body.max_cost_usd)

        agent_kwargs: dict[str, Any] = {
            "pattern": AgentPattern(body.pattern),
            "model": body.model or DEFAULT_MODEL,
            "tools": tuple(body.tools),
            "settings": agent_settings,
        }
        if body.instructions:
            agent_kwargs["instructions"] = body.instructions
        agent_config = Agent(**agent_kwargs)

        # Optional per-request permission policy — ASK rules for the
        # listed glob patterns. Combined with the auto-rules built into
        # PermissionPolicy.evaluate (destructiveHint=True → ASK).
        permission_policy = None
        if body.require_approval_for_tools:
            from kaos_agents.runtime.permissions import PermissionPolicy
            from kaos_agents.types.permissions import PermissionDecision, PermissionRule

            permission_policy = PermissionPolicy(
                rules=tuple(
                    PermissionRule(
                        pattern=pat,
                        action=PermissionDecision.ASK,
                        reason="per-request require_approval_for_tools",
                    )
                    for pat in body.require_approval_for_tools
                )
            )

        # Per-request CircuitBreaker (#506). Defaults trip after 5
        # consecutive failures-or-uninformative returns on a single tool;
        # see the CircuitBreaker docstring for the rationale anchored on
        # session 01KS2DEBYT341F1F16B3BRQRV0 (12 zero-result web searches
        # in a row with no breaker wired). Per-request scope is correct
        # because each new HTTP request is a fresh user intent — we do
        # not want cross-request memory of consecutive failures.
        from kaos_agents.action.circuit import CircuitBreaker

        runner_hooks = (CircuitBreaker(),)

        runner = Runner(
            agent_config,
            runtime=app.state.runtime,
            vfs=app.state.vfs,
            permission_policy=permission_policy,
            hooks=runner_hooks,
        )

        # Content negotiation: JSON-only clients take the blocking path.
        if "application/json" in accept and "text/event-stream" not in accept:
            from kaos_agents.tools.registry import _run_turn_with_status

            response, status = await _run_turn_with_status(
                runner,
                body.message,
                effective_session_id,
                is_internal_iteration=body.is_internal_iteration,
            )
            payload = {
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
                "budget_exceeded": status["budget_exceeded"],
                "paused_for_approval": status["paused_for_approval"],
            }
            if status["paused_for_approval"]:
                payload["pending_tool_name"] = status["pending_tool_name"]
                payload["run_state_ref"] = status["run_state_ref"]
            if status["budget_exceeded"]:
                payload["budget_kind"] = status["budget_kind"]
            return JSONResponse(payload)

        # Default: SSE streaming
        event_stream = runner.run(
            body.message,
            effective_session_id,
            is_internal_iteration=body.is_internal_iteration,
        )
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
        run_id: Annotated[
            str, Path(description="Run identifier from a paused ToolCallApprovalRequired")
        ],
        body: Annotated[ApprovalRequest, Body()],
        tenant_id: Annotated[str | None, Depends(_require_auth)],
    ) -> StreamingResponse:
        """Approve or deny a paused run, then stream the continuation as SSE.

        Loads the persisted ``RunState`` for the given ``run_id`` from VFS,
        invokes ``Runner.resume(state, approved=body.approved)``, and
        streams the resulting events back as SSE.

        Returns 404 if no RunState exists for the run_id (the run was
        never paused, or the VFS state has expired) OR if the loaded
        run belongs to a different tenant. Both return 404 (not 403) to
        avoid leaking run-id existence across tenants — see KC17-P0-3.
        """
        from kaos_agents.runtime.interrupts import load_run_state

        try:
            state = await load_run_state(run_id, app.state.vfs)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No paused run found for run_id={run_id!r}: {exc}. "
                    "Either the run was never paused, or the VFS state has expired. "
                    "Check the run_state_ref returned with the original "
                    "ToolCallApprovalRequired event."
                ),
            ) from exc

        # KC17-P0-3 — tenant scoping. A paused run's session_id carries
        # the tenant prefix (because it was scoped at message-creation
        # time). If this caller's tenant doesn't own the session, return
        # 404 — explicitly NOT 403 to avoid confirming the run exists.
        if tenant_id is not None and not state.session_id.startswith(f"{tenant_id}:"):
            logger.warning(
                "kaos-agents approve: cross-tenant access blocked "
                "(run_id=%s state_session=%s caller_tenant=%s)",
                run_id,
                state.session_id,
                tenant_id,
            )
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No paused run found for run_id={run_id!r}. "
                    "Either the run was never paused, or the VFS state has expired."
                ),
            )

        # WS-0.3: reconstruct the Runner with the original Agent config
        # when available. Pre-WS-0.3 this fell back to ``Agent()`` defaults,
        # silently losing pattern / model / tools / instructions from the
        # paused run (so a paused plan would resume as default chat).
        # The snapshot is captured at pause time on ``state.agent_config``;
        # older RunStates without it continue to fall back to defaults.
        agent_config = state.agent_config.to_agent() if state.agent_config is not None else Agent()

        # Same per-request CircuitBreaker as the start-turn handler
        # (#506). A resumed run gets a fresh breaker — the prior run's
        # state is opaque to us, and re-applying old per-tool counts
        # could mis-trip on legitimate retries after the user approved
        # an asked-for tool.
        from kaos_agents.action.circuit import CircuitBreaker

        runner = Runner(
            agent_config,
            runtime=app.state.runtime,
            vfs=app.state.vfs,
            hooks=(CircuitBreaker(),),
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
        tenant_id: Annotated[str | None, Depends(_require_auth)],
    ) -> SessionResponse:
        """Create a new session (initializes memory)."""
        from kaos_agents.memory.store import SessionStore

        effective_session_id = scope_session_id(body.session_id, tenant_id)
        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create(effective_session_id)
        section_names = [mt.value for mt in memory.sections]
        return SessionResponse(
            session_id=body.session_id,
            turn_count=memory.turn_count,
            sections=section_names,
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(
        tenant_id: Annotated[str | None, Depends(_require_auth)],
        session_id: str = Path(description="Session identifier"),
    ) -> SessionResponse:
        """Get session state (turn count, configured sections)."""
        from kaos_agents.memory.store import SessionStore

        effective_session_id = scope_session_id(session_id, tenant_id)
        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(effective_session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found: {exc}. "
                "Check the session_id matches a previous message call.",
            ) from exc
        section_names = [mt.value for mt in memory.sections]
        return SessionResponse(
            session_id=session_id,
            turn_count=memory.turn_count,
            sections=section_names,
        )

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(
        tenant_id: Annotated[str | None, Depends(_require_auth)],
        session_id: str = Path(description="Session identifier"),
    ) -> dict[str, str]:
        """Delete a session and its memory.

        KC17-P1-1: actually deletes the persisted session directory.
        Pre-KC17 this called ``vfs.cleanup_context(session_id)`` only,
        which left ``sessions/{id}/memory.json`` intact and made the
        DELETE a no-op for subsequent ``SessionStore.exists()`` calls.
        """
        from kaos_agents.memory.store import SessionStore

        effective_session_id = scope_session_id(session_id, tenant_id)
        vfs: VirtualFileSystem = app.state.vfs

        # 1. Remove persisted SessionStore files (memory.json + graph.ttl).
        store = SessionStore(vfs)
        await store.delete(effective_session_id)

        # 2. Sweep any VFS scratch keyed off the session id (artifacts,
        #    run-state, etc.). cleanup_context is idempotent on a fresh
        #    namespace so it's safe to call even when nothing was created.
        await vfs.cleanup_context(effective_session_id)

        return {"status": "deleted", "session_id": session_id}

    @app.post(
        "/v1/sessions/{session_id}/memory/messages/turn",
        response_model=MemoryTurnResponse,
    )
    async def append_memory_turn(
        session_id: Annotated[str, Path(description="Session identifier")],
        body: Annotated[MemoryTurnRequest, Body()],
        tenant_id: Annotated[str | None, Depends(_require_auth)],
    ) -> MemoryTurnResponse:
        """Append a canonical (user, final-assistant) pair to memory.

        Used by outer multi-iteration loops (e.g. kaos-agents'
        AgenticLoop) that drove each iteration with
        ``is_internal_iteration=True`` and now need to persist the
        resolved turn exactly once. No LLM call — pure memory write.

        Writes are skipped when the corresponding field is empty so the
        caller can omit either side (e.g. assistant-only on a failure
        recovery path). The session is created on demand if it does not
        yet exist, matching the message endpoint's behaviour.

        See ``docs/plans/2026-05-19-agentic-loop-honesty.md`` §3.1.a for
        the iteration-leak fix this endpoint supports.
        """
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        effective_session_id = scope_session_id(session_id, tenant_id)
        store = SessionStore(app.state.vfs)
        memory = await store.load_or_create(effective_session_id)

        appended = 0
        user_text = body.user_message.strip()
        assistant_text = body.assistant_message.strip()
        if user_text:
            memory.add(MemoryType.MESSAGES, f"user: {user_text}")
            appended += 1
        if assistant_text:
            memory.add(MemoryType.MESSAGES, f"assistant: {assistant_text}")
            appended += 1

        if appended > 0:
            # B0.3: fire end-of-turn summarization on the canonical
            # append path. Pre-fix this endpoint wrote via memory.add()
            # only and never called summarize_turn() / end_turn() —
            # MESSAGES grew unbounded across long sessions until the
            # assemble-context call OOM'd the planner's prompt budget.
            # Summary trigger is best-effort: ON_OVERFLOW sections that
            # are over budget get summarized; LLM failures (no API
            # key, transport error) fall through to a logged warning
            # so the canonical write still completes. ``end_turn()``
            # always runs to keep ``turn_count`` honest.
            try:
                await memory.summarize_turn()
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "memory.summarize_turn failed (session=%s): %s — "
                    "canonical turn append continues without summary",
                    effective_session_id,
                    exc,
                )
            memory.end_turn()
            await store.save(memory)

        return MemoryTurnResponse(
            session_id=session_id,
            turn_count=memory.turn_count,
            appended=appended,
        )

    # Note: register /memory/search BEFORE /memory/{section} so FastAPI's
    # first-match routing sends "search" to the search endpoint rather than
    # treating it as a section name.
    @app.get(
        "/v1/sessions/{session_id}/memory/search",
        response_model=MemorySearchResponse,
    )
    async def search_memory_endpoint(
        tenant_id: Annotated[str | None, Depends(_require_auth)],
        session_id: str = Path(description="Session identifier"),
        query: str = Query(description="Search query"),
        top_k: int = Query(default=10, ge=1, le=50, description="Max results"),
    ) -> MemorySearchResponse:
        """BM25 search across memory sections."""
        from kaos_agents.memory.search import search_memory
        from kaos_agents.memory.store import SessionStore

        effective_session_id = scope_session_id(session_id, tenant_id)
        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(effective_session_id)
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
                    content=r.content,
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
        tenant_id: Annotated[str | None, Depends(_require_auth)],
        session_id: str = Path(description="Session identifier"),
        section: str = Path(description="Memory section name"),
        limit: int = Query(default=20, ge=1, le=100, description="Max items"),
    ) -> MemoryQueryResponse:
        """Read a memory section."""
        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        effective_session_id = scope_session_id(session_id, tenant_id)
        store = SessionStore(app.state.vfs)
        try:
            memory = await store.load(effective_session_id)
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
    """Get VFS from runtime, falling back to the kaos-core disk-backed default.

    The disk-backed default matches ``kaos_core.KaosRuntime`` and
    ``kaos_core.vfs.core.VirtualFileSystem`` — both default to
    ``StorageBackend.DISK`` rooted at ``.kaos-vfs/``. Callers that need an
    in-memory VFS (most commonly tests) should construct one explicitly
    and pass it via ``create_app(runtime=KaosRuntime(vfs=...))``.
    """
    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs
    return VirtualFileSystem()
