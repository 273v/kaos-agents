"""S4 — permission gating: ASK rule pauses run and surfaces at the wire.

2 surfaces (API + MCP) x 2 providers = 4 tests. CLI variant covered
by ladder T09.

Pre-this-change, neither the API nor the MCP tool surface accepted a
per-request permission policy. The MCP tool's Runner construction
omitted ``permission_policy=`` entirely, so the gating path was a
no-op for every MCP caller. We now thread
``require_approval_for_tools`` (a comma-separated glob list on MCP, a
list on the API request body) into a ``PermissionPolicy`` with ASK
rules.

When the agent tries to call a gated tool, the Runner pauses and
yields ``ToolCallApprovalRequired``. The non-streaming wire paths
(JSON API + MCP tool result) surface this as
``paused_for_approval=true`` with the ``pending_tool_name`` and
``run_state_ref`` so callers can resume via
``POST /v1/runs/{run_id}/approve`` (API) or by re-issuing the call
after explicit approval (out-of-band).
"""

from __future__ import annotations

import pytest

from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    _require_keys,
    assert_no_error,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

# A prompt that forces the agent to attempt a tool call we then gate.
MESSAGE = (
    "Use kaos-source-fr-search exactly once: term=PFAS, "
    "agency=environmental-protection-agency, doc_type=NOTICE, "
    "per_page=1, order=newest. Report results[0].document_number."
)


async def _api_call_with_policy(provider: str, *, session_id: str) -> dict:
    """Hit the API in JSON mode with require_approval_for_tools=['kaos-source-fr-*']."""
    _require_keys()
    from httpx import ASGITransport, AsyncClient
    from kaos_core import KaosRuntime
    from kaos_source.runtime.tools import register_source_tools

    from kaos_agents.api.server import create_app

    runtime = KaosRuntime.default()
    register_source_tools(runtime)
    app = create_app(runtime=runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "message": MESSAGE,
                "model": model_for(provider),
                "pattern": "chat",
                "tools": ["kaos-source-fr-search"],
                "require_approval_for_tools": ["kaos-source-fr-*"],
            },
            headers={"Accept": "application/json"},
            timeout=120.0,
        )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:400]}"
    return resp.json()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s4_permission_via_api(provider: str) -> None:
    """ASK on kaos-source-fr-* → paused_for_approval=true at the wire."""
    data = await _api_call_with_policy(provider, session_id=f"s4-api-{provider}")
    assert data.get("paused_for_approval") is True, (
        f"api/{provider}: expected paused_for_approval=true when policy "
        f"ASK matches the gated tool. Response: {data}"
    )
    pending = str(data.get("pending_tool_name") or "")
    assert "kaos-source-fr" in pending, (
        f"api/{provider}: expected pending_tool_name to reference kaos-source-fr-*; got {pending!r}"
    )
    # The run must have persisted state for resume.
    assert data.get("run_state_ref"), (
        f"api/{provider}: expected non-empty run_state_ref so the caller "
        f"can resume via POST /v1/runs/{{run_id}}/approve. Got: {data}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s4_permission_via_mcp(provider: str) -> None:
    """Same gate via MCP kaos-agent-chat with require_approval_for_tools."""
    result = await mcp_call(
        "kaos-agent-chat",
        arguments={
            "message": MESSAGE,
            "session_id": f"s4-mcp-{provider}",
            "model": model_for(provider),
            "tool_filter": "kaos-source-fr-search",
            "require_approval_for_tools": "kaos-source-fr-*",
        },
        server_args=("--with-source",),
        timeout=120.0,
    )
    assert_no_error(result)
    raw = result.raw
    structured = getattr(raw, "structuredContent", None) or {}
    assert structured.get("paused_for_approval") is True, (
        f"mcp/{provider}: expected paused_for_approval=true. structuredContent: {structured}"
    )
    pending = str(structured.get("pending_tool_name") or "")
    assert "kaos-source-fr" in pending, (
        f"mcp/{provider}: expected pending_tool_name to reference kaos-source-fr-*; got {pending!r}"
    )
    assert structured.get("run_state_ref"), (
        f"mcp/{provider}: expected non-empty run_state_ref. structuredContent: {structured}"
    )
