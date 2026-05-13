"""S5 — budget cap: BudgetExceeded fires when plan_max_cost_usd is tight.

2 surfaces (API + MCP) x 2 providers = 4 tests. CLI variant covered
by ladder T10.

Pre-this-change, neither the FastAPI surface nor the MCP tool surface
accepted a per-request cost cap, so a real budget regression in
plan-execute could ship with green API/MCP tests. The surfaces now
take ``max_cost_usd`` and surface ``budget_exceeded=true`` (plus
``budget_kind="cost"``) in the response payload, so a runaway plan is
visible to wire callers, not just streaming SSE consumers.

Setting ``max_cost_usd=0.001`` forces the cap to bite on the first
LLM call — every model on our shortlist costs more than $0.001 per
plan-step. The pass condition is: the response indicates the budget
was exceeded.
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

# Plan-execute prompt: must be a goal the planner decomposes into 2+ steps
# so the budget evaluator actually runs between steps.
MESSAGE = (
    "Goal: find the latest EPA Federal Register notice mentioning PFAS, "
    "extract its document_number, title, and publication_date, then summarize. "
    "Plan it in at least 3 explicit steps: search, extract, summarize."
)


async def _api_call_plan(provider: str, *, max_cost_usd: float, session_id: str) -> dict:
    """Hit the API in JSON mode with pattern=plan + max_cost_usd."""
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
                "pattern": "plan",
                "tools": ["kaos-source-fr-search"],
                "max_cost_usd": max_cost_usd,
            },
            headers={"Accept": "application/json"},
            timeout=180.0,
        )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:400]}"
    return resp.json()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s5_budget_via_api(provider: str) -> None:
    """Tight cost cap (0.001 USD) on a plan-execute run → budget_exceeded=true."""
    data = await _api_call_plan(provider, max_cost_usd=0.001, session_id=f"s5-api-{provider}")
    assert data.get("budget_exceeded") is True, (
        f"api/{provider}: expected budget_exceeded=true with "
        f"max_cost_usd=0.001 on plan-execute. Response: {data}"
    )
    assert data.get("budget_kind") == "cost", (
        f"api/{provider}: expected budget_kind='cost'; got {data.get('budget_kind')!r}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s5_budget_via_mcp(provider: str) -> None:
    """Same tight cap via the MCP kaos-agent-plan tool."""
    result = await mcp_call(
        "kaos-agent-plan",
        arguments={
            "message": MESSAGE,
            "session_id": f"s5-mcp-{provider}",
            "model": model_for(provider),
            "tool_filter": "kaos-source-fr-search",
            "max_cost_usd": 0.001,
        },
        server_args=("--with-source",),
        timeout=180.0,
    )
    assert_no_error(result)
    # Pull the structured payload that AgentPlanTool returns.
    raw = result.raw
    structured = getattr(raw, "structuredContent", None) or {}
    assert structured.get("budget_exceeded") is True, (
        f"mcp/{provider}: expected budget_exceeded=true. structuredContent: {structured}"
    )
    assert structured.get("budget_kind") == "cost", (
        f"mcp/{provider}: expected budget_kind='cost'; got {structured.get('budget_kind')!r}"
    )
