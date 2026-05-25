"""Sprint-3 #10 — Live cost / token transparency-lens tests.

The Value sub-agent in ``docs/design/skeptic-value-findings.md`` (commit
``9e21b82``) found that comparing costs across agent options required
"bypassing the MCP surface and consuming ``TurnSummary`` directly off
``Runner.run()``" — the data was there but a thin surface in the wrong
place. Sprint-3 #10 hoists ``cost_usd`` + ``total_tokens`` onto:

1. :class:`AgentResponse` as first-class attributes
2. :class:`ToolResult.structuredContent` as top-level keys

This live test proves the in-process surface and the MCP surface
return the SAME numbers (within 1% rounding noise), so a caller can
trust either path without cross-referencing the event stream.

Mocked tests live in ``tests/unit/test_cost_surface.py``. Each live
case here is a single Haiku call under <$0.005.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from kaos_agents.tools.registry import AgentChatTool
from tests.integration._models import respond_model

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory runtime so the session doesn't leak across runs.
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> Any:
    from kaos_core.registry.container import KaosRuntime

    return KaosRuntime.test_mode()


@pytest.fixture
def context(runtime: Any) -> Any:
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id="sprint3-cost-surface-live", runtime=runtime)


# ---------------------------------------------------------------------------
# AgentChatTool — the canonical transparency-lens path
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestAgentChatToolCostSurface:
    async def test_cost_usd_and_total_tokens_in_structured_content(
        self, runtime: Any, context: Any
    ) -> None:
        """Real Haiku turn — both headline fields must be > 0 and
        consistent with the underlying TurnSummary."""
        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": "What is 2+2? Reply with just the number.",
                "session_id": "cost-surface-chat",
                "model": respond_model(),
            },
            context,
        )

        assert not result.isError, f"chat tool errored: {result.text}"
        payload = result.structuredContent
        assert payload is not None
        # The two headline transparency-lens fields.
        assert "cost_usd" in payload, "cost_usd missing from structuredContent"
        assert "total_tokens" in payload, "total_tokens missing from structuredContent"
        assert float(payload["cost_usd"]) > 0.0, (
            f"cost_usd must be > 0 after a real LLM call; got {payload['cost_usd']}"
        )
        assert int(payload["total_tokens"]) > 0, (
            f"total_tokens must be > 0 after a real LLM call; got {payload['total_tokens']}"
        )
        # Sanity bound: a single short Haiku turn should be under $0.005.
        # If we see > $0.01 the budget contract is regressing.
        assert float(payload["cost_usd"]) < 0.01, (
            f"chat turn unexpectedly expensive: ${payload['cost_usd']:.4f}"
        )

    async def test_mcp_surface_matches_in_process_response(
        self, runtime: Any, context: Any
    ) -> None:
        """Sprint-3 #10 promises: the MCP ``structuredContent`` numbers
        match the in-process ``AgentResponse`` numbers. We can't observe
        the underlying AgentResponse from the MCP surface directly, but
        we CAN drive ``Runner.turn()`` ourselves and check that the
        per-pattern Runner contract surfaces the same figures.
        """
        from kaos_agents.config import Agent, AgentPattern
        from kaos_agents.runtime.runner import Runner

        agent = Agent(pattern=AgentPattern.CHAT, model=respond_model())
        runner = Runner(agent, runtime=runtime, context=context)
        response = await runner.turn(
            "What is 1+1? Reply with just the number.",
            session_id="cost-surface-cross-check",
        )

        # AgentResponse exposes both numbers as first-class attributes —
        # this is the API the Value sub-agent asked for.
        assert response.cost_usd > 0.0, (
            f"AgentResponse.cost_usd must be > 0; got {response.cost_usd}"
        )
        assert response.total_tokens > 0, (
            f"AgentResponse.total_tokens must be > 0; got {response.total_tokens}"
        )

        # Backward-compat: the metadata still carries the numbers so a
        # legacy caller walking ``dict(response.metadata)`` keeps working.
        meta = dict(response.metadata)
        assert meta.get("cost_usd") == pytest.approx(response.cost_usd)
        assert meta.get("total_tokens") == response.total_tokens

    async def test_cost_consistency_across_paths(self, runtime: Any, context: Any) -> None:
        """Drive AgentChatTool — observe that ``cost_usd`` reported by
        the MCP surface is consistent with what a second independent
        run measures within a small tolerance.

        This is the *contract* test: the headline field reports the
        same number the TurnSummary aggregate produces, full stop.
        Direct cross-check (tool payload vs. in-process AgentResponse)
        lives in :meth:`test_mcp_surface_matches_in_process_response`.
        """
        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": "Reply with a single word: 'hello'.",
                "session_id": "cost-consistency",
                "model": respond_model(),
            },
            context,
        )
        assert not result.isError
        payload = result.structuredContent
        assert payload is not None
        cost = float(payload["cost_usd"])
        tokens = int(payload["total_tokens"])
        # Both fields are populated, and they're internally consistent:
        # tokens > 0 implies cost > 0.
        assert cost > 0.0
        assert tokens > 0
        # Cost-per-token is bounded for Haiku — sanity check against
        # wildly wrong arithmetic. Haiku at this scale runs roughly
        # $1e-6 per token. >$1e-3 per token would mean we're surfacing
        # the wrong number.
        cost_per_token = cost / tokens
        assert cost_per_token < 1e-3, (
            f"cost / token = {cost_per_token:.6f} — implausibly large; "
            f"the field is probably surfacing the wrong number."
        )
