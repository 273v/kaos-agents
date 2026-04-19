"""Phase 3: MCP client integration test.

Spawns `kaos-agents-serve` as a subprocess, connects via the MCP
Python SDK's stdio_client, and calls the agent tools. This proves
the full MCP → Runner → tools → response path works from OUTSIDE
the process — the A2A (agent-to-agent) surface.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


def _tool_text(result: Any) -> str:
    parts: list[str] = []
    for content in result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_list_tools() -> None:
    """Connect to kaos-agents-serve via stdio and list available tools."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaos_agents.serve"],
        env={**os.environ},
    )

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.list_tools()
        tool_names = [t.name for t in result.tools]

        assert "kaos-agent-chat" in tool_names, f"kaos-agent-chat not in tools: {tool_names}"
        assert len(tool_names) >= 6, f"Expected >=6 tools, got {len(tool_names)}: {tool_names}"
        print(f"MCP tools: {tool_names}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_agent_chat_single_turn() -> None:
    """Call kaos-agent-chat via MCP and get a real response."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaos_agents.serve"],
        env={**os.environ},
    )

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        result = await session.call_tool(
            "kaos-agent-chat",
            arguments={
                "message": "What is habeas corpus? Answer in one sentence.",
                "session_id": "mcp-test-1",
            },
        )

        assert not result.isError, f"Tool returned error: {result.content}"
        text = _tool_text(result)
        assert len(text) > 20, f"Response too short: {text}"
        assert "habeas corpus" in text.lower() or "custody" in text.lower(), (
            f"Response doesn't answer the question: {text[:200]}"
        )
        print(f"MCP response: {text[:200]}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_agent_chat_multi_turn_memory() -> None:
    """Multi-turn via MCP: turn 2 recalls turn 1 context."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaos_agents.serve"],
        env={**os.environ},
    )

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Turn 1: give the agent a fact
        r1 = await session.call_tool(
            "kaos-agent-chat",
            arguments={
                "message": (
                    "Remember: the defendant is John Smith and the case number is 24-CV-1234."
                ),
                "session_id": "mcp-memory-test",
            },
        )
        assert not r1.isError

        # Turn 2: ask about the fact
        r2 = await session.call_tool(
            "kaos-agent-chat",
            arguments={
                "message": "What is the case number I just told you?",
                "session_id": "mcp-memory-test",
            },
        )
        assert not r2.isError
        text2 = _tool_text(r2)
        assert "1234" in text2 or "24-CV" in text2, (
            f"Agent didn't recall case number across MCP turns: {text2[:200]}"
        )
        print(f"MCP turn 2: {text2[:200]}")
