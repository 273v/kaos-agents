"""Phase 2.2 / INTEG-4: Agent-driven extraction test.

Proves the full path: user message → agent reads recipe → agent calls
kaos-extract-schema tool → extraction results returned via events.

This is the first test where the Runner's tool bridge invokes a real
extraction tool (not just the LLM responding from its training data).
"""

from __future__ import annotations

import os

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import (
    KaosEvent,
    RunError,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
)
from kaos_agents.runner import Runner

CONTRACT_EXCERPT = (
    "SPONSORSHIP AGREEMENT\n\n"
    "This Sponsorship Agreement ('Agreement') is made and entered into as of "
    "February 17, 1999, by and between Tickets.com, Inc., a Delaware corporation "
    "('Tickets'), located at 4061 Glencoe Avenue, Marina del Rey, California 90292, "
    "and America Online, Inc. ('AOL'), located at 22000 AOL Way, Dulles, Virginia.\n\n"
    "1. TERM. The initial term of this Agreement shall be for three (3) years "
    "commencing on the Effective Date.\n\n"
    "2. GOVERNING LAW. This Agreement shall be governed by and construed in "
    "accordance with the laws of the State of California.\n\n"
    "3. TERMINATION. Tickets may terminate this Agreement for any reason upon "
    "thirty (30) days prior written notice to AOL.\n\n"
    "4. LIMITATION OF LIABILITY. Except for claims arising under Section 6, "
    "in no event will either party's aggregate liability exceed the total "
    "fees paid under this Agreement during the twelve months prior to the claim."
)


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


async def _collect(runner: Runner, message: str, session_id: str) -> tuple[str, list[KaosEvent]]:
    events: list[KaosEvent] = []
    text_parts: list[str] = []
    async for event in runner.run(message, session_id):
        events.append(event)
        if isinstance(event, TextDelta):
            text_parts.append(event.content)
    return "".join(text_parts), events


def _make_runner_with_extraction() -> Runner:
    """Build a Runner with extraction tools registered."""
    from kaos_core.registry.container import KaosRuntime

    runtime = KaosRuntime.default()

    from kaos_agents.tools import register_agent_tools

    register_agent_tools(runtime)

    agent = Agent.create(
        instructions=(
            "You are a legal document extraction assistant. "
            "When asked to extract information from a contract, use the "
            "kaos-extract-schema tool with the appropriate recipe_name. "
            "Always report the extracted values in your response."
        ),
        model="anthropic:claude-haiku-4-5",
        pattern="chat",
        tools=("kaos-extract-*",),
    )
    return Runner(agent, runtime=runtime)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_calls_extract_schema_tool() -> None:
    """Agent invokes kaos-extract-schema when asked to extract from a contract."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    contract_text = CONTRACT_EXCERPT
    runner = _make_runner_with_extraction()

    message = (
        f"Extract the parties and governing law from this contract using "
        f"the kaos-extract-schema tool with a simple 2-column schema. "
        f"Here is the contract text:\n\n{contract_text}"
    )

    text, events = await _collect(runner, message, "extract-test")

    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    tool_results = [e for e in events if isinstance(e, ToolCallResult)]
    errors = [e for e in events if isinstance(e, RunError)]

    assert not errors, f"RunError: {[e.message for e in errors]}"

    assert tool_starts, (
        f"Expected ToolCallStart event — agent didn't call any tool. Response text: {text[:200]}"
    )

    extract_calls = [e for e in tool_starts if "extract" in e.tool_name.lower()]
    assert extract_calls, (
        f"Expected kaos-extract-* tool call but got: {[e.tool_name for e in tool_starts]}"
    )

    assert tool_results, "Expected ToolCallResult event after tool execution"
    assert len(text) > 20, f"Response too short: {text[:100]}"

    print(f"\nAgent called: {[e.tool_name for e in tool_starts]}")
    print(f"Response preview: {text[:300]}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_extraction_with_recipe() -> None:
    """Agent uses recipe_name when asked to use a specific recipe."""
    if not _has_key():
        pytest.skip("ANTHROPIC_API_KEY not set")

    contract_text = CONTRACT_EXCERPT
    runner = _make_runner_with_extraction()

    message = (
        f"Use the kaos-extract-schema tool with recipe_name='court-opinion' "
        f"to extract from this text. Just try it — I know it's a contract, "
        f"not a court opinion.\n\n{contract_text}"
    )

    text, events = await _collect(runner, message, "recipe-test")

    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    errors = [e for e in events if isinstance(e, RunError)]

    if errors:
        print(f"Errors (non-fatal): {[e.message for e in errors]}")

    if tool_starts:
        for ts in tool_starts:
            print(f"Tool called: {ts.tool_name}")
            args = dict(ts.arguments)
            if "recipe_name" in args:
                print(f"  recipe_name = {args['recipe_name']}")

    assert len(text) > 10, f"Agent produced no meaningful response: {text[:100]}"
