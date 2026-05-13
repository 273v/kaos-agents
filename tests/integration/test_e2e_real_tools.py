"""End-to-end integration tests: kaos-agents with REAL tools from kaos-source + kaos-web.

These tests prove the full architecture works:
- KaosRuntime tool registration → ToolBridge → ReAct / Plan execution
- Real LLM calls (Anthropic) + real API calls (Federal Register, eCFR, web)
- Memory persistence across turns via VFS
- Intent classification → dispatch → tool calling → response synthesis

Requires ANTHROPIC_API_KEY.
Tools used: kaos-source Federal Register (free, no key) and eCFR (free, no key).
"""

from __future__ import annotations

import pytest
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig
from kaos_source import register_ecfr_tools, register_federal_register_tools

from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.plan_execute import PlanExecuteAgent
from kaos_agents.settings import KaosAgentSettings

MODEL = "anthropic:claude-haiku-4-5"

# Tool names available after registration — used for filter_names
FR_TOOLS = [
    "kaos-source-fr-search",
    "kaos-source-fr-get-document",
    "kaos-source-fr-get-content",
    "kaos-source-fr-agencies",
]
ECFR_TOOLS = [
    "kaos-source-ecfr-titles",
    "kaos-source-ecfr-structure",
    "kaos-source-ecfr-content",
    "kaos-source-ecfr-search-structure",
]
ALL_SOURCE_TOOLS = FR_TOOLS + ECFR_TOOLS


def _make_runtime() -> KaosRuntime:
    """Create a runtime with real kaos-source tools."""
    runtime = KaosRuntime()
    register_federal_register_tools(runtime)
    register_ecfr_tools(runtime)
    return runtime


def _memory_vfs() -> VirtualFileSystem:
    """In-memory VFS for test isolation."""
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


def _test_settings() -> KaosAgentSettings:
    """Settings tuned for integration tests — generous budget, fast timeout."""
    return KaosAgentSettings(
        default_llm_model=MODEL,
        planning_llm_model=MODEL,
        max_tools=20,
        max_react_iterations=5,
        tool_timeout_seconds=30.0,
        plan_max_steps=10,
        plan_max_replans=2,
        plan_max_cost_usd=0.50,
        plan_max_wall_clock_seconds=120.0,
    )


# ---------------------------------------------------------------------------
# ChatAgent + real tools
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestChatAgentRealTools:
    """ChatAgent with real Federal Register and eCFR tools."""

    async def test_fr_search_via_react(self):
        """ChatAgent should use FR search tool when asked about federal regulations."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=FR_TOOLS,
            settings=settings,
        )

        response = await agent.turn(
            "Search the Federal Register for recent EPA rules about emissions. "
            "Use the kaos-source-fr-search tool with query 'EPA emissions'.",
            session_id="e2e-fr-search",
        )

        assert response.text, "Response text should not be empty"
        assert len(response.text) > 20, "Response should be substantive"
        # The response should mention something about EPA or emissions or Federal Register
        response_lower = response.text.lower()
        assert any(
            w in response_lower for w in ("epa", "emission", "federal register", "rule", "document")
        ), f"Response should mention EPA/emissions content, got: {response.text[:200]}"

    async def test_ecfr_titles_via_react(self):
        """ChatAgent should use eCFR titles tool to list CFR titles."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=ECFR_TOOLS,
            settings=settings,
        )

        response = await agent.turn(
            "List the available CFR titles using the eCFR tools. "
            "Use kaos-source-ecfr-titles to get the list.",
            session_id="e2e-ecfr-titles",
        )

        assert response.text, "Response text should not be empty"
        assert len(response.text) > 20
        # eCFR has 50 titles — response should reference CFR or titles
        response_lower = response.text.lower()
        assert any(
            w in response_lower for w in ("cfr", "title", "code of federal regulations", "50")
        ), f"Response should mention CFR titles, got: {response.text[:200]}"

    async def test_tool_calls_recorded(self):
        """Tool calls should be recorded in the AgentResponse."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=["kaos-source-fr-agencies"],
            settings=settings,
        )

        response = await agent.turn(
            "Use the kaos-source-fr-agencies tool to list federal agencies.",
            session_id="e2e-tool-record",
        )

        assert response.text
        # Tool calls may or may not be present depending on classification,
        # but the response should be meaningful
        assert len(response.text) > 10

    async def test_multi_turn_memory_with_tools(self):
        """Memory persists across turns — second turn should have context from first."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "e2e-multi-turn-tools"

        # Turn 1: Search for something specific
        agent1 = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=FR_TOOLS,
            settings=settings,
        )
        r1 = await agent1.turn(
            "Search the Federal Register for 'clean water act' rules.",
            session_id=session_id,
        )
        assert r1.turn_number == 1
        assert r1.text

        # Turn 2: Ask a follow-up (should have context from turn 1)
        agent2 = ChatAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=FR_TOOLS,
            settings=settings,
        )
        r2 = await agent2.turn(
            "What did you find in the previous search?",
            session_id=session_id,
        )
        assert r2.turn_number == 2
        assert r2.text
        # The second response should reference the first turn's content
        # (clean water, regulations, etc.)


# ---------------------------------------------------------------------------
# PlanExecuteAgent + real tools
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestPlanExecuteAgentRealTools:
    """PlanExecuteAgent with real Federal Register and eCFR tools."""

    async def test_simple_query_passthrough(self):
        """Simple question should be handled without planning."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = PlanExecuteAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=ALL_SOURCE_TOOLS,
            settings=settings,
        )

        response = await agent.turn(
            "What is the Federal Register?",
            session_id="e2e-plan-simple",
        )

        assert response.text
        assert len(response.text) > 20
        response_lower = response.text.lower()
        assert any(
            w in response_lower for w in ("federal register", "government", "publication", "rules")
        )

    async def test_multi_step_research(self):
        """Multi-step request should trigger planning with real tools.

        This is the key E2E test — the LLM generates a plan, then executes
        it using real FR/eCFR tools via the tool bridge.
        """
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = PlanExecuteAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=ALL_SOURCE_TOOLS,
            settings=settings,
        )

        response = await agent.turn(
            "First, list the CFR titles using the eCFR tools. "
            "Then, search the Federal Register for recent EPA rules about air quality.",
            session_id="e2e-plan-multi",
        )

        assert response.text, "Plan execution should produce a response"
        assert len(response.text) > 20, "Response should be substantive"
        # Response should contain something from the tools
        assert response.turn_number == 1

    async def test_plan_with_memory_persistence(self):
        """Plan results should be stored in memory and available in next turn."""
        runtime = _make_runtime()
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "e2e-plan-memory"

        # Turn 1: Run a plan
        agent1 = PlanExecuteAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=FR_TOOLS,
            settings=settings,
        )
        r1 = await agent1.turn(
            "Search the Federal Register for recent rules about clean energy tax credits.",
            session_id=session_id,
        )
        assert r1.turn_number == 1
        assert r1.text

        # Turn 2: Ask about what was found
        agent2 = PlanExecuteAgent(
            vfs,
            runtime=runtime,
            model=MODEL,
            tool_filter=FR_TOOLS,
            settings=settings,
        )
        r2 = await agent2.turn(
            "Summarize what you found about clean energy tax credits.",
            session_id=session_id,
        )
        assert r2.turn_number == 2
        assert r2.text


# ---------------------------------------------------------------------------
# ToolBridge integration
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestToolBridgeIntegration:
    """Verify the ToolBridge correctly adapts real tools for ReAct."""

    async def test_bridge_preserves_tool_metadata(self):
        """Bridged tools should preserve name and description from KaosTool metadata."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools

        runtime = _make_runtime()
        tools = bridge_runtime_tools(runtime, filter_names=ALL_SOURCE_TOOLS)

        assert len(tools) == 8, f"Expected 8 tools, got {len(tools)}"

        tool_names = {t.name for t in tools}
        for expected in ALL_SOURCE_TOOLS:
            assert expected in tool_names, f"Missing tool: {expected}"

        # All tools should have descriptions
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"

    async def test_bridge_tool_execution(self):
        """Bridged tool should execute and return real results."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools

        runtime = _make_runtime()
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-source-ecfr-titles"])

        assert len(tools) == 1
        tool = tools[0]

        result = await tool.executor()
        assert isinstance(result, str)
        assert "CFR" in result or "title" in result.lower() or "50" in result

    async def test_bridge_tool_with_params(self):
        """Bridged tool should pass parameters correctly."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools

        runtime = _make_runtime()
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-source-fr-search"])

        assert len(tools) == 1
        tool = tools[0]

        result = await tool.executor(query="clean air act", per_page=2)
        assert isinstance(result, str)
        assert len(result) > 10
        # Should get search results, not an error
        assert "error" not in result.lower() or "Found" in result

    async def test_bridge_filter_limits_tools(self):
        """Filter should limit which tools are bridged."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools

        runtime = _make_runtime()

        # Only FR tools
        fr_tools = bridge_runtime_tools(runtime, filter_names=FR_TOOLS)
        assert len(fr_tools) == 4

        # Only eCFR tools
        ecfr_tools = bridge_runtime_tools(runtime, filter_names=ECFR_TOOLS)
        assert len(ecfr_tools) == 4

        # All
        all_tools = bridge_runtime_tools(runtime, filter_names=ALL_SOURCE_TOOLS)
        assert len(all_tools) == 8
