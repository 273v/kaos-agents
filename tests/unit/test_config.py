"""Tests for Agent configuration (Phase 2).

Covers:
- Agent is frozen (immutable)
- AgentPattern enum values
- Agent.create() convenience constructor
- Agent.resolve_settings() defaults
- Agent.effective_model() fallback
- Agent.tool_filter() glob-to-list conversion
"""

from __future__ import annotations

import pytest

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.settings import KaosAgentSettings


@pytest.mark.unit
class TestAgentPattern:
    def test_pattern_values(self) -> None:
        assert AgentPattern.CHAT == "chat"
        assert AgentPattern.PLAN == "plan"
        assert AgentPattern.RESEARCH == "research"

    def test_pattern_from_string(self) -> None:
        assert AgentPattern("chat") == AgentPattern.CHAT
        assert AgentPattern("plan") == AgentPattern.PLAN
        assert AgentPattern("research") == AgentPattern.RESEARCH

    def test_invalid_pattern(self) -> None:
        with pytest.raises(ValueError, match="nonexistent"):
            AgentPattern("nonexistent")


@pytest.mark.unit
class TestAgentConfig:
    def test_frozen(self) -> None:
        agent = Agent()
        with pytest.raises(AttributeError):
            setattr(agent, "model", "different")  # noqa: B010

    def test_defaults(self) -> None:
        agent = Agent()
        assert agent.instructions == "You are a helpful assistant."
        assert agent.pattern == AgentPattern.CHAT
        assert agent.tools == ()
        assert agent.settings is None
        assert agent.max_tools is None
        assert agent.name is None

    def test_custom_fields(self) -> None:
        agent = Agent(
            instructions="You are a legal assistant.",
            model="anthropic:claude-sonnet-4-6",
            pattern=AgentPattern.PLAN,
            tools=("kaos-source-*", "kaos-web-*"),
            max_plan_steps=10,
            name="legal-planner",
        )
        assert agent.instructions == "You are a legal assistant."
        assert agent.model == "anthropic:claude-sonnet-4-6"
        assert agent.pattern == AgentPattern.PLAN
        assert agent.tools == ("kaos-source-*", "kaos-web-*")
        assert agent.max_plan_steps == 10
        assert agent.name == "legal-planner"


@pytest.mark.unit
class TestAgentCreate:
    def test_create_with_string_pattern(self) -> None:
        agent = Agent.create(pattern="plan")
        assert agent.pattern == AgentPattern.PLAN

    def test_create_with_list_tools(self) -> None:
        agent = Agent.create(tools=["kaos-source-*", "kaos-web-*"])
        assert agent.tools == ("kaos-source-*", "kaos-web-*")

    def test_create_with_dict_metadata(self) -> None:
        agent = Agent.create(metadata={"team": "legal", "version": "1"})
        assert ("team", "legal") in agent.metadata
        assert ("version", "1") in agent.metadata


@pytest.mark.unit
class TestAgentMethods:
    def test_resolve_settings_default(self) -> None:
        agent = Agent()
        settings = agent.resolve_settings()
        assert isinstance(settings, KaosAgentSettings)

    def test_resolve_settings_provided(self) -> None:
        custom = KaosAgentSettings(default_context_budget_tokens=8000)
        agent = Agent(settings=custom)
        settings = agent.resolve_settings()
        assert settings.default_context_budget_tokens == 8000

    def test_effective_model_explicit(self) -> None:
        agent = Agent(model="anthropic:claude-sonnet-4-6")
        assert agent.effective_model() == "anthropic:claude-sonnet-4-6"

    def test_tool_filter_empty(self) -> None:
        agent = Agent()
        assert agent.tool_filter() is None

    def test_tool_filter_with_patterns(self) -> None:
        agent = Agent(tools=("kaos-source-*", "kaos-web-*"))
        result = agent.tool_filter()
        assert result == ["kaos-source-*", "kaos-web-*"]
