"""Tests for pattern metadata() classmethods (Track 3 chunk A4).

Confirms each pattern subclass overrides metadata() with the correct
:class:`AgentMetadata` shape:
- pattern field matches AgentPattern enum value
- name is canonical kebab-cased
- tags include pattern-specific markers
"""

from __future__ import annotations

import pytest

from kaos_agents.config import AgentPattern
from kaos_agents.patterns import ChatAgent, PlanExecuteAgent, ResearchAgent
from kaos_agents.types import AgentMetadata


@pytest.mark.unit
class TestPatternMetadata:
    def test_chat_metadata(self) -> None:
        meta = ChatAgent.metadata()
        assert isinstance(meta, AgentMetadata)
        assert meta.name == "chat-agent"
        assert meta.pattern == AgentPattern.CHAT.value
        assert "chat" in meta.tags
        assert "react" in meta.tags

    def test_plan_metadata(self) -> None:
        meta = PlanExecuteAgent.metadata()
        assert isinstance(meta, AgentMetadata)
        assert meta.name == "plan-execute-agent"
        assert meta.pattern == AgentPattern.PLAN.value
        assert "plan" in meta.tags
        assert "multi-step" in meta.tags

    def test_research_metadata(self) -> None:
        meta = ResearchAgent.metadata()
        assert isinstance(meta, AgentMetadata)
        assert meta.name == "research-agent"
        assert meta.pattern == AgentPattern.RESEARCH.value
        assert "research" in meta.tags
        assert "rag" in meta.tags
        assert "citation" in meta.tags

    def test_metadata_overrides_baseagent_default(self) -> None:
        """BaseAgent's default returns ``pattern='chat'`` because that's the
        ``AgentMetadata`` default. Each subclass overrides to its own value."""
        from kaos_agents.runtime.agent import BaseAgent

        chat_meta = ChatAgent.metadata()
        plan_meta = PlanExecuteAgent.metadata()
        research_meta = ResearchAgent.metadata()
        base_meta = BaseAgent.metadata()

        # All distinct (the override matters)
        assert chat_meta.pattern == "chat"
        assert plan_meta.pattern == "plan"
        assert research_meta.pattern == "research"
        # BaseAgent's default name
        assert base_meta.name == "base-agent"
