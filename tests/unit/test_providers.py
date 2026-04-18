"""Tests for provider configuration (Phase 5).

Covers:
- ModelRole enum values
- ProviderConfig.model_for() role-based resolution
- ProviderConfig.model_for() fallback to default
- FAST/BALANCED/STRONG presets
- Agent.model_for_role() with and without provider
- Agent.effective_model() with provider
"""

from __future__ import annotations

import pytest

from kaos_agents.config import Agent
from kaos_agents.providers import BALANCED, FAST, STRONG, ModelRole, ProviderConfig


@pytest.mark.unit
class TestModelRole:
    def test_role_values(self) -> None:
        assert ModelRole.CLASSIFY == "classify"
        assert ModelRole.RESPOND == "respond"
        assert ModelRole.PLAN == "plan"
        assert ModelRole.RESEARCH == "research"
        assert ModelRole.EVALUATE == "evaluate"

    def test_from_string(self) -> None:
        assert ModelRole("plan") == ModelRole.PLAN


@pytest.mark.unit
class TestProviderConfig:
    def test_model_for_explicit_role(self) -> None:
        config = ProviderConfig(
            default="model-a",
            role_models={ModelRole.PLAN: "model-b"},
        )
        assert config.model_for(ModelRole.PLAN) == "model-b"

    def test_model_for_fallback(self) -> None:
        config = ProviderConfig(
            default="model-a",
            role_models={ModelRole.PLAN: "model-b"},
        )
        assert config.model_for(ModelRole.CLASSIFY) == "model-a"

    def test_empty_role_models(self) -> None:
        config = ProviderConfig(default="model-x")
        assert config.model_for(ModelRole.PLAN) == "model-x"
        assert config.model_for(ModelRole.RESEARCH) == "model-x"

    def test_frozen(self) -> None:
        config = ProviderConfig()
        with pytest.raises(AttributeError):
            setattr(config, "default", "other")  # noqa: B010


@pytest.mark.unit
class TestPresets:
    def test_fast_all_haiku(self) -> None:
        for role in ModelRole:
            assert "haiku" in FAST.model_for(role)

    def test_balanced_classify_cheap(self) -> None:
        assert "haiku" in BALANCED.model_for(ModelRole.CLASSIFY)
        assert "haiku" in BALANCED.model_for(ModelRole.RESPOND)

    def test_balanced_plan_strong(self) -> None:
        assert "sonnet" in BALANCED.model_for(ModelRole.PLAN)
        assert "sonnet" in BALANCED.model_for(ModelRole.RESEARCH)

    def test_strong_default_sonnet(self) -> None:
        assert "sonnet" in STRONG.model_for(ModelRole.CLASSIFY)

    def test_strong_plan_opus(self) -> None:
        assert "opus" in STRONG.model_for(ModelRole.PLAN)


@pytest.mark.unit
class TestAgentWithProvider:
    def test_effective_model_with_provider(self) -> None:
        agent = Agent(provider=BALANCED)
        assert "haiku" in agent.effective_model()

    def test_effective_model_without_provider(self) -> None:
        agent = Agent(model="anthropic:claude-sonnet-4-6")
        assert agent.effective_model() == "anthropic:claude-sonnet-4-6"

    def test_model_for_role_with_provider(self) -> None:
        agent = Agent(provider=BALANCED)
        assert "sonnet" in agent.model_for_role("plan")
        assert "haiku" in agent.model_for_role("classify")

    def test_model_for_role_without_provider(self) -> None:
        agent = Agent(model="anthropic:claude-sonnet-4-6")
        # All roles return the same model when no provider is set
        assert agent.model_for_role("plan") == "anthropic:claude-sonnet-4-6"
        assert agent.model_for_role("classify") == "anthropic:claude-sonnet-4-6"

    def test_model_for_role_unknown_role(self) -> None:
        agent = Agent(provider=BALANCED)
        # Unknown role falls back to provider default
        result = agent.model_for_role("unknown_role")
        assert "haiku" in result

    def test_create_with_provider(self) -> None:
        agent = Agent.create(provider=BALANCED, pattern="plan")
        assert agent.provider is BALANCED
        assert "sonnet" in agent.model_for_role("plan")


# ---------------------------------------------------------------------------
# Gap 8: ProviderConfig is actually threaded through the execution pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderPipelineIntegration:
    """Verify classify/respond/plan/research actually consult provider.model_for()."""

    def test_base_agent_model_for_role_with_provider(self) -> None:
        """BaseAgent._model_for_role returns the role-specific model."""
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.agent import BaseAgent

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        agent = BaseAgent(vfs, provider=BALANCED)

        assert "haiku" in agent._model_for_role("classify")
        assert "sonnet" in agent._model_for_role("plan")
        assert "sonnet" in agent._model_for_role("research")

    def test_base_agent_model_for_role_without_provider(self) -> None:
        """Without provider, classify/respond use self._model; plan uses settings."""
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.agent import BaseAgent

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        agent = BaseAgent(vfs, model="custom:model-x")

        assert agent._model_for_role("classify") == "custom:model-x"
        assert agent._model_for_role("respond") == "custom:model-x"
        # plan falls back to settings.planning_llm_model
        assert agent._model_for_role("plan") == agent._settings.planning_llm_model

    def test_runner_threads_provider_to_internal_agent(self) -> None:
        """Runner passes agent.provider to the constructed internal agent."""
        from kaos_agents.runner import Runner

        agent = Agent(provider=BALANCED)
        runner = Runner(agent)
        internal = runner._build_internal_agent()

        # Internal agent is a ChatAgent with provider set
        assert internal._provider is BALANCED
        # And role resolution works end-to-end
        assert "sonnet" in internal._model_for_role("plan")
        assert "haiku" in internal._model_for_role("classify")
