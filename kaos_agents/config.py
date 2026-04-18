"""Agent — frozen configuration for an agent instance.

An ``Agent`` is a declaration, not an executor. It describes what the agent
is (instructions, model, tools, pattern) but does not run anything. Pass
an ``Agent`` to a ``Runner`` to execute.

This separation enables:
- Multiple Runners sharing one Agent config
- Hooks and provider adapters on the Runner, not the Agent
- Serializable, hashable agent definitions
- Clean testability (mock the Runner, not the Agent)

Design follows the OpenAI Agents SDK pattern (Agent = config, Runner = engine)
adapted for KAOS's typed settings and tool glob patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Any

from kaos_agents.providers import ProviderConfig
from kaos_agents.settings import DEFAULT_MODEL, KaosAgentSettings


@unique
class AgentPattern(StrEnum):
    """Which execution pattern the Runner should use."""

    CHAT = "chat"  # ReAct tool calling (ChatAgent)
    PLAN = "plan"  # Adaptive plan-execute (PlanExecuteAgent)
    RESEARCH = "research"  # RAG with citation verification (ResearchAgent)


@dataclass(frozen=True, slots=True)
class Agent:
    """Frozen configuration for an agent — a declaration, not an executor.

    Pass to ``Runner`` to execute::

        agent = Agent(
            instructions="You are a research assistant.",
            model="anthropic:claude-sonnet-4-6",
            tools=("kaos-source-*", "kaos-web-*"),
            pattern=AgentPattern.PLAN,
        )
        runner = Runner(agent, runtime=runtime)
        async for event in runner.run("Find recent policy changes", "session-1"):
            ...

    All fields are immutable. Pattern-specific parameters (max_plan_steps,
    rag_top_k) are set via ``settings`` or per-field overrides — the Agent
    carries the full configuration so the Runner doesn't need to.
    """

    # Core identity
    instructions: str = "You are a helpful assistant."
    model: str = DEFAULT_MODEL
    pattern: AgentPattern = AgentPattern.CHAT

    # Tool selection — glob patterns matched against runtime tool names
    tools: tuple[str, ...] = ()

    # Provider — role-based model selection. Takes precedence over ``model``
    # when set. None means use the ``model`` field for all roles.
    provider: ProviderConfig | None = None

    # Settings — all 23 KaosAgentSettings fields as a single object.
    # None means "use environment defaults" (resolved at Runner construction).
    settings: KaosAgentSettings | None = None

    # Pattern-specific overrides (None means "use settings default").
    # These exist as top-level fields for ergonomic construction without
    # needing to create a full KaosAgentSettings instance.
    max_tools: int | None = None
    max_react_iterations: int | None = None
    max_plan_steps: int | None = None
    rag_top_k: int | None = None
    rag_max_retries: int | None = None

    # Delegation (Phase 7).
    # delegated_agents: sub-agents callable as tools (agent_as_tool pattern).
    # handoffs: agents this one can transfer control to (routing pattern).
    # Declared here as forward references since delegation.py imports Agent.
    delegated_agents: tuple[Any, ...] = ()  # tuple[DelegatedAgent, ...]
    handoffs: tuple[Agent, ...] = ()  # Agents to route to on handoff
    max_delegation_depth: int = 3  # Safety limit for nested delegation

    # Grounding policy (FUND-8). When set, the Runner applies the
    # policy's confidence threshold to GroundedAnswer results from
    # ResearchAgent and any Call/Program that produces Answer[T].
    # Answers below min_confidence collapse to InsufficientEvidence
    # with an explicit refusal event for visibility.
    refusal_policy: Any | None = None  # RefusalPolicy from kaos-llm-core

    # Metadata (not used by Runner, but available for tracing/logging)
    name: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        instructions: str = "You are a helpful assistant.",
        model: str = DEFAULT_MODEL,
        pattern: str | AgentPattern = AgentPattern.CHAT,
        tools: tuple[str, ...] | list[str] = (),
        provider: ProviderConfig | None = None,
        settings: KaosAgentSettings | None = None,
        max_tools: int | None = None,
        max_react_iterations: int | None = None,
        max_plan_steps: int | None = None,
        rag_top_k: int | None = None,
        rag_max_retries: int | None = None,
        delegated_agents: tuple[Any, ...] | list[Any] = (),
        handoffs: tuple[Agent, ...] | list[Agent] = (),
        max_delegation_depth: int = 3,
        refusal_policy: Any | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Create an Agent with convenience conversions.

        Accepts string pattern names and list tools for ergonomic usage.
        Converts to the frozen internal representation.
        """
        return cls(
            instructions=instructions,
            model=model,
            pattern=AgentPattern(pattern) if isinstance(pattern, str) else pattern,
            tools=tuple(tools),
            provider=provider,
            settings=settings,
            max_tools=max_tools,
            max_react_iterations=max_react_iterations,
            max_plan_steps=max_plan_steps,
            rag_top_k=rag_top_k,
            rag_max_retries=rag_max_retries,
            delegated_agents=tuple(delegated_agents),
            handoffs=tuple(handoffs),
            max_delegation_depth=max_delegation_depth,
            refusal_policy=refusal_policy,
            name=name,
            metadata=tuple(sorted(metadata.items())) if metadata else (),
        )

    def resolve_settings(self) -> KaosAgentSettings:
        """Resolve settings: use provided or construct from environment."""
        return KaosAgentSettings.resolve(self.settings)

    def effective_model(self) -> str:
        """The model string to use, considering provider and settings fallback.

        Priority: provider.default > explicit model > settings default.
        """
        if self.provider is not None:
            return self.provider.default
        if self.model != DEFAULT_MODEL:
            return self.model
        return self.resolve_settings().default_llm_model

    def model_for_role(self, role: str) -> str:
        """Get the model for a specific role (classify, plan, research, etc.).

        If a ``provider`` is set, delegates to ``ProviderConfig.model_for()``.
        Otherwise returns ``effective_model()`` for all roles.
        """
        if self.provider is not None:
            from kaos_agents.providers import ModelRole

            try:
                return self.provider.model_for(ModelRole(role))
            except ValueError:
                return self.provider.default
        return self.effective_model()

    def tool_filter(self) -> list[str] | None:
        """Convert tool glob patterns to a filter list, or None for 'all tools'."""
        return list(self.tools) if self.tools else None
