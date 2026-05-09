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
from typing import TYPE_CHECKING, Any

from kaos_agents.settings import DEFAULT_MODEL, KaosAgentSettings
from kaos_agents.types.providers import ProviderConfig

if TYPE_CHECKING:
    # Imported only for type hints — kaos_agents.core.envelope imports
    # ``AgentPattern`` from this module, so a top-level import would be
    # circular. ``to_envelope`` / ``from_envelope`` defer the runtime
    # import inside the method bodies.
    from kaos_agents.core.envelope import AgentEnvelope


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

    def effective_refusal_policy(self) -> Any | None:
        """Resolve the refusal policy: caller-supplied wins; otherwise
        derive from settings.verifier_min_confidence > 0.

        Order of precedence:
        1. Explicit ``refusal_policy=`` passed at construction (highest).
        2. ``settings.verifier_min_confidence > 0`` → build a default
           ``RefusalPolicy(min_confidence=that)``.
        3. None (legacy permissive — no enforcement).

        N4 — exposes the verifier-confidence floor as an env-var/CLI
        knob so a partner can deploy ``KAOS_AGENT_VERIFIER_MIN_CONFIDENCE=0.7``
        and have low-confidence answers collapse to InsufficientEvidence
        without writing code. The kaos-llm-core ``RefusalPolicy`` is the
        underlying mechanism; this method is the convenience surface.
        """
        if self.refusal_policy is not None:
            return self.refusal_policy
        try:
            settings = self.resolve_settings()
        except Exception:
            return None
        threshold = float(getattr(settings, "verifier_min_confidence", 0.0) or 0.0)
        if threshold <= 0.0:
            return None
        try:
            from kaos_llm_core.signatures.grounding import RefusalPolicy

            return RefusalPolicy(min_confidence=threshold)
        except ImportError:
            return None

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
            from kaos_agents.types.providers import ModelRole

            try:
                return self.provider.model_for(ModelRole(role))
            except ValueError:
                return self.provider.default
        return self.effective_model()

    def tool_filter(self) -> list[str] | None:
        """Convert tool glob patterns to a filter list, or None for 'all tools'."""
        return list(self.tools) if self.tools else None

    def clone_with(self, **overrides: Any) -> Agent:
        """Return a new Agent with the given fields replaced.

        Mirrors ``kaos_llm_core.programs.cloning.clone_call``. Use for
        delegation and routing — sub-agents are typically a parent agent
        with a few fields swapped (instructions, model, tools).

        Frozen dataclass + ``dataclasses.replace`` makes this a one-liner.
        Unknown field names raise ``TypeError`` (the standard library's
        behaviour, no extra validation needed).
        """
        from dataclasses import replace

        return replace(self, **overrides)

    def to_envelope(self) -> AgentEnvelope:
        """Return a content-addressed AgentEnvelope mirroring this agent.

        Phase 0.D limitation: ``settings`` (the full ``KaosAgentSettings``)
        and ``refusal_policy`` are NOT round-tripped — they live in
        environment variables and a typed kaos-llm-core type
        respectively. Phase 1 will add ``settings_overrides`` projection
        and refusal-policy capture.

        Recursion: ``delegated_agents`` typically contain
        ``DelegatedAgent`` wrappers (see ``kaos_agents.runtime.delegation``);
        each wrapper exposes ``.agent`` which is an ``Agent`` we can
        envelope. Anything that doesn't expose an Agent via ``.agent``
        is silently dropped (with a debug log) — recoverable in Phase 4
        when delegation is rewritten. ``handoffs`` are plain Agents, so
        they envelope directly.
        """
        # Local import — see TYPE_CHECKING note at the top of this module.
        from kaos_core.logging import get_logger

        from kaos_agents.core.envelope import AgentEnvelope

        logger = get_logger("kaos.agents.config")

        # Project provider — the AgentEnvelope schema stores it as a dict.
        # ``ProviderConfig`` is a frozen dataclass (not Pydantic) so we use
        # ``dataclasses.asdict``. ``model_dump`` is checked first to keep
        # the door open for a future Pydantic provider type without
        # changing this method.
        provider_payload: dict[str, Any] | None
        if self.provider is None:
            provider_payload = None
        else:
            model_dump = getattr(self.provider, "model_dump", None)
            if callable(model_dump):
                # Pydantic-style providers (future extensibility).
                provider_payload = model_dump()
            else:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(self.provider) and not isinstance(self.provider, type):
                    provider_payload = asdict(self.provider)
                else:
                    # Last-resort fallback — best-effort dict projection.
                    provider_payload = dict(getattr(self.provider, "__dict__", {}))

        # Project delegated_agents → tuple[AgentEnvelope, ...].
        delegated_envelopes: list[AgentEnvelope] = []
        for entry in self.delegated_agents:
            inner = getattr(entry, "agent", None)
            if isinstance(inner, Agent):
                delegated_envelopes.append(inner.to_envelope())
            elif isinstance(entry, Agent):
                # Defensive: support callers that pass plain Agents.
                delegated_envelopes.append(entry.to_envelope())
            else:
                logger.debug(
                    "to_envelope: skipping delegated entry without .agent: type=%s",
                    type(entry).__name__,
                )

        # Project handoffs → tuple[AgentEnvelope, ...]. Skip non-Agent
        # entries the same way (defensive — the field is typed as Agent
        # but the dataclass doesn't enforce that at runtime).
        handoff_envelopes: list[AgentEnvelope] = []
        for entry in self.handoffs:
            if isinstance(entry, Agent):
                handoff_envelopes.append(entry.to_envelope())
            else:
                logger.debug(
                    "to_envelope: skipping handoff entry that is not an Agent: type=%s",
                    type(entry).__name__,
                )

        return AgentEnvelope(
            pattern=self.pattern,
            instructions=self.instructions,
            model=self.model,
            tools=self.tools,
            provider=provider_payload,
            settings_overrides={},  # Phase 0.D: settings not round-tripped.
            max_tools=self.max_tools,
            max_react_iterations=self.max_react_iterations,
            max_plan_steps=self.max_plan_steps,
            rag_top_k=self.rag_top_k,
            rag_max_retries=self.rag_max_retries,
            delegated_agents=tuple(delegated_envelopes),
            handoffs=tuple(handoff_envelopes),
            max_delegation_depth=self.max_delegation_depth,
            name=self.name,
            metadata=self.metadata,
            recipe_id=None,
        )

    @classmethod
    def from_envelope(cls, envelope: AgentEnvelope | dict[str, Any]) -> Agent:
        """Reconstruct an Agent from a (possibly serialized) AgentEnvelope.

        Inverse of :meth:`to_envelope`. Round-trips bit-identically modulo
        the Phase 0.D limitations:

        - ``settings`` (full ``KaosAgentSettings``) is dropped — the
          rebuilt Agent uses ``None`` and resolves from the environment.
        - ``refusal_policy`` is dropped — set None on the rebuilt Agent.
        - ``delegated_agents`` is the lossy direction: the envelope only
          stores nested ``AgentEnvelope`` payloads, not the
          ``DelegatedAgent`` wrappers the Runner expects. We rebuild
          them as plain ``Agent`` instances and stash them back into
          ``delegated_agents``. Callers wanting Runner-ready wrappers
          must re-wrap via ``kaos_agents.runtime.delegation.agent_as_tool``.
          Phase 4 will tighten this contract when delegation is
          rewritten.
        """
        # Local import — see TYPE_CHECKING note at the top of this module.
        from kaos_agents.core.envelope import AgentEnvelope

        # Accept dict input by validating through AgentEnvelope so
        # defaults/types are applied.
        if isinstance(envelope, dict):
            envelope = AgentEnvelope.model_validate(envelope)

        # Rebuild provider — ProviderConfig is a frozen dataclass.
        provider: ProviderConfig | None
        if envelope.provider is None:
            provider = None
        else:
            from kaos_agents.types.providers import ModelRole

            payload = dict(envelope.provider)
            raw_role_models = payload.pop("role_models", {}) or {}
            # Coerce the role keys back to ModelRole — JSON round-trip
            # leaves them as strings.
            role_models: dict[ModelRole, str] = {}
            for key, value in raw_role_models.items():
                role_models[ModelRole(key) if isinstance(key, str) else key] = value
            provider = ProviderConfig(
                default=payload.get("default", DEFAULT_MODEL),
                role_models=role_models,
            )

        # Rebuild delegated_agents recursively. The envelope only knows
        # nested AgentEnvelopes — return raw Agents (lossy direction
        # documented in the docstring).
        delegated_agents: tuple[Any, ...] = tuple(
            cls.from_envelope(item) for item in envelope.delegated_agents
        )
        handoffs: tuple[Agent, ...] = tuple(cls.from_envelope(item) for item in envelope.handoffs)

        return cls(
            instructions=envelope.instructions,
            model=envelope.model,
            pattern=envelope.pattern,
            tools=envelope.tools,
            provider=provider,
            settings=None,  # Phase 0.D: not round-tripped.
            max_tools=envelope.max_tools,
            max_react_iterations=envelope.max_react_iterations,
            max_plan_steps=envelope.max_plan_steps,
            rag_top_k=envelope.rag_top_k,
            rag_max_retries=envelope.rag_max_retries,
            delegated_agents=delegated_agents,
            handoffs=handoffs,
            max_delegation_depth=envelope.max_delegation_depth,
            refusal_policy=None,  # Phase 0.D: not round-tripped.
            name=envelope.name,
            metadata=envelope.metadata,
        )
