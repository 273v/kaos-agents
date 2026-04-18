"""Provider configuration — role-based model selection.

Real workflows need different models for different tasks:
- Classification is cheap (haiku)
- Planning needs strong reasoning (sonnet/opus)
- Research needs grounded generation (sonnet)
- Simple responses can use the cheapest model

``ProviderConfig`` maps ``ModelRole`` to model identifiers.
Presets (FAST, BALANCED, STRONG) provide common configurations.

Usage::

    from kaos_agents.providers import BALANCED

    agent = Agent(
        instructions="Research assistant.",
        provider=BALANCED,
        tools=("kaos-source-*",),
    )
    # Classification uses haiku, planning uses sonnet, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, unique

from kaos_agents.settings import DEFAULT_MODEL


@unique
class ModelRole(StrEnum):
    """What kind of work the model is doing."""

    CLASSIFY = "classify"  # Intent classification (cheap, fast)
    RESPOND = "respond"  # Simple conversational response
    PLAN = "plan"  # Plan expansion and decomposition
    RESEARCH = "research"  # RAG and document Q&A
    EVALUATE = "evaluate"  # Result quality judgment


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Maps model roles to model identifiers.

    Roles not in ``role_models`` fall back to ``default``.
    Model identifiers use the kaos-llm-client format:
    ``provider:model-name`` (e.g., ``anthropic:claude-haiku-4-5``).

    Usage::

        config = ProviderConfig(
            default="anthropic:claude-haiku-4-5",
            role_models={
                ModelRole.PLAN: "anthropic:claude-sonnet-4-6",
                ModelRole.RESEARCH: "anthropic:claude-sonnet-4-6",
            },
        )
        model = config.model_for(ModelRole.PLAN)  # -> "anthropic:claude-sonnet-4-6"
        model = config.model_for(ModelRole.CLASSIFY)  # -> "anthropic:claude-haiku-4-5"
    """

    default: str = DEFAULT_MODEL
    role_models: dict[ModelRole, str] = field(default_factory=dict)

    def model_for(self, role: ModelRole) -> str:
        """Get the model identifier for a given role."""
        return self.role_models.get(role, self.default)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

FAST = ProviderConfig(
    default="anthropic:claude-haiku-4-5",
    role_models={},
)
"""Use the cheapest model for everything. Good for development and testing."""

BALANCED = ProviderConfig(
    default="anthropic:claude-haiku-4-5",
    role_models={
        ModelRole.PLAN: "anthropic:claude-sonnet-4-6",
        ModelRole.RESEARCH: "anthropic:claude-sonnet-4-6",
    },
)
"""Cheap for classification/response, strong for planning/research."""

STRONG = ProviderConfig(
    default="anthropic:claude-sonnet-4-6",
    role_models={
        ModelRole.PLAN: "anthropic:claude-opus-4-6",
        ModelRole.RESEARCH: "anthropic:claude-opus-4-6",
    },
)
"""Strong for everything, strongest for planning/research. Higher cost."""
