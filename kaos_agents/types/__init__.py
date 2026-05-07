"""Frozen value types for kaos-agents.

This package holds *only* pure data types — frozen ``@dataclass`` value
records and frozen pydantic ``KaosModel`` metadata. No behavior, no I/O,
no module-level state. Mirrors the layout of ``kaos_core.types``.

The split:

- :mod:`kaos_agents.types.intents` — ``IntentType`` / ``IntentResult``
- :mod:`kaos_agents.types.tool_call` — ``ToolCallRecord``
- :mod:`kaos_agents.types.response` — ``AgentResponse``
- :mod:`kaos_agents.types.usage` — ``InvocationUsage`` / ``ZERO_USAGE``
- :mod:`kaos_agents.types.plan` — planning system types (``Step``, ``Decision``, ...)
- :mod:`kaos_agents.types.memory` — memory system types (``MemoryItem``, ``SectionConfig``, ...)
- :mod:`kaos_agents.types.providers` — ``ProviderConfig`` / ``ModelRole`` + presets
- :mod:`kaos_agents.types.permissions` — ``PermissionRule`` / ``PermissionDecision``
- :mod:`kaos_agents.types.metadata` — ``AgentMetadata`` / ``EventMetadata`` /
  ``HookMetadata`` / ``PatternMetadata`` / ``RecipeMetadata`` (frozen pydantic)
"""

from __future__ import annotations

from kaos_agents.types.agent_tool_spec import AgentToolRegistration, AgentToolSpec
from kaos_agents.types.data_type import DataType, ToolDataTypeSpec
from kaos_agents.types.intents import IntentResult, IntentType
from kaos_agents.types.memory import (
    DEFAULT_SECTIONS,
    EvictionPolicy,
    MemoryItem,
    MemoryType,
    PersistenceMode,
    SectionConfig,
    SummarizationPolicy,
    create_item,
    estimate_tokens,
)
from kaos_agents.types.metadata import (
    AgentMetadata,
    EventMetadata,
    HookMetadata,
    PatternMetadata,
    RecipeMetadata,
)
from kaos_agents.types.permissions import PermissionDecision, PermissionRule
from kaos_agents.types.plan import (
    ComposeResult,
    Decision,
    EvalMode,
    Judgment,
    PlanBudget,
    PrimitiveTrace,
    RouteResult,
    Step,
    StepStatus,
    StepType,
    StopReason,
)
from kaos_agents.types.providers import (
    BALANCED,
    FAST,
    STRONG,
    ModelRole,
    ProviderConfig,
)
from kaos_agents.types.response import AgentResponse
from kaos_agents.types.tool_call import ToolCallRecord, ToolExecution
from kaos_agents.types.tool_group import ToolGroup
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage

__all__ = [
    "BALANCED",
    "DEFAULT_SECTIONS",
    "FAST",
    "STRONG",
    "ZERO_USAGE",
    "AgentMetadata",
    "AgentResponse",
    "AgentToolRegistration",
    "AgentToolSpec",
    "ComposeResult",
    "DataType",
    "Decision",
    "EvalMode",
    "EventMetadata",
    "EvictionPolicy",
    "HookMetadata",
    "IntentResult",
    "IntentType",
    "InvocationUsage",
    "Judgment",
    "MemoryItem",
    "MemoryType",
    "ModelRole",
    "PatternMetadata",
    "PermissionDecision",
    "PermissionRule",
    "PersistenceMode",
    "PlanBudget",
    "PrimitiveTrace",
    "ProviderConfig",
    "RecipeMetadata",
    "RouteResult",
    "SectionConfig",
    "Step",
    "StepStatus",
    "StepType",
    "StopReason",
    "SummarizationPolicy",
    "ToolCallRecord",
    "ToolDataTypeSpec",
    "ToolExecution",
    "ToolGroup",
    "create_item",
    "estimate_tokens",
]
