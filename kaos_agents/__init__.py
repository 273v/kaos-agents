"""kaos-agents — Agentic runtime with persistent memory for KAOS.

Base-install discipline (KC17-P0-1)
-----------------------------------

A large fraction of the public surface lives in modules that pull in
optional extras at import time:

* ``[llm]``  → ``kaos_llm_core``  (Programs, Signatures, ReAct, Call, …)
* ``[api]``  → ``fastapi`` + ``starlette``  (HTTP server)

Eagerly importing those names from this top-level ``__init__`` breaks
``import kaos_agents`` for users who installed only the base
distribution — the audited regression that this module's lazy
attribute table closes.

The fix
^^^^^^^

Names whose home module transitively requires ``kaos_llm_core`` (or
``fastapi``) are listed in :data:`_LAZY_ATTRS` and resolved on first
access via :pep:`562` ``__getattr__``.  Names whose home module is
fully importable without any optional extra remain eagerly imported,
keeping the static-analysis ergonomics (IDE jump-to-definition,
``ty`` / ``mypy`` reachability) intact for the always-on subset.

Consumer ergonomics are unchanged::

    from kaos_agents import Actor, Runner, FindingsAgent  # still works

The difference is purely *when* the optional-extra dependency is
materialised: at first use of the name, not at top-of-package
import.  Consumers without the ``[llm]`` extra still successfully
``import kaos_agents`` and can use the always-on names
(``KaosEvent``, ``Agent``, ``AgentPattern``, ``KaosAgentSettings``,
…) without ever pulling in ``kaos_llm_core``.

The :class:`Actor` resolution path layers an explicit install-hint
:class:`ImportError` on top of the underlying error so the message
points at the right ``pip install`` invocation; this lives in
:mod:`kaos_agents.action`'s ``__getattr__``.

TYPE_CHECKING re-exports
^^^^^^^^^^^^^^^^^^^^^^^^

Type checkers (``ty``, ``mypy``) don't execute ``__getattr__`` — they
only see the static ``import`` statements.  All lazy names therefore
also live under an ``if TYPE_CHECKING:`` guard so static analysers
keep finding them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from kaos_agents._version import __version__

# ---------------------------------------------------------------------------
# Eager imports — modules that do NOT transitively require optional extras.
# ---------------------------------------------------------------------------
from kaos_agents.action import (
    ActionPlan,
    ActionRefusal,
    ActionResult,
    ApprovalWorkflow,
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    Reversibility,
)
from kaos_agents.base import KaosAgent, KaosPattern
from kaos_agents.config import Agent, AgentPattern
from kaos_agents.core import (
    AgentEnvelope,
    TurnInvocation,
    TurnPlan,
    agent_hash,
    current_turn,
)
from kaos_agents.decorators import FunctionHook, hook
from kaos_agents.errors import (
    ERROR_KIND_AUTH,
    ERROR_KIND_CONTEXT_TOO_LARGE,
    ERROR_KIND_PROVIDER,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_SERVICE_UNAVAILABLE,
    ERROR_KIND_TRANSPORT,
    SURFACING_FAILURE_KINDS,
    AgentFailureClassification,
    EventDeserializationError,
    EventSerializationError,
    EvictionError,
    KaosAgentError,
    MemoryBudgetExceededError,
    SectionNotConfiguredError,
    SessionCorruptedError,
    SessionNotFoundError,
    classify_agent_failure,
)
from kaos_agents.escalation import (
    EscalationContext,
    EscalationDecision,
    EscalationKind,
    EscalationPolicy,
    EscalationResumePayload,
    HITLBridge,
    HITLChannel,
    escalation_resource_uri,
)
from kaos_agents.events import (
    BudgetExceeded,
    CitationFound,
    EscalationRequired,
    EventEmitter,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
    IntentClassified,
    KaosEvent,
    LifecycleEvent,
    MemoryEvent,
    MemoryEventKind,
    PlanProposed,
    PlanStepSummary,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallApprovalRequired,
    ToolCallArgsDelta,
    ToolCallSummary,
    TurnSummary,
    deserialize_event,
    deserialize_event_json,
    serialize_event,
    serialize_event_json,
)
from kaos_agents.hooks import (
    AuditHook,
    CostTrackingHook,
    HookAction,
    KaosHook,
    LoggingHook,
    dispatch_hook,
)
from kaos_agents.hooks.otel import OTelHook
from kaos_agents.registry import (
    AgentToolSpecRegistry,
    EventRegistry,
    HookRegistry,
    PatternRegistry,
    ToolDataTypeRegistry,
    ToolGroupRegistry,
    default_event_registry,
    default_hook_registry,
    default_pattern_registry,
    default_tool_data_type_registry,
    default_tool_group_registry,
    default_tool_spec_registry,
)
from kaos_agents.runtime.delegation import (
    DelegatedAgent,
    DelegationDepthExceeded,
    agent_as_tool,
    current_delegation_depth,
)
from kaos_agents.runtime.interrupts import PendingToolCall, RunState
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.tools import register_agent_tools
from kaos_agents.triggers import (
    CLIPromptTrigger,
    DelegationTrigger,
    EscalationTrigger,
    FileSystemTrigger,
    HTTPMessageTrigger,
    MCPToolTrigger,
    ScheduledTrigger,
    Trigger,
    TriggerKind,
    TriggerSource,
    WebhookTrigger,
)
from kaos_agents.types import (
    AgentMetadata,
    AgentResponse,
    AgentToolRegistration,
    AgentToolSpec,
    DataType,
    EventMetadata,
    FieldSet,
    HookMetadata,
    IntentResult,
    IntentType,
    PatternMetadata,
    PermissionDecision,
    PermissionRule,
    RecipeMetadata,
    SessionToolSet,
    ToolCallRecord,
    ToolDataTypeSpec,
    ToolExecution,
    ToolGroup,
)
from kaos_agents.types.providers import BALANCED, FAST, STRONG, ModelRole, ProviderConfig

# ---------------------------------------------------------------------------
# TYPE_CHECKING re-exports — give static analysers visibility into names
# that are resolved lazily at runtime via ``__getattr__`` below.  These
# imports are stripped at runtime so they cost nothing on a base install.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover — typing-only
    from kaos_agents.action.actor import Actor
    from kaos_agents.api.wire import events_to_jsonl, events_to_sse
    from kaos_agents.context import (
        CatalogMode,
        TriageResult,
        assemble_context,
        expand_with_graph,
        filter_tools,
        render_tool_catalog,
        triage_corpus,
    )
    from kaos_agents.governance import (
        JSONLAuditLogger,
        OverrideHook,
        OverrideKind,
        StateSnapshot,
        load_snapshot,
        save_snapshot,
    )
    from kaos_agents.intent import (
        Ambiguity,
        AmbiguityKind,
        ClarificationLoop,
        Constraint,
        ConstraintKind,
        Goal,
        IntentExtractor,
        IntentSignature,
    )
    from kaos_agents.intent import IntentResult as IntentResultV2  # type: ignore[attr-defined]
    from kaos_agents.loop import AgentLoop
    from kaos_agents.memory import (
        EvictionPolicy,
        KBEntry,
        KBQuery,
        KBResult,
        KnowledgeBase,
        MatterClientGuard,
        MatterIsolationError,
        MemoryItem,
        MemoryType,
        PersistenceMode,
        PromotionDecision,
        PromotionPolicy,
        Section,
        SectionConfig,
        SessionMemory,
        SessionStore,
        SummarizationPolicy,
        WorkingMemory,
        agent_iri,
        doc_iri,
        emit_from_event,
        finding_iri,
        step_iri,
        tool_call_iri,
    )
    from kaos_agents.optimization import (
        AgentEvalResult,
        AgentExample,
        AgentMetricFn,
        evaluate_agent,
    )
    from kaos_agents.patterns import ChatAgent, PlanExecuteAgent, ResearchAgent
    from kaos_agents.patterns.findings import FindingsAgent
    from kaos_agents.patterns.reflexion import ReflexionLoop
    from kaos_agents.patterns.router import RouterAgent
    from kaos_agents.perception import (
        Perceiver,
        PerceptionItem,
        PerceptionQuery,
        PerceptionQueryKind,
        PerceptionRefusal,
        PerceptionResult,
        read_only_tools,
    )
    from kaos_agents.planning import (
        ComposeResult,
        Decision,
        EvalMode,
        HierarchicalPlan,
        HierarchicalPlanner,
        Judgment,
        Plan,
        PlanBudget,
        PlanExecutePlan,
        PlanExecutePlanner,
        PlanGraph,
        Planner,
        PlanResult,
        PlanStep,
        PrimitiveTrace,
        ReActPlan,
        ReActPlanner,
        RouteResult,
        Step,
        StepStatus,
        StepType,
        StopReason,
        SubAgentSpec,
    )
    from kaos_agents.runtime.agent import BaseAgent
    from kaos_agents.runtime.runner import Runner
    from kaos_agents.termination import Decision as TerminationDecision
    from kaos_agents.termination import (
        DecisionKind,
        DegradationOutcome,
        DegradationPolicy,
        LoopDetector,
        LoopDetectorResult,
        SuccessCriteria,
        TerminationJudge,
    )


# ---------------------------------------------------------------------------
# Lazy attribute table — name → (module path, source attribute name).
#
# The source attribute name is usually identical to the public name; the
# tuple form lets us alias (e.g. ``IntentResultV2`` ← ``IntentResult`` from
# :mod:`kaos_agents.intent`, or ``TerminationDecision`` ←
# ``Decision`` from :mod:`kaos_agents.termination`).
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # --- action / api: explicit install hints live in their own __getattr__.
    "Actor": ("kaos_agents.action", "Actor"),
    "events_to_jsonl": ("kaos_agents.api.wire", "events_to_jsonl"),
    "events_to_sse": ("kaos_agents.api.wire", "events_to_sse"),
    # --- context (transitively imports kaos_llm_core for classify/doc2query/retrieval).
    "CatalogMode": ("kaos_agents.context", "CatalogMode"),
    "TriageResult": ("kaos_agents.context", "TriageResult"),
    "assemble_context": ("kaos_agents.context", "assemble_context"),
    "expand_with_graph": ("kaos_agents.context", "expand_with_graph"),
    "filter_tools": ("kaos_agents.context", "filter_tools"),
    "render_tool_catalog": ("kaos_agents.context", "render_tool_catalog"),
    "triage_corpus": ("kaos_agents.context", "triage_corpus"),
    # --- governance (action.circuit re-export is fine, but the package's
    #     own modules pull kaos_llm_core via the Actor surface).
    "JSONLAuditLogger": ("kaos_agents.governance", "JSONLAuditLogger"),
    "OverrideHook": ("kaos_agents.governance", "OverrideHook"),
    "OverrideKind": ("kaos_agents.governance", "OverrideKind"),
    "StateSnapshot": ("kaos_agents.governance", "StateSnapshot"),
    "load_snapshot": ("kaos_agents.governance", "load_snapshot"),
    "save_snapshot": ("kaos_agents.governance", "save_snapshot"),
    # --- intent (clarify + extractor use Program / Call / ChainOfThought).
    "Ambiguity": ("kaos_agents.intent", "Ambiguity"),
    "AmbiguityKind": ("kaos_agents.intent", "AmbiguityKind"),
    "ClarificationLoop": ("kaos_agents.intent", "ClarificationLoop"),
    "Constraint": ("kaos_agents.intent", "Constraint"),
    "ConstraintKind": ("kaos_agents.intent", "ConstraintKind"),
    "Goal": ("kaos_agents.intent", "Goal"),
    "IntentExtractor": ("kaos_agents.intent", "IntentExtractor"),
    "IntentSignature": ("kaos_agents.intent", "IntentSignature"),
    "IntentResultV2": ("kaos_agents.intent", "IntentResult"),
    # --- loop (8-step turn loop drives a Program).
    "AgentLoop": ("kaos_agents.loop", "AgentLoop"),
    # --- memory (institutional.KnowledgeBase subclasses Program).
    "EvictionPolicy": ("kaos_agents.memory", "EvictionPolicy"),
    "KBEntry": ("kaos_agents.memory", "KBEntry"),
    "KBQuery": ("kaos_agents.memory", "KBQuery"),
    "KBResult": ("kaos_agents.memory", "KBResult"),
    "KnowledgeBase": ("kaos_agents.memory", "KnowledgeBase"),
    "MatterClientGuard": ("kaos_agents.memory", "MatterClientGuard"),
    "MatterIsolationError": ("kaos_agents.memory", "MatterIsolationError"),
    "MemoryItem": ("kaos_agents.memory", "MemoryItem"),
    "MemoryType": ("kaos_agents.memory", "MemoryType"),
    "PersistenceMode": ("kaos_agents.memory", "PersistenceMode"),
    "PromotionDecision": ("kaos_agents.memory", "PromotionDecision"),
    "PromotionPolicy": ("kaos_agents.memory", "PromotionPolicy"),
    "Section": ("kaos_agents.memory", "Section"),
    "SectionConfig": ("kaos_agents.memory", "SectionConfig"),
    "SessionMemory": ("kaos_agents.memory", "SessionMemory"),
    "SessionStore": ("kaos_agents.memory", "SessionStore"),
    "SummarizationPolicy": ("kaos_agents.memory", "SummarizationPolicy"),
    "WorkingMemory": ("kaos_agents.memory", "WorkingMemory"),
    "agent_iri": ("kaos_agents.memory", "agent_iri"),
    "doc_iri": ("kaos_agents.memory", "doc_iri"),
    "emit_from_event": ("kaos_agents.memory", "emit_from_event"),
    "finding_iri": ("kaos_agents.memory", "finding_iri"),
    "step_iri": ("kaos_agents.memory", "step_iri"),
    "tool_call_iri": ("kaos_agents.memory", "tool_call_iri"),
    # --- optimization (eval pulls kaos_llm_core types).
    "AgentEvalResult": ("kaos_agents.optimization", "AgentEvalResult"),
    "AgentExample": ("kaos_agents.optimization", "AgentExample"),
    "AgentMetricFn": ("kaos_agents.optimization", "AgentMetricFn"),
    "evaluate_agent": ("kaos_agents.optimization", "evaluate_agent"),
    # --- patterns (chat.py imports Signature/InputField/OutputField at top level).
    "ChatAgent": ("kaos_agents.patterns", "ChatAgent"),
    "PlanExecuteAgent": ("kaos_agents.patterns", "PlanExecuteAgent"),
    "ResearchAgent": ("kaos_agents.patterns", "ResearchAgent"),
    "FindingsAgent": ("kaos_agents.patterns.findings", "FindingsAgent"),
    "ReflexionLoop": ("kaos_agents.patterns.reflexion", "ReflexionLoop"),
    "RouterAgent": ("kaos_agents.patterns.router", "RouterAgent"),
    # --- perception (Perceiver subclasses Program).
    "Perceiver": ("kaos_agents.perception", "Perceiver"),
    "PerceptionItem": ("kaos_agents.perception", "PerceptionItem"),
    "PerceptionQuery": ("kaos_agents.perception", "PerceptionQuery"),
    "PerceptionQueryKind": ("kaos_agents.perception", "PerceptionQueryKind"),
    "PerceptionRefusal": ("kaos_agents.perception", "PerceptionRefusal"),
    "PerceptionResult": ("kaos_agents.perception", "PerceptionResult"),
    "read_only_tools": ("kaos_agents.perception", "read_only_tools"),
    # --- planning (most planners use kaos_llm_core).
    "ComposeResult": ("kaos_agents.planning", "ComposeResult"),
    "Decision": ("kaos_agents.planning", "Decision"),
    "EvalMode": ("kaos_agents.planning", "EvalMode"),
    "HierarchicalPlan": ("kaos_agents.planning", "HierarchicalPlan"),
    "HierarchicalPlanner": ("kaos_agents.planning", "HierarchicalPlanner"),
    "Judgment": ("kaos_agents.planning", "Judgment"),
    "Plan": ("kaos_agents.planning", "Plan"),
    "PlanBudget": ("kaos_agents.planning", "PlanBudget"),
    "PlanExecutePlan": ("kaos_agents.planning", "PlanExecutePlan"),
    "PlanExecutePlanner": ("kaos_agents.planning", "PlanExecutePlanner"),
    "PlanGraph": ("kaos_agents.planning", "PlanGraph"),
    "Planner": ("kaos_agents.planning", "Planner"),
    "PlanResult": ("kaos_agents.planning", "PlanResult"),
    "PlanStep": ("kaos_agents.planning", "PlanStep"),
    "PrimitiveTrace": ("kaos_agents.planning", "PrimitiveTrace"),
    "ReActPlan": ("kaos_agents.planning", "ReActPlan"),
    "ReActPlanner": ("kaos_agents.planning", "ReActPlanner"),
    "RouteResult": ("kaos_agents.planning", "RouteResult"),
    "Step": ("kaos_agents.planning", "Step"),
    "StepStatus": ("kaos_agents.planning", "StepStatus"),
    "StepType": ("kaos_agents.planning", "StepType"),
    "StopReason": ("kaos_agents.planning", "StopReason"),
    "SubAgentSpec": ("kaos_agents.planning", "SubAgentSpec"),
    # --- runtime.agent + runtime.runner (BaseAgent / Runner consume Programs).
    "BaseAgent": ("kaos_agents.runtime.agent", "BaseAgent"),
    "Runner": ("kaos_agents.runtime.runner", "Runner"),
    # --- termination (TerminationJudge subclasses Program; Decision aliased).
    "TerminationDecision": ("kaos_agents.termination", "Decision"),
    "DecisionKind": ("kaos_agents.termination", "DecisionKind"),
    "DegradationOutcome": ("kaos_agents.termination", "DegradationOutcome"),
    "DegradationPolicy": ("kaos_agents.termination", "DegradationPolicy"),
    "LoopDetector": ("kaos_agents.termination", "LoopDetector"),
    "LoopDetectorResult": ("kaos_agents.termination", "LoopDetectorResult"),
    "SuccessCriteria": ("kaos_agents.termination", "SuccessCriteria"),
    "TerminationJudge": ("kaos_agents.termination", "TerminationJudge"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolver for optional-extra-dependent names.

    Resolves each lazy name on first access by importing its home
    module and reading the attribute.  Caches the result back into
    this module's globals so subsequent accesses are O(1).

    Raises :class:`AttributeError` for unknown names — never silently
    returns ``None`` — so ``hasattr(kaos_agents, "bogus")`` is honest.
    """
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    # Cache for subsequent lookups — PEP 562 __getattr__ is called only
    # when the regular attribute lookup misses.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(__all__))


__all__ = [
    "BALANCED",
    "ERROR_KIND_AUTH",
    "ERROR_KIND_CONTEXT_TOO_LARGE",
    "ERROR_KIND_PROVIDER",
    "ERROR_KIND_RATE_LIMIT",
    "ERROR_KIND_SERVICE_UNAVAILABLE",
    "ERROR_KIND_TRANSPORT",
    "FAST",
    "STRONG",
    "SURFACING_FAILURE_KINDS",
    "ActionPlan",
    "ActionRefusal",
    "ActionResult",
    "Actor",
    "Agent",
    "AgentEnvelope",
    "AgentEvalResult",
    "AgentExample",
    "AgentFailureClassification",
    "AgentLoop",
    "AgentMetadata",
    "AgentMetricFn",
    "AgentPattern",
    "AgentResponse",
    "AgentToolRegistration",
    "AgentToolSpec",
    "AgentToolSpecRegistry",
    "Ambiguity",
    "AmbiguityKind",
    "ApprovalWorkflow",
    "AuditHook",
    "BaseAgent",
    "BudgetExceeded",
    "CLIPromptTrigger",
    "CatalogMode",
    "ChatAgent",
    "CircuitBreaker",
    "CircuitState",
    "CitationFound",
    "ClarificationLoop",
    "ComposeResult",
    "Constraint",
    "ConstraintKind",
    "CostTrackingHook",
    "DataType",
    "Decision",
    "DecisionKind",
    "DegradationOutcome",
    "DegradationPolicy",
    "DelegatedAgent",
    "DelegationDepthExceeded",
    "DelegationTrigger",
    "EscalationContext",
    "EscalationDecision",
    "EscalationKind",
    "EscalationPolicy",
    "EscalationRequired",
    "EscalationResumePayload",
    "EscalationTrigger",
    "EvalMode",
    "EventDeserializationError",
    "EventEmitter",
    "EventMetadata",
    "EventRegistry",
    "EventSerializationError",
    "EvictionError",
    "EvictionPolicy",
    "EvidenceInsufficient",
    "FieldSet",
    "FileSystemTrigger",
    "FindingsAgent",
    "FunctionHook",
    "Goal",
    "GroundingRefusalTriggered",
    "HITLBridge",
    "HITLChannel",
    "HTTPMessageTrigger",
    "HierarchicalPlan",
    "HierarchicalPlanner",
    "HookAction",
    "HookMetadata",
    "HookRegistry",
    "IntentClassified",
    "IntentExtractor",
    "IntentResult",
    "IntentResultV2",
    "IntentSignature",
    "IntentType",
    "JSONLAuditLogger",
    "Judgment",
    "KBEntry",
    "KBQuery",
    "KBResult",
    "KaosAgent",
    "KaosAgentError",
    "KaosAgentSettings",
    "KaosEvent",
    "KaosHook",
    "KaosPattern",
    "KnowledgeBase",
    "LifecycleEvent",
    "LoggingHook",
    "LoopDetector",
    "LoopDetectorResult",
    "MCPToolTrigger",
    "MatterClientGuard",
    "MatterIsolationError",
    "MemoryBudgetExceededError",
    "MemoryEvent",
    "MemoryEventKind",
    "MemoryItem",
    "MemoryType",
    "ModelRole",
    "OTelHook",
    "OverrideHook",
    "OverrideKind",
    "PatternMetadata",
    "PatternRegistry",
    "PendingToolCall",
    "Perceiver",
    "PerceptionItem",
    "PerceptionQuery",
    "PerceptionQueryKind",
    "PerceptionRefusal",
    "PerceptionResult",
    "PermissionDecision",
    "PermissionPolicy",
    "PermissionRule",
    "PersistenceMode",
    "Plan",
    "PlanBudget",
    "PlanExecuteAgent",
    "PlanExecutePlan",
    "PlanExecutePlanner",
    "PlanGraph",
    "PlanProposed",
    "PlanResult",
    "PlanStep",
    "PlanStepSummary",
    "Planner",
    "PrimitiveTrace",
    "PromotionDecision",
    "PromotionPolicy",
    "ProviderConfig",
    "RateLimiter",
    "ReActPlan",
    "ReActPlanner",
    "RecipeMetadata",
    "ReflexionLoop",
    "ResearchAgent",
    "Reversibility",
    "RouteResult",
    "RouterAgent",
    "RunError",
    "RunState",
    "Runner",
    "ScheduledTrigger",
    "Section",
    "SectionConfig",
    "SectionNotConfiguredError",
    "SessionCorruptedError",
    "SessionMemory",
    "SessionNotFoundError",
    "SessionStore",
    "SessionToolSet",
    "Span",
    "SpanPhase",
    "SpanSubject",
    "StateSnapshot",
    "Step",
    "StepStatus",
    "StepType",
    "StopReason",
    "StreamDelta",
    "SubAgentSpec",
    "SuccessCriteria",
    "SummarizationPolicy",
    "TerminationDecision",
    "TerminationJudge",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallApprovalRequired",
    "ToolCallArgsDelta",
    "ToolCallRecord",
    "ToolCallSummary",
    "ToolDataTypeRegistry",
    "ToolDataTypeSpec",
    "ToolExecution",
    "ToolGroup",
    "ToolGroupRegistry",
    "TriageResult",
    "Trigger",
    "TriggerKind",
    "TriggerSource",
    "TurnInvocation",
    "TurnPlan",
    "TurnSummary",
    "WebhookTrigger",
    "WorkingMemory",
    "__version__",
    "agent_as_tool",
    "agent_hash",
    "agent_iri",
    "assemble_context",
    "classify_agent_failure",
    "current_delegation_depth",
    "current_turn",
    "default_event_registry",
    "default_hook_registry",
    "default_pattern_registry",
    "default_tool_data_type_registry",
    "default_tool_group_registry",
    "default_tool_spec_registry",
    "deserialize_event",
    "deserialize_event_json",
    "dispatch_hook",
    "doc_iri",
    "emit_from_event",
    "escalation_resource_uri",
    "evaluate_agent",
    "events_to_jsonl",
    "events_to_sse",
    "expand_with_graph",
    "filter_tools",
    "finding_iri",
    "hook",
    "load_snapshot",
    "read_only_tools",
    "register_agent_tools",
    "render_tool_catalog",
    "save_snapshot",
    "serialize_event",
    "serialize_event_json",
    "step_iri",
    "tool_call_iri",
    "triage_corpus",
]
