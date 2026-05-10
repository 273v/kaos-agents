"""Session memory — section-based context management for the agent runtime."""

from __future__ import annotations

from kaos_agents.memory.institutional import KBEntry, KBQuery, KBResult, KnowledgeBase
from kaos_agents.memory.isolation import MatterClientGuard, MatterIsolationError
from kaos_agents.memory.promotion import PromotionDecision, PromotionPolicy
from kaos_agents.memory.sections import Section
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.triples import (
    agent_iri,
    doc_iri,
    emit_from_event,
    finding_iri,
    step_iri,
    tool_call_iri,
)
from kaos_agents.memory.working import WorkingMemory
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

__all__ = [
    "DEFAULT_SECTIONS",
    "EvictionPolicy",
    "KBEntry",
    "KBQuery",
    "KBResult",
    "KnowledgeBase",
    "MatterClientGuard",
    "MatterIsolationError",
    "MemoryItem",
    "MemoryType",
    "PersistenceMode",
    "PromotionDecision",
    "PromotionPolicy",
    "Section",
    "SectionConfig",
    "SessionMemory",
    "SessionStore",
    "SummarizationPolicy",
    "WorkingMemory",
    "agent_iri",
    "create_item",
    "doc_iri",
    "emit_from_event",
    "estimate_tokens",
    "finding_iri",
    "step_iri",
    "tool_call_iri",
]
