"""TurnPlan — the public composition surface for one agent turn.

Mirrors kaos_llm_core.programs.call.CallPlan at the turn boundary:
external composers (delegation, MCP wrappers, FastAPI routes,
evaluation harnesses) consume a TurnPlan instead of reaching into
private agent state.

Phase 0 ships a minimal TurnPlan with the fields known to be needed
by the existing Runner — later phases (1-4) extend it with intent,
perceiver, actor, planner, termination_judge, escalation_policy, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_agents.base.event import KaosEvent
from kaos_agents.events.emitter import EventEmitter


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """Resolved, frozen bundle for one agent turn.

    Built by AgentLoop.prepare_turn(trigger). Consumed by every step
    method on AgentLoop and by external composers. No instance scratch
    on AgentLoop / Runner; all per-turn data flows through TurnPlan
    plus the active TurnInvocation (for mutated state).
    """

    # Identity
    session_id: str
    run_id: str
    turn_number: int

    # The trigger that opened this turn (Phase 1 will tighten the type).
    trigger: Any

    # Active emitter (auto-fills timestamp/sequence/session_id/run_id).
    emitter: EventEmitter

    # Span linkage — set when this turn is a sub-agent of an outer turn.
    parent_span_id: str | None = None

    # Phase-1+ subsystems (typed Any in Phase 0; tightened later)
    intent: Any = None  # IntentResult
    memory: Any = None  # SessionMemory hydrated for this session
    working_memory: dict[str, Any] = field(default_factory=dict)
    perceiver: Any = None
    actor: Any = None
    planner: Any = None
    termination_judge: Any = None
    escalation_policy: Any = None
    permission_policy: Any = None

    # Optional initial event seed (e.g. an inbound MCP message
    # represented as a KaosEvent before the turn began). Most callers
    # leave this empty.
    seed_events: tuple[KaosEvent, ...] = ()


__all__ = ["TurnPlan"]
