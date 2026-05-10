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

    # DEFECT-5 (May 2026): IntentExtractor usage from prepare_turn.
    # The IntentExtractor's LLM call happens inside prepare_turn
    # before the AgentLoop's collect_events scope is open, so its
    # usage would otherwise be invisible to the loop's roll-up.
    # AgentLoop._run_8_step_turn emits a synthetic UsageObserved
    # from this field at Step 1 so the cost lands in the active
    # collector. ``None`` when no usage info was captured (e.g.
    # the intent_extractor stub didn't return an Invocation).
    intent_usage: Any = None


__all__ = ["TurnPlan"]
