"""kaos_agents.core — Phase 0 ground-up rewrite primitives.

Re-exports the canonical runtime contracts for an agent turn:

- :class:`TurnInvocation` + :func:`current_turn` + :data:`_active_turn_var`
  — the per-turn execution record (mirror of
  :class:`kaos_llm_core.programs._invocation.Invocation`).
- :class:`TurnPlan` — the public composition surface for one turn
  (mirror of :class:`kaos_llm_core.programs.call.CallPlan`).
- :class:`AgentEnvelope` + :func:`agent_hash` — content-addressed
  declarative form of an Agent (mirror of
  :class:`kaos_llm_core.programs.envelope.ProgramEnvelope`).

Phase 0.A is purely additive — no existing behavior changes. Later
phases (1-4) wire these into ``runtime/agent.py`` and ``patterns/*``.
"""

from kaos_agents.core.envelope import AgentEnvelope, agent_hash
from kaos_agents.core.invocation import (
    TurnInvocation,
    _active_turn_var,
    current_turn,
)
from kaos_agents.core.plan import TurnPlan

__all__ = [
    "AgentEnvelope",
    "TurnInvocation",
    "TurnPlan",
    "_active_turn_var",
    "agent_hash",
    "current_turn",
]
