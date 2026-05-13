"""kaos_agents.loop — Phase 2.B AgentLoop skeleton (Ten Questions §6).

Public surface:

* :class:`AgentLoop` — the canonical agent turn loop, a kaos-llm-core
  :class:`~kaos_llm_core.programs.base.Program` subclass that owns the
  outer 8-step turn loop. Provides:

  - :meth:`AgentLoop.prepare_turn` — composition surface (intent
    extraction + memory pass-through + emitter construction).
  - :meth:`AgentLoop.forward` — blocking turn execution that returns a
    :class:`~kaos_agents.core.invocation.TurnInvocation`.
  - :meth:`AgentLoop.stream` — Task-backed Queue wrapper that yields
    :class:`~kaos_agents.base.event.KaosEvent` instances live as
    ``forward()`` runs.
  - :meth:`AgentLoop.invoke` — Program-conformant blocking invocation
    that delegates to ``forward()``.

Phase 2.B is purely additive. Phase 6 cuts the existing BaseAgent /
Runner machinery over to AgentLoop.
"""

from kaos_agents.loop.agent_loop import AgentLoop

__all__ = ["AgentLoop"]
