"""Action subsystem — paper Q4 ("how does the agent make things happen?").

Phase 1.C of the kaos-agents ground-up rewrite. Purely additive — no
existing module wires through it yet.

Public surface:

- :class:`Reversibility` — four-tier action classification.
- :class:`ActionPlan` / :class:`ActionResult` / :class:`ActionRefusal`
  — frozen value types for the propose → dispatch → result/refuse
  pipeline.
- :class:`Actor` — the gating Program; consumes an :class:`ActionPlan`
  and returns a result or a refusal.
- :class:`ApprovalWorkflow` — predicate + event builder for human
  approval.
- :class:`RateLimiter` / :class:`CircuitBreaker` /
  :class:`CircuitState` — :class:`KaosHook` subclasses with sync
  ``allow`` predicates for inline use by the Actor.
"""

from __future__ import annotations

from kaos_agents.action.actor import Actor
from kaos_agents.action.approval import ApprovalWorkflow
from kaos_agents.action.circuit import CircuitBreaker, CircuitState
from kaos_agents.action.rate_limit import RateLimiter
from kaos_agents.action.reversibility import Reversibility, infer_reversibility
from kaos_agents.action.types import ActionPlan, ActionRefusal, ActionResult

__all__ = [
    "ActionPlan",
    "ActionRefusal",
    "ActionResult",
    "Actor",
    "ApprovalWorkflow",
    "CircuitBreaker",
    "CircuitState",
    "RateLimiter",
    "Reversibility",
    "infer_reversibility",
]
