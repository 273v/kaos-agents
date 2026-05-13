"""ActionPlan, ActionResult, ActionRefusal — typed value types.

Phase 1.C of the kaos-agents rewrite. The Action subsystem answers
paper Q4 ("how does the agent make things happen?"); these are the
frozen value types it produces and consumes.

- :class:`ActionPlan` — a *proposed* action. Distinct from a tool call
  as-issued because it adds the reversibility / approval / reversal
  fields the actor uses to gate dispatch.
- :class:`ActionResult` — returned after a successful tool dispatch.
- :class:`ActionRefusal` — returned when the actor refuses a proposed
  action (rate-limited, circuit-open, permission-denied, awaiting
  approval, reversibility-denied).

All three are pydantic ``BaseModel`` (extra=forbid, frozen) so they
can be passed through the wire surface and round-tripped through JSON.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kaos_agents.action.reversibility import Reversibility


class ActionPlan(BaseModel):
    """A proposed action — produced by the Actor's planning step.

    Distinct from a tool call as-issued: ``ActionPlan`` adds
    ``reversibility`` + ``approval_required`` + ``reversal_strategy``
    fields the actor uses for gating before dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    reversibility: Reversibility = Reversibility.IRREVERSIBLE
    approval_required: bool = True
    reversal_strategy: str | None = None
    """Human/machine-readable reversal recipe (e.g. ``"DELETE FROM ... WHERE ..."``)."""
    rationale: str = ""


class ActionResult(BaseModel):
    """Returned after a successful tool dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    success: bool = True
    output: Any = None
    side_effects: tuple[str, ...] = ()
    """Human-readable list of side effects observed during dispatch."""
    reversibility_actually: Reversibility = Reversibility.IRREVERSIBLE
    """The reversibility tier observed *after* the action ran. May
    differ from the planned tier if the tool revealed a stronger
    side-effect than declared."""
    duration_ms: float | None = None
    cost_usd: float | None = None


class ActionRefusal(BaseModel):
    """Returned when the actor refuses a proposed action.

    Common ``kind`` values:

    - ``"rate_limit"`` — RateLimiter hook denied the call.
    - ``"circuit_open"`` — CircuitBreaker hook is OPEN.
    - ``"permission_denied"`` — :class:`PermissionPolicy` returned DENY.
    - ``"approval_pending"`` — caller should re-issue after
      ``Runner.resume`` once the human approves.
    - ``"reversibility_denied"`` — e.g. IRREVERSIBLE without dual-key.
    - ``"not_configured"`` — no dispatch callback wired (Phase 1.C).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    reason: str
    kind: str
    recommended_alternatives: tuple[str, ...] = ()
    retry_after_seconds: float | None = None


__all__ = ["ActionPlan", "ActionRefusal", "ActionResult"]
