"""Actor — the mutation-surface :class:`Program`.

Composes:

- kaos-llm-core :class:`ReAct` (the reasoning + tool-call loop).
- Reversibility-aware permission gating (consults
  :class:`PermissionPolicy`).
- :class:`ApprovalWorkflow` for high-tier actions.
- :class:`RateLimiter` and :class:`CircuitBreaker` (as KaosHooks;
  injected at the Runner layer in Phase 2 — Phase 1.C just records
  them as known hooks on the Actor instance).

Phase 1.C ships the constructor and the gating logic. Full ReAct
integration into the AgentLoop is Phase 3 (when planners are
introduced) — for Phase 1.C, the actor exposes::

    actor.classify_plan(action_plan) -> "auto_allow" | "approval_required" | "deny"

and the wrapping :meth:`forward` is a thin pass-through that returns
:class:`ActionResult` / :class:`ActionRefusal` based on
``classify_plan``'s verdict and a stubbed dispatch callback (so unit
tests can verify the gating without standing up a real ReAct).

Real ReAct dispatch (the actual LLM-driven inner loop) is wired in
Phase 3 by replacing the stub callback.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from kaos_llm_core.programs.base import Program

from kaos_agents.action.approval import ApprovalWorkflow
from kaos_agents.action.circuit import CircuitBreaker
from kaos_agents.action.rate_limit import RateLimiter
from kaos_agents.action.reversibility import Reversibility
from kaos_agents.action.types import ActionPlan, ActionRefusal, ActionResult

ClassifyVerdict = Literal["auto_allow", "approval_required", "deny"]

DispatchCallback = Callable[
    [ActionPlan],
    "ActionResult | ActionRefusal | Awaitable[ActionResult | ActionRefusal]",
]


class Actor(Program):
    """Mutation-surface Program — gates and dispatches :class:`ActionPlan`.

    Phase 1.C is the gating + structure stage. Real ReAct dispatch is
    Phase 3; the ``dispatch`` callback is the seam where that wiring
    will land.
    """

    def __init__(
        self,
        *,
        permission_policy: Any | None = None,
        approval: ApprovalWorkflow | None = None,
        dispatch: DispatchCallback | None = None,
        reversibility_default: Reversibility = Reversibility.IRREVERSIBLE,
        rate_limiter: RateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        super().__init__()
        # Private attributes (leading underscore) so Program.__setattr__
        # doesn't try to register them as Call/Program children. The
        # rate_limiter and circuit_breaker are KaosHook subclasses, not
        # Program children, but they would still be filtered out by the
        # isinstance check — we keep them private to be explicit.
        self._permissions = permission_policy
        self._approval = approval or ApprovalWorkflow()
        self._dispatch = dispatch
        self._default_reversibility = reversibility_default
        self._rate_limiter = rate_limiter
        self._circuit_breaker = circuit_breaker

    # ------------------------------------------------------------------ #
    # Read-only accessors for tests / Runner wiring.                      #
    # ------------------------------------------------------------------ #

    @property
    def approval(self) -> ApprovalWorkflow:
        return self._approval

    @property
    def rate_limiter(self) -> RateLimiter | None:
        return self._rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        return self._circuit_breaker

    @property
    def permission_policy(self) -> Any | None:
        return self._permissions

    # ------------------------------------------------------------------ #
    # Gating.                                                             #
    # ------------------------------------------------------------------ #

    def classify_plan(self, plan: ActionPlan) -> ClassifyVerdict:
        """Return one of ``"auto_allow" | "approval_required" | "deny"``.

        Consults reversibility and :class:`PermissionPolicy`. Phase 1.C
        does not actually pause — that's a Runner-layer concern.
        ``classify_plan`` returns a verdict that the Runner uses to
        either dispatch or emit
        :class:`ToolCallApprovalRequired`.

        Resolution order:

        1. PermissionPolicy explicit DENY → ``"deny"``.
        2. PermissionPolicy explicit ALLOW + approval_required from the
           workflow → if approval is needed, still ``"approval_required"``
           (ALLOW does not bypass dual-key for IRREVERSIBLE actions); if
           not needed, ``"auto_allow"``.
           For non-IRREVERSIBLE tiers, an explicit ALLOW with
           ``ActionPlan.approval_required=False`` short-circuits to
           ``"auto_allow"``.
        3. PermissionPolicy ASK → ``"approval_required"``.
        4. No policy / no decision → fall back to the
           :class:`ApprovalWorkflow.required_for` rule.
        """
        # 1-3. PermissionPolicy may short-circuit.
        if self._permissions is not None:
            decision = self._evaluate_policy(plan)
            if decision is not None:
                # Local import to avoid circular issues.
                from kaos_agents.types.permissions import PermissionDecision

                if decision == PermissionDecision.DENY:
                    return "deny"
                if decision == PermissionDecision.ASK:
                    return "approval_required"
                # ALLOW: still consult ApprovalWorkflow for IRREVERSIBLE
                # so explicit ALLOW does not bypass dual-key.
                if (
                    decision == PermissionDecision.ALLOW
                    and plan.reversibility == Reversibility.IRREVERSIBLE
                ):
                    return "approval_required"
                if decision == PermissionDecision.ALLOW:
                    return "auto_allow"

        # 4. Fall through to reversibility-driven approval.
        return "approval_required" if self._approval.required_for(plan) else "auto_allow"

    def _evaluate_policy(self, plan: ActionPlan) -> Any:
        """Call ``permission_policy.evaluate(tool_name, annotations=None)``.

        Returns whatever the policy returns (typically a
        :class:`PermissionDecision`), or ``None`` if the policy doesn't
        expose an ``evaluate`` callable.
        """
        evaluate = getattr(self._permissions, "evaluate", None)
        if evaluate is None:
            return None
        try:
            return evaluate(plan.tool_name, None)
        except TypeError:
            # Policy stub that takes only the tool name.
            return evaluate(plan.tool_name)

    # ------------------------------------------------------------------ #
    # Dispatch (Phase 1.C: thin pass-through).                            #
    # ------------------------------------------------------------------ #

    async def forward(self, **kwargs: Any) -> ActionResult | ActionRefusal:
        """Gate ``plan`` and either dispatch or refuse.

        Signature matches :class:`Program.forward` (``**kwargs``) so
        Liskov holds; the only meaningful kwarg is ``plan``.

        Phase 1.C uses a stub-friendly contract: ``self._dispatch`` is
        a callable that takes the plan and returns an
        :class:`ActionResult` (or :class:`ActionRefusal`). It may be
        sync or async. If absent, ``forward`` returns a refusal of
        kind ``"not_configured"``.
        """
        plan = kwargs.get("plan")
        if not isinstance(plan, ActionPlan):
            raise TypeError(
                "Actor.forward requires a `plan: ActionPlan` keyword argument. "
                f"Got plan={plan!r}. "
                "Construct an ActionPlan and pass it as `await actor(plan=...)` "
                "or `await actor.forward(plan=...)`. "
                "Real ReAct dispatch (Phase 3) will replace this with a higher-level "
                "user-message → ActionPlan planner."
            )
        verdict = self.classify_plan(plan)
        if verdict == "deny":
            return ActionRefusal(
                tool_name=plan.tool_name,
                reason="permission denied",
                kind="permission_denied",
            )
        if verdict == "approval_required":
            return ActionRefusal(
                tool_name=plan.tool_name,
                reason=f"approval pending (reversibility={plan.reversibility.value})",
                kind="approval_pending",
            )
        # auto_allow.
        if self._dispatch is None:
            return ActionRefusal(
                tool_name=plan.tool_name,
                reason="no dispatch callback configured",
                kind="not_configured",
            )
        return await _maybe_await(self._dispatch(plan))


async def _maybe_await(x: Any) -> Any:
    """If ``x`` is awaitable, await it; otherwise return it as-is."""
    if inspect.isawaitable(x):
        return await x
    return x


__all__ = ["Actor", "ClassifyVerdict"]
