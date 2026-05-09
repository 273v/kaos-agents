"""Actor.classify_plan + Actor.forward gating."""

from __future__ import annotations

from kaos_agents.action.actor import Actor
from kaos_agents.action.approval import ApprovalWorkflow
from kaos_agents.action.reversibility import Reversibility
from kaos_agents.action.types import ActionPlan, ActionRefusal, ActionResult
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.types.permissions import PermissionDecision, PermissionRule


class TestClassifyPlanReversibility:
    def test_reversible_auto_allow(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
            approval_required=True,
        )
        assert actor.classify_plan(plan) == "auto_allow"

    def test_recoverable_auto_allow(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-tabular-insert",
            reversibility=Reversibility.RECOVERABLE,
            approval_required=True,
        )
        assert actor.classify_plan(plan) == "auto_allow"

    def test_externally_visible_with_flag(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            approval_required=True,
        )
        assert actor.classify_plan(plan) == "approval_required"

    def test_externally_visible_no_flag(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            approval_required=False,
        )
        assert actor.classify_plan(plan) == "auto_allow"

    def test_irreversible_always_approval(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.IRREVERSIBLE,
            approval_required=False,  # ignored for IRREVERSIBLE
        )
        assert actor.classify_plan(plan) == "approval_required"


class TestClassifyPlanWithPolicy:
    def test_explicit_deny(self) -> None:
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-source-*", action=PermissionDecision.DENY),)
        )
        actor = Actor(permission_policy=policy)
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.REVERSIBLE,  # would otherwise auto-allow
        )
        assert actor.classify_plan(plan) == "deny"

    def test_explicit_ask_overrides_reversible(self) -> None:
        # ASK from the policy overrides REVERSIBLE auto-allow.
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-source-*", action=PermissionDecision.ASK),)
        )
        actor = Actor(permission_policy=policy)
        plan = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            reversibility=Reversibility.REVERSIBLE,
        )
        assert actor.classify_plan(plan) == "approval_required"

    def test_explicit_allow_does_not_bypass_irreversible(self) -> None:
        # ALLOW does not let IRREVERSIBLE skip dual-key approval.
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-source-*", action=PermissionDecision.ALLOW),)
        )
        actor = Actor(permission_policy=policy)
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.IRREVERSIBLE,
            approval_required=False,
        )
        assert actor.classify_plan(plan) == "approval_required"

    def test_explicit_allow_short_circuits_externally_visible(self) -> None:
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-source-*", action=PermissionDecision.ALLOW),)
        )
        actor = Actor(permission_policy=policy)
        plan = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            approval_required=False,  # explicit caller intent
        )
        assert actor.classify_plan(plan) == "auto_allow"


class TestForward:
    async def test_forward_deny(self) -> None:
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-source-*", action=PermissionDecision.DENY),)
        )
        actor = Actor(permission_policy=policy)
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.REVERSIBLE,
        )
        result = await actor.forward(plan=plan)
        assert isinstance(result, ActionRefusal)
        assert result.kind == "permission_denied"

    async def test_forward_approval_pending(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.IRREVERSIBLE,
        )
        result = await actor.forward(plan=plan)
        assert isinstance(result, ActionRefusal)
        assert result.kind == "approval_pending"
        assert "irreversible" in result.reason

    async def test_forward_auto_allow_no_dispatch(self) -> None:
        actor = Actor()
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
        )
        result = await actor.forward(plan=plan)
        assert isinstance(result, ActionRefusal)
        assert result.kind == "not_configured"

    async def test_forward_auto_allow_sync_dispatch(self) -> None:
        def stub(plan: ActionPlan) -> ActionResult:
            return ActionResult(
                tool_name=plan.tool_name,
                output={"ok": True},
                duration_ms=1.0,
                cost_usd=0.0,
            )

        actor = Actor(dispatch=stub)
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
        )
        result = await actor.forward(plan=plan)
        assert isinstance(result, ActionResult)
        assert result.output == {"ok": True}

    async def test_forward_auto_allow_async_dispatch(self) -> None:
        async def stub(plan: ActionPlan) -> ActionResult:
            return ActionResult(tool_name=plan.tool_name, output="async!")

        actor = Actor(dispatch=stub)
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
        )
        result = await actor.forward(plan=plan)
        assert isinstance(result, ActionResult)
        assert result.output == "async!"

    async def test_forward_invokable_via_program_call(self) -> None:
        # Phase 1.C — Actor is a Program; ensure __call__ unwraps to
        # forward() output without raising (the trace tree is empty
        # because we don't call any kaos-llm-core Calls).
        async def stub(plan: ActionPlan) -> ActionResult:
            return ActionResult(tool_name=plan.tool_name)

        actor = Actor(dispatch=stub)
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
        )
        out = await actor(plan=plan)
        assert isinstance(out, ActionResult)
        assert out.tool_name == "kaos-pdf-render"


class TestActorAccessors:
    def test_default_components(self) -> None:
        actor = Actor()
        assert isinstance(actor.approval, ApprovalWorkflow)
        assert actor.rate_limiter is None
        assert actor.circuit_breaker is None
        assert actor.permission_policy is None

    def test_custom_components(self) -> None:
        from kaos_agents.action.circuit import CircuitBreaker
        from kaos_agents.action.rate_limit import RateLimiter

        rl = RateLimiter()
        cb = CircuitBreaker()
        wf = ApprovalWorkflow()
        policy = PermissionPolicy()

        actor = Actor(
            permission_policy=policy,
            approval=wf,
            rate_limiter=rl,
            circuit_breaker=cb,
        )
        assert actor.approval is wf
        assert actor.rate_limiter is rl
        assert actor.circuit_breaker is cb
        assert actor.permission_policy is policy
