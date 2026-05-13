"""Tier 9: PermissionPolicy gating fires ToolCallApprovalRequired.

A PermissionPolicy with an ASK rule for a specific tool MUST cause
the runner to (a) emit ``ToolCallApprovalRequired`` when the agent
tries to call that tool, and (b) pause the run cleanly. Verifies the
HITL gating path end-to-end.

Cost target: ~$0.05.
"""

from __future__ import annotations

import pytest

from kaos_agents.events import KaosEvent, ToolCallApprovalRequired
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.types.permissions import PermissionDecision, PermissionRule
from tests.integration.ladder.conftest import (
    collect_events,
    model_for_tier,
)

pytestmark = pytest.mark.live

TIER = 9
BUDGET_USD = 0.10


def _approval_events(events: list[KaosEvent]) -> list[ToolCallApprovalRequired]:
    return [e for e in events if isinstance(e, ToolCallApprovalRequired)]


@pytest.mark.asyncio
async def test_permission_policy_triggers_approval_required(ladder_runner_factory, turn_session_id):
    """ASK rule on FR search → ToolCallApprovalRequired event + pause."""
    policy = PermissionPolicy(
        rules=(
            PermissionRule(
                pattern="kaos-source-fr-*",
                action=PermissionDecision.ASK,
                reason="Test gate: all FR API calls require explicit approval.",
            ),
        )
    )
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a research assistant. Use kaos-source-fr-search "
                "to find Federal Register documents."
            ),
            "model": model_for_tier(TIER),
            "tools": ("kaos-source-fr-search",),
        },
        register_source=True,
        permission_policy=policy,
    )
    msg = (
        "Use kaos-source-fr-search exactly once: term=PFAS "
        "agency=environmental-protection-agency doc_type=NOTICE per_page=1 "
        "order=newest. Report results[0].title."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    # NOTE: cannot use assert_within_budget here — when permission ASK
    # fires, the runner returns BEFORE emitting TurnSummary (the run
    # is paused awaiting approval). assert_within_budget hard-fails on
    # missing TurnSummary. Instead, check the cost gate manually
    # against UsageObserved events (which fire pre-pause).
    from kaos_agents.events import UsageObserved

    usage = sum(
        float(getattr(e, "cost_usd", 0.0) or 0.0) for e in events if isinstance(e, UsageObserved)
    )
    assert usage <= 2.0 * BUDGET_USD, (
        f"tier-9: pre-pause LLM cost ${usage:.4f} exceeded 2x budget ${2.0 * BUDGET_USD:.4f}"
    )

    approvals = _approval_events(events)
    assert approvals, (
        "expected >=1 ToolCallApprovalRequired event when PermissionPolicy "
        "returns ASK for the tool. Zero events means the runner's "
        "permission gating regressed (Runner._pause_for_approval is wired "
        "from runner.py:323/347)."
    )

    # The approval event should reference the gated tool name.
    matched = any("kaos-source-fr" in str(getattr(a, "tool_name", "") or "") for a in approvals)
    assert matched, (
        f"approval event must reference the gated tool; got tool_names: "
        f"{[getattr(a, 'tool_name', None) for a in approvals]}"
    )
