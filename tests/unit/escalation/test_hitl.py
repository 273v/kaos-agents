"""Unit tests for :class:`kaos_agents.escalation.hitl.HITLBridge`."""

from __future__ import annotations

import pytest

from kaos_agents.escalation.hitl import (
    EscalationContext,
    HITLBridge,
    HITLChannel,
    escalation_resource_uri,
)
from kaos_agents.escalation.kinds import EscalationKind


class TestEscalationResourceUri:
    def test_uri_form(self) -> None:
        assert escalation_resource_uri("abc") == "kaos-agents://escalation/abc"

    def test_uri_form_with_long_id(self) -> None:
        eid = "esc_" + "a" * 12
        assert escalation_resource_uri(eid) == f"kaos-agents://escalation/{eid}"


class TestHITLBridgeSurface:
    def test_default_channel_is_mcp_resource(self) -> None:
        bridge = HITLBridge()
        ctx = bridge.surface(
            kind=EscalationKind.OUTSIDE_COMPETENCE,
            reason="not in competence",
            resume_token="run-1",
        )
        assert isinstance(ctx, EscalationContext)
        assert ctx.channel is HITLChannel.MCP_RESOURCE
        assert ctx.kind is EscalationKind.OUTSIDE_COMPETENCE
        assert ctx.reason == "not in competence"
        assert ctx.resume_token == "run-1"
        assert ctx.escalation_id.startswith("esc_")
        assert ctx.uri == f"kaos-agents://escalation/{ctx.escalation_id}"

    def test_custom_default_channel(self) -> None:
        bridge = HITLBridge(default_channel=HITLChannel.HTTP)
        ctx = bridge.surface(
            kind=EscalationKind.APPROVAL_REQUIRED,
            reason="destructive",
            resume_token="run-2",
        )
        assert ctx.channel is HITLChannel.HTTP

    def test_cli_channel_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        bridge = HITLBridge()
        ctx = bridge.surface(
            kind=EscalationKind.CLARIFICATION_NEEDED,
            reason="need clarification on 'the contract'",
            resume_token="run-3",
            channel=HITLChannel.CLI,
            escalation_id="esc_known001",
        )
        captured = capsys.readouterr()
        assert captured.err  # something was written
        assert "esc_known001" in captured.err
        assert "clarification_needed" in captured.err
        assert ctx.uri in captured.err

    def test_http_and_mcp_resource_channels_do_not_write_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bridge = HITLBridge()
        bridge.surface(
            kind=EscalationKind.APPROVAL_REQUIRED,
            reason="x",
            resume_token="r",
            channel=HITLChannel.HTTP,
        )
        bridge.surface(
            kind=EscalationKind.APPROVAL_REQUIRED,
            reason="x",
            resume_token="r",
            channel=HITLChannel.MCP_RESOURCE,
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_escalation_id_is_unique_per_call(self) -> None:
        bridge = HITLBridge()
        seen: set[str] = set()
        for _ in range(10):
            ctx = bridge.surface(
                kind=EscalationKind.DOMAIN_SPECIFIC,
                reason="x",
                resume_token="r",
            )
            assert ctx.escalation_id not in seen
            seen.add(ctx.escalation_id)

    def test_explicit_escalation_id_is_preserved(self) -> None:
        bridge = HITLBridge()
        ctx = bridge.surface(
            kind=EscalationKind.LOOP_DETECTED,
            reason="loop",
            resume_token="r",
            escalation_id="esc_supplied",
        )
        assert ctx.escalation_id == "esc_supplied"
        assert ctx.uri == "kaos-agents://escalation/esc_supplied"

    def test_details_propagate(self) -> None:
        bridge = HITLBridge()
        ctx = bridge.surface(
            kind=EscalationKind.BUDGET_EXCEEDED,
            reason="cost cap",
            resume_token="r",
            details={"limit_usd": 1.0, "actual_usd": 1.5},
        )
        assert ctx.details == {"limit_usd": 1.0, "actual_usd": 1.5}

    def test_empty_details_default(self) -> None:
        bridge = HITLBridge()
        ctx = bridge.surface(
            kind=EscalationKind.DOMAIN_SPECIFIC,
            reason="x",
            resume_token="r",
        )
        assert ctx.details == {}


class TestHITLChannelEnum:
    def test_three_members(self) -> None:
        names = {member.value for member in HITLChannel}
        assert names == {"cli", "http", "mcp-resource"}
