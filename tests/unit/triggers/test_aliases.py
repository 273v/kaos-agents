"""Unit tests for the eight source-tag thin re-export modules.

Each ``triggers/<source>.py`` module exposes a stable name aliased to
the corresponding :class:`Trigger` factory ``classmethod``. Phase 4+
may swap the alias for a real :class:`TriggerSource` subclass without
changing the import path — these tests pin the Phase 2.A contract.
"""

from __future__ import annotations

from kaos_agents.triggers import (
    CLIPromptTrigger,
    DelegationTrigger,
    EscalationTrigger,
    FileSystemTrigger,
    HTTPMessageTrigger,
    MCPToolTrigger,
    ScheduledTrigger,
    WebhookTrigger,
)
from kaos_agents.triggers.base import Trigger, TriggerKind


def _same_classmethod(a: object, b: object) -> bool:
    """Bound classmethods compare by underlying ``__func__``.

    Each attribute access on a class produces a fresh ``MethodType``
    descriptor, so ``Trigger.mcp is Trigger.mcp`` is ``False``. The
    underlying ``__func__`` is the stable identity.
    """
    return getattr(a, "__func__", a) is getattr(b, "__func__", b)


class TestAliasIdentity:
    """Each alias is exactly the corresponding ``Trigger.<source>`` factory.

    Bound-classmethod identity is checked via ``__func__`` because each
    attribute access returns a fresh descriptor object.
    """

    def test_mcp(self) -> None:
        assert _same_classmethod(MCPToolTrigger, Trigger.mcp)

    def test_http(self) -> None:
        assert _same_classmethod(HTTPMessageTrigger, Trigger.http)

    def test_cli(self) -> None:
        assert _same_classmethod(CLIPromptTrigger, Trigger.cli)

    def test_scheduled(self) -> None:
        assert _same_classmethod(ScheduledTrigger, Trigger.scheduled)

    def test_escalation(self) -> None:
        assert _same_classmethod(EscalationTrigger, Trigger.escalation)

    def test_webhook(self) -> None:
        assert _same_classmethod(WebhookTrigger, Trigger.webhook)

    def test_filesystem(self) -> None:
        assert _same_classmethod(FileSystemTrigger, Trigger.filesystem)

    def test_delegation(self) -> None:
        assert _same_classmethod(DelegationTrigger, Trigger.delegation)


class TestAliasProducesCorrectKind:
    """Calling the alias produces a :class:`Trigger` of the right ``kind``."""

    def test_mcp(self) -> None:
        t = MCPToolTrigger("hi", session_id="s")
        assert t.kind is TriggerKind.MCP

    def test_http(self) -> None:
        t = HTTPMessageTrigger("hi", request_id="r")
        assert t.kind is TriggerKind.HTTP

    def test_cli(self) -> None:
        t = CLIPromptTrigger("hi", session_id="c")
        assert t.kind is TriggerKind.CLI

    def test_scheduled(self) -> None:
        t = ScheduledTrigger(job_name="j")
        assert t.kind is TriggerKind.SCHEDULED

    def test_escalation(self) -> None:
        t = EscalationTrigger(reason="needs help")
        assert t.kind is TriggerKind.ESCALATION

    def test_webhook(self) -> None:
        t = WebhookTrigger(event_type="evt")
        assert t.kind is TriggerKind.WEBHOOK

    def test_filesystem(self) -> None:
        t = FileSystemTrigger(path="/p")
        assert t.kind is TriggerKind.FILESYSTEM

    def test_delegation(self) -> None:
        t = DelegationTrigger(goal="g", parent_turn_id="p")
        assert t.kind is TriggerKind.DELEGATION


class TestSubmodulePathStability:
    """Phase 2.A → 4+ stability: importing from the per-source submodule
    must keep working even after the alias is swapped for a real class."""

    def test_mcp_submodule_path(self) -> None:
        from kaos_agents.triggers.mcp import MCPToolTrigger as M

        assert _same_classmethod(M, MCPToolTrigger)

    def test_http_submodule_path(self) -> None:
        from kaos_agents.triggers.http import HTTPMessageTrigger as M

        assert _same_classmethod(M, HTTPMessageTrigger)

    def test_cli_submodule_path(self) -> None:
        from kaos_agents.triggers.cli import CLIPromptTrigger as M

        assert _same_classmethod(M, CLIPromptTrigger)

    def test_schedule_submodule_path(self) -> None:
        from kaos_agents.triggers.schedule import ScheduledTrigger as M

        assert _same_classmethod(M, ScheduledTrigger)

    def test_escalation_submodule_path(self) -> None:
        from kaos_agents.triggers.escalation import EscalationTrigger as M

        assert _same_classmethod(M, EscalationTrigger)

    def test_webhook_submodule_path(self) -> None:
        from kaos_agents.triggers.webhook import WebhookTrigger as M

        assert _same_classmethod(M, WebhookTrigger)

    def test_fs_submodule_path(self) -> None:
        from kaos_agents.triggers.fs import FileSystemTrigger as M

        assert _same_classmethod(M, FileSystemTrigger)

    def test_delegation_submodule_path(self) -> None:
        from kaos_agents.triggers.delegation import DelegationTrigger as M

        assert _same_classmethod(M, DelegationTrigger)
