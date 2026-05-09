"""Trigger taxonomy — paper §Q1 / plan §4 Q1.

A :class:`Trigger` is the typed event that opens an agent turn. The
agent loop is trigger-source agnostic: MCP tool calls, HTTP requests,
CLI prompts, scheduled jobs, sub-agent escalations, webhooks,
filesystem changes, and parent → sub-agent delegations all produce
the same value type with a ``kind`` discriminator.

Phase 2.A surface:

- :class:`TriggerKind` — the 8-member ``StrEnum`` discriminator.
- :class:`Trigger` — the frozen pydantic event with eight source-tagged
  ``classmethod`` factories.
- :class:`TriggerSource` — the async-iterator ABC contract for
  pull-based sources (concrete impls land in Phase 4+).
- :data:`MCPToolTrigger` / :data:`HTTPMessageTrigger` /
  :data:`CLIPromptTrigger` / :data:`ScheduledTrigger` /
  :data:`EscalationTrigger` / :data:`WebhookTrigger` /
  :data:`FileSystemTrigger` / :data:`DelegationTrigger` — stable-name
  aliases for the eight ``Trigger.<source>`` factories.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger, TriggerKind, TriggerSource
from kaos_agents.triggers.cli import CLIPromptTrigger
from kaos_agents.triggers.delegation import DelegationTrigger
from kaos_agents.triggers.escalation import EscalationTrigger
from kaos_agents.triggers.fs import FileSystemTrigger
from kaos_agents.triggers.http import HTTPMessageTrigger
from kaos_agents.triggers.mcp import MCPToolTrigger
from kaos_agents.triggers.schedule import ScheduledTrigger
from kaos_agents.triggers.webhook import WebhookTrigger

__all__ = [
    "CLIPromptTrigger",
    "DelegationTrigger",
    "EscalationTrigger",
    "FileSystemTrigger",
    "HTTPMessageTrigger",
    "MCPToolTrigger",
    "ScheduledTrigger",
    "Trigger",
    "TriggerKind",
    "TriggerSource",
    "WebhookTrigger",
]
