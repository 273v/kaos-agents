"""Permission value types — rules and decisions.

The matching engine (:class:`kaos_agents.permissions.PermissionPolicy`)
lives separately because it has behavior; this module holds only the
frozen value types it composes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from fnmatch import fnmatch


@unique
class PermissionDecision(StrEnum):
    """Result of a permission policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """A single permission rule matching tool names by glob pattern.

    Rules are evaluated in order. The first matching rule wins.

    Args:
        pattern: Glob pattern for tool names (e.g., "kaos-web-delete-*").
        action: What to do when the pattern matches.
        reason: Human-readable explanation (shown in approval UI).
    """

    pattern: str
    action: PermissionDecision
    reason: str = ""

    def matches(self, tool_name: str) -> bool:
        """Check if this rule matches the given tool name."""
        return fnmatch(tool_name, self.pattern)
