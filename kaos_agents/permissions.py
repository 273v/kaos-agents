"""Permission policy for tool execution control.

Evaluates whether a tool call should be allowed, denied, or require
human approval before execution. Rules are matched against tool names
using glob patterns (``fnmatch``).

The evaluation order follows the KAOS permission hierarchy:
1. ``humanConfirmationRequired`` annotation → always ask
2. Explicit deny rules → deny
3. Explicit allow rules → allow
4. ``readOnlyHint=True`` annotation → auto-allow
5. ``destructiveHint=True`` annotation → ask
6. Default → allow

Usage::

    policy = PermissionPolicy(rules=(
        PermissionRule(pattern="kaos-web-delete-*", action="deny", reason="No deletions"),
        PermissionRule(pattern="kaos-source-*", action="allow"),
    ))

    decision = policy.evaluate("kaos-web-delete-page", annotations)
    # -> "deny"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from fnmatch import fnmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_core.types.annotations import ToolAnnotations


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


class PermissionPolicy:
    """Evaluates whether a tool call is permitted.

    Combines explicit rules with tool annotation hints to produce
    a ``PermissionDecision``. Rules take precedence over annotations.

    The evaluation order:
    1. ``humanConfirmationRequired`` annotation → ASK
    2. First matching deny rule → DENY
    3. First matching allow rule → ALLOW
    4. ``readOnlyHint=True`` → ALLOW (safe by default)
    5. ``destructiveHint=True`` → ASK
    6. No match → ALLOW (permissive default)
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: tuple[PermissionRule, ...] = ()) -> None:
        self._rules = rules

    @property
    def rules(self) -> tuple[PermissionRule, ...]:
        """The configured permission rules."""
        return self._rules

    def evaluate(
        self,
        tool_name: str,
        annotations: ToolAnnotations | None = None,
    ) -> PermissionDecision:
        """Evaluate whether a tool call is permitted.

        Args:
            tool_name: The tool being called.
            annotations: Tool annotations (readOnlyHint, destructiveHint, etc.).

        Returns:
            PermissionDecision: ALLOW, DENY, or ASK.
        """
        # 1. humanConfirmationRequired always triggers ASK
        if annotations and annotations.humanConfirmationRequired:
            return PermissionDecision.ASK

        # 2-3. Explicit rules (first match wins)
        for rule in self._rules:
            if rule.matches(tool_name):
                return rule.action

        # 4. readOnlyHint → auto-allow
        if annotations and annotations.readOnlyHint:
            return PermissionDecision.ALLOW

        # 5. destructiveHint → ask
        if annotations and annotations.destructiveHint:
            return PermissionDecision.ASK

        # 6. Default: permissive
        return PermissionDecision.ALLOW
