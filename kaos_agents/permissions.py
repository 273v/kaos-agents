"""Permission policy for tool execution control.

The frozen value types (:class:`PermissionRule`, :class:`PermissionDecision`)
live in :mod:`kaos_agents.types.permissions`. This module owns the
matching engine — :class:`PermissionPolicy` — which composes them with
tool annotations to render a decision.

The evaluation order follows the KAOS permission hierarchy:
1. ``humanConfirmationRequired`` annotation → always ask
2. Explicit deny rules → deny
3. Explicit allow rules → allow
4. ``readOnlyHint=True`` annotation → auto-allow
5. ``destructiveHint=True`` annotation → ask
6. Default → allow

Usage::

    from kaos_agents.permissions import PermissionPolicy
    from kaos_agents.types import PermissionDecision, PermissionRule

    policy = PermissionPolicy(rules=(
        PermissionRule(pattern="kaos-web-delete-*", action=PermissionDecision.DENY,
                       reason="No deletions"),
        PermissionRule(pattern="kaos-source-*", action=PermissionDecision.ALLOW),
    ))

    decision = policy.evaluate("kaos-web-delete-page", annotations)
    # -> PermissionDecision.DENY
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_agents.types.permissions import PermissionDecision, PermissionRule

if TYPE_CHECKING:
    from kaos_core.types.annotations import ToolAnnotations


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
