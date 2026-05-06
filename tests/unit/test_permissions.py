"""Tests for permission policy (Phase 6).

Covers:
- PermissionRule glob matching
- PermissionPolicy evaluation order
- readOnlyHint auto-allow
- destructiveHint auto-ask
- humanConfirmationRequired always ask
- Explicit deny/allow rules take precedence
- Default permissive behavior
"""

from __future__ import annotations

import pytest
from kaos_core.types.annotations import ToolAnnotations

from kaos_agents.permissions import PermissionPolicy
from kaos_agents.types import PermissionDecision, PermissionRule


def _make_annotations(
    *,
    read_only: bool = False,
    destructive: bool = False,
    human_confirm: bool = False,
) -> ToolAnnotations:
    """Create a ToolAnnotations instance with the specified hints."""
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        humanConfirmationRequired=human_confirm,
    )


@pytest.mark.unit
class TestPermissionRule:
    def test_exact_match(self) -> None:
        rule = PermissionRule(pattern="kaos-web-delete", action=PermissionDecision.DENY)
        assert rule.matches("kaos-web-delete")
        assert not rule.matches("kaos-web-get")

    def test_glob_wildcard(self) -> None:
        rule = PermissionRule(pattern="kaos-web-*", action=PermissionDecision.ALLOW)
        assert rule.matches("kaos-web-delete")
        assert rule.matches("kaos-web-get-markdown")
        assert not rule.matches("kaos-source-fr-search")

    def test_frozen(self) -> None:
        rule = PermissionRule(pattern="*", action=PermissionDecision.ALLOW)
        with pytest.raises(AttributeError):
            setattr(rule, "pattern", "other")  # noqa: B010


@pytest.mark.unit
class TestPermissionPolicy:
    def test_default_allows(self) -> None:
        """Empty policy with no annotations → ALLOW."""
        policy = PermissionPolicy()
        assert policy.evaluate("any-tool") == PermissionDecision.ALLOW

    def test_read_only_auto_allow(self) -> None:
        """readOnlyHint=True → ALLOW (even without rules)."""
        policy = PermissionPolicy()
        ann = _make_annotations(read_only=True)
        assert policy.evaluate("any-tool", ann) == PermissionDecision.ALLOW

    def test_destructive_auto_ask(self) -> None:
        """destructiveHint=True → ASK (when no rules match)."""
        policy = PermissionPolicy()
        ann = _make_annotations(destructive=True)
        assert policy.evaluate("any-tool", ann) == PermissionDecision.ASK

    def test_human_confirmation_always_ask(self) -> None:
        """humanConfirmationRequired=True → ASK (overrides everything)."""
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="*", action=PermissionDecision.ALLOW),)
        )
        ann = _make_annotations(human_confirm=True)
        assert policy.evaluate("any-tool", ann) == PermissionDecision.ASK

    def test_explicit_deny(self) -> None:
        """Explicit deny rule → DENY."""
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-web-delete-*", action=PermissionDecision.DENY),)
        )
        assert policy.evaluate("kaos-web-delete-page") == PermissionDecision.DENY

    def test_explicit_allow_overrides_destructive(self) -> None:
        """Explicit allow rule takes precedence over destructiveHint."""
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="kaos-web-*", action=PermissionDecision.ALLOW),)
        )
        ann = _make_annotations(destructive=True)
        assert policy.evaluate("kaos-web-delete", ann) == PermissionDecision.ALLOW

    def test_first_matching_rule_wins(self) -> None:
        """Rules are evaluated in order; first match wins."""
        policy = PermissionPolicy(
            rules=(
                PermissionRule(pattern="kaos-web-delete-*", action=PermissionDecision.DENY),
                PermissionRule(pattern="kaos-web-*", action=PermissionDecision.ALLOW),
            )
        )
        assert policy.evaluate("kaos-web-delete-page") == PermissionDecision.DENY
        assert policy.evaluate("kaos-web-get-markdown") == PermissionDecision.ALLOW

    def test_no_annotations_no_rules(self) -> None:
        """No annotations, no rules → ALLOW."""
        policy = PermissionPolicy()
        assert policy.evaluate("unknown-tool") == PermissionDecision.ALLOW
