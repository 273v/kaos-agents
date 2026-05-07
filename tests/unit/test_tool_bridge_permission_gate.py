"""Regression tests for WS-0.1 — permission policy is a pre-execution gate.

The bug: the Runner's permission check fired only on ``ToolCallStart``
events replayed from the ReAct trajectory, which meant destructive tools
had already executed by the time ``ASK`` / ``DENY`` was evaluated.

The fix: ``kaos_tool_to_llm_tool`` accepts a ``permission_policy`` kwarg
that is consulted **before** the wrapped ``KaosTool.execute()`` runs.
The test set proves:

1. DENY → underlying tool's ``execute()`` is never called.
2. ASK → underlying tool's ``execute()`` is never called.
3. ALLOW → underlying tool is called normally.
4. No policy (``None``) → legacy permissive behavior.
5. ``destructiveHint`` annotation triggers ASK even without an explicit rule.
6. ``readOnlyHint`` annotation does not block execution.

The ``_ObservableTool`` fixture records every ``execute()`` invocation
in a list so the "never called" assertion is concrete, not inferred.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult

from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.types import PermissionDecision, PermissionRule


class _ObservableTool(KaosTool):
    """KaosTool that records every execute() call — proves whether the
    gate actually blocked the underlying side effect."""

    def __init__(self, name: str, *, annotations: ToolAnnotations | None = None) -> None:
        self._name = name
        self._annotations = annotations or ToolAnnotations(readOnlyHint=True)
        self.invocations: list[dict[str, Any]] = []

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description="Observable test tool.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=self._annotations,
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        self.invocations.append(dict(inputs))
        return ToolResult.create_text(f"{self._name} ran with {inputs}")


@pytest.mark.unit
class TestPermissionGateDeny:
    async def test_deny_rule_blocks_execution(self) -> None:
        tool = _ObservableTool("dangerous-delete-everything")
        policy = PermissionPolicy(
            rules=(
                PermissionRule(
                    pattern="dangerous-*",
                    action=PermissionDecision.DENY,
                    reason="Explicit deny rule",
                ),
            )
        )
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        result = await wrapped.invoke({"foo": "bar"})

        assert tool.invocations == [], (
            "DENY must prevent the underlying tool from running, but "
            f"got {len(tool.invocations)} invocation(s): {tool.invocations}"
        )
        parsed = json.loads(result)
        assert parsed["error"] is True
        assert parsed["permission_decision"] == "deny"
        assert "denied by permission policy" in parsed["message"]


@pytest.mark.unit
class TestPermissionGateAsk:
    async def test_ask_rule_blocks_execution(self) -> None:
        tool = _ObservableTool("kaos-test-send-email")
        policy = PermissionPolicy(
            rules=(
                PermissionRule(
                    pattern="kaos-test-send-email",
                    action=PermissionDecision.ASK,
                    reason="Email requires human approval",
                ),
            )
        )
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        result = await wrapped.invoke({"to": "user@example.com"})

        assert tool.invocations == [], (
            "ASK must prevent the underlying tool from running until "
            "approval resolves, but got "
            f"{len(tool.invocations)} invocation(s): {tool.invocations}"
        )
        parsed = json.loads(result)
        assert parsed["permission_decision"] == "ask"
        assert "approval" in parsed["message"].lower()


@pytest.mark.unit
class TestPermissionGateAllow:
    async def test_allow_rule_permits_execution(self) -> None:
        tool = _ObservableTool("read-only-search")
        policy = PermissionPolicy(
            rules=(
                PermissionRule(
                    pattern="read-only-*",
                    action=PermissionDecision.ALLOW,
                ),
            )
        )
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        result = await wrapped.invoke({"query": "hello"})

        assert len(tool.invocations) == 1
        assert tool.invocations[0] == {"query": "hello"}
        assert "ran with" in result

    async def test_no_policy_permits_execution(self) -> None:
        """Legacy path — when no policy is supplied, behavior is permissive."""
        tool = _ObservableTool("kaos-test-unrestricted-tool")
        wrapped = kaos_tool_to_llm_tool(tool)  # no permission_policy

        await wrapped.invoke({"q": "x"})

        assert len(tool.invocations) == 1


@pytest.mark.unit
class TestPermissionGateAnnotations:
    async def test_destructive_hint_triggers_ask(self) -> None:
        """A tool marked destructiveHint=True triggers ASK even without
        an explicit rule — the PermissionPolicy default behavior."""
        tool = _ObservableTool(
            "db-drop-table",
            annotations=ToolAnnotations(destructiveHint=True),
        )
        policy = PermissionPolicy()  # no explicit rules
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        result = await wrapped.invoke({"table": "users"})

        assert tool.invocations == [], (
            "destructiveHint annotation must cause ASK and block execution"
        )
        parsed = json.loads(result)
        assert parsed["permission_decision"] == "ask"

    async def test_read_only_hint_allows(self) -> None:
        """A tool marked readOnlyHint=True is auto-allowed."""
        tool = _ObservableTool(
            "kaos-test-search-documents",
            annotations=ToolAnnotations(readOnlyHint=True),
        )
        policy = PermissionPolicy()
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        await wrapped.invoke({"query": "x"})

        assert len(tool.invocations) == 1

    async def test_human_confirmation_required_always_asks(self) -> None:
        """humanConfirmationRequired trumps any allow rule."""
        tool = _ObservableTool(
            "human-in-loop",
            annotations=ToolAnnotations(humanConfirmationRequired=True, readOnlyHint=True),
        )
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="*", action=PermissionDecision.ALLOW),)
        )
        wrapped = kaos_tool_to_llm_tool(tool, permission_policy=policy)

        result = await wrapped.invoke({})

        assert tool.invocations == [], (
            "humanConfirmationRequired must block execution even when an "
            "allow rule matches — approval is mandatory"
        )
        parsed = json.loads(result)
        assert parsed["permission_decision"] == "ask"
