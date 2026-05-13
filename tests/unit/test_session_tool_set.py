"""Tests for SessionToolSet + filter_tools (Track 4 chunk T4-5).

Confirms:

- ``SessionToolSet`` is frozen, slotted, defaults to fully-unrestricted
- ``is_unrestricted`` flips when any of the three sets is non-empty
- ``unrestricted()`` returns an identity-stable singleton
- ``filter_tools`` honours all four resolution rules:
  1. denied_tools always wins
  2. allowed_tools restricts to the explicit list
  3. allowed_groups restricts via group membership
  4. empty config = pass-through
- Tools without a registered group fail allowed_groups but pass
  allowed_tools (no silent group-coupling)
- Public surface: top-level + types/context re-exports
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaos_agents.context import filter_tools
from kaos_agents.registry import ToolGroupRegistry
from kaos_agents.types import SessionToolSet, ToolGroup

# ---------------------------------------------------------------------------
# Stub tool — minimum interface filter_tools needs
# ---------------------------------------------------------------------------


@dataclass
class _StubMetadata:
    name: str


@dataclass
class _StubTool:
    """Minimum surface: ``.metadata.name`` like a KaosTool."""

    _name: str

    @property
    def metadata(self) -> _StubMetadata:
        return _StubMetadata(name=self._name)


def _t(name: str) -> _StubTool:
    return _StubTool(_name=name)


# ---------------------------------------------------------------------------
# SessionToolSet value type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionToolSetValueType:
    def test_default_unrestricted(self) -> None:
        ts = SessionToolSet()
        assert ts.allowed_tools == frozenset()
        assert ts.allowed_groups == frozenset()
        assert ts.denied_tools == frozenset()
        assert ts.is_unrestricted

    def test_any_constraint_flips_unrestricted(self) -> None:
        assert SessionToolSet(allowed_tools=frozenset({"a"})).is_unrestricted is False
        assert SessionToolSet(allowed_groups=frozenset({"g"})).is_unrestricted is False
        assert SessionToolSet(denied_tools=frozenset({"a"})).is_unrestricted is False

    def test_unrestricted_singleton(self) -> None:
        a = SessionToolSet.unrestricted()
        b = SessionToolSet.unrestricted()
        assert a is b
        assert a.is_unrestricted

    def test_frozen_no_mutation(self) -> None:
        ts = SessionToolSet(allowed_tools=frozenset({"a"}))
        with pytest.raises(Exception):  # noqa: B017
            object.__setattr__(ts, "allowed_tools", frozenset({"b"}))
            ts.__dict__["allowed_tools"] = frozenset({"b"})

    def test_equality_by_value(self) -> None:
        a = SessionToolSet(allowed_tools=frozenset({"a", "b"}))
        b = SessionToolSet(allowed_tools=frozenset({"b", "a"}))  # set order
        assert a == b


# ---------------------------------------------------------------------------
# filter_tools resolution rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterToolsRules:
    def test_unrestricted_returns_input(self) -> None:
        tools = [_t("a"), _t("b"), _t("c")]
        result = filter_tools(tools, SessionToolSet())
        # Returns a list (defensively copied — list never aliases)
        assert [_StubTool, _StubTool, _StubTool] == [type(x) for x in result]
        assert [x._name for x in result] == ["a", "b", "c"]

    def test_denied_tools_blocks_even_with_allow(self) -> None:
        ts = SessionToolSet(
            allowed_tools=frozenset({"a", "b"}),
            denied_tools=frozenset({"a"}),
        )
        result = filter_tools([_t("a"), _t("b"), _t("c")], ts)
        # 'a' is denied (wins), 'b' is allowed, 'c' not in allow-list
        assert [x._name for x in result] == ["b"]

    def test_allowed_tools_restricts(self) -> None:
        ts = SessionToolSet(allowed_tools=frozenset({"a", "c"}))
        result = filter_tools([_t("a"), _t("b"), _t("c")], ts)
        assert [x._name for x in result] == ["a", "c"]

    def test_allowed_groups_restricts_via_registry(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="research", description="d", tool_names=("a", "b")))
        reg.register(ToolGroup(name="billing", description="d", tool_names=("c",)))

        ts = SessionToolSet(allowed_groups=frozenset({"research"}))
        result = filter_tools([_t("a"), _t("b"), _t("c")], ts, group_registry=reg)
        # Only research-group tools pass
        assert [x._name for x in result] == ["a", "b"]

    def test_tool_in_no_group_fails_group_filter(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="research", description="d", tool_names=("a",)))

        ts = SessionToolSet(allowed_groups=frozenset({"research"}))
        # 'b' belongs to no group — blocked when only group filter is set
        result = filter_tools([_t("a"), _t("b")], ts, group_registry=reg)
        assert [x._name for x in result] == ["a"]

    def test_explicit_allow_overrides_group_membership(self) -> None:
        """Tool in allowed_tools passes even if it has no group."""
        reg = ToolGroupRegistry()  # empty registry — no groups
        ts = SessionToolSet(allowed_tools=frozenset({"a"}))
        result = filter_tools([_t("a"), _t("b")], ts, group_registry=reg)
        # 'a' explicitly allowed (passes), 'b' not in allow-list (blocked)
        assert [x._name for x in result] == ["a"]

    def test_unnamed_tool_dropped(self) -> None:
        """Tools with no extractable name are dropped (fail-closed)."""
        # Plain object — no metadata, no name
        plain = object()
        ts = SessionToolSet(allowed_tools=frozenset({"a"}))
        result = filter_tools([plain, _t("a")], ts)
        assert len(result) == 1
        assert result[0]._name == "a"  # type: ignore[union-attr]

    def test_input_not_mutated(self) -> None:
        tools = [_t("a"), _t("b")]
        original_ids = [id(x) for x in tools]
        ts = SessionToolSet(allowed_tools=frozenset({"a"}))
        _ = filter_tools(tools, ts)
        # Input list unchanged
        assert [id(x) for x in tools] == original_ids
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Combined rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedFilterRules:
    def test_allow_tools_plus_allow_groups(self) -> None:
        """When both allow lists are set, a tool passes if it's in EITHER."""
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="research", description="d", tool_names=("b", "c")))

        ts = SessionToolSet(
            allowed_tools=frozenset({"a"}),
            allowed_groups=frozenset({"research"}),
        )
        result = filter_tools([_t("a"), _t("b"), _t("c"), _t("d")], ts, group_registry=reg)
        # 'a' via allowed_tools; 'b' and 'c' via research group; 'd' nowhere
        names = sorted(x._name for x in result)
        assert names == ["a", "b", "c"]

    def test_deny_overrides_combined_allow(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="research", description="d", tool_names=("b",)))

        ts = SessionToolSet(
            allowed_tools=frozenset({"a"}),
            allowed_groups=frozenset({"research"}),
            denied_tools=frozenset({"a", "b"}),
        )
        # Both 'a' (explicit allow) and 'b' (group allow) are denied
        result = filter_tools([_t("a"), _t("b")], ts, group_registry=reg)
        assert result == []


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicSurface:
    def test_top_level_imports(self) -> None:
        import kaos_agents

        for name in ("SessionToolSet", "filter_tools"):
            assert hasattr(kaos_agents, name), f"{name} missing from public surface"
            assert name in kaos_agents.__all__, f"{name} missing from __all__"
