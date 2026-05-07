"""Tests for ToolGroup + ToolGroupRegistry (Track 4 chunk T4-2).

Confirms:

- ``ToolGroup`` is a frozen value type with a non-empty-name invariant
- iter / contains / len work over the tool-name list
- ``ToolGroupRegistry`` register / get / has / clear / list_names
  match the AgentToolSpecRegistry shape
- Conflict detection raises ``RegistryError``; ``force=True`` overrides;
  same-group re-register is a no-op
- 2-level discovery: ``groups()`` returns ToolGroups; ``tools_in()``
  returns the names; ``groups_for_tool()`` returns sorted membership
- Unknown groups return ``None`` from ``get`` (no empty-spec fallback —
  see registry docstring for rationale)
- ``register_agent_tools`` populates the default registry with the
  three built-in groups: agent, extraction, graph
- Top-level ``kaos_agents`` re-exports the new symbols
"""

from __future__ import annotations

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.registry import ToolGroupRegistry, default_tool_group_registry
from kaos_agents.types import ToolGroup

# ---------------------------------------------------------------------------
# ToolGroup value type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolGroupValueType:
    def test_basic_construction(self) -> None:
        group = ToolGroup(
            name="extraction",
            description="Schema-driven extraction tools.",
            tool_names=("kaos-extract-schema", "kaos-extract-corpus"),
        )
        assert group.name == "extraction"
        assert group.description == "Schema-driven extraction tools."
        assert group.tool_names == ("kaos-extract-schema", "kaos-extract-corpus")
        assert group.tags == ()

    def test_empty_name_raises(self) -> None:
        # An empty name would silently shadow whichever group registered
        # last — much better to fail loudly at construction.
        with pytest.raises(ValueError):
            ToolGroup(name="", description="x", tool_names=())

    def test_default_tool_names_empty(self) -> None:
        # A group can register before its tools (tools added later by
        # appending or recreating). Empty default is meaningful.
        group = ToolGroup(name="future", description="placeholder")
        assert group.tool_names == ()

    def test_iter_yields_tool_names(self) -> None:
        group = ToolGroup(name="g", description="d", tool_names=("a", "b", "c"))
        assert list(group) == ["a", "b", "c"]

    def test_contains_membership(self) -> None:
        group = ToolGroup(name="g", description="d", tool_names=("a", "b"))
        assert "a" in group
        assert "z" not in group
        # Non-string membership tests are always False (no TypeError)
        assert 42 not in group  # type: ignore[operator]

    def test_len_is_tool_count(self) -> None:
        group = ToolGroup(name="g", description="d", tool_names=("a", "b", "c"))
        assert len(group) == 3

    def test_frozen_no_mutation(self) -> None:
        group = ToolGroup(name="g", description="d", tool_names=())
        with pytest.raises(Exception):  # noqa: B017
            object.__setattr__(group, "name", "changed")
            group.__dict__["name"] = "changed"

    def test_equality_by_value(self) -> None:
        a = ToolGroup(name="g", description="d", tool_names=("a",))
        b = ToolGroup(name="g", description="d", tool_names=("a",))
        c = ToolGroup(name="g", description="d", tool_names=("a", "b"))
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Registry mutation + lookup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolGroupRegistryMutation:
    def test_register_and_get(self) -> None:
        reg = ToolGroupRegistry()
        group = ToolGroup(name="g", description="d", tool_names=("a", "b"))
        reg.register(group)
        assert reg.get("g") == group
        assert reg.has("g")
        assert "g" in reg

    def test_get_unknown_returns_none(self) -> None:
        # Unlike ToolDataTypeRegistry's empty-spec fallback, a missing
        # group is genuinely absent — pretending otherwise would lie
        # about the catalogue.
        reg = ToolGroupRegistry()
        assert reg.get("nope") is None

    def test_register_conflict_raises(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="g", description="d", tool_names=("a",)))
        with pytest.raises(RegistryError):
            reg.register(ToolGroup(name="g", description="other", tool_names=("z",)))

    def test_register_same_group_twice_is_idempotent(self) -> None:
        reg = ToolGroupRegistry()
        group = ToolGroup(name="g", description="d", tool_names=("a",))
        reg.register(group)
        reg.register(group)  # no raise — equal value
        assert len(reg) == 1

    def test_force_replaces(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="g", description="old", tool_names=()))
        new_group = ToolGroup(name="g", description="new", tool_names=("z",))
        reg.register(new_group, force=True)
        assert reg.get("g") == new_group

    def test_unregister(self) -> None:
        reg = ToolGroupRegistry()
        group = ToolGroup(name="g", description="d", tool_names=())
        reg.register(group)
        removed = reg.unregister("g")
        assert removed == group
        assert not reg.has("g")

    def test_unregister_unknown_returns_none(self) -> None:
        reg = ToolGroupRegistry()
        assert reg.unregister("nope") is None

    def test_clear(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="a", description="d"))
        reg.register(ToolGroup(name="b", description="d"))
        reg.clear()
        assert len(reg) == 0
        assert reg.list_names() == []

    def test_list_names_sorted(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="zeta", description="d"))
        reg.register(ToolGroup(name="alpha", description="d"))
        reg.register(ToolGroup(name="mu", description="d"))
        assert reg.list_names() == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# 2-level discovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTwoLevelDiscovery:
    def test_groups_returns_all(self) -> None:
        reg = ToolGroupRegistry()
        a = ToolGroup(name="a", description="d", tool_names=("t1",))
        b = ToolGroup(name="b", description="d", tool_names=("t2",))
        reg.register(a)
        reg.register(b)
        groups = reg.groups()
        assert set(groups) == {a, b}

    def test_tools_in_known_group(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="g", description="d", tool_names=("a", "b", "c")))
        assert reg.tools_in("g") == ("a", "b", "c")

    def test_tools_in_unknown_group_returns_empty(self) -> None:
        reg = ToolGroupRegistry()
        assert reg.tools_in("nope") == ()

    def test_groups_for_tool_returns_membership(self) -> None:
        # A tool may appear in multiple groups; surface them sorted
        reg = ToolGroupRegistry()
        reg.register(
            ToolGroup(name="extraction", description="d", tool_names=("kaos-extract-schema",))
        )
        reg.register(ToolGroup(name="legal", description="d", tool_names=("kaos-extract-schema",)))
        reg.register(ToolGroup(name="other", description="d", tool_names=("z",)))

        memberships = reg.groups_for_tool("kaos-extract-schema")
        assert memberships == ("extraction", "legal")

    def test_groups_for_unknown_tool_returns_empty(self) -> None:
        reg = ToolGroupRegistry()
        reg.register(ToolGroup(name="g", description="d", tool_names=("a",)))
        assert reg.groups_for_tool("z") == ()


# ---------------------------------------------------------------------------
# Default registry — populated by register_agent_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultRegistryPopulation:
    def test_register_agent_tools_creates_three_groups(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None
        register_agent_tools(runtime)

        for group_name in ("agent", "extraction", "graph"):
            assert default_tool_group_registry.has(group_name), f"{group_name} group not registered"

    def test_extraction_group_has_three_tools(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None
        register_agent_tools(runtime)

        extraction = default_tool_group_registry.get("extraction")
        assert extraction is not None
        assert len(extraction) == 3
        assert "kaos-extract-schema" in extraction
        assert "kaos-extract-corpus" in extraction
        assert "kaos-extract-verify" in extraction

    def test_graph_group_holds_b3_tools(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None
        register_agent_tools(runtime)

        graph_group = default_tool_group_registry.get("graph")
        assert graph_group is not None
        assert "kaos-agent-graph-walk" in graph_group
        assert "kaos-agent-graph-sparql" in graph_group
        assert "kaos-agent-graph-projection" in graph_group

    def test_groups_for_tool_finds_membership_in_default(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None
        register_agent_tools(runtime)

        membership = default_tool_group_registry.groups_for_tool("kaos-agent-memory-search")
        assert "agent" in membership


# ---------------------------------------------------------------------------
# Top-level surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicSurface:
    def test_top_level_imports(self) -> None:
        import kaos_agents

        for name in (
            "ToolGroup",
            "ToolGroupRegistry",
            "default_tool_group_registry",
        ):
            assert hasattr(kaos_agents, name), f"{name} missing from public surface"
            assert name in kaos_agents.__all__, f"{name} missing from __all__"
