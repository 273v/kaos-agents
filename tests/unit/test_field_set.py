"""Tests for FieldSet size-tiered serialization (Track 4 chunk T4-3).

Confirms:

- ``FieldSet`` enum has 4 values matching kelvin (SMALL/MEDIUM/LARGE/ALL)
- ``project()`` dispatches by ``isinstance`` to registered projectors
- Built-in projectors handle ToolMetadata, ToolGroup, AgentMetadata
- Tier subset relation: SMALL ⊆ MEDIUM ⊆ LARGE keys-wise
- ``register_projector`` lets out-of-tree types opt in
- Generic fallback returns ``{"name": ...}`` for unknown types at
  small tiers and the type's full dump at ALL
- ``ToolMetadata`` LARGE projection compacts ``input_schema`` to
  parameter names (no full type / constraint blowup)
- ``ToolGroup`` LARGE projection includes ``tool_count`` + ``tool_names``
- Top-level + types-package re-exports
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.parameters import ParameterSchema

from kaos_agents.types import (
    AgentMetadata,
    FieldSet,
    ToolGroup,
    project,
    register_projector,
)

# ---------------------------------------------------------------------------
# Fixtures — ToolMetadata sample with input_schema
# ---------------------------------------------------------------------------


def _sample_tool_metadata() -> ToolMetadata:
    return ToolMetadata(
        name="kaos-test-tool",
        display_name="My Tool",
        description="Does a thing.",
        category=ToolCategory.DATA,
        capability=ToolCapability.QUERY,
        module_name="kaos-agents",
        version="0.1.0",
        annotations=ToolAnnotations(readOnlyHint=True),
        input_schema=[
            ParameterSchema(name="query", type="string", description="What to ask."),
            ParameterSchema(name="top_k", type="integer", description="Cap.", required=False),
        ],
    )


def _sample_agent_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="research-agent",
        description="A research-pattern agent.",
        pattern="research",
        tags=("research", "rag", "citation"),
    )


# ---------------------------------------------------------------------------
# FieldSet enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFieldSetEnum:
    def test_four_kelvin_values(self) -> None:
        names = {fs.name for fs in FieldSet}
        assert names == {"SMALL", "MEDIUM", "LARGE", "ALL"}

    def test_string_values(self) -> None:
        assert FieldSet.SMALL.value == "small"
        assert FieldSet.MEDIUM.value == "medium"
        assert FieldSet.LARGE.value == "large"
        assert FieldSet.ALL.value == "all"


# ---------------------------------------------------------------------------
# ToolMetadata projection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProjectToolMetadata:
    def test_small_tier_just_name(self) -> None:
        meta = _sample_tool_metadata()
        result = project(meta, FieldSet.SMALL)
        assert result == {"name": "kaos-test-tool"}

    def test_medium_tier_adds_description(self) -> None:
        meta = _sample_tool_metadata()
        result = project(meta, FieldSet.MEDIUM)
        assert result["name"] == "kaos-test-tool"
        assert result["display_name"] == "My Tool"
        assert result["description"] == "Does a thing."

    def test_large_tier_adds_category_and_compact_schema(self) -> None:
        meta = _sample_tool_metadata()
        result = project(meta, FieldSet.LARGE)
        # MEDIUM keys still present
        assert result["name"] == "kaos-test-tool"
        assert result["description"] == "Does a thing."
        # LARGE adds category + capability + compact parameter names
        assert result["category"] == "data"
        assert result["capability"] == "query"
        # Compact: just names, no full ParameterSchema dump
        assert result["parameter_names"] == ["query", "top_k"]

    def test_all_tier_full_model_dump(self) -> None:
        meta = _sample_tool_metadata()
        result = project(meta, FieldSet.ALL)
        # Full dump includes all fields including input_schema as
        # full ParameterSchema dicts
        assert result["name"] == "kaos-test-tool"
        assert "input_schema" in result
        assert isinstance(result["input_schema"], list)
        # Every input_schema entry has full type/description metadata
        first = result["input_schema"][0]
        assert "name" in first
        assert "type" in first
        assert "description" in first

    def test_subset_relation_keys(self) -> None:
        """SMALL keys ⊆ MEDIUM keys ⊆ LARGE keys."""
        meta = _sample_tool_metadata()
        small_keys = set(project(meta, FieldSet.SMALL).keys())
        medium_keys = set(project(meta, FieldSet.MEDIUM).keys())
        large_keys = set(project(meta, FieldSet.LARGE).keys())
        assert small_keys <= medium_keys
        assert medium_keys <= large_keys


# ---------------------------------------------------------------------------
# ToolGroup projection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProjectToolGroup:
    def test_small_tier(self) -> None:
        group = ToolGroup(
            name="extraction",
            description="Schema-driven tools.",
            tool_names=("a", "b"),
        )
        assert project(group, FieldSet.SMALL) == {"name": "extraction"}

    def test_medium_tier(self) -> None:
        group = ToolGroup(
            name="extraction",
            description="Schema-driven tools.",
            tool_names=("a", "b"),
        )
        result = project(group, FieldSet.MEDIUM)
        assert result["name"] == "extraction"
        assert result["description"] == "Schema-driven tools."
        # Tool list NOT in MEDIUM — that's LARGE territory
        assert "tool_names" not in result

    def test_large_tier_includes_tool_list(self) -> None:
        group = ToolGroup(
            name="extraction",
            description="Schema-driven tools.",
            tool_names=("a", "b", "c"),
        )
        result = project(group, FieldSet.LARGE)
        assert result["tool_count"] == 3
        assert result["tool_names"] == ["a", "b", "c"]

    def test_all_tier_full_dataclass_dump(self) -> None:
        group = ToolGroup(
            name="extraction",
            description="d",
            tool_names=("a",),
            tags=("read-only",),
        )
        result = project(group, FieldSet.ALL)
        assert result["name"] == "extraction"
        assert result["tags"] == ("read-only",)


# ---------------------------------------------------------------------------
# AgentMetadata projection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProjectAgentMetadata:
    def test_small_tier_just_name(self) -> None:
        meta = _sample_agent_metadata()
        result = project(meta, FieldSet.SMALL)
        assert result == {"name": "research-agent"}

    def test_medium_tier_adds_description_and_pattern(self) -> None:
        meta = _sample_agent_metadata()
        result = project(meta, FieldSet.MEDIUM)
        assert result["pattern"] == "research"
        assert result["description"] == "A research-pattern agent."

    def test_large_tier_adds_tags(self) -> None:
        meta = _sample_agent_metadata()
        result = project(meta, FieldSet.LARGE)
        assert result["tags"] == ["research", "rag", "citation"]


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenericFallback:
    def test_unknown_type_at_small_returns_name_field(self) -> None:
        @dataclass(frozen=True)
        class Custom:
            name: str
            extra: str = "ignored"

        c = Custom(name="thing")
        result = project(c, FieldSet.SMALL)
        assert result == {"name": "thing"}

    def test_unknown_type_at_all_uses_dataclass_asdict(self) -> None:
        @dataclass(frozen=True)
        class Custom:
            name: str
            extra: str

        c = Custom(name="t", extra="e")
        result = project(c, FieldSet.ALL)
        assert result == {"name": "t", "extra": "e"}

    def test_object_without_name_returns_empty(self) -> None:
        # No registered projector, no name field, not a dataclass —
        # SMALL tier returns empty dict (no identifier to surface)
        class Plain:
            pass

        result = project(Plain(), FieldSet.SMALL)
        assert result == {}


# ---------------------------------------------------------------------------
# register_projector — out-of-tree types opt in
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterProjector:
    def test_register_custom_projector(self) -> None:
        @dataclass(frozen=True)
        class Widget:
            name: str
            color: str
            secret: str

        def _project_widget(w: Widget, fs: FieldSet) -> dict:
            if fs == FieldSet.ALL:
                return {"name": w.name, "color": w.color, "secret": w.secret}
            if fs == FieldSet.LARGE:
                return {"name": w.name, "color": w.color}
            if fs == FieldSet.MEDIUM:
                return {"name": w.name, "color": w.color}
            return {"name": w.name}  # SMALL

        register_projector(Widget, _project_widget)

        w = Widget(name="W1", color="blue", secret="hush")
        # Custom projector hides 'secret' from non-ALL tiers
        assert project(w, FieldSet.SMALL) == {"name": "W1"}
        assert project(w, FieldSet.LARGE) == {"name": "W1", "color": "blue"}
        assert "secret" not in project(w, FieldSet.LARGE)
        assert "secret" in project(w, FieldSet.ALL)

    def test_register_replaces_prior(self) -> None:
        @dataclass(frozen=True)
        class Gizmo:
            name: str

        def _v1(g: Gizmo, fs: FieldSet) -> dict:
            return {"version": 1, "name": g.name}

        def _v2(g: Gizmo, fs: FieldSet) -> dict:
            return {"version": 2, "name": g.name}

        register_projector(Gizmo, _v1)
        register_projector(Gizmo, _v2)
        assert project(Gizmo(name="g"), FieldSet.SMALL) == {"version": 2, "name": "g"}


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicSurface:
    def test_top_level_has_field_set(self) -> None:
        import kaos_agents

        assert hasattr(kaos_agents, "FieldSet")
        assert "FieldSet" in kaos_agents.__all__

    def test_types_package_exports_project(self) -> None:
        import kaos_agents.types as t

        for name in ("FieldSet", "project", "register_projector"):
            assert hasattr(t, name)
            assert name in t.__all__
