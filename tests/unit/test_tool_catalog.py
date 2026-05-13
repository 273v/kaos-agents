"""Tests for tool-catalog rendering modes (Track 4 chunk T4-4).

Confirms:

- ``render_tool_catalog`` accepts both ``KaosTool`` instances (with
  ``.metadata`` property) and raw ``ToolMetadata`` records
- Empty input → empty string (never raises during prompt assembly)
- ``compact`` mode: comma-separated names, no descriptions
- ``flat`` mode: bullet list at MEDIUM verbosity by default; honours
  ``field_set`` override
- ``grouped`` mode: groups discovered via the registry; tools in no
  group land in a synthetic "other" section
- ``full`` mode: JSON-serialised full dump (debug only, but valid)
- Unknown mode raises ``ValueError`` (loud failure for typos)
"""

from __future__ import annotations

import json

import pytest
from kaos_core.types.metadata import (
    ToolAnnotations,
    ToolCapability,
    ToolCategory,
    ToolMetadata,
)
from kaos_core.types.parameters import ParameterSchema

from kaos_agents.context import render_tool_catalog
from kaos_agents.registry import ToolGroupRegistry
from kaos_agents.types import FieldSet, ToolGroup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_meta(name: str, description: str = "") -> ToolMetadata:
    return ToolMetadata(
        name=name,
        display_name=name.replace("-", " ").title(),
        description=description,
        category=ToolCategory.DATA,
        capability=ToolCapability.QUERY,
        module_name="kaos-agents",
        version="0.1.0",
        annotations=ToolAnnotations(readOnlyHint=True),
        input_schema=[
            ParameterSchema(name="x", type="string", description="x"),
        ],
    )


@pytest.fixture
def sample_metas() -> list[ToolMetadata]:
    return [
        _make_tool_meta("kaos-extract-schema", "Schema-driven extraction."),
        _make_tool_meta("kaos-extract-corpus", "Corpus-fan-out extraction."),
        _make_tool_meta("kaos-agent-graph-walk", "N-hop walk on session graph."),
    ]


@pytest.fixture
def custom_group_registry() -> ToolGroupRegistry:
    """Registry with two groups for predictable group rendering."""
    reg = ToolGroupRegistry()
    reg.register(
        ToolGroup(
            name="extraction",
            description="Schema-driven extraction tools.",
            tool_names=("kaos-extract-schema", "kaos-extract-corpus"),
        )
    )
    reg.register(
        ToolGroup(
            name="graph",
            description="Session-graph tools.",
            tool_names=("kaos-agent-graph-walk",),
        )
    )
    return reg


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyInput:
    def test_empty_list_returns_empty_string(self) -> None:
        assert render_tool_catalog([]) == ""

    def test_unrenderable_inputs_skipped(self) -> None:
        # Plain strings have no metadata or .name fields with the right
        # shape — they're skipped, not raised on.
        assert render_tool_catalog(["not-a-tool", 42]) == ""


# ---------------------------------------------------------------------------
# compact mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompactMode:
    def test_returns_comma_separated_names(self, sample_metas) -> None:
        result = render_tool_catalog(sample_metas, mode="compact")
        assert result == "kaos-extract-schema, kaos-extract-corpus, kaos-agent-graph-walk"

    def test_no_descriptions_in_compact(self, sample_metas) -> None:
        result = render_tool_catalog(sample_metas, mode="compact")
        # Compact dropps descriptions to keep the prompt tight
        assert "Schema-driven" not in result
        assert "Corpus" not in result


# ---------------------------------------------------------------------------
# flat mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlatMode:
    def test_default_flat_includes_descriptions(self, sample_metas) -> None:
        result = render_tool_catalog(sample_metas, mode="flat")
        # Each tool on its own line, "name: description" shape
        assert "- kaos-extract-schema: Schema-driven extraction." in result
        assert "- kaos-extract-corpus: Corpus-fan-out extraction." in result
        assert "- kaos-agent-graph-walk: N-hop walk on session graph." in result
        # Three lines, three tools
        assert result.count("\n") == 2

    def test_flat_with_small_tier_drops_descriptions(self, sample_metas) -> None:
        result = render_tool_catalog(sample_metas, mode="flat", field_set=FieldSet.SMALL)
        assert "- kaos-extract-schema" in result
        assert "Schema-driven" not in result

    def test_flat_with_no_description_falls_back_cleanly(self) -> None:
        meta = _make_tool_meta("kaos-bare-tool", description="")
        # Empty description → just the name (no trailing ": ")
        result = render_tool_catalog([meta], mode="flat")
        # Falls back to display_name when description is empty
        assert "kaos-bare-tool" in result


# ---------------------------------------------------------------------------
# grouped mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGroupedMode:
    def test_grouped_renders_each_group_section(self, sample_metas, custom_group_registry) -> None:
        result = render_tool_catalog(
            sample_metas, mode="grouped", group_registry=custom_group_registry
        )
        # Each group has a header line "## name: description"
        assert "## extraction: Schema-driven extraction tools." in result
        assert "## graph: Session-graph tools." in result
        # Tools land in the right group's bullet block
        ext_idx = result.index("## extraction")
        graph_idx = result.index("## graph")
        assert "kaos-extract-schema" in result[ext_idx:graph_idx]
        assert "kaos-agent-graph-walk" in result[graph_idx:]

    def test_grouped_with_ungrouped_tool_creates_other_section(self, custom_group_registry) -> None:
        ungrouped = _make_tool_meta("kaos-orphan-tool", "Has no group.")
        in_extraction = _make_tool_meta("kaos-extract-schema", "Schema.")

        result = render_tool_catalog(
            [in_extraction, ungrouped],
            mode="grouped",
            group_registry=custom_group_registry,
        )
        # Both: the named group AND the synthetic "other" group
        assert "## extraction:" in result
        assert "## other:" in result
        # The orphan lands in "other"
        orphan_section = result[result.index("## other") :]
        assert "kaos-orphan-tool" in orphan_section

    def test_grouped_skips_empty_groups(self, custom_group_registry) -> None:
        # Pass tools that only match one group — the other group's
        # section should not appear (no header for empty groups)
        only_graph = [_make_tool_meta("kaos-agent-graph-walk", "Walk.")]
        result = render_tool_catalog(
            only_graph, mode="grouped", group_registry=custom_group_registry
        )
        assert "## graph:" in result
        assert "## extraction:" not in result


# ---------------------------------------------------------------------------
# full mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFullMode:
    def test_full_returns_valid_json(self, sample_metas) -> None:
        result = render_tool_catalog(sample_metas, mode="full")
        # Must parse as JSON (debug/audit format)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        # Full tier — input_schema dumps with full ParameterSchema records
        for record in parsed:
            assert "input_schema" in record


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorHandling:
    def test_unknown_mode_raises_value_error(self, sample_metas) -> None:
        # Cast through Any so ty's Literal-narrowing check passes; the
        # runtime check is what we care about — bad mode strings must
        # surface a clear error.
        from typing import Any, cast

        with pytest.raises(ValueError, match="Unknown catalog mode"):
            render_tool_catalog(sample_metas, mode=cast(Any, "unknown"))


# ---------------------------------------------------------------------------
# KaosTool acceptance (mock without the full base class)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKaosToolAcceptance:
    def test_accepts_object_with_metadata_property(self) -> None:
        """Instances with a ``.metadata`` property (as KaosTool does)
        are accepted alongside raw ToolMetadata records."""

        class FakeTool:
            @property
            def metadata(self) -> ToolMetadata:
                return _make_tool_meta("kaos-fake-tool", "Just a fake.")

        result = render_tool_catalog([FakeTool()], mode="flat")
        assert "kaos-fake-tool" in result
        assert "Just a fake." in result


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicSurface:
    def test_top_level_exports(self) -> None:
        import kaos_agents

        assert hasattr(kaos_agents, "render_tool_catalog")
        assert hasattr(kaos_agents, "CatalogMode")
        assert "render_tool_catalog" in kaos_agents.__all__
        assert "CatalogMode" in kaos_agents.__all__

    def test_context_package_exports(self) -> None:
        import kaos_agents.context as ctx

        for name in ("render_tool_catalog", "CatalogMode"):
            assert hasattr(ctx, name)
            assert name in ctx.__all__
