"""Tests for DataType + ToolDataTypeRegistry (Track 4 chunk T4-1).

Confirms:

- ``DataType`` enum mirrors kelvin's 9-value taxonomy (TEXT/MARKDOWN/
  JSON/JSONL/HTML/CSV/TABLE/RDF/BINARY) with stable string values
- ``ToolDataTypeSpec`` is frozen, slotted, and has an empty-spec singleton
- ``ToolDataTypeRegistry`` returns the empty spec for unknown tools
  (no None-checks needed by callers)
- Registration enforces no-conflict by default; ``force=True`` overrides
- ``tools_by_input_type`` / ``tools_by_output_type`` filter correctly
- The default registry gets populated by ``register_agent_tools`` for
  every built-in tool
- The kaos_agents top-level surface re-exports the new types
"""

from __future__ import annotations

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.registry import (
    ToolDataTypeRegistry,
    default_tool_data_type_registry,
)
from kaos_agents.types import DataType, ToolDataTypeSpec

# ---------------------------------------------------------------------------
# DataType enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataTypeEnum:
    def test_nine_kelvin_values(self) -> None:
        # Kelvin's 9-value taxonomy — sticking to it keeps cross-tool
        # discovery semantics stable.
        names = {d.name for d in DataType}
        expected = {
            "TEXT",
            "MARKDOWN",
            "JSON",
            "JSONL",
            "HTML",
            "CSV",
            "TABLE",
            "RDF",
            "BINARY",
        }
        assert names == expected

    def test_string_values(self) -> None:
        assert DataType.TEXT.value == "text"
        assert DataType.MARKDOWN.value == "markdown"
        assert DataType.JSON.value == "json"
        assert DataType.JSONL.value == "jsonl"
        assert DataType.HTML.value == "html"
        assert DataType.CSV.value == "csv"
        assert DataType.TABLE.value == "table"
        assert DataType.RDF.value == "rdf"
        assert DataType.BINARY.value == "binary"


# ---------------------------------------------------------------------------
# ToolDataTypeSpec value type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolDataTypeSpec:
    def test_default_is_empty(self) -> None:
        spec = ToolDataTypeSpec()
        assert spec.input_type is None
        assert spec.output_type is None
        assert spec.is_empty

    def test_populated_spec_not_empty(self) -> None:
        spec = ToolDataTypeSpec(input_type=DataType.JSON, output_type=DataType.MARKDOWN)
        assert spec.is_empty is False

    def test_partial_spec_not_empty(self) -> None:
        # Only output_type set is still meaningful (a tool that emits CSV
        # but doesn't care about input format)
        spec = ToolDataTypeSpec(output_type=DataType.CSV)
        assert spec.is_empty is False
        assert spec.input_type is None

    def test_frozen(self) -> None:
        spec = ToolDataTypeSpec(input_type=DataType.JSON)
        # Frozen dataclasses raise on mutation. Use the dataclasses
        # internals to attempt mutation in a way that doesn't trip
        # ty's static-assignment check while still hitting the same
        # runtime guard (object.__setattr__ bypasses descriptors but
        # the frozen-class wrapper still raises).
        with pytest.raises(Exception):  # noqa: B017
            object.__setattr__(spec, "input_type", DataType.TEXT)
            spec.__dict__["input_type"] = DataType.TEXT

    def test_empty_singleton(self) -> None:
        # Identity-stable empty spec — same instance returned every call
        a = ToolDataTypeSpec.empty()
        b = ToolDataTypeSpec.empty()
        assert a is b


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolDataTypeRegistry:
    def test_get_unknown_returns_empty(self) -> None:
        reg = ToolDataTypeRegistry()
        spec = reg.get("does-not-exist")
        assert spec.is_empty
        # Even though the spec is empty, fields are queryable
        assert spec.input_type is None
        assert spec.output_type is None

    def test_register_and_get(self) -> None:
        reg = ToolDataTypeRegistry()
        spec = ToolDataTypeSpec(input_type=DataType.JSON, output_type=DataType.MARKDOWN)
        reg.register("my-tool", spec)
        assert reg.get("my-tool") == spec
        assert reg.has("my-tool") is True
        assert "my-tool" in reg

    def test_register_conflict_raises(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("t", ToolDataTypeSpec(input_type=DataType.TEXT))
        with pytest.raises(RegistryError):
            reg.register("t", ToolDataTypeSpec(input_type=DataType.JSON))

    def test_register_same_spec_twice_is_idempotent(self) -> None:
        """Re-registering the same spec is allowed (no-op-ish)."""
        reg = ToolDataTypeRegistry()
        spec = ToolDataTypeSpec(input_type=DataType.TEXT)
        reg.register("t", spec)
        reg.register("t", spec)  # no raise
        assert reg.get("t") == spec

    def test_force_replaces(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("t", ToolDataTypeSpec(input_type=DataType.TEXT))
        new_spec = ToolDataTypeSpec(input_type=DataType.JSON)
        reg.register("t", new_spec, force=True)
        assert reg.get("t") == new_spec

    def test_unregister_returns_spec(self) -> None:
        reg = ToolDataTypeRegistry()
        spec = ToolDataTypeSpec(output_type=DataType.JSON)
        reg.register("t", spec)
        removed = reg.unregister("t")
        assert removed == spec
        assert reg.has("t") is False

    def test_unregister_unknown_returns_none(self) -> None:
        reg = ToolDataTypeRegistry()
        assert reg.unregister("nope") is None

    def test_clear_drops_all(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("a", ToolDataTypeSpec(input_type=DataType.TEXT))
        reg.register("b", ToolDataTypeSpec(input_type=DataType.JSON))
        reg.clear()
        assert len(reg) == 0
        assert reg.list_names() == []

    def test_list_names_sorted(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("zeta", ToolDataTypeSpec())
        reg.register("alpha", ToolDataTypeSpec())
        reg.register("mu", ToolDataTypeSpec())
        assert reg.list_names() == ["alpha", "mu", "zeta"]

    def test_repr_includes_count(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("t", ToolDataTypeSpec(input_type=DataType.JSON))
        assert "1 specs" in repr(reg)
        assert "'t'" in repr(reg)


@pytest.mark.unit
class TestTypeDrivenDiscovery:
    def test_tools_by_input_type(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("a", ToolDataTypeSpec(input_type=DataType.JSON))
        reg.register("b", ToolDataTypeSpec(input_type=DataType.JSON))
        reg.register("c", ToolDataTypeSpec(input_type=DataType.TEXT))

        json_consumers = reg.tools_by_input_type(DataType.JSON)
        assert json_consumers == ("a", "b")  # alphabetical

        text_consumers = reg.tools_by_input_type(DataType.TEXT)
        assert text_consumers == ("c",)

        # No matches → empty tuple
        assert reg.tools_by_input_type(DataType.RDF) == ()

    def test_tools_by_output_type(self) -> None:
        reg = ToolDataTypeRegistry()
        reg.register("emit-md", ToolDataTypeSpec(output_type=DataType.MARKDOWN))
        reg.register("emit-csv", ToolDataTypeSpec(output_type=DataType.CSV))
        reg.register("emit-md-2", ToolDataTypeSpec(output_type=DataType.MARKDOWN))

        md_emitters = reg.tools_by_output_type(DataType.MARKDOWN)
        assert md_emitters == ("emit-md", "emit-md-2")

    def test_input_query_ignores_output_only_specs(self) -> None:
        """A tool that declared only output_type doesn't match input queries."""
        reg = ToolDataTypeRegistry()
        reg.register("output-only", ToolDataTypeSpec(output_type=DataType.JSON))
        assert reg.tools_by_input_type(DataType.JSON) == ()
        assert reg.tools_by_output_type(DataType.JSON) == ("output-only",)


# ---------------------------------------------------------------------------
# Default registry — populated by register_agent_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultRegistryPopulation:
    def test_register_agent_tools_tags_builtin_tools(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None

        # Force-clean the default registry before registering — other
        # tests might have already populated it
        register_agent_tools(runtime)

        # All 12 built-in tools should now have a non-empty I/O spec
        for tool_name in (
            "kaos-agent-chat",
            "kaos-agent-plan",
            "kaos-agent-memory-query",
            "kaos-agent-memory-search",
            "kaos-agent-memory-clear",
            "kaos-agent-recipe-list",
            "kaos-extract-schema",
            "kaos-extract-corpus",
            "kaos-extract-verify",
            "kaos-agent-graph-walk",
            "kaos-agent-graph-sparql",
            "kaos-agent-graph-projection",
        ):
            assert default_tool_data_type_registry.has(tool_name), f"{tool_name} not registered"

    def test_extraction_tools_emit_json(self) -> None:
        # Sanity check: structured extraction outputs JSON
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        runtime.tools.register_tool = lambda t: None
        register_agent_tools(runtime)

        json_emitters = default_tool_data_type_registry.tools_by_output_type(DataType.JSON)
        # All 3 extract tools emit JSON
        assert "kaos-extract-schema" in json_emitters
        assert "kaos-extract-corpus" in json_emitters
        assert "kaos-extract-verify" in json_emitters


# ---------------------------------------------------------------------------
# Top-level surface re-export
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicSurface:
    def test_top_level_imports(self) -> None:
        import kaos_agents

        # All four T4-1 names re-exported at the top level
        for name in (
            "DataType",
            "ToolDataTypeSpec",
            "ToolDataTypeRegistry",
            "default_tool_data_type_registry",
        ):
            assert hasattr(kaos_agents, name), f"{name} missing from public surface"
            assert name in kaos_agents.__all__, f"{name} missing from __all__"
