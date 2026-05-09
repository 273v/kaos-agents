"""Unit tests for :func:`kaos_agents.perception.registry.read_only_tools`.

Covers each duck-typed shape:

- :class:`kaos_core.base.tool.KaosTool` (real subclass, real metadata).
- :class:`kaos_llm_core.programs.tool.Tool` (no annotations surface,
  always skipped).
- ``dict`` with nested ``"annotations"``.
- ``dict`` with top-level ``"readOnlyHint"``.

Plus the empty-input and missing-annotations edge cases.
"""

from __future__ import annotations

from typing import Any

from kaos_core.base.context import KaosContext
from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult

from kaos_agents.perception.registry import read_only_tools

# --- Fixtures ---------------------------------------------------------


def _make_tool_metadata(*, name: str, annotations: ToolAnnotations | None) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        display_name=name,
        description="x",
        category=ToolCategory.TEXT,
        capability=ToolCapability.TRANSFORM,
        module_name="test",
        version="0.0.1",
        annotations=annotations,
    )


class _ReadOnlyKaosTool(KaosTool):
    @property
    def metadata(self) -> ToolMetadata:
        return _make_tool_metadata(
            name="kaos-test-readonly",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        return ToolResult.create_success(output={"items": []}, summary="x")


class _WriteKaosTool(KaosTool):
    @property
    def metadata(self) -> ToolMetadata:
        return _make_tool_metadata(
            name="kaos-test-write",
            annotations=ToolAnnotations(readOnlyHint=False),
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        return ToolResult.create_success(output={"items": []}, summary="x")


class _UnannotatedKaosTool(KaosTool):
    @property
    def metadata(self) -> ToolMetadata:
        return _make_tool_metadata(name="kaos-test-unann", annotations=None)

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        return ToolResult.create_success(output={"items": []}, summary="x")


# --- Tests ------------------------------------------------------------


def test_empty_input_returns_empty_tuple() -> None:
    assert read_only_tools([]) == ()


def test_kaos_tool_read_only_kept() -> None:
    tool = _ReadOnlyKaosTool()
    assert read_only_tools([tool]) == (tool,)


def test_kaos_tool_write_dropped() -> None:
    tool = _WriteKaosTool()
    assert read_only_tools([tool]) == ()


def test_kaos_tool_no_annotations_skipped() -> None:
    """Tools with annotations=None are skipped — fail closed."""
    tool = _UnannotatedKaosTool()
    assert read_only_tools([tool]) == ()


def test_dict_nested_annotations_kept() -> None:
    spec = {
        "name": "dict-readonly",
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
    }
    assert read_only_tools([spec]) == (spec,)


def test_dict_nested_annotations_write_dropped() -> None:
    spec = {
        "name": "dict-write",
        "annotations": {"readOnlyHint": False, "destructiveHint": True},
    }
    assert read_only_tools([spec]) == ()


def test_dict_flat_read_only_hint_kept() -> None:
    spec = {"name": "flat", "readOnlyHint": True}
    assert read_only_tools([spec]) == (spec,)


def test_dict_no_annotations_or_flag_skipped() -> None:
    spec = {"name": "no-info"}
    assert read_only_tools([spec]) == ()


def test_object_with_direct_annotations_attr_kept() -> None:
    """Duck-typed object exposing .annotations directly."""

    class _Duck:
        annotations = ToolAnnotations(readOnlyHint=True)

    duck = _Duck()
    assert read_only_tools([duck]) == (duck,)


def test_kaos_llm_core_tool_skipped_silently() -> None:
    """kaos-llm-core ``Tool`` has no annotations — fail closed."""
    from kaos_llm_client.types import ToolDefinition
    from kaos_llm_core.programs.tool import Tool

    def _fn(query: str) -> str:
        return query

    tool = Tool(
        definition=ToolDefinition(name="x", description="y", parameters={"type": "object"}),
        executor=_fn,
    )
    assert read_only_tools([tool]) == ()


def test_mixed_input_filters_correctly() -> None:
    ro = _ReadOnlyKaosTool()
    wr = _WriteKaosTool()
    spec_ro = {"name": "d-ro", "readOnlyHint": True}
    spec_unknown = {"name": "d-unk"}
    result = read_only_tools([ro, wr, spec_ro, spec_unknown])
    assert ro in result
    assert spec_ro in result
    assert wr not in result
    assert spec_unknown not in result
    assert len(result) == 2


def test_read_only_hint_truthy_string_not_accepted() -> None:
    """Strict ``True`` is required — truthy strings don't qualify."""
    spec = {"name": "n", "readOnlyHint": "yes"}
    assert read_only_tools([spec]) == ()
