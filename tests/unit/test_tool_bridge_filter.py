"""Regression tests for WS-0.5 — glob-pattern tool filtering.

Pre-fix behavior (bug): ``bridge_runtime_tools(filter_names=["kaos-source-*"])``
did exact-name match and returned zero tools.

Post-fix: ``filter_names`` accepts :func:`fnmatch` glob patterns plus
exact names plus any mix of the two. The test set asserts the four
canonical cases — exact name, wildcard suffix, wildcard prefix, mixed
list — all route correctly through ``bridge_runtime_tools``.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult

from kaos_agents.actions.tool_bridge import _matches_filter, bridge_runtime_tools


def _make_tool(name: str) -> KaosTool:
    """Minimal KaosTool factory for filter-matching tests."""

    class _Tool(KaosTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name=name,
                description=f"Test tool {name}.",
                category=ToolCategory.TEXT,
                capability=ToolCapability.EXTRACT,
                module_name="kaos-agents-test",
                version="0.1.0",
                annotations=ToolAnnotations(readOnlyHint=True),
            )

        async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
            return ToolResult.create_text("")

    return _Tool()


@pytest.fixture()
def runtime() -> KaosRuntime:
    """Runtime preloaded with 6 tools across 3 fictional modules."""
    rt = KaosRuntime()
    for tool_name in (
        "kaos-pdf-extract",
        "kaos-pdf-search",
        "kaos-source-discover",
        "kaos-source-materialize",
        "kaos-source-edgar-lookup",
        "kaos-web-fetch",
    ):
        rt.tools.register_tool(_make_tool(tool_name))
    return rt


@pytest.mark.unit
class TestMatchesFilter:
    def test_exact_name_matches(self) -> None:
        assert _matches_filter("kaos-pdf-extract", ["kaos-pdf-extract"])

    def test_exact_name_no_match(self) -> None:
        assert not _matches_filter("kaos-pdf-extract", ["kaos-pdf-search"])

    def test_wildcard_suffix_matches(self) -> None:
        assert _matches_filter("kaos-source-discover", ["kaos-source-*"])
        assert _matches_filter("kaos-source-edgar-lookup", ["kaos-source-*"])

    def test_wildcard_suffix_does_not_cross_module(self) -> None:
        assert not _matches_filter("kaos-pdf-extract", ["kaos-source-*"])

    def test_wildcard_prefix_matches(self) -> None:
        assert _matches_filter("kaos-pdf-search", ["kaos-*-search"])
        assert _matches_filter("kaos-web-search", ["kaos-*-search"])

    def test_empty_patterns_matches_nothing(self) -> None:
        assert not _matches_filter("kaos-pdf-extract", [])

    def test_multiple_patterns_any_match(self) -> None:
        patterns = ["kaos-pdf-extract", "kaos-source-*"]
        assert _matches_filter("kaos-pdf-extract", patterns)
        assert _matches_filter("kaos-source-discover", patterns)
        assert not _matches_filter("kaos-web-fetch", patterns)


@pytest.mark.unit
class TestBridgeRuntimeToolsFilter:
    def test_no_filter_bridges_everything(self, runtime: KaosRuntime) -> None:
        tools = bridge_runtime_tools(runtime)
        assert len(tools) == 6

    def test_exact_name_filter(self, runtime: KaosRuntime) -> None:
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-pdf-extract"])
        assert len(tools) == 1
        assert tools[0].name == "kaos-pdf-extract"

    def test_glob_suffix_filter_selects_module(self, runtime: KaosRuntime) -> None:
        """The WS-0.5 headline case — ``kaos-source-*`` must match the
        three kaos-source tools, not zero."""
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-source-*"])
        names = sorted(t.name for t in tools)
        assert names == [
            "kaos-source-discover",
            "kaos-source-edgar-lookup",
            "kaos-source-materialize",
        ]

    def test_glob_prefix_filter(self, runtime: KaosRuntime) -> None:
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-*-search"])
        names = sorted(t.name for t in tools)
        assert names == ["kaos-pdf-search"]

    def test_mixed_exact_and_glob(self, runtime: KaosRuntime) -> None:
        tools = bridge_runtime_tools(
            runtime,
            filter_names=["kaos-web-fetch", "kaos-source-*"],
        )
        names = sorted(t.name for t in tools)
        assert names == [
            "kaos-source-discover",
            "kaos-source-edgar-lookup",
            "kaos-source-materialize",
            "kaos-web-fetch",
        ]

    def test_glob_matching_zero_tools_returns_empty_list(self, runtime: KaosRuntime) -> None:
        """Filter that matches nothing yields an empty list, not an error."""
        tools = bridge_runtime_tools(runtime, filter_names=["kaos-nomatch-*"])
        assert tools == []
