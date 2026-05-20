"""Unit tests for :mod:`kaos_agents.capabilities.retrieve`.

Stub-runtime + stub-tool coverage of the Step 2 federator. No live
LLM, no real kaos-web / kaos-source / kaos-content tools — every
backing dependency is hand-built here so we can exercise the
federation logic in isolation.

See plan §1 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.types import (
    Capability,
    CapabilityKind,
    CostClass,
    ToolAnnotations,
    ToolCapability,
    ToolCategory,
    ToolMetadata,
)
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

from kaos_agents.capabilities.retrieve import (
    RETRIEVE_CAPABILITY_NAME,
    RetrievalHit,
    retrieve,
)
from kaos_agents.registry.capability_registry import CapabilityRegistry

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTool:
    """Minimal :class:`KaosTool`-shaped stub.

    Holds a ``ToolMetadata`` and an async ``execute`` that returns a
    pre-baked :class:`ToolResult` (or raises if ``raise_on_execute``
    is set).
    """

    def __init__(
        self,
        name: str,
        *,
        result: ToolResult | None = None,
        raise_on_execute: Exception | None = None,
        input_schema: tuple[ParameterSchema, ...] | None = None,
    ) -> None:
        self._name = name
        self._result = result
        self._raise = raise_on_execute
        schema = input_schema or (
            ParameterSchema(name="query", type="string", description="q"),
            ParameterSchema(
                name="max_results",
                type="integer",
                description="n",
                required=False,
                default=5,
            ),
        )
        self._metadata = ToolMetadata(
            name=name,
            description=f"Stub tool {name}",
            module_name="kaos-test",
            version="0.0.0",
            category=ToolCategory.TEXT,
            capability=ToolCapability.QUERY,
            input_schema=list(schema),
            annotations=ToolAnnotations(readOnlyHint=True),
        )
        self.calls: list[dict[str, Any]] = []

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        self.calls.append(dict(inputs))
        if self._raise is not None:
            raise self._raise
        assert self._result is not None
        return self._result


class _StubToolRegistry:
    def __init__(self, tools: dict[str, _StubTool]) -> None:
        self._tools = tools

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def get_tool(self, name: str) -> _StubTool | None:
        return self._tools.get(name)


class _StubRuntime:
    def __init__(self, tools: dict[str, _StubTool]) -> None:
        self.tools = _StubToolRegistry(tools)


def _structured_results(*items: dict[str, Any]) -> ToolResult:
    """Wrap a list of dicts in the kaos-* search structuredContent shape."""
    return ToolResult.create_success(
        output={"results": list(items)},
        summary=f"{len(items)} results",
    )


def _text_result(text: str) -> ToolResult:
    return ToolResult.create_text(text)


def _search_capability(
    name: str,
    backing: tuple[str, ...],
    *,
    tags: tuple[str, ...] = (),
    kind: CapabilityKind = CapabilityKind.SEARCH,
) -> Capability:
    return Capability(
        name=name,
        kind=kind,
        description=f"Search capability {name}",
        cost_class=CostClass.CHEAP,
        tags=tags,
        backing_tool_names=backing,
    )


# ---------------------------------------------------------------------------
# Module-import-time registration
# ---------------------------------------------------------------------------


def test_retrieve_capability_registered_on_import() -> None:
    """Importing the module registers the sample ``retrieve`` capability."""
    from kaos_agents.registry import default_capability_registry

    cap = default_capability_registry.get(RETRIEVE_CAPABILITY_NAME)
    assert cap is not None
    assert cap.kind == CapabilityKind.SEARCH
    assert cap.cost_class == CostClass.CHEAP
    assert "query" in cap.inputs
    assert "mode" in cap.inputs
    assert "source_filter" in cap.inputs
    assert cap.outputs == ("RetrievalHit[]",)


# ---------------------------------------------------------------------------
# Discover mode — search-tool aggregation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_aggregates_across_search_tools() -> None:
    """Discover mode invokes every SEARCH capability's backing tool."""
    web = _StubTool(
        "kaos-web-search",
        result=_structured_results(
            {"title": "T1", "url": "https://a.example", "snippet": "snip1", "score": 0.9}
        ),
    )
    fr = _StubTool(
        "kaos-source-fr-search",
        result=_structured_results(
            {"title": "Reg X", "url": "fr://X", "snippet": "rule text", "score": 0.7}
        ),
    )
    runtime = _StubRuntime({"kaos-web-search": web, "kaos-source-fr-search": fr})

    registry = CapabilityRegistry()
    registry.register(_search_capability("web", ("kaos-web-search",), tags=("web",)))
    registry.register(_search_capability("fr", ("kaos-source-fr-search",), tags=("legal-source",)))

    hits = await retrieve(
        "agency rulemaking",
        runtime,  # ty: ignore[invalid-argument-type]
        mode="discover",
        registry=registry,
    )

    assert len(hits) == 2
    source_ids = {h.source_id for h in hits}
    assert source_ids == {"kaos-web-search", "kaos-source-fr-search"}
    by_source = {h.source_id: h for h in hits}
    assert by_source["kaos-web-search"].text == "snip1"
    assert by_source["kaos-web-search"].provenance == "https://a.example"
    assert by_source["kaos-web-search"].score == pytest.approx(0.9)
    # Each tool received the query.
    assert web.calls == [{"query": "agency rulemaking", "max_results": 5}]
    assert fr.calls == [{"query": "agency rulemaking", "max_results": 5}]


@pytest.mark.asyncio
async def test_source_filter_narrows_to_matching_tags() -> None:
    """Non-empty ``source_filter`` keeps only capabilities with matching tags."""
    web = _StubTool(
        "kaos-web-search",
        result=_structured_results(
            {"snippet": "web hit", "url": "https://a.example", "score": 0.5}
        ),
    )
    fr = _StubTool(
        "kaos-source-fr-search",
        result=_structured_results({"snippet": "fr hit", "url": "fr://X", "score": 0.8}),
    )
    runtime = _StubRuntime({"kaos-web-search": web, "kaos-source-fr-search": fr})

    registry = CapabilityRegistry()
    registry.register(_search_capability("web", ("kaos-web-search",), tags=("web",)))
    registry.register(_search_capability("fr", ("kaos-source-fr-search",), tags=("legal-source",)))

    hits = await retrieve(
        "query",
        runtime,  # ty: ignore[invalid-argument-type]
        source_filter=("legal-source",),
        mode="discover",
        registry=registry,
    )

    assert len(hits) == 1
    assert hits[0].source_id == "kaos-source-fr-search"
    # web tool was filtered out by tag; never invoked.
    assert web.calls == []
    assert len(fr.calls) == 1


# ---------------------------------------------------------------------------
# Mode flag — read / grep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_mode_picks_fetch_tools() -> None:
    """``read`` mode invokes READ-kind capabilities whose tool name fetches."""
    fetch = _StubTool(
        "kaos-web-fetch-page",
        result=_text_result("# Page body\nMarkdown content."),
    )
    search = _StubTool(
        "kaos-web-search",
        result=_structured_results({"snippet": "should not be picked", "url": "x"}),
    )
    runtime = _StubRuntime({"kaos-web-fetch-page": fetch, "kaos-web-search": search})

    registry = CapabilityRegistry()
    registry.register(
        Capability(
            name="web-read",
            kind=CapabilityKind.READ,
            description="Fetch a web page",
            backing_tool_names=("kaos-web-fetch-page",),
        )
    )
    registry.register(_search_capability("web-search", ("kaos-web-search",)))

    hits = await retrieve(
        "https://example.com",
        runtime,  # ty: ignore[invalid-argument-type]
        mode="read",
        registry=registry,
    )

    assert len(hits) == 1
    assert hits[0].source_id == "kaos-web-fetch-page"
    assert "Markdown content." in hits[0].text
    # search tool not invoked (wrong kind + wrong name hint).
    assert search.calls == []
    assert len(fetch.calls) == 1


@pytest.mark.asyncio
async def test_grep_mode_picks_search_document_tools() -> None:
    """``grep`` mode invokes tools whose name contains ``search-document``."""
    grep_tool = _StubTool(
        "kaos-content-search-document",
        result=_structured_results(
            {
                "preview": "matching paragraph",
                "block_ref": "#/body/3",
                "score": 1.2,
            }
        ),
    )
    web_search = _StubTool(
        "kaos-web-search",
        result=_structured_results({"snippet": "unrelated", "url": "x", "score": 0.1}),
    )
    runtime = _StubRuntime(
        {
            "kaos-content-search-document": grep_tool,
            "kaos-web-search": web_search,
        }
    )

    registry = CapabilityRegistry()
    registry.register(_search_capability("search-document", ("kaos-content-search-document",)))
    registry.register(_search_capability("web", ("kaos-web-search",)))

    hits = await retrieve(
        "merger clause",
        runtime,  # ty: ignore[invalid-argument-type]
        mode="grep",
        registry=registry,
    )

    assert len(hits) == 1
    assert hits[0].source_id == "kaos-content-search-document"
    assert hits[0].provenance == "#/body/3"
    assert hits[0].text == "matching paragraph"
    # web-search not selected (no ``search-document`` substring).
    assert web_search.calls == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_registry_returns_no_hits() -> None:
    """An empty registry returns an empty tuple without raising."""
    runtime = _StubRuntime({})
    registry = CapabilityRegistry()  # nothing registered
    hits = await retrieve(
        "any query",
        runtime,  # ty: ignore[invalid-argument-type]
        registry=registry,
    )
    assert hits == ()


@pytest.mark.asyncio
async def test_tool_execution_error_is_silently_skipped() -> None:
    """A tool that raises during ``execute`` contributes no hits."""
    broken = _StubTool(
        "kaos-source-edgar-search",
        raise_on_execute=RuntimeError("provider 500"),
    )
    healthy = _StubTool(
        "kaos-web-search",
        result=_structured_results({"snippet": "ok", "url": "https://a.example", "score": 0.3}),
    )
    runtime = _StubRuntime({"kaos-source-edgar-search": broken, "kaos-web-search": healthy})

    registry = CapabilityRegistry()
    registry.register(
        _search_capability("edgar", ("kaos-source-edgar-search",), tags=("legal-source",))
    )
    registry.register(_search_capability("web", ("kaos-web-search",), tags=("web",)))

    hits = await retrieve(
        "query",
        runtime,  # ty: ignore[invalid-argument-type]
        mode="discover",
        registry=registry,
    )

    # Only the healthy tool's hit survives — the failure is swallowed.
    assert len(hits) == 1
    assert hits[0].source_id == "kaos-web-search"
    # Both tools were attempted.
    assert len(broken.calls) == 1
    assert len(healthy.calls) == 1


@pytest.mark.asyncio
async def test_tool_missing_query_param_is_skipped() -> None:
    """Tools whose schema lacks ``query`` are skipped silently."""
    no_query = _StubTool(
        "kaos-misshaped-search",
        result=_text_result("never reached"),
        input_schema=(ParameterSchema(name="topic", type="string", description="t"),),
    )
    runtime = _StubRuntime({"kaos-misshaped-search": no_query})

    registry = CapabilityRegistry()
    registry.register(_search_capability("mis", ("kaos-misshaped-search",)))

    hits = await retrieve(
        "ignored",
        runtime,  # ty: ignore[invalid-argument-type]
        registry=registry,
    )

    assert hits == ()
    # ``execute`` was never invoked — we never attempt the call.
    assert no_query.calls == []


@pytest.mark.asyncio
async def test_source_filter_excludes_everything_returns_empty() -> None:
    """A source_filter that no capability satisfies yields an empty tuple."""
    web = _StubTool(
        "kaos-web-search",
        result=_structured_results({"snippet": "x", "url": "y", "score": 0.1}),
    )
    runtime = _StubRuntime({"kaos-web-search": web})

    registry = CapabilityRegistry()
    registry.register(_search_capability("web", ("kaos-web-search",), tags=("web",)))

    hits = await retrieve(
        "q",
        runtime,  # ty: ignore[invalid-argument-type]
        source_filter=("nonexistent-domain",),
        registry=registry,
    )
    assert hits == ()
    assert web.calls == []


@pytest.mark.asyncio
async def test_max_n_caps_per_tool_results() -> None:
    """``max_n`` caps per-tool hits (federated total may exceed it)."""
    web = _StubTool(
        "kaos-web-search",
        result=_structured_results(
            *[{"snippet": f"r{i}", "url": f"u{i}", "score": float(i)} for i in range(10)]
        ),
    )
    runtime = _StubRuntime({"kaos-web-search": web})

    registry = CapabilityRegistry()
    registry.register(_search_capability("web", ("kaos-web-search",)))

    hits = await retrieve(
        "query",
        runtime,  # ty: ignore[invalid-argument-type]
        max_n=3,
        registry=registry,
    )

    assert len(hits) == 3
    assert {h.text for h in hits} == {"r0", "r1", "r2"}
    assert isinstance(hits[0], RetrievalHit)
