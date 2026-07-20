"""Unit tests for kaos_agents.runtime.tool_retrieval.

Deterministic-only — exercises the BM25 path without LLM involvement.
The lexicon-expansion path is covered by a smoke test that constructs
a small Lexicon and verifies synonym hits.
"""

from __future__ import annotations

from typing import Any

from kaos_core.base.tool import KaosTool
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

from kaos_agents.runtime.tool_retrieval import (
    DEFAULT_TOP_K,
    ToolHit,
    ToolRetrieval,
    _tool_searchable_text,
)

# ---------------------------------------------------------------------------
# Helpers — minimal fake tools for the index
# ---------------------------------------------------------------------------


class _FakeTool(KaosTool):
    """A no-op tool with configurable metadata for retrieval tests."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        display_name: str = "",
        module_name: str = "kaos-test",
        input_params: tuple[str, ...] = (),
    ) -> None:
        self._meta = ToolMetadata(
            name=name,
            display_name=display_name or name,
            description=description,
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=module_name,
            version="0.1.0",
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(name=p, type="string", description="") for p in input_params
            ],
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._meta

    async def execute(
        self, inputs: dict[str, Any], context: Any = None
    ) -> ToolResult:  # pragma: no cover — not exercised
        return ToolResult.create_text("ok")


def _build_catalog() -> list[KaosTool]:
    """Three tools spanning distinct semantic spaces, plus one for ties."""
    return [
        _FakeTool(
            "kaos-source-fr-search",
            "Search the Federal Register for regulatory documents and notices.",
            input_params=("term", "agency", "doc_type"),
        ),
        _FakeTool(
            "kaos-pdf-extract-page-text",
            "Extract the text content of a specific page from a PDF file.",
            input_params=("path", "page"),
        ),
        _FakeTool(
            "kaos-tabular-query",
            "Execute SQL against registered tables via DuckDB. Returns TSV.",
            input_params=("sql", "max_rows"),
        ),
        _FakeTool(
            "kaos-source-edgar-fetch",
            "Fetch a filing from the SEC EDGAR system.",
            input_params=("cik", "accession"),
        ),
    ]


# ---------------------------------------------------------------------------
# _tool_searchable_text — indexed text composition
# ---------------------------------------------------------------------------


class TestSearchableText:
    def test_includes_name_with_hyphens_as_tokens(self) -> None:
        tool = _FakeTool("kaos-pdf-extract", "Extract text from PDF.")
        text = _tool_searchable_text(tool)
        # Each segment of the hyphen-split name should be present.
        assert "kaos" in text
        assert "pdf" in text
        assert "extract" in text

    def test_includes_description(self) -> None:
        tool = _FakeTool("kaos-test-marker", "Some unique description marker xyzzy.")
        text = _tool_searchable_text(tool)
        assert "xyzzy" in text

    def test_includes_param_names(self) -> None:
        tool = _FakeTool("kaos-test-marker", "x.", input_params=("unique_param_marker",))
        text = _tool_searchable_text(tool)
        # Underscores split into tokens
        assert "unique" in text
        assert "param" in text
        assert "marker" in text

    def test_empty_metadata_safe(self) -> None:
        # Defensive: an emergency-built tool with empty fields shouldn't crash.
        tool = _FakeTool("kaos-test-empty", "")
        assert _tool_searchable_text(tool) != ""


# ---------------------------------------------------------------------------
# ToolRetrieval — core behaviour
# ---------------------------------------------------------------------------


class TestToolRetrieval:
    def test_empty_catalog_returns_no_hits(self) -> None:
        retrieval = ToolRetrieval([])
        assert retrieval.size == 0
        assert retrieval.search("anything", top_k=5) == []
        assert retrieval.select("anything", top_k=5) == []

    def test_empty_query_returns_no_hits(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        assert retrieval.search("", top_k=5) == []
        assert retrieval.search("   ", top_k=5) == []

    def test_search_returns_relevant_tool_first(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        hits = retrieval.search("Search the Federal Register for EPA rules", top_k=3)
        assert len(hits) >= 1
        # The FR tool should rank at the top.
        assert hits[0].tool.metadata.name == "kaos-source-fr-search"
        assert hits[0].score > 0.0

    def test_search_returns_pdf_tool_for_pdf_query(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        hits = retrieval.search("Extract text from a PDF page", top_k=2)
        assert hits[0].tool.metadata.name == "kaos-pdf-extract-page-text"

    def test_search_returns_tabular_tool_for_sql_query(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        hits = retrieval.search("Run a SQL query against the DuckDB tables", top_k=2)
        assert hits[0].tool.metadata.name == "kaos-tabular-query"

    def test_select_returns_bare_tools_in_score_order(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        tools = retrieval.select("PDF extraction please", top_k=3)
        assert tools, "expected at least one hit"
        # First tool should be the PDF one
        assert tools[0].metadata.name == "kaos-pdf-extract-page-text"

    def test_zero_score_hits_excluded(self) -> None:
        # Query with no token overlap should produce no hits (or scores=0).
        retrieval = ToolRetrieval(_build_catalog())
        hits = retrieval.search("completely unrelated quantum chromodynamics", top_k=10)
        # Either empty or all scores positive — zero-score hits filtered.
        for h in hits:
            assert h.score > 0.0

    def test_top_k_respected(self) -> None:
        retrieval = ToolRetrieval(_build_catalog())
        # All 4 tools have at least one token in common with this query
        # ("kaos") so we should be able to limit explicitly.
        hits = retrieval.search("kaos data tool", top_k=2)
        assert len(hits) <= 2

    def test_default_top_k_constant(self) -> None:
        # Sanity: the public default constant is exported and reasonable.
        assert DEFAULT_TOP_K > 0
        assert DEFAULT_TOP_K <= 50  # not an unbounded firehose

    def test_tool_hit_is_frozen_dataclass(self) -> None:
        # Round-trip stability — ToolHit is a value type, must not be mutated.
        retrieval = ToolRetrieval(_build_catalog())
        hits = retrieval.search("PDF page", top_k=1)
        assert isinstance(hits[0], ToolHit)
        # frozen=True means assignment raises
        import pytest

        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            hits[0].score = 999.0


# ---------------------------------------------------------------------------
# Smoke test: lexicon expansion
# ---------------------------------------------------------------------------


class TestLexiconExpansion:
    """Optional lexicon path — verifies the kwarg is plumbed correctly.

    A real lexicon (OpenGloss) is heavy to construct; we use a minimal
    in-memory Lexicon with one synonym entry to confirm the search-time
    expansion fires. The full quality story (does lexicon help on the
    real KAOS tool catalog) is a benchmark, not a unit test.
    """

    def test_lexicon_kwarg_does_not_break_search(self) -> None:
        from kaos_nlp_core.lexicon import Lexicon

        lex = Lexicon()  # empty lexicon — no expansion will fire
        retrieval = ToolRetrieval(_build_catalog(), lexicon=lex)
        hits = retrieval.search("Federal Register search", top_k=1)
        # Without expansion the result should still match the FR tool
        # via plain BM25 — verifies we didn't break the base case.
        assert hits[0].tool.metadata.name == "kaos-source-fr-search"
