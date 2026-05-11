"""Labeled benchmark queries for tool-retrieval evaluation.

Each entry maps a *natural-language query* (the kind a real user might
type into an agent) to the set of *correct tool names* a retrieval
system should surface. The catalog covers 4 KAOS modules:
``kaos-pdf``, ``kaos-web``, ``kaos-tabular``, ``kaos-office`` — 50
tools total when all four are loaded.

Three categories:

- ``direct``     — high lexical overlap with the tool name/description.
                    Plain BM25 should crush these. Sanity floor.
- ``synonym``    — the query uses a synonym that doesn't appear in any
                    tool's text. Lexicon expansion is expected to help.
                    Multi-query may or may not.
- ``conceptual`` — the query describes the task at a higher level
                    of abstraction than the tool descriptions. The LLM
                    needs to bridge from goal → tool vocabulary. Multi-
                    query is expected to help.

Truth labels were derived by hand from the catalog (see test
fixtures `_make_real_catalog_runtime` for the loader). Keep this list
under version control — anchoring eval to a fixed labeled set is the
only way the BEIR-style cross-domain check from
``feedback_benchmark_first.md`` makes sense for tool retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    """One benchmark query with ground-truth tool labels."""

    query: str
    relevant: frozenset[str]
    category: str  # "direct" | "synonym" | "conceptual"


BENCHMARK_QUERIES: tuple[LabeledQuery, ...] = (
    # =====================================================================
    # DIRECT — straightforward lexical overlap.
    # =====================================================================
    LabeledQuery(
        query="extract text from a PDF page",
        relevant=frozenset(
            {
                "kaos-pdf-extract-page-text",
                "kaos-pdf-extract-parse",
            }
        ),
        category="direct",
    ),
    LabeledQuery(
        query="render a PDF page as an image",
        relevant=frozenset({"kaos-pdf-render-page"}),
        category="direct",
    ),
    LabeledQuery(
        query="get PDF metadata",
        relevant=frozenset({"kaos-pdf-metadata"}),
        category="direct",
    ),
    LabeledQuery(
        query="search the web with a query",
        relevant=frozenset({"kaos-web-search"}),
        category="direct",
    ),
    LabeledQuery(
        query="fetch a URL and get plain text",
        relevant=frozenset(
            {
                "kaos-web-get-text",
                "kaos-web-fetch-page",
                "kaos-web-get-markdown",
            }
        ),
        category="direct",
    ),
    LabeledQuery(
        query="extract HTML tables from a webpage",
        relevant=frozenset({"kaos-web-get-tables"}),
        category="direct",
    ),
    LabeledQuery(
        query="run a SQL query against a registered table",
        relevant=frozenset(
            {
                "kaos-tabular-query",
                "kaos-tabular-filter",
            }
        ),
        category="direct",
    ),
    LabeledQuery(
        query="join two tables on a shared key column",
        relevant=frozenset({"kaos-tabular-join"}),
        category="direct",
    ),
    LabeledQuery(
        query="extract text from a DOCX file",
        relevant=frozenset(
            {
                "kaos-office-get-text",
                "kaos-office-parse-docx",
                "kaos-office-get-markdown",
            }
        ),
        category="direct",
    ),
    LabeledQuery(
        query="list sheets in an Excel workbook",
        relevant=frozenset({"kaos-office-list-sheets-xlsx"}),
        category="direct",
    ),
    # =====================================================================
    # SYNONYM — query uses lexical alternatives. Lexicon should help.
    # =====================================================================
    LabeledQuery(
        query="download a webpage",  # 'download' not in any tool text → 'fetch'
        relevant=frozenset(
            {
                "kaos-web-fetch-page",
                "kaos-web-get-text",
                "kaos-web-get-markdown",
            }
        ),
        category="synonym",
    ),
    LabeledQuery(
        query="look up information online",  # 'look up' → 'search'
        relevant=frozenset({"kaos-web-search"}),
        category="synonym",
    ),
    LabeledQuery(
        query="get the document properties",  # 'properties' → 'metadata'
        relevant=frozenset(
            {
                "kaos-pdf-metadata",
                "kaos-office-metadata",
                "kaos-office-xlsx-metadata",
                "kaos-web-get-metadata",
            }
        ),
        category="synonym",
    ),
    LabeledQuery(
        query="convert a spreadsheet into a table I can query",  # spreadsheet → xlsx
        relevant=frozenset(
            {
                "kaos-office-parse-xlsx",
                "kaos-office-get-sheet-xlsx",
                "kaos-tabular-register",
                "kaos-tabular-read-file",
            }
        ),
        category="synonym",
    ),
    LabeledQuery(
        query="show me the document outline",  # outline → bookmarks / TOC
        relevant=frozenset({"kaos-pdf-get-outline"}),
        category="synonym",
    ),
    LabeledQuery(
        query="how many rows are in the table",  # row count → 'count'
        relevant=frozenset({"kaos-tabular-count"}),
        category="synonym",
    ),
    LabeledQuery(
        query="show me sample data from this table",  # sample → 'sample'/'top-k'/'describe'
        relevant=frozenset(
            {
                "kaos-tabular-sample",
                "kaos-tabular-describe",
                "kaos-tabular-top-k",
            }
        ),
        category="synonym",
    ),
    # =====================================================================
    # CONCEPTUAL — task-level intent, requires bridging from goal vocab
    # to tool vocab. Multi-query (LLM paraphrasing) should help.
    # =====================================================================
    LabeledQuery(
        query="I want to find duplicate customer records",
        relevant=frozenset({"kaos-tabular-find-duplicates"}),
        category="conceptual",
    ),
    LabeledQuery(
        query="reshape my data from wide to long format",
        relevant=frozenset({"kaos-tabular-unpivot"}),
        category="conceptual",
    ),
    LabeledQuery(
        query="show me the relationship between two columns",  # → correlation
        relevant=frozenset({"kaos-tabular-correlation"}),
        category="conceptual",
    ),
    LabeledQuery(
        query="I need slide-by-slide notes from my deck",  # → slide notes
        relevant=frozenset(
            {
                "kaos-office-get-slide-notes",
                "kaos-office-list-slides",
                "kaos-office-parse-pptx",
            }
        ),
        category="conceptual",
    ),
    LabeledQuery(
        query="categorize each page of this scan as blank or text",  # classify-page
        relevant=frozenset({"kaos-pdf-classify-page"}),
        category="conceptual",
    ),
    LabeledQuery(
        query="give me the top sales by quarter",  # top-k + groupby
        relevant=frozenset(
            {
                "kaos-tabular-top-k",
                "kaos-tabular-aggregate",
            }
        ),
        category="conceptual",
    ),
    LabeledQuery(
        query="find passages in the report mentioning revenue",  # search-document
        relevant=frozenset(
            {
                "kaos-pdf-search-document",
                "kaos-office-search",
            }
        ),
        category="conceptual",
    ),
    LabeledQuery(
        query="grab the navigation menu from this page",  # links classified as nav
        relevant=frozenset({"kaos-web-get-links"}),
        category="conceptual",
    ),
)


def queries_by_category() -> dict[str, tuple[LabeledQuery, ...]]:
    """Group the benchmark queries by their category for slice reporting."""
    out: dict[str, list[LabeledQuery]] = {}
    for q in BENCHMARK_QUERIES:
        out.setdefault(q.category, []).append(q)
    return {k: tuple(v) for k, v in out.items()}


__all__ = ["BENCHMARK_QUERIES", "LabeledQuery", "queries_by_category"]
