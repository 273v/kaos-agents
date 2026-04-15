"""WS-TR.PR-4 — structured-extraction MCP tools.

Three tools that expose the WS-TR :class:`kaos_llm_core.programs.extract.Extract`
primitive (PR-2) and :func:`extract_corpus` (PR-3) to MCP clients like Claude
Code / Codex / Gemini:

- ``kaos-extract-schema``   — extract a schema from a single document.
- ``kaos-extract-corpus``   — fan a schema out over a corpus (batch_run).
- ``kaos-extract-verify``   — verify a Cited[T]-shaped claim against a source.

All three tools are read-only / closed-world / idempotent — Claude Code
auto-approves them. They return :class:`ToolResult` with structured output;
large results route through the artifact manifest tier per
``docs/guides/mcp-data-flow.md``.

Schema is supplied as inline JSON (``schema_json``) or by recipe name
(``recipe_name``, one of the WS-TR bundled recipes:
``merger-agreement`` / ``spa-deal-points`` / ``lease`` / ``lpa`` /
``court-opinion``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"

# Read-only, closed-world — auto-approved in Claude Code per docs/guides/tool-design.md.
_EXTRACT_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _resolve_schema(
    *,
    schema_json: str | None,
    recipe_name: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve an :class:`ExtractionSchema`-compatible dict from user input.

    Exactly one of ``schema_json`` or ``recipe_name`` must be provided.
    Returns ``(schema_dict, error_message)``. Either is ``None`` — at most
    one will be non-None.
    """
    if schema_json and recipe_name:
        return None, (
            "Pass exactly one of 'schema_json' or 'recipe_name', not both. "
            "Use 'recipe_name' for the WS-TR bundled recipes "
            "(merger-agreement, spa-deal-points, lease, lpa, court-opinion) "
            "or 'schema_json' for a custom inline schema."
        )
    if not schema_json and not recipe_name:
        return None, (
            "Missing schema. Pass either 'recipe_name' (e.g., 'merger-agreement') "
            "or 'schema_json' — an inline JSON string matching "
            "ExtractionSchema.from_dict. See "
            "docs/design/structured-extraction-integration.md §1.4 for shape."
        )

    if recipe_name:
        from kaos_agents.recipes import extraction_recipe_names, load_extraction_recipe

        recipe = load_extraction_recipe(recipe_name)
        if recipe is None:
            available = ", ".join(extraction_recipe_names()) or "<none>"
            return None, (
                f"Unknown extraction recipe '{recipe_name}'. "
                f"Available recipes: {available}. "
                "See kaos_agents/recipes/extraction/ for the full set."
            )
        return recipe.get("schema", {}), None

    # schema_json path — schema_json is non-None here.
    try:
        assert schema_json is not None  # for ty
        spec = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        return None, (
            f"Invalid schema_json: {exc}. "
            "Expected a JSON string parseable by ExtractionSchema.from_dict. "
            "Tip: pass the JSON directly, not a path. "
            'Minimal example: {"id":"x","columns":[{"id":"name","column_type":"string"}]}'
        )
    return spec, None


class ExtractSchemaTool(KaosTool):
    """Extract a schema from a single document via the WS-TR Extract program."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-extract-schema",
            display_name="Extract Schema from Document",
            description=(
                "Run schema-driven structured extraction on one document using "
                "provider-native structured output (OpenAI / Anthropic / Google). "
                "Pass either 'recipe_name' (merger-agreement / spa-deal-points / "
                "lease / lpa / court-opinion) or an inline 'schema_json' string. "
                "Returns one ExtractionCell per column with ai_value, citations, "
                "status, confidence, and cost. For multi-doc fan-out use "
                "kaos-extract-corpus instead."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.EXTRACT,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_EXTRACT_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="text",
                    type="string",
                    description="The full source text to extract from.",
                ),
                ParameterSchema(
                    name="schema_json",
                    type="string",
                    description=(
                        "Inline JSON string matching ExtractionSchema.from_dict. "
                        "Mutually exclusive with recipe_name."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="recipe_name",
                    type="string",
                    description=(
                        "Name of a bundled extraction recipe. Mutually exclusive with schema_json."
                    ),
                    required=False,
                    constraints={
                        "enum": [
                            "merger-agreement",
                            "spa-deal-points",
                            "lease",
                            "lpa",
                            "court-opinion",
                        ]
                    },
                ),
                ParameterSchema(
                    name="doc_id",
                    type="string",
                    description="Stable identifier for this document. Defaults to 'doc-1'.",
                    required=False,
                    default="doc-1",
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "Provider:model identifier (e.g., 'anthropic:claude-haiku-4-5'). "
                        "Defaults to the KAOS_LLM_CORE_DEFAULT_MODEL env var if set."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="provenance",
                    type="string",
                    description=(
                        "Provenance mode: 'none' (bare values, fastest), "
                        "'cited' (values + citations, default), or 'grounded' "
                        "(full verification with retry-on-span-failure)."
                    ),
                    required=False,
                    default="cited",
                    constraints={"enum": ["none", "cited", "grounded"]},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        text = inputs.get("text", "")
        if not text:
            return ToolResult.create_error(
                "Missing 'text' parameter. Provide the source text to extract from. "
                "For multi-doc fan-out, use kaos-extract-corpus instead."
            )

        schema_dict, err = _resolve_schema(
            schema_json=inputs.get("schema_json"),
            recipe_name=inputs.get("recipe_name"),
        )
        if err is not None:
            return ToolResult.create_error(err)

        doc_id = inputs.get("doc_id", "doc-1")
        model = inputs.get("model") or None
        provenance = inputs.get("provenance", "cited")

        try:
            from kaos_llm_core.programs.extract import Extract
            from kaos_llm_core.signatures.extraction import ExtractionSchema
        except ImportError as exc:
            return ToolResult.create_error(
                f"kaos-llm-core is not installed: {exc}. "
                "Install kaos-agents with the [llm] extra or install kaos-llm-core directly. "
                "Fix: uv add kaos-llm-core."
            )

        try:
            schema = ExtractionSchema.from_dict(schema_dict or {})
            extractor = Extract(schema, model=model, provenance=provenance)
            result = await extractor.extract(text=text, doc_id=doc_id)
        except Exception as exc:
            logger.warning("kaos-extract-schema failed: %s", exc, exc_info=True)
            return ToolResult.create_error(
                f"Extraction failed: {exc}. "
                "Check that the provider API key is configured and that the schema is valid. "
                "Alternative: try kaos-llm-core-call for a simpler one-field extraction."
            )

        cells_summary = [
            {
                "column_id": c.column_id,
                "status": c.status,
                "ai_value": c.ai_value,
                "confidence": c.confidence,
                "n_citations": len(c.citations),
            }
            for c in result.cells
        ]
        output = {
            "doc_id": result.doc_id,
            "schema_id": result.schema_id,
            "schema_version": result.schema_version,
            "is_verified": result.is_verified,
            "refused_columns": list(result.refused_columns),
            "cost_usd": result.cost_usd,
            "tokens_total": result.tokens_total,
            "latency_ms": result.latency_ms,
            "cells": cells_summary,
            # Full typed rows round-trip-safe via raw_output_json.
            "raw_output_json": result.raw_output_json,
        }
        n_extracted = sum(1 for c in result.cells if c.status == "extracted")
        summary = (
            f"Extracted {n_extracted}/{len(result.cells)} columns from '{doc_id}' "
            f"(${result.cost_usd:.4f}, {result.latency_ms:.0f}ms)"
        )
        return ToolResult.create_success(output=output, summary=summary)


class ExtractCorpusTool(KaosTool):
    """Fan a schema out over a corpus using the WS-TR extract_corpus primitive."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-extract-corpus",
            display_name="Extract Schema over Corpus",
            description=(
                "Fan a schema out over a corpus of documents via batch_run "
                "(resumable, content-addressed, cost-attributed). Pass the corpus "
                "as a JSON-encoded dict of {doc_id: text}. Returns an aggregate "
                "result + the path where the JSONL log and manifest live. A "
                "second invocation with the same output_dir skips completed docs. "
                "For a single document, use kaos-extract-schema instead."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.EXTRACT,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_EXTRACT_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="corpus_json",
                    type="string",
                    description=(
                        "JSON-encoded object {doc_id: text}, e.g., "
                        '{"doc-1": "first source...", "doc-2": "second source..."}. '
                        "Each value is the full source text for that doc."
                    ),
                ),
                ParameterSchema(
                    name="output_dir",
                    type="string",
                    description=(
                        "Directory where the items.jsonl log and manifest are "
                        "written. Pass an absolute path. Same path on a subsequent "
                        "call triggers resume."
                    ),
                ),
                ParameterSchema(
                    name="schema_json",
                    type="string",
                    description="Inline JSON schema. Mutually exclusive with recipe_name.",
                    required=False,
                ),
                ParameterSchema(
                    name="recipe_name",
                    type="string",
                    description="Bundled recipe name. Mutually exclusive with schema_json.",
                    required=False,
                    constraints={
                        "enum": [
                            "merger-agreement",
                            "spa-deal-points",
                            "lease",
                            "lpa",
                            "court-opinion",
                        ]
                    },
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description="Provider:model identifier. Default: KAOS_LLM_CORE_DEFAULT_MODEL.",
                    required=False,
                ),
                ParameterSchema(
                    name="provenance",
                    type="string",
                    description="'none' | 'cited' (default) | 'grounded'.",
                    required=False,
                    default="cited",
                    constraints={"enum": ["none", "cited", "grounded"]},
                ),
                ParameterSchema(
                    name="max_concurrency",
                    type="integer",
                    description="Max parallel LLM calls. Default 8.",
                    required=False,
                    default=8,
                    constraints={"min": 1, "max": 64},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        corpus_json = inputs.get("corpus_json", "")
        output_dir = inputs.get("output_dir", "")
        if not corpus_json:
            return ToolResult.create_error(
                "Missing 'corpus_json'. Pass a JSON object {doc_id: text}."
            )
        if not output_dir:
            return ToolResult.create_error(
                "Missing 'output_dir'. Pass an absolute path where the batch log "
                "should be written. The directory is created if missing."
            )
        try:
            corpus = json.loads(corpus_json)
        except json.JSONDecodeError as exc:
            return ToolResult.create_error(
                f"Invalid corpus_json: {exc}. "
                'Expected JSON object mapping doc_id to text, e.g., {"doc-1": "..."}.'
            )
        if not isinstance(corpus, dict) or not corpus:
            return ToolResult.create_error(
                "corpus_json must be a non-empty JSON object {doc_id: text}."
            )
        for k, v in corpus.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return ToolResult.create_error(
                    "Every corpus entry must be str → str. "
                    f"Got key type {type(k).__name__}, value type {type(v).__name__}."
                )

        schema_dict, err = _resolve_schema(
            schema_json=inputs.get("schema_json"),
            recipe_name=inputs.get("recipe_name"),
        )
        if err is not None:
            return ToolResult.create_error(err)

        model = inputs.get("model") or None
        provenance = inputs.get("provenance", "cited")
        max_concurrency = int(inputs.get("max_concurrency", 8))

        try:
            from kaos_llm_core.programs.extract import extract_corpus
            from kaos_llm_core.signatures.extraction import ExtractionSchema
        except ImportError as exc:
            return ToolResult.create_error(
                f"kaos-llm-core is not installed: {exc}. "
                "Install kaos-agents with the [llm] extra or install kaos-llm-core directly."
            )

        try:
            schema = ExtractionSchema.from_dict(schema_dict or {})
            result = await extract_corpus(
                schema,
                corpus,
                output_dir=output_dir,
                model=model,
                provenance=provenance,
                max_concurrency=max_concurrency,
            )
        except Exception as exc:
            logger.warning("kaos-extract-corpus failed: %s", exc, exc_info=True)
            return ToolResult.create_error(
                f"Corpus extraction failed: {exc}. "
                "Check provider API key and output_dir permissions. "
                "Alternative: run kaos-extract-schema per document instead."
            )

        # Summarize rows without blowing past the 16KB inline threshold.
        row_summary = [
            {
                "doc_id": r.doc_id,
                "is_verified": r.is_verified,
                "refused_columns": list(r.refused_columns),
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "n_cells": len(r.cells),
                "n_extracted": sum(1 for c in r.cells if c.status == "extracted"),
            }
            for r in result.rows
        ]
        output = {
            "schema_id": result.schema_id,
            "schema_version": result.schema_version,
            "output_dir": result.output_dir,
            "n_docs_processed": result.n_docs_processed,
            "n_docs_errored": result.n_docs_errored,
            "n_docs_skipped": result.n_docs_skipped,
            "cost_usd": result.cost_usd,
            "tokens_total": result.tokens_total,
            "rows": row_summary,
            "table_row_count": (result.table.tables[0].row_count if result.table.tables else 0),
        }
        summary = (
            f"Extracted schema '{result.schema_id}' over {result.n_docs_processed} docs "
            f"({result.n_docs_errored} errors, {result.n_docs_skipped} skipped) "
            f"— ${result.cost_usd:.4f}, {result.tokens_total} tokens. "
            f"Log: {result.output_dir}/items.jsonl"
        )
        return ToolResult.create_success(output=output, summary=summary)


class ExtractVerifyTool(KaosTool):
    """Verify a Cited[T]-shaped claim against a source text."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-extract-verify",
            display_name="Verify Cited Claim",
            description=(
                "Verify that a claim's quoted spans actually appear in the provided "
                "source text. Takes a claim (value + spans) and source_text, tries "
                "the WS-1 match strategies (STRICT → SUBSTRING → CASE_INSENSITIVE "
                "→ NORMALIZED_TOKEN → FUZZY_CHAR → FUZZY_TOKEN), returns per-span "
                "verification results. Useful for post-hoc audit of any cited "
                "extraction or for standalone claim grounding."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_EXTRACT_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="source_text",
                    type="string",
                    description="The source text to verify claims against.",
                ),
                ParameterSchema(
                    name="source_uri",
                    type="string",
                    description="URI identifying the source (used in span bookkeeping).",
                    required=False,
                    default="doc:inline",
                ),
                ParameterSchema(
                    name="claim_json",
                    type="string",
                    description=(
                        "JSON-encoded Cited[T] claim: "
                        '{"value": <any>, "spans": [{"source_uri":"...","char_span":[s,e],"quote":"..."}]}. '
                        "Spans must use the matching source_uri."
                    ),
                ),
                ParameterSchema(
                    name="threshold",
                    type="number",
                    description="Fuzzy-strategy similarity threshold in [0, 1]. Default 0.9.",
                    required=False,
                    default=0.9,
                    constraints={"min": 0.0, "max": 1.0},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        source_text = inputs.get("source_text", "")
        source_uri = inputs.get("source_uri", "doc:inline")
        claim_json = inputs.get("claim_json", "")
        threshold = float(inputs.get("threshold", 0.9))

        if not source_text:
            return ToolResult.create_error("Missing 'source_text'.")
        if not claim_json:
            return ToolResult.create_error(
                "Missing 'claim_json'. Pass a JSON-encoded Cited[T] claim. "
                'Example: {"value":"Acme","spans":[{"source_uri":"doc:inline",'
                '"char_span":[0,4],"quote":"Acme"}]}.'
            )

        try:
            claim = json.loads(claim_json)
        except json.JSONDecodeError as exc:
            return ToolResult.create_error(
                f"Invalid claim_json: {exc}. Must be a JSON object with 'value' and 'spans' fields."
            )

        try:
            from typing import Any as _Any

            from kaos_llm_core.signatures.grounding import (
                DEFAULT_MATCH_STRATEGIES,
                Cited,
            )

            # Parametrize on Any — the Cited wrapper's verification only
            # inspects spans, not value. Keeping T=Any lets the tool accept
            # claims of any shape without a type-juggling round-trip.
            cited = Cited[_Any].model_validate(claim)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to parse claim as Cited[T]: {exc}. "
                'Shape: {"value": <any>, "spans": [{"source_uri":"...", '
                '"char_span":[start,end], "quote":"..."}]}. '
                "spans must be a non-empty list."
            )

        corpus = {source_uri: source_text}
        errors = cited.verify(
            corpus,
            strategies=DEFAULT_MATCH_STRATEGIES,
            threshold=threshold,
        )

        per_span = [
            {
                "span_index": i,
                "quote": s.quote,
                "source_uri": s.source_uri,
                "char_span": list(s.char_span),
                "verified": not any(e.span_index == i for e in errors),
            }
            for i, s in enumerate(cited.spans)
        ]
        output = {
            "value": cited.value,
            "n_spans": len(cited.spans),
            "n_verified": sum(1 for row in per_span if row["verified"]),
            "n_errors": len(errors),
            "all_verified": len(errors) == 0,
            "per_span": per_span,
            "errors": [{"span_index": e.span_index, "reason": e.reason} for e in errors],
        }
        summary = f"{output['n_verified']}/{output['n_spans']} spans verified" + (
            " (all verified)" if output["all_verified"] else f" ({len(errors)} failed)"
        )
        return ToolResult.create_success(output=output, summary=summary)
