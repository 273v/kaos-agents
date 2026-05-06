"""Unit tests for WS-TR.PR-4 extraction MCP tools.

Covers:
- Metadata (names, annotations, input schemas).
- Error handling (missing params, conflicting params, bad JSON).
- Schema resolution (inline schema_json vs recipe_name).
- Verify-tool happy path against a synthetic source.

Live extraction-over-an-LLM is covered in
``tests/integration/test_mcp_extract_live.py``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from kaos_agents.mcp_extract import (
    ExtractCorpusTool,
    ExtractSchemaTool,
    ExtractVerifyTool,
    _resolve_schema,
)
from kaos_agents.recipes import (
    extraction_recipe_names,
    load_extraction_recipe,
    load_extraction_recipes,
)


class TestExtractionRecipes:
    def test_all_recipes_load(self) -> None:
        names = set(extraction_recipe_names())
        assert names == {
            "merger-agreement",
            "spa-deal-points",
            "lease",
            "lpa",
            "court-opinion",
            "privilege-classification",
            "change-of-control",
        }

    def test_recipe_has_expected_shape(self) -> None:
        recipe = load_extraction_recipe("merger-agreement")
        assert recipe is not None
        assert recipe["name"] == "merger-agreement"
        assert "schema" in recipe
        schema = recipe["schema"]
        assert "columns" in schema
        assert len(schema["columns"]) > 0
        # Every column has an id + column_type.
        for col in schema["columns"]:
            assert "id" in col
            assert "column_type" in col

    def test_recipe_schemas_compile(self) -> None:
        """Every shipped recipe must produce a valid ExtractionSchema."""
        from kaos_llm_core.signatures.extraction import ExtractionSchema

        for recipe in load_extraction_recipes():
            schema = ExtractionSchema.from_dict(recipe["schema"])
            assert schema.id
            assert len(schema.columns) > 0
            # to_signature() must not raise for any provenance mode.
            for prov in ("none", "cited", "grounded"):
                Sig = schema.to_signature(provenance=cast(Any, prov))
                assert Sig.__name__.startswith("Extract_")

    def test_unknown_recipe_returns_none(self) -> None:
        assert load_extraction_recipe("not-a-real-recipe") is None


class TestResolveSchemaHelper:
    def test_both_fails(self) -> None:
        spec, err = _resolve_schema(schema_json='{"id":"x"}', recipe_name="lease")
        assert spec is None
        assert err is not None
        assert "exactly one" in err

    def test_neither_fails(self) -> None:
        spec, err = _resolve_schema(schema_json=None, recipe_name=None)
        assert spec is None
        assert err is not None
        assert "Missing schema" in err

    def test_recipe_path(self) -> None:
        spec, err = _resolve_schema(schema_json=None, recipe_name="lease")
        assert err is None
        assert spec is not None
        assert spec["id"] == "lease-v1"

    def test_unknown_recipe(self) -> None:
        spec, err = _resolve_schema(schema_json=None, recipe_name="xyz")
        assert spec is None
        assert err is not None
        assert "Unknown extraction recipe" in err

    def test_inline_json(self) -> None:
        schema_json = json.dumps(
            {"id": "inline", "columns": [{"id": "name", "column_type": "string"}]}
        )
        spec, err = _resolve_schema(schema_json=schema_json, recipe_name=None)
        assert err is None
        assert spec == {
            "id": "inline",
            "columns": [{"id": "name", "column_type": "string"}],
        }

    def test_bad_json(self) -> None:
        spec, err = _resolve_schema(schema_json="{not-json", recipe_name=None)
        assert spec is None
        assert err is not None
        assert "Invalid schema_json" in err


class TestExtractSchemaToolMetadata:
    def test_metadata(self) -> None:
        tool = ExtractSchemaTool()
        meta = tool.metadata
        assert meta.name == "kaos-extract-schema"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.destructiveHint is False
        assert meta.annotations.openWorldHint is False
        assert meta.annotations.idempotentHint is True
        params = {p.name for p in meta.input_schema}
        assert {"text", "schema_json", "recipe_name", "doc_id", "model", "provenance"} <= params

    @pytest.mark.asyncio
    async def test_missing_text_errors(self) -> None:
        tool = ExtractSchemaTool()
        result = await tool.execute({"schema_json": "{}"}, None)
        assert result.isError
        assert "text" in (result.text or "").lower()

    @pytest.mark.asyncio
    async def test_missing_schema_errors(self) -> None:
        tool = ExtractSchemaTool()
        result = await tool.execute({"text": "hello"}, None)
        assert result.isError
        assert "schema" in (result.text or "").lower()


class TestExtractCorpusToolMetadata:
    def test_metadata(self) -> None:
        tool = ExtractCorpusTool()
        meta = tool.metadata
        assert meta.name == "kaos-extract-corpus"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is False
        params = {p.name for p in meta.input_schema}
        assert {
            "corpus_json",
            "output_dir",
            "schema_json",
            "recipe_name",
            "model",
            "provenance",
            "max_concurrency",
        } <= params

    @pytest.mark.asyncio
    async def test_missing_corpus_errors(self) -> None:
        tool = ExtractCorpusTool()
        result = await tool.execute({"output_dir": "/tmp/x"}, None)
        assert result.isError

    @pytest.mark.asyncio
    async def test_bad_corpus_json(self) -> None:
        tool = ExtractCorpusTool()
        result = await tool.execute(
            {"corpus_json": "{not-json", "output_dir": "/tmp/x", "recipe_name": "lease"},
            None,
        )
        assert result.isError
        assert "corpus_json" in (result.text or "").lower()

    @pytest.mark.asyncio
    async def test_corpus_wrong_shape(self) -> None:
        tool = ExtractCorpusTool()
        result = await tool.execute(
            {
                "corpus_json": json.dumps(["doc1", "doc2"]),
                "output_dir": "/tmp/x",
                "recipe_name": "lease",
            },
            None,
        )
        assert result.isError


class TestExtractVerifyTool:
    def test_metadata(self) -> None:
        tool = ExtractVerifyTool()
        meta = tool.metadata
        assert meta.name == "kaos-extract-verify"
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.openWorldHint is False

    @pytest.mark.asyncio
    async def test_verifies_matching_span(self) -> None:
        """Verification happy path — a real span is reported as verified."""
        tool = ExtractVerifyTool()
        source_text = "This Agreement is effective as of January 1, 2025."
        claim = {
            "value": "2025-01-01",
            "spans": [
                {
                    "source_uri": "doc:inline",
                    "char_span": [34, 49],
                    "quote": "January 1, 2025",
                }
            ],
        }
        result = await tool.execute(
            {
                "source_text": source_text,
                "source_uri": "doc:inline",
                "claim_json": json.dumps(claim),
            },
            None,
        )
        assert not result.isError
        output = result.structuredContent
        assert output is not None
        assert output["all_verified"] is True
        assert output["n_spans"] == 1
        assert output["n_verified"] == 1

    @pytest.mark.asyncio
    async def test_detects_fabricated_span(self) -> None:
        """A quote that doesn't appear in the source is flagged."""
        tool = ExtractVerifyTool()
        source_text = "This Agreement is effective as of January 1, 2025."
        claim = {
            "value": "2026-12-31",
            "spans": [
                {
                    "source_uri": "doc:inline",
                    "char_span": [0, 10],
                    "quote": "December 31, 2026",
                }
            ],
        }
        result = await tool.execute(
            {
                "source_text": source_text,
                "source_uri": "doc:inline",
                "claim_json": json.dumps(claim),
            },
            None,
        )
        assert not result.isError
        output = result.structuredContent
        assert output is not None
        assert output["all_verified"] is False
        assert output["n_errors"] >= 1

    @pytest.mark.asyncio
    async def test_missing_source_text_errors(self) -> None:
        tool = ExtractVerifyTool()
        result = await tool.execute({"claim_json": "{}"}, None)
        assert result.isError

    @pytest.mark.asyncio
    async def test_missing_claim_errors(self) -> None:
        tool = ExtractVerifyTool()
        result = await tool.execute({"source_text": "hi"}, None)
        assert result.isError
        assert "claim_json" in (result.text or "").lower()


class TestRegistrationCount:
    def test_register_agent_tools_count(self) -> None:
        """Registering tools should now yield 9 (was 6; +3 from WS-TR.PR-4)."""
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.tools import register_agent_tools

        runtime = KaosRuntime()
        count = register_agent_tools(runtime)
        assert count == 9
