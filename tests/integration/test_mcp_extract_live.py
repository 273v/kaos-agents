"""Live integration tests for WS-TR.PR-4 extraction MCP tools.

Exercises the 3 tools end-to-end against a real Anthropic API:
- kaos-extract-schema with an inline schema
- kaos-extract-schema with a bundled recipe (court-opinion) on a short fixture
- kaos-extract-corpus with a 2-doc inline corpus
- kaos-extract-verify against a real claim + fabricated claim

Per CLAUDE.md: this is the acceptance gate. Unit tests alone are not
sufficient.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kaos_agents.tools.extract import (
    ExtractCorpusTool,
    ExtractSchemaTool,
    ExtractVerifyTool,
)

pytestmark = pytest.mark.live

requires_anthropic = pytest.mark.skipif(
    not (os.getenv("KAOS_LLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
    reason="Anthropic API key not set",
)


CONTRACT_TEXT = (
    "SOFTWARE LICENSE AGREEMENT\n\n"
    "This Software License Agreement (the 'Agreement') is made and entered into "
    "effective as of January 15, 2025 (the 'Effective Date'), by and between "
    "Acme Corporation, a Delaware corporation ('Licensor'), and Beta LLC, "
    "a Texas limited liability company ('Licensee').\n\n"
    "GOVERNING LAW. This Agreement shall be governed by and construed in "
    "accordance with the laws of the State of Delaware."
)

SHORT_OPINION_TEXT = (
    "SUPREME COURT OF THE UNITED STATES\n"
    "No. 759. — Argued February 28, 1966. — Decided June 13, 1966.\n\n"
    "MIRANDA v. ARIZONA\n\n"
    "Appeal from the Supreme Court of Arizona.\n\n"
    "MR. CHIEF JUSTICE WARREN delivered the opinion of the Court.\n\n"
    "The prosecution may not use statements, whether exculpatory or "
    "inculpatory, stemming from custodial interrogation of the defendant "
    "unless it demonstrates the use of procedural safeguards effective to "
    "secure the privilege against self-incrimination. The judgment of the "
    "Supreme Court of Arizona is reversed."
)


@pytest.mark.integration
@requires_anthropic
class TestExtractSchemaToolLive:
    async def test_inline_schema_end_to_end(self) -> None:
        tool = ExtractSchemaTool()
        schema = {
            "id": "contract-smoke",
            "columns": [
                {"id": "parties", "column_type": "list", "constraints": {"inner": "string"}},
                {"id": "effective_date", "column_type": "date"},
                {"id": "governing_law", "column_type": "string"},
            ],
        }
        result = await tool.execute(
            {
                "text": CONTRACT_TEXT,
                "schema_json": json.dumps(schema),
                "doc_id": "doc-1",
                "model": "anthropic:claude-haiku-4-5",
                "provenance": "none",
            },
            None,
        )
        assert not result.isError, result.text
        output = result.structuredContent
        assert output is not None
        assert output["doc_id"] == "doc-1"
        assert len(output["cells"]) == 3
        by_id = {c["column_id"]: c for c in output["cells"]}
        # Contents must appear in the extracted values.
        parties_str = json.dumps(by_id["parties"]["ai_value"])
        assert "Acme" in parties_str
        assert "Beta" in parties_str
        assert "Delaware" in str(by_id["governing_law"]["ai_value"])

    async def test_recipe_name_end_to_end(self) -> None:
        tool = ExtractSchemaTool()
        result = await tool.execute(
            {
                "text": SHORT_OPINION_TEXT,
                "recipe_name": "court-opinion",
                "doc_id": "miranda-case",
                "model": "anthropic:claude-haiku-4-5",
                "provenance": "none",
            },
            None,
        )
        assert not result.isError, result.text
        output = result.structuredContent
        assert output is not None
        assert output["schema_id"] == "court-opinion-v2"
        by_id = {c["column_id"]: c for c in output["cells"]}
        # case_name should include Miranda.
        case_name = by_id.get("case_name", {}).get("ai_value")
        assert case_name is not None
        assert "Miranda" in str(case_name)
        # court should include Supreme Court.
        court = by_id.get("court", {}).get("ai_value")
        assert court is not None
        assert "Supreme Court" in str(court)


@pytest.mark.integration
@requires_anthropic
class TestExtractCorpusToolLive:
    async def test_2_doc_corpus(self, tmp_path: Path) -> None:
        tool = ExtractCorpusTool()
        corpus = {
            "doc-1": CONTRACT_TEXT,
            "doc-2": SHORT_OPINION_TEXT,
        }
        # Keep schema small and applicable to both docs.
        schema = {
            "id": "tiny-corpus-schema",
            "columns": [
                {
                    "id": "document_type",
                    "column_type": "string",
                    "description": (
                        "The document type (e.g., 'Software License Agreement', 'court opinion')."
                    ),
                },
                {
                    "id": "primary_entity",
                    "column_type": "string",
                    "description": "The primary named entity (organization, party, or case name).",
                },
            ],
        }
        result = await tool.execute(
            {
                "corpus_json": json.dumps(corpus),
                "output_dir": str(tmp_path / "mcp-corpus"),
                "schema_json": json.dumps(schema),
                "model": "anthropic:claude-haiku-4-5",
                "provenance": "none",
                "max_concurrency": 2,
            },
            None,
        )
        assert not result.isError, result.text
        output = result.structuredContent
        assert output is not None
        assert output["n_docs_processed"] == 2
        assert output["n_docs_errored"] == 0
        assert output["table_row_count"] == 2


@pytest.mark.integration
class TestExtractVerifyToolLive:
    """No LLM required — verification is deterministic."""

    async def test_real_claim_verified(self) -> None:
        tool = ExtractVerifyTool()
        text = "This Agreement is effective as of January 15, 2025."
        claim = {
            "value": "2025-01-15",
            "spans": [
                {
                    "source_uri": "doc:contract",
                    "char_span": [34, 50],
                    "quote": "January 15, 2025",
                }
            ],
        }
        result = await tool.execute(
            {
                "source_text": text,
                "source_uri": "doc:contract",
                "claim_json": json.dumps(claim),
            },
            None,
        )
        assert not result.isError
        output = result.structuredContent
        assert output is not None
        assert output["all_verified"] is True

    async def test_fabricated_claim_flagged(self) -> None:
        tool = ExtractVerifyTool()
        text = "This Agreement is effective as of January 15, 2025."
        claim = {
            "value": "December 31, 1999",
            "spans": [
                {
                    "source_uri": "doc:contract",
                    "char_span": [0, 18],
                    "quote": "December 31, 1999",
                }
            ],
        }
        result = await tool.execute(
            {
                "source_text": text,
                "source_uri": "doc:contract",
                "claim_json": json.dumps(claim),
            },
            None,
        )
        assert not result.isError
        output = result.structuredContent
        assert output is not None
        assert output["all_verified"] is False
        assert output["n_errors"] >= 1
