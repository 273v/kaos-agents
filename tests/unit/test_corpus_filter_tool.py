"""Unit tests for the K8 kaos-agent-corpus-filter MCP tool."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from kaos_content.artifacts import store_document
from kaos_content.model.document import ContentDocument
from kaos_content.shortcuts import heading, paragraph
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.results import ToolResult

from kaos_agents.tools.corpus_filter import AgentCorpusFilterTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> KaosRuntime:
    return KaosRuntime()


@pytest.fixture
def context(runtime: KaosRuntime) -> KaosContext:
    return KaosContext.create(session_id="test", runtime=runtime)


def _nda_doc(label: str) -> ContentDocument:
    return ContentDocument(
        body=(
            heading(1, f"Mutual NDA ({label})"),
            paragraph(
                "Confidential Information includes business plans, financial "
                "projections, and customer lists."
            ),
            paragraph("The Term is twenty-four months from the Effective Date."),
        ),
    )


async def _store(doc: ContentDocument, ctx: KaosContext, name: str) -> str:
    manifest = await store_document(doc, ctx.runtime, ctx, name=name)
    return manifest.artifact_id


def _payload(result: ToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None
    return result.structuredContent


def _error_text(result: ToolResult) -> str:
    first = result.content[0]
    text = getattr(first, "text", None)
    assert text is not None
    return str(text)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self) -> None:
        tool = AgentCorpusFilterTool()
        assert tool.metadata.name == "kaos-agent-corpus-filter"

    def test_annotations_not_read_only(self) -> None:
        tool = AgentCorpusFilterTool()
        ann = tool.metadata.annotations
        assert ann is not None
        # This tool spends money — must not auto-approve.
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False


# ---------------------------------------------------------------------------
# Execute — input validation (no LLM)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_context(self) -> None:
        tool = AgentCorpusFilterTool()
        result = asyncio.run(tool.execute({"intent": "x", "artifact_ids": ["a"]}, None))
        assert result.isError

    def test_missing_intent(self, context: KaosContext) -> None:
        tool = AgentCorpusFilterTool()
        result = asyncio.run(tool.execute({"artifact_ids": ["a"]}, context))
        assert result.isError
        assert "intent" in _error_text(result).lower()

    def test_missing_artifact_ids(self, context: KaosContext) -> None:
        tool = AgentCorpusFilterTool()
        result = asyncio.run(tool.execute({"intent": "x"}, context))
        assert result.isError
        assert "artifact_ids" in _error_text(result)

    def test_max_keep_zero_rejected(self, context: KaosContext) -> None:
        tool = AgentCorpusFilterTool()
        result = asyncio.run(
            tool.execute(
                {"intent": "x", "artifact_ids": ["a"], "max_keep": 0},
                context,
            ),
        )
        assert result.isError
        assert "max_keep" in _error_text(result)


# ---------------------------------------------------------------------------
# Execute — no loadable artifacts (no LLM call)
# ---------------------------------------------------------------------------


class TestNoLoadableArtifacts:
    def test_all_artifacts_missing(self, context: KaosContext) -> None:
        """All artifact_ids fail to load → tool returns success with
        empty kept/dropped, total_loadable=0. Does NOT call the LLM."""
        tool = AgentCorpusFilterTool()
        result = asyncio.run(
            tool.execute(
                {
                    "intent": "find anything",
                    "artifact_ids": ["fake-1", "fake-2"],
                },
                context,
            ),
        )
        assert not result.isError
        out = _payload(result)
        assert out["total_input"] == 2
        assert out["total_loadable"] == 0
        assert out["kept"] == []
        assert out["dropped"] == []
        assert out["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Execute — happy path (stubbed LLM)
# ---------------------------------------------------------------------------


async def _stub_filter_keep_first(
    *,
    intent: str,
    artifacts: list[dict[str, Any]],
    max_keep: int,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Stub: keep the first artifact, drop the rest."""
    if not artifacts:
        return [], [], 0.0
    kept = [
        {
            "artifact_id": artifacts[0]["id"],
            "relevance": 0.95,
            "reasoning": "stub kept first",
        },
    ]
    dropped = [{"artifact_id": a["id"], "reason": "stub dropped"} for a in artifacts[1:]]
    return kept, dropped, 0.002


class TestHappyPath:
    def test_filter_returns_kept_and_dropped(self, context: KaosContext) -> None:
        async def _go() -> Any:
            a1 = await _store(_nda_doc("Acme"), context, name="acme")
            a2 = await _store(_nda_doc("Beta"), context, name="beta")
            a3 = await _store(_nda_doc("Gamma"), context, name="gamma")

            tool = AgentCorpusFilterTool()
            with patch(
                "kaos_agents.tools.corpus_filter._run_corpus_filter_llm",
                side_effect=_stub_filter_keep_first,
            ):
                return (
                    a1,
                    await tool.execute(
                        {
                            "intent": "find confidential information clauses",
                            "artifact_ids": [a1, a2, a3],
                        },
                        context,
                    ),
                )

        first_id, result = asyncio.run(_go())
        assert not result.isError
        out = _payload(result)
        assert out["total_input"] == 3
        assert out["total_loadable"] == 3
        assert len(out["kept"]) == 1
        assert len(out["dropped"]) == 2
        assert out["kept"][0]["artifact_id"] == first_id
        assert out["kept"][0]["relevance"] == 0.95
        assert out["cost_usd"] == pytest.approx(0.002)

    def test_unloadable_artifacts_skipped(self, context: KaosContext) -> None:
        async def _go() -> Any:
            a1 = await _store(_nda_doc("Acme"), context, name="acme")
            tool = AgentCorpusFilterTool()
            with patch(
                "kaos_agents.tools.corpus_filter._run_corpus_filter_llm",
                side_effect=_stub_filter_keep_first,
            ):
                return await tool.execute(
                    {"intent": "x", "artifact_ids": [a1, "bogus-id"]},
                    context,
                )

        result = asyncio.run(_go())
        out = _payload(result)
        assert out["total_input"] == 2
        assert out["total_loadable"] == 1


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tool_registered_via_register_agent_tools(self) -> None:
        from kaos_agents.tools.registry import register_agent_tools

        rt = KaosRuntime()
        register_agent_tools(rt)
        names = [t.metadata.name for t in rt.tools.list_tool_objects()]
        assert "kaos-agent-corpus-filter" in names
