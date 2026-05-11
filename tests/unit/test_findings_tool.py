"""Unit tests for the K7 kaos-agent-findings MCP tool."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kaos_agents.patterns import findings as findings_mod
from kaos_agents.patterns.findings import FilteredFinding, FindingCandidate
from kaos_agents.tools.findings import AgentFindingsTool, _build_selector

# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_name(self) -> None:
        tool = AgentFindingsTool()
        assert tool.metadata.name == "kaos-agent-findings"

    def test_annotations_not_read_only(self) -> None:
        """This tool spends money. readOnlyHint=False prevents auto-approval."""
        tool = AgentFindingsTool()
        ann = tool.metadata.annotations
        assert ann is not None
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False
        assert ann.openWorldHint is False

    def test_input_schema_has_required_fields(self) -> None:
        tool = AgentFindingsTool()
        params = {p.name for p in tool.metadata.input_schema}
        assert "artifact_id" in params
        assert "question" in params
        assert "select_by" in params
        assert "selector_arg" in params
        assert "filter_model" in params
        assert "synthesis_model" in params

    def test_select_by_enum_constraint(self) -> None:
        tool = AgentFindingsTool()
        select_by = next(p for p in tool.metadata.input_schema if p.name == "select_by")
        assert "enum" in select_by.constraints
        assert set(select_by.constraints["enum"]) == {
            "every_sentence",
            "token",
            "entity",
        }


# ---------------------------------------------------------------------------
# _build_selector — input validation
# ---------------------------------------------------------------------------


class TestBuildSelector:
    def test_every_sentence(self) -> None:
        sel = _build_selector("every_sentence", None)
        assert callable(sel)

    def test_token_requires_arg(self) -> None:
        with pytest.raises(ValueError, match="select_by='token' requires"):
            _build_selector("token", None)
        with pytest.raises(ValueError, match="select_by='token' requires"):
            _build_selector("token", "")
        with pytest.raises(ValueError, match="select_by='token' requires"):
            _build_selector("token", "   ")

    def test_token_with_arg(self) -> None:
        sel = _build_selector("token", "indemnif")
        assert callable(sel)

    def test_entity_requires_arg(self) -> None:
        with pytest.raises(ValueError, match="select_by='entity' requires"):
            _build_selector("entity", None)

    def test_entity_with_arg(self) -> None:
        sel = _build_selector("entity", "dates")
        assert callable(sel)

    def test_unknown_select_by(self) -> None:
        with pytest.raises(ValueError, match="Unknown select_by"):
            _build_selector("nonsense", None)


# ---------------------------------------------------------------------------
# Execute — error paths (no LLM)
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    def test_missing_context(self) -> None:
        tool = AgentFindingsTool()
        result = asyncio.run(tool.execute({"artifact_id": "x", "question": "q"}, None))
        assert result.isError
        first = result.content[0]
        assert "runtime context" in getattr(first, "text", "")

    def test_missing_artifact_id(self) -> None:
        """KaosContext with a runtime but no artifact_id."""
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        ctx = KaosContext.create(session_id="s", runtime=KaosRuntime())
        tool = AgentFindingsTool()
        result = asyncio.run(tool.execute({"question": "q"}, ctx))
        assert result.isError
        first = result.content[0]
        assert "artifact_id" in getattr(first, "text", "")

    def test_missing_question(self) -> None:
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        ctx = KaosContext.create(session_id="s", runtime=KaosRuntime())
        tool = AgentFindingsTool()
        result = asyncio.run(tool.execute({"artifact_id": "x"}, ctx))
        assert result.isError
        first = result.content[0]
        assert "question" in getattr(first, "text", "")

    def test_invalid_select_by(self) -> None:
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        ctx = KaosContext.create(session_id="s", runtime=KaosRuntime())
        tool = AgentFindingsTool()
        result = asyncio.run(
            tool.execute(
                {"artifact_id": "x", "question": "q", "select_by": "nonsense"},
                ctx,
            ),
        )
        assert result.isError
        first = result.content[0]
        assert "Invalid select_by" in getattr(first, "text", "")

    def test_token_select_without_arg(self) -> None:
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        ctx = KaosContext.create(session_id="s", runtime=KaosRuntime())
        tool = AgentFindingsTool()
        result = asyncio.run(
            tool.execute(
                {"artifact_id": "x", "question": "q", "select_by": "token"},
                ctx,
            ),
        )
        assert result.isError
        first = result.content[0]
        assert "selector_arg" in getattr(first, "text", "")

    def test_unloadable_artifact(self) -> None:
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        ctx = KaosContext.create(session_id="s", runtime=KaosRuntime())
        tool = AgentFindingsTool()
        result = asyncio.run(
            tool.execute(
                {"artifact_id": "definitely-not-real", "question": "q"},
                ctx,
            ),
        )
        assert result.isError
        first = result.content[0]
        assert "Failed to load" in getattr(first, "text", "")


# ---------------------------------------------------------------------------
# Execute — happy path (stub LLM)
# ---------------------------------------------------------------------------


async def _stub_filter_keep_all(
    chunk: tuple[FindingCandidate, ...],
    **_kwargs: Any,
) -> tuple[tuple[FilteredFinding, ...], float]:
    return (
        tuple(FilteredFinding(candidate=c, relevance=0.9, reasoning="ok") for c in chunk),
        0.001,
    )


async def _stub_synthesize(
    *,
    question: str,
    findings: tuple[FilteredFinding, ...],
    model: str,
) -> tuple[str, float]:
    cited = " ".join(f"[{f.candidate.finding_id}]" for f in findings)
    return f"Synthesized: {cited}", 0.005


class TestExecuteHappyPath:
    def test_full_pipeline_with_stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end via the tool surface — load → run agent → return payload.

        Patches the LLM helpers so the test is deterministic and free.
        """
        from kaos_content.artifacts import store_document
        from kaos_content.model.document import ContentDocument
        from kaos_content.shortcuts import paragraph
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_keep_all)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize)

        runtime = KaosRuntime()
        ctx = KaosContext.create(session_id="s", runtime=runtime)

        # Store a small doc with the substring we'll select on.
        doc = ContentDocument(
            body=(
                paragraph("Indemnification carve-outs apply for gross negligence."),
                paragraph("The Term is 24 months from the Effective Date."),
                paragraph("Indemnification is capped at $100,000."),
            ),
        )

        async def _go() -> Any:
            manifest = await store_document(doc, runtime, ctx, name="findings-test")
            tool = AgentFindingsTool()
            return await tool.execute(
                {
                    "artifact_id": manifest.artifact_id,
                    "question": "What about indemnification?",
                    "select_by": "token",
                    "selector_arg": "indemnif",
                },
                ctx,
            )

        result = asyncio.run(_go())
        assert not result.isError
        out = result.structuredContent
        assert out is not None
        # Two sentences contain 'indemnif'
        assert out["total_enumerated"] == 2
        # Stub filter kept all
        assert out["total_filtered"] == 2
        # 1 filter call (chunk_size default 20, 2 candidates)
        assert out["filter_calls"] == 1
        assert "Synthesized" in out["answer"]
        assert len(out["findings"]) == 2
        for f in out["findings"]:
            assert "indemnif" in f["text"].lower()
            assert f["relevance"] == 0.9
            assert f["reasoning"] == "ok"
            assert "finding_id" in f


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tool_registers(self) -> None:
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.tools.registry import register_agent_tools

        rt = KaosRuntime()
        register_agent_tools(rt)
        names = [t.metadata.name for t in rt.tools.list_tool_objects()]
        assert "kaos-agent-findings" in names
