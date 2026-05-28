"""Unit tests for ``kaos-agent-design-extraction`` (PR-1a).

The tool wraps ``kaos_llm_core.programs.designers.design_schema``.
Unit tests pin the contract surface (input validation, output shape,
error handling) with a mocked ``design_schema`` so this suite stays
fast + deterministic. The live integration test that exercises the
real designer on real persona prompts is run manually per the plan
§7 step-3 gate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.tools.design_extraction import (
    AgentDesignExtractionTool,
    _block_text,
    _doc_head_text,
)


def _err_text(result: Any) -> str:
    """Lift the first content item's text payload off a ToolResult.

    The ``ToolResult.content`` union includes non-text content types
    (image, audio, embedded resource, resource link) that ty cannot
    narrow at the call site. Tests only ever produce text content,
    so accessing ``.text`` via ``getattr`` with a fallback keeps the
    runtime behavior while satisfying ty's union check.
    """
    return str(getattr(result.content[0], "text", ""))


def _ok_inputs(**overrides: Any) -> dict[str, Any]:
    """Default-good inputs for the tool. By default ``design_only=True``
    so unit tests can exercise the designer leg without mocking the
    per-document fan-out's ``Call(Extract_<schema>).invoke``. Fan-out
    behavior is tested explicitly in :class:`TestFanOut`.
    """
    base: dict[str, Any] = {
        "question": "What is the governing law of each contract?",
        "artifact_ids": ["doc-1", "doc-2", "doc-3"],
        "design_only": True,
    }
    base.update(overrides)
    return base


def _fake_schema(*, schema_id: str = "sd_abc123", n_cols: int = 3) -> Any:
    """Build a minimal ExtractionSchema-shaped object for tests.

    Avoids importing kaos-llm-core in the test file's top scope so
    these tests run even without the [llm] extra installed.
    """
    from kaos_llm_core.signatures.extraction import ColumnSpec, ExtractionSchema

    columns = tuple(
        ColumnSpec(
            id=f"col_{i}",
            label=f"Col {i}",
            column_type="string",
            description=f"Column {i} description",
            required=True,
        )
        for i in range(n_cols)
    )
    return ExtractionSchema(id=schema_id, version=1, columns=columns)


def _make_context(runtime: Any = None) -> Any:
    """Build a minimal `KaosContext` for tests.

    The unit tests only need ``context.runtime`` to be truthy; the
    cheapest way to satisfy ty's type check is to instantiate the
    real KaosContext with a placeholder runtime.
    """
    from kaos_core.base.context import KaosContext

    return KaosContext(
        session_id="test-session",
        runtime=runtime if runtime is not None else object(),
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_missing_context_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute(_ok_inputs(), context=None)
        assert result.isError
        assert "runtime context" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_missing_runtime_errors(self) -> None:
        ctx = _make_context()
        ctx.runtime = None  # type: ignore[assignment]
        tool = AgentDesignExtractionTool()
        result = await tool.execute(_ok_inputs(), context=ctx)
        assert result.isError

    @pytest.mark.asyncio
    async def test_missing_question_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute({"artifact_ids": ["doc-1"]}, context=_make_context())
        assert result.isError
        assert "question" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_blank_question_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute(_ok_inputs(question="   "), context=_make_context())
        assert result.isError

    @pytest.mark.asyncio
    async def test_missing_artifact_ids_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute({"question": "what governing law?"}, context=_make_context())
        assert result.isError
        assert "artifact_ids" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_empty_artifact_ids_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute(_ok_inputs(artifact_ids=[]), context=_make_context())
        assert result.isError

    @pytest.mark.asyncio
    async def test_non_list_artifact_ids_errors(self) -> None:
        tool = AgentDesignExtractionTool()
        result = await tool.execute(_ok_inputs(artifact_ids="doc-1"), context=_make_context())
        assert result.isError


# ---------------------------------------------------------------------
# Happy path — schema returned, structured content matches
# ---------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_schema_in_structured_content(self) -> None:
        fake_doc = type("_Doc", (), {"body": ()})()  # empty body — no sample
        fake_schema = _fake_schema(schema_id="sd_test_happy", n_cols=2)
        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=fake_schema),
            ),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(artifact_ids=["a", "b", "c"]),
                context=_make_context(),
            )
        assert not result.isError
        sc = result.structuredContent
        assert sc is not None
        assert sc["schema_id"] == "sd_test_happy"
        assert sc["schema_version"] == 1
        assert len(sc["columns"]) == 2
        assert sc["columns"][0]["id"] == "col_0"
        assert sc["columns"][0]["column_type"] == "string"
        assert sc["columns"][0]["required"] is True
        assert sc["artifacts_sampled"] == 3
        assert sc["artifacts_requested"] == 3
        assert "cost_usd" in sc
        assert "total_tokens" in sc

    @pytest.mark.asyncio
    async def test_designer_invoked_with_concatenated_corpus_sample(self) -> None:
        """The designer receives one corpus_sample string with all
        ``=== <id> ===`` headers. Pin the formatting so downstream
        designer-quality measurement can assume the contract."""
        fake_doc = type("_Doc", (), {"body": ()})()
        fake_schema = _fake_schema()
        spy_design = AsyncMock(return_value=fake_schema)
        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch("kaos_llm_core.programs.designers.design_schema", new=spy_design),
        ):
            tool = AgentDesignExtractionTool()
            await tool.execute(
                _ok_inputs(artifact_ids=["doc-1", "doc-2"]),
                context=_make_context(),
            )
        assert spy_design.await_count == 1
        assert spy_design.await_args is not None
        kwargs = spy_design.await_args.kwargs
        sample = kwargs["corpus_sample"]
        assert "=== doc-1 ===" in sample
        assert "=== doc-2 ===" in sample
        assert kwargs["question"] == "What is the governing law of each contract?"
        # schema_id NOT passed — relies on kaos-llm-core's auto-derivation
        assert "schema_id" not in kwargs

    @pytest.mark.asyncio
    async def test_explicit_model_passed_through(self) -> None:
        fake_doc = type("_Doc", (), {"body": ()})()
        fake_schema = _fake_schema()
        spy_design = AsyncMock(return_value=fake_schema)
        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch("kaos_llm_core.programs.designers.design_schema", new=spy_design),
        ):
            tool = AgentDesignExtractionTool()
            await tool.execute(
                _ok_inputs(model="openai:gpt-5.4-mini"),
                context=_make_context(),
            )
        assert spy_design.await_args is not None
        assert spy_design.await_args.kwargs["model"] == "openai:gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_domain_hint_passed_through(self) -> None:
        fake_doc = type("_Doc", (), {"body": ()})()
        fake_schema = _fake_schema()
        spy_design = AsyncMock(return_value=fake_schema)
        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch("kaos_llm_core.programs.designers.design_schema", new=spy_design),
        ):
            tool = AgentDesignExtractionTool()
            await tool.execute(
                _ok_inputs(domain_hint="mutual NDAs"),
                context=_make_context(),
            )
        assert spy_design.await_args is not None
        assert spy_design.await_args.kwargs["domain_hint"] == "mutual NDAs"


# ---------------------------------------------------------------------
# Error handling — load failures + designer failures degrade cleanly
# ---------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_all_artifacts_unloadable_errors(self) -> None:
        with patch(
            "kaos_content.artifacts.load_document",
            new=AsyncMock(side_effect=RuntimeError("not found")),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(artifact_ids=["bad-1", "bad-2"]),
                context=_make_context(),
            )
        assert result.isError
        assert "loadable" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_partial_loadable_proceeds(self) -> None:
        """If 1 of 3 loads, the schema is still designed against the
        loadable subset. The result reports the count mismatch so the
        caller can decide whether the partial sample was sufficient."""
        fake_doc = type("_Doc", (), {"body": ()})()
        fake_schema = _fake_schema()

        load_calls = {"n": 0}

        async def fake_load(aid: str, _runtime: Any) -> Any:
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                return fake_doc
            raise RuntimeError("not found")

        with (
            patch("kaos_content.artifacts.load_document", new=fake_load),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=fake_schema),
            ),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(artifact_ids=["good", "bad-1", "bad-2"]),
                context=_make_context(),
            )
        assert not result.isError
        assert result.structuredContent is not None
        assert result.structuredContent["artifacts_sampled"] == 1
        assert result.structuredContent["artifacts_requested"] == 3

    @pytest.mark.asyncio
    async def test_designer_failure_errors_with_actionable_message(self) -> None:
        fake_doc = type("_Doc", (), {"body": ()})()
        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(side_effect=RuntimeError("provider 500")),
            ),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())
        assert result.isError
        msg = _err_text(result).lower()
        # error message must include (a) what failed (b) how to fix
        assert "designer" in msg or "schemadesigner" in msg
        assert "anthropic_api_key" in msg or "verify" in msg


# ---------------------------------------------------------------------
# Helpers — _doc_head_text + _block_text
# ---------------------------------------------------------------------


class TestDocHeadText:
    def test_empty_body_returns_empty_string(self) -> None:
        doc = type("_Doc", (), {"body": ()})()
        assert _doc_head_text(doc, max_chars=100) == ""

    def test_truncates_at_max_chars(self) -> None:
        block = type("_B", (), {"text": "x" * 1000})()
        doc = type("_Doc", (), {"body": (block,)})()
        out = _doc_head_text(doc, max_chars=50)
        assert len(out) == 50

    def test_concatenates_multiple_blocks(self) -> None:
        b1 = type("_B", (), {"text": "Alpha"})()
        b2 = type("_B", (), {"text": "Beta"})()
        doc = type("_Doc", (), {"body": (b1, b2)})()
        out = _doc_head_text(doc, max_chars=100)
        assert "Alpha" in out
        assert "Beta" in out

    def test_stops_when_budget_exhausted(self) -> None:
        b1 = type("_B", (), {"text": "ABCDEFGHIJ"})()  # 10 chars
        b2 = type("_B", (), {"text": "SHOULD_NOT_APPEAR"})()
        doc = type("_Doc", (), {"body": (b1, b2)})()
        out = _doc_head_text(doc, max_chars=10)
        assert out == "ABCDEFGHIJ"
        assert "SHOULD_NOT_APPEAR" not in out


class TestBlockText:
    def test_direct_text_attribute(self) -> None:
        b = type("_B", (), {"text": "hello"})()
        assert _block_text(b) == "hello"

    def test_walks_children_with_value(self) -> None:
        child = type("_Text", (), {"value": "world"})()
        block = type("_B", (), {"children": (child,)})()
        assert _block_text(block) == "world"

    def test_nested_children(self) -> None:
        leaf = type("_Text", (), {"value": "leaf"})()
        inner = type("_B", (), {"children": (leaf,)})()
        outer = type("_B", (), {"children": (inner,)})()
        assert _block_text(outer) == "leaf"

    def test_empty_block_returns_empty_string(self) -> None:
        block = type("_B", (), {"children": ()})()
        assert _block_text(block) == ""


# ---------------------------------------------------------------------
# Metadata sanity — registration contract
# ---------------------------------------------------------------------


class TestMetadata:
    def test_name_is_dashed_three_segments(self) -> None:
        tool = AgentDesignExtractionTool()
        assert tool.metadata.name == "kaos-agent-design-extraction"
        assert tool.metadata.name.count("-") >= 2

    def test_required_inputs_declared(self) -> None:
        tool = AgentDesignExtractionTool()
        names = {p.name for p in tool.metadata.input_schema}
        assert "question" in names
        assert "artifact_ids" in names

    def test_optional_inputs_marked(self) -> None:
        tool = AgentDesignExtractionTool()
        by_name = {p.name: p for p in tool.metadata.input_schema}
        assert by_name["domain_hint"].required is False
        assert by_name["designer_model"].required is False
        assert by_name["extract_model"].required is False
        assert by_name["design_only"].required is False
        assert by_name["max_concurrency"].required is False

    def test_annotations_lock_destructive_off(self) -> None:
        tool = AgentDesignExtractionTool()
        ann = tool.metadata.annotations
        assert ann is not None
        assert ann.destructiveHint is False
        # readOnlyHint=False because this spends money on an LLM call
        assert ann.readOnlyHint is False


# ---------------------------------------------------------------------
# Fan-out — per-document extracts + stamp_source_uri JOIN
# ---------------------------------------------------------------------


class TestFanOut:
    """PR-1b: the tool fans out per-document Call(Extract_<schema>) in
    parallel and stamps source_uri on every Cited cell. These tests
    mock Call.invoke so the suite stays fast + deterministic; the
    live integration test that exercises the real fan-out on real
    NDA fixtures is the §7 step-4 gate.
    """

    @staticmethod
    def _make_invocation(cells: dict[str, Any], *, cost: float = 0.01, tokens: int = 500) -> Any:
        """Build a fake Invocation-shaped object matching what
        Call.invoke returns: ``.output`` is an attribute bag of the
        runtime Extract Signature's output fields; ``.usage`` carries
        cost + tokens.
        """
        output_obj = type("_Output", (), cells)()
        usage_obj = type("_Usage", (), {"cost_usd": cost, "total_tokens": tokens})()
        return type("_Inv", (), {"output": output_obj, "usage": usage_obj})()

    @staticmethod
    def _cited(value: str, *, source_uri: str = "LLM-FABRICATED.docx") -> Any:
        """Build a Cited[str] with an LLM-emitted source_uri the
        dispatcher should override."""
        from kaos_llm_core.signatures.grounding import Cited, Span

        return Cited[str](
            value=value,
            spans=[
                Span(
                    source_uri=source_uri,
                    char_span=(0, len(value)),
                    quote=value,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_fan_out_returns_one_row_per_doc(self) -> None:
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=2)
        cited = self._cited("Delaware")
        inv = self._make_invocation({"col_0": cited, "col_1": cited})

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return inv

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(
                    artifact_ids=["doc-A", "doc-B", "doc-C"],
                    design_only=False,
                ),
                context=_make_context(),
            )

        assert not result.isError, _err_text(result)
        sc = result.structuredContent
        assert sc is not None
        assert sc["execution_mode"] == "full"
        assert sc["row_count"] == 3
        assert len(sc["rows"]) == 3
        for row in sc["rows"]:
            assert row["artifact_id"] in ("doc-A", "doc-B", "doc-C")
            assert set(row["cells"].keys()) == {"col_0", "col_1"}

    @pytest.mark.asyncio
    async def test_source_uri_is_stamped_from_artifact_id(self) -> None:
        """The dispatcher MUST override the LLM-emitted source_uri
        with the artifact_id the dispatcher passed to the Call.
        This is the central P3-class fix from the plan §3.6.
        """
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=1)
        cited = self._cited(
            "Delaware governs",
            source_uri="LLM-FABRICATED-NOT-REAL.docx",
        )
        inv = self._make_invocation({"col_0": cited})

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return inv

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(
                    artifact_ids=["EMNA Mutual NDA.docx"],
                    design_only=False,
                ),
                context=_make_context(),
            )

        sc = result.structuredContent
        assert sc is not None
        row = sc["rows"][0]
        cell = row["cells"]["col_0"]
        # Every span carries the dispatcher-assigned source_uri,
        # NOT the LLM-emitted one.
        for span in cell["spans"]:
            assert span["source_uri"] == "EMNA Mutual NDA.docx"
            assert span["source_uri"] != "LLM-FABRICATED-NOT-REAL.docx"
        # The Cited value itself is preserved.
        assert cell["value"] == "Delaware governs"

    @pytest.mark.asyncio
    async def test_null_cells_counted_when_field_optional(self) -> None:
        """When the extraction returns None for an optional field, the
        tool surfaces it as None in the row dict and bumps
        null_cell_count so the agent can decide whether to retry.
        """
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=3)
        # Two cells valid, one None
        cited = self._cited("X")
        inv = self._make_invocation({"col_0": cited, "col_1": None, "col_2": cited})

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return inv

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(artifact_ids=["doc-A"], design_only=False),
                context=_make_context(),
            )

        sc = result.structuredContent
        assert sc is not None
        assert sc["null_cell_count"] == 1
        assert sc["rows"][0]["cells"]["col_1"] is None
        assert sc["rows"][0]["cells"]["col_0"] is not None
        assert sc["rows"][0]["cells"]["col_2"] is not None

    @pytest.mark.asyncio
    async def test_extract_failure_per_doc_surfaces_error_row(self) -> None:
        """One doc's extract failing must not abort the whole run.
        The failure becomes a row with all-null cells + ``error``."""
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=2)
        good_cited = self._cited("OK")
        good_inv = self._make_invocation({"col_0": good_cited, "col_1": good_cited})

        # Per-doc dispatch decides return-vs-raise based on call order.
        call_n = {"n": 0}

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            call_n["n"] += 1
            if call_n["n"] == 2:
                raise RuntimeError("provider 500 on doc-B")
            return good_inv

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(
                    artifact_ids=["doc-A", "doc-B", "doc-C"],
                    design_only=False,
                    max_concurrency=1,  # force deterministic order
                ),
                context=_make_context(),
            )

        sc = result.structuredContent
        assert sc is not None
        assert sc["row_count"] == 3
        assert sc["failed_doc_count"] == 1
        # 2 null cells (from the failed doc) + 0 from the good docs
        assert sc["null_cell_count"] == 2
        failed_rows = [r for r in sc["rows"] if "error" in r]
        assert len(failed_rows) == 1
        assert "provider 500" in failed_rows[0]["error"]

    @pytest.mark.asyncio
    async def test_cost_aggregated_across_doc_extracts(self) -> None:
        """The result's ``extraction_cost_usd`` is the sum of per-doc
        usage; ``cost_usd`` is designer + extraction."""
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=1)
        cited = self._cited("X")
        inv = self._make_invocation({"col_0": cited}, cost=0.05, tokens=1234)

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return inv

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(
                    artifact_ids=["a", "b", "c", "d"],
                    design_only=False,
                ),
                context=_make_context(),
            )

        sc = result.structuredContent
        assert sc is not None
        # 4 docs at $0.05 each = $0.20 extraction
        assert abs(sc["extraction_cost_usd"] - 0.20) < 1e-6
        assert abs(sc["total_tokens"] - 4 * 1234) < 1
        # designer is still 0.0 (design_schema doesn't expose usage today)
        assert sc["designer_cost_usd"] == 0.0
        # total = designer + extraction
        assert abs(sc["cost_usd"] - 0.20) < 1e-6

    @pytest.mark.asyncio
    async def test_design_only_skips_fan_out(self) -> None:
        """``design_only=True`` returns the schema with rows=[] and
        execution_mode="design_only". No Call.invoke happens."""
        fake_doc = type("_Doc", (), {"body": ()})()
        schema = _fake_schema(n_cols=2)
        invoke_called = {"n": 0}

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            invoke_called["n"] += 1
            return self._make_invocation({"col_0": self._cited("X"), "col_1": self._cited("Y")})

        from kaos_llm_core.programs.call import Call

        with (
            patch(
                "kaos_content.artifacts.load_document",
                new=AsyncMock(return_value=fake_doc),
            ),
            patch(
                "kaos_llm_core.programs.designers.design_schema",
                new=AsyncMock(return_value=schema),
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentDesignExtractionTool()
            result = await tool.execute(
                _ok_inputs(artifact_ids=["doc-A"], design_only=True),
                context=_make_context(),
            )

        sc = result.structuredContent
        assert sc is not None
        assert sc["execution_mode"] == "design_only"
        assert sc["rows"] == []
        assert invoke_called["n"] == 0
