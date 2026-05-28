"""Unit tests for ``kaos-agent-interpret-extraction`` (PR-2).

The tool composes:

1. :class:`AgentDesignExtractionTool` — typed extraction (mocked here
   so unit tests don't pay for real LLM calls)
2. :class:`InterpretExtractionSignature` — synthesizer (mocked Call)

Tests pin the loop control surface — convergence, max_iters cap,
budget cap, schema augmentation between iterations, cumulative row
merging, error handling at each phase. The live integration test
(against real NDA fixtures) lives outside the unit gate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.tools.interpret_extraction import (
    AgentInterpretExtractionTool,
    _project_rows_for_synth,
)


def _err_text(result: Any) -> str:
    """Lift first content item's text payload off a ToolResult."""
    if not result.content:
        return ""
    return str(getattr(result.content[0], "text", "") or "")


def _make_context() -> Any:
    """Real ``KaosContext`` bound to ``KaosRuntime.test_mode()`` — the
    in-memory + GLOBAL-isolation default per kaos-agents/CLAUDE.md."""
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime

    runtime = KaosRuntime.test_mode()
    return KaosContext(session_id="test", runtime=runtime)


def _ok_inputs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question": "Summarize the governing law of each contract.",
        "artifact_ids": ["doc-A", "doc-B"],
    }
    base.update(overrides)
    return base


def _fake_extract_result(
    *,
    columns: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    cost_usd: float = 0.05,
    is_error: bool = False,
    error_text: str = "",
) -> Any:
    """Build a ToolResult-shaped object matching design_extraction's
    return contract."""
    from kaos_core.types.results import ToolResult

    if is_error:
        return ToolResult.create_error(error_text)

    return ToolResult.create_success(
        output={
            "schema_id": "sd_fake",
            "columns": columns
            or [
                {
                    "id": "jurisdiction",
                    "column_type": "string",
                    "description": "GL",
                    "required": True,
                },
            ],
            "rows": rows
            or [
                {
                    "artifact_id": "doc-A",
                    "cells": {"jurisdiction": {"value": "Delaware", "spans": []}},
                },
                {
                    "artifact_id": "doc-B",
                    "cells": {"jurisdiction": {"value": "Michigan", "spans": []}},
                },
            ],
            "row_count": 2,
            "null_cell_count": 0,
            "failed_doc_count": 0,
            "cost_usd": cost_usd,
        },
    )


def _make_invocation(
    *,
    memo: str = "Both contracts use US state law.",
    score: int = 9,
    needs_more: bool = False,
    requested: tuple[str, ...] = (),
    cost: float = 0.02,
) -> Any:
    """Build a fake Invocation matching what Call.invoke returns."""
    output_obj = type(
        "_Out",
        (),
        {
            "memo": memo,
            "score": score,
            "needs_more_extraction": needs_more,
            "requested_columns": requested,
        },
    )()
    usage_obj = type("_Usage", (), {"cost_usd": cost, "total_tokens": 500})()
    return type("_Inv", (), {"output": output_obj, "usage": usage_obj})()


# ─── Input validation ─────────────────────────────────────────────────


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_missing_question_errors(self) -> None:
        tool = AgentInterpretExtractionTool()
        result = await tool.execute({"artifact_ids": ["a"]}, context=_make_context())
        assert result.isError
        assert "question" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_empty_question_errors(self) -> None:
        tool = AgentInterpretExtractionTool()
        result = await tool.execute(
            {"question": "   ", "artifact_ids": ["a"]}, context=_make_context()
        )
        assert result.isError

    @pytest.mark.asyncio
    async def test_missing_artifact_ids_errors(self) -> None:
        tool = AgentInterpretExtractionTool()
        result = await tool.execute({"question": "x"}, context=_make_context())
        assert result.isError
        assert "artifact_ids" in _err_text(result).lower()

    @pytest.mark.asyncio
    async def test_artifact_ids_not_a_list_errors(self) -> None:
        tool = AgentInterpretExtractionTool()
        result = await tool.execute(
            {"question": "x", "artifact_ids": "not-a-list"}, context=_make_context()
        )
        assert result.isError

    @pytest.mark.asyncio
    async def test_no_runtime_errors(self) -> None:
        tool = AgentInterpretExtractionTool()
        result = await tool.execute(_ok_inputs(), context=None)
        assert result.isError
        assert "runtime" in _err_text(result).lower()


# ─── Convergence — synthesizer says done on iter 1 ───────────────────


class TestConvergeImmediately:
    """When the first synthesizer pass returns needs_more=false, the
    loop must stop after iter 1 — no second extraction call."""

    @pytest.mark.asyncio
    async def test_converges_at_iter_1_when_needs_more_false(self) -> None:
        from kaos_llm_core.programs.call import Call

        extract_mock = AsyncMock(return_value=_fake_extract_result())
        invoke_mock = AsyncMock(
            return_value=_make_invocation(
                memo="Final memo",
                score=10,
                needs_more=False,
                requested=(),
            )
        )

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=extract_mock,
            ),
            patch.object(Call, "invoke", new=invoke_mock),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())

        assert not result.isError, _err_text(result)
        assert extract_mock.call_count == 1, "second extract must not run on convergence"
        sc = result.structuredContent
        assert sc is not None
        assert sc is not None
        assert sc["loop_status"] == "converged"
        assert sc["converged_at_iter"] == 1
        assert sc["iterations_run"] == 1
        assert sc["memo"] == "Final memo"
        assert sc["score"] == 10


# ─── Iteration — synthesizer requests more columns ───────────────────


class TestIteration:
    """When the synthesizer reports needs_more=true with requested
    columns, the loop must:

    1. Make a second extraction call
    2. The second call's ``question`` must include the augmenting hint
       so the schema designer focuses on the requested columns
    3. Merge the new columns + cells into the cumulative state
    4. Stop when the second synth call returns needs_more=false
    """

    @pytest.mark.asyncio
    async def test_iter_2_extract_question_includes_augmenting_hint(self) -> None:
        from kaos_llm_core.programs.call import Call

        # Track what question the extractor sees on each call.
        seen_questions: list[str] = []

        async def fake_extract(self_, inputs: dict[str, Any], *, context: Any = None) -> Any:
            seen_questions.append(inputs["question"])
            return _fake_extract_result()

        # First synth: needs_more=True with proposals. Second: done.
        invoke_results = [
            _make_invocation(needs_more=True, requested=("gov_law: the state",)),
            _make_invocation(needs_more=False),
        ]
        invoke_iter = iter(invoke_results)

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return next(invoke_iter)

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=fake_extract,
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())

        assert not result.isError, _err_text(result)
        assert len(seen_questions) == 2
        # First call: bare question; second: includes augmenting hint
        assert "follow-up" not in seen_questions[0]
        assert "gov_law" in seen_questions[1]
        assert "Iteration 2" in seen_questions[1]
        sc = result.structuredContent
        assert sc is not None
        assert sc["iterations_run"] == 2
        assert sc["converged_at_iter"] == 2

    @pytest.mark.asyncio
    async def test_max_iters_caps_loop(self) -> None:
        """When the synthesizer keeps signaling needs_more, the loop
        must stop at ``max_iters`` and report loop_status=max_iters_reached."""
        from kaos_llm_core.programs.call import Call

        async def fake_extract(*_args: Any, **_kwargs: Any) -> Any:
            return _fake_extract_result()

        async def always_needs_more(*_args: Any, **_kwargs: Any) -> Any:
            return _make_invocation(needs_more=True, requested=("col_x: foo",))

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=fake_extract,
            ),
            patch.object(Call, "invoke", new=always_needs_more),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(max_iters=2), context=_make_context())

        assert not result.isError, _err_text(result)
        sc = result.structuredContent
        assert sc is not None
        assert sc["loop_status"] == "max_iters_reached"
        assert sc["converged_at_iter"] is None
        assert sc["iterations_run"] == 2

    @pytest.mark.asyncio
    async def test_budget_cap_stops_loop(self) -> None:
        """When cumulative cost reaches the budget, the loop breaks
        BEFORE the next iteration — even if the synthesizer would keep
        asking for more."""
        from kaos_llm_core.programs.call import Call

        async def fake_extract(*_args: Any, **_kwargs: Any) -> Any:
            # Big extract cost so we blow the budget on iter 1
            return _fake_extract_result(cost_usd=0.50)

        async def always_needs_more(*_args: Any, **_kwargs: Any) -> Any:
            return _make_invocation(needs_more=True, requested=("col_x: foo",), cost=0.10)

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=fake_extract,
            ),
            patch.object(Call, "invoke", new=always_needs_more),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(
                _ok_inputs(max_iters=10, budget_usd=0.40),
                context=_make_context(),
            )

        assert not result.isError, _err_text(result)
        sc = result.structuredContent
        assert sc is not None
        assert sc["loop_status"] == "budget_exhausted"
        assert sc["converged_at_iter"] is None
        assert sc["iterations_run"] == 1
        assert sc["cost_usd"] >= 0.40


# ─── Cumulative state merging ─────────────────────────────────────────


class TestCumulativeMerge:
    @pytest.mark.asyncio
    async def test_columns_union_across_iterations(self) -> None:
        from kaos_llm_core.programs.call import Call

        # Iter 1 returns cols [a]; iter 2 returns cols [b]. Final should
        # have both.
        extract_results = [
            _fake_extract_result(
                columns=[
                    {"id": "a", "column_type": "string", "description": "A", "required": True}
                ],
                rows=[{"artifact_id": "doc-A", "cells": {"a": {"value": "v1"}}}],
            ),
            _fake_extract_result(
                columns=[
                    {"id": "b", "column_type": "string", "description": "B", "required": True}
                ],
                rows=[{"artifact_id": "doc-A", "cells": {"b": {"value": "v2"}}}],
            ),
        ]
        extract_iter = iter(extract_results)

        async def fake_extract(*_args: Any, **_kwargs: Any) -> Any:
            return next(extract_iter)

        invoke_iter = iter(
            [
                _make_invocation(needs_more=True, requested=("b: B",)),
                _make_invocation(needs_more=False),
            ]
        )

        async def fake_invoke(*_args: Any, **_kwargs: Any) -> Any:
            return next(invoke_iter)

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=fake_extract,
            ),
            patch.object(Call, "invoke", new=fake_invoke),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())

        assert not result.isError, _err_text(result)
        sc = result.structuredContent
        assert sc is not None
        merged = sc["extracted"]
        col_ids = {c["id"] for c in merged["columns"]}
        assert col_ids == {"a", "b"}, f"expected both cols, got {col_ids}"
        # The single row should have both cells now
        assert merged["row_count"] == 1
        row_cells = merged["rows"][0]["cells"]
        assert "a" in row_cells and "b" in row_cells


# ─── Error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_first_iter_extract_fails_surfaces_error(self) -> None:
        async def fake_extract(*_args: Any, **_kwargs: Any) -> Any:
            return _fake_extract_result(is_error=True, error_text="extract failed for reasons")

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
            new=fake_extract,
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())

        assert result.isError
        assert (
            "extract failed" in _err_text(result).lower()
            or "design_extraction" in _err_text(result).lower()
        )

    @pytest.mark.asyncio
    async def test_synth_call_failure_surfaces_error(self) -> None:
        from kaos_llm_core.programs.call import Call

        async def fake_extract(*_args: Any, **_kwargs: Any) -> Any:
            return _fake_extract_result()

        async def synth_raises(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("synth boom")

        with (
            patch(
                "kaos_agents.tools.interpret_extraction.AgentDesignExtractionTool.execute",
                new=fake_extract,
            ),
            patch.object(Call, "invoke", new=synth_raises),
        ):
            tool = AgentInterpretExtractionTool()
            result = await tool.execute(_ok_inputs(), context=_make_context())

        assert result.isError
        text = _err_text(result).lower()
        assert "synth" in text or "interpretextraction" in text


# ─── Helper: _project_rows_for_synth ─────────────────────────────────


class TestProjection:
    def test_drops_spans_keeps_values(self) -> None:
        rows = {
            "doc-A": {
                "artifact_id": "doc-A",
                "cells": {
                    "x": {"value": "hello", "spans": [{"source_uri": "doc-A", "text": "hello"}]},
                },
            },
        }
        cols = {"x": {"id": "x", "description": "field x"}}
        out = _project_rows_for_synth(rows, cols)
        import json

        parsed = json.loads(out)
        assert parsed["rows"][0]["cells"]["x"] == "hello"
        # Spans should be dropped from the synth input
        assert "spans" not in str(parsed["rows"][0])

    def test_handles_null_cells(self) -> None:
        rows = {"doc-A": {"artifact_id": "doc-A", "cells": {"x": None}}}
        cols = {"x": {"id": "x", "description": "x"}}
        out = _project_rows_for_synth(rows, cols)
        import json

        parsed = json.loads(out)
        assert parsed["rows"][0]["cells"]["x"] is None

    def test_handles_raw_value_cells(self) -> None:
        """Some cells may already be raw values (not dicts)."""
        rows = {"doc-A": {"artifact_id": "doc-A", "cells": {"x": "raw-string"}}}
        cols = {"x": {"id": "x", "description": "x"}}
        out = _project_rows_for_synth(rows, cols)
        import json

        parsed = json.loads(out)
        assert parsed["rows"][0]["cells"]["x"] == "raw-string"


# ─── Metadata sanity ──────────────────────────────────────────────────


class TestMetadata:
    def test_name_is_dashed_three_segments(self) -> None:
        tool = AgentInterpretExtractionTool()
        assert tool.metadata.name == "kaos-agent-interpret-extraction"

    def test_inputs_include_required_and_optional(self) -> None:
        tool = AgentInterpretExtractionTool()
        by_name = {p.name: p for p in tool.metadata.input_schema}
        assert by_name["question"].required is True
        assert by_name["artifact_ids"].required is True
        assert by_name["domain_hint"].required is False
        assert by_name["deliverable_hint"].required is False
        assert by_name["max_iters"].required is False
        assert by_name["budget_usd"].required is False

    def test_annotations_cost_incurring_no_writes(self) -> None:
        tool = AgentInterpretExtractionTool()
        ann = tool.metadata.annotations
        assert ann is not None
        assert ann.destructiveHint is False
        # Spends money on LLM calls → readOnlyHint=False
        assert ann.readOnlyHint is False
