"""Live integration test — kaos-office (XLSX) + kaos-tabular (DuckDB) + kaos-agent-chat.

This is the "lawyer says build me a dashboard" composition: parse an
Excel P&L, register it with the session's DuckDB engine, hand the
agent the ``kaos-tabular-query`` tool, and ask it which product line
saw the largest Q4-vs-Q1 margin compression.

The fixture is generated **in-process** via ``kaos_office.xlsx``
writer (we deliberately avoid ``openpyxl`` because it is not
available in the kaos-agents venv — the native ``write_xlsx``
serializer round-trips through ``parse_xlsx`` correctly and gives us
a real binary XLSX on disk).

Composition under test:
1. Build a ``TabularDocument`` with a deterministic 5-product P&L.
2. ``write_xlsx`` → real ``.xlsx`` on disk.
3. ``parse_xlsx`` → ``TabularDocument`` (round-trip).
4. Register both with the **session-scoped** ``TabularEngine`` so the
   MCP ``kaos-tabular-query`` tool sees the same tables (the engine is
   keyed by ``context.session_id`` via ``SESSION_REGISTRY``).
5. Ground-truth: run the YoY-margin SQL directly via the engine to
   pin the right answer.
6. Hand the agent the chat tool with ``tool_filter="kaos-tabular-*"``
   and ask the natural-language question.
7. Assert the answer names the right product line, cites a
   coordinate, and the F2 recorder's captured cost is bounded.

Skipped without ``ANTHROPIC_API_KEY``. Bounded to one short ReAct
loop on Haiku — typical spend < $0.05 per run.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.integration._models import respond_model

MODEL = respond_model()

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)


# ---------------------------------------------------------------------------
# Fixture: deterministic Q1/Q4 P&L workbook
# ---------------------------------------------------------------------------


# Hand-picked so the YoY margin compression has a unique, unambiguous
# winner — Gizmos drops from a 50% margin in Q1 to a 20% margin in Q4
# (-30 pp). Widgets and Gadgets stay roughly flat / improve.
PNL_ROWS: tuple[tuple[str, float, float, float, float], ...] = (
    # (product, q1_revenue, q1_cogs, q4_revenue, q4_cogs)
    ("Widgets", 100.0, 60.0, 120.0, 80.0),  # 40% -> 33% : -7 pp
    ("Gizmos", 200.0, 100.0, 250.0, 200.0),  # 50% -> 20% : -30 pp  <-- the answer
    ("Gadgets", 150.0, 90.0, 180.0, 100.0),  # 40% -> 44% : +4 pp
    ("Doodads", 80.0, 40.0, 95.0, 50.0),  # 50% -> 47% : -3 pp
    ("Sprockets", 120.0, 72.0, 140.0, 90.0),  # 40% -> 36% : -4 pp
)


def _build_pnl_xlsx(out_path: Path) -> str:
    """Write a deterministic P&L workbook and return the sheet name as
    DuckDB will see it after parse_xlsx round-trip.

    The native xlsx writer normalizes table names to title case
    (e.g. ``pnl_2024`` → ``Pnl 2024``). Picking a name that survives
    normalization to itself (``PnL2024``) keeps the SQL readable.
    """
    from kaos_content import TabularDocument
    from kaos_content.model.tabular import Column, ColumnType, Table
    from kaos_office.xlsx import write_xlsx

    cols = (
        Column(name="product", column_type=ColumnType.TEXT),
        Column(name="q1_revenue", column_type=ColumnType.FLOAT),
        Column(name="q1_cogs", column_type=ColumnType.FLOAT),
        Column(name="q4_revenue", column_type=ColumnType.FLOAT),
        Column(name="q4_cogs", column_type=ColumnType.FLOAT),
    )
    table = Table(
        name="PnL2024",
        columns=cols,
        rows=PNL_ROWS,
        row_count=len(PNL_ROWS),
    )
    doc = TabularDocument(tables=(table,))
    write_xlsx(doc, out_path)
    return "PnL2024"


# ---------------------------------------------------------------------------
# Pytest fixtures: workbook, runtime, context, pre-registered engine
# ---------------------------------------------------------------------------


@pytest.fixture
def pnl_workbook(tmp_path: Path) -> tuple[Path, str]:
    """Materialize the P&L workbook on disk. Returns (path, sheet_name)."""
    out = tmp_path / "pnl_2024.xlsx"
    sheet_name = _build_pnl_xlsx(out)
    assert out.exists() and out.stat().st_size > 0, "xlsx writer produced an empty file"
    return out, sheet_name


@pytest.fixture
def runtime() -> Any:
    """Per-test runtime with an in-memory VFS.

    The default ``KaosRuntime()`` uses a disk-backed VFS at
    ``.kaos-vfs``, which means session memory persists across pytest
    invocations. That cross-run leakage causes the agent to "answer
    from memory" on the second-and-later runs without ever calling
    ``kaos-tabular-query`` — false-green on the composition contract.
    ``KaosRuntime.test_mode()`` (kaos-core ≥ 0.1.0a5) installs an
    in-memory, globally-scoped VFS in one line and keeps
    ``runtime.artifacts`` lazy-bound to that VFS, so we don't have
    to rebuild the ArtifactStore by hand any more.
    """
    from kaos_core.registry.container import KaosRuntime

    rt = KaosRuntime.test_mode()

    # Register the tabular MCP tools onto this runtime so the agent
    # can discover + dispatch them via ReAct.
    from kaos_tabular.tools import register_tabular_tools

    register_tabular_tools(rt)
    return rt


@pytest.fixture
def session_id() -> str:
    # Random-ish session id per test so even shared SESSION_REGISTRY
    # state from a previous run can't leak the table in.
    import uuid

    return f"excel-fin-live-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def context(runtime: Any, session_id: str) -> Any:
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id=session_id, runtime=runtime)


@pytest.fixture
def preregistered_engine(pnl_workbook: tuple[Path, str], session_id: str) -> Iterator[Any]:
    """Round-trip the workbook through parse_xlsx and pre-register
    the result with the same session-scoped TabularEngine the MCP
    ``kaos-tabular-query`` tool will resolve via
    :data:`kaos_tabular._session.SESSION_REGISTRY`.

    Yields (engine, registered_table_name).
    """
    import asyncio
    import contextlib

    from kaos_office.xlsx import parse_xlsx
    from kaos_tabular._session import SESSION_REGISTRY

    path, _sheet_name = pnl_workbook
    parsed = parse_xlsx(path)
    assert parsed.tables, "parse_xlsx returned an empty document"

    engine = asyncio.get_event_loop().run_until_complete(SESSION_REGISTRY.get(session_id))
    names = engine.register_document(parsed)
    assert names, "register_document registered nothing"
    yield engine, names[0]
    # Best-effort cleanup; SESSION_REGISTRY is LRU-bounded so even if
    # we leak the engine, the test runner won't grow unboundedly.
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(SESSION_REGISTRY.close_all())


# ---------------------------------------------------------------------------
# Helpers — ground-truth SQL + answer-parsing
# ---------------------------------------------------------------------------


def _yoy_margin_compression(engine: Any, table_name: str) -> list[tuple[str, float]]:
    """Run the canonical YoY-margin SQL directly. Returns rows sorted
    by yoy_margin_change ascending — first row is the biggest
    compression.
    """
    sql = (
        "SELECT product, "
        "(q4_revenue - q4_cogs) / q4_revenue AS q4_margin, "
        "(q1_revenue - q1_cogs) / q1_revenue AS q1_margin, "
        "((q4_revenue - q4_cogs) / q4_revenue) "
        "- ((q1_revenue - q1_cogs) / q1_revenue) AS yoy_margin_change "
        f'FROM "{table_name}" '
        "ORDER BY yoy_margin_change ASC"
    )
    result = engine.execute(sql)
    # Each row is (product, q4_margin, q1_margin, yoy_margin_change).
    return [(str(r[0]), float(r[3])) for r in result.rows]


# ---------------------------------------------------------------------------
# Live composition test
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestExcelFinancialComposition:
    """End-to-end: XLSX → DuckDB → agent answers a margin question."""

    async def test_agent_identifies_biggest_margin_compression(
        self,
        runtime: Any,
        context: Any,
        session_id: str,
        pnl_workbook: tuple[Path, str],
        preregistered_engine: tuple[Any, str],
    ) -> None:
        """The agent, armed with ``kaos-tabular-*`` tools and a
        pre-registered P&L table, should:

        - run SQL (via the ``kaos-tabular-query`` tool) against the
          session engine, and
        - return an answer that names Gizmos (the deterministic
          winner) and cites the table/product as the coordinate.

        We also assert the F2 recorder + cost-tracking machinery sees
        a non-zero, bounded spend.
        """
        engine, table_name = preregistered_engine

        # Ground truth — fail loudly if the fixture itself drifts.
        ranked = _yoy_margin_compression(engine, table_name)
        assert ranked, "ground-truth SQL returned no rows"
        winner_product, winner_delta = ranked[0]
        assert winner_product == "Gizmos", (
            f"fixture drift: expected Gizmos to be the biggest "
            f"margin compression but got {winner_product!r} with delta "
            f"{winner_delta:+.4f}. Adjust PNL_ROWS or the assertion."
        )
        assert winner_delta < -0.20, (
            "fixture drift: Gizmos's YoY margin compression collapsed; "
            "the agent's answer would be ambiguous."
        )

        # Now ask the agent. We name the table explicitly so the model
        # doesn't have to guess (this test exercises the agent's
        # SQL-writing + tool-dispatch ability, not its schema-discovery
        # ability — that's a separate test).
        #
        # The prompt is deliberately blunt about tool use: Haiku
        # occasionally tries to answer from "training data" or just
        # guesses when the question feels analytically simple. The
        # explicit "you do NOT know the row values" sentence forces
        # the only legal path through kaos-tabular-query.
        question = (
            f'There is a DuckDB table in the current session called "{table_name}" '
            "with columns product (text), q1_revenue, q1_cogs, "
            "q4_revenue, q4_cogs (all float, one row per product line). "
            "You do NOT know the row values — you must read them by "
            "calling the kaos-tabular-query tool. "
            f"Run this SQL via that tool: SELECT product, "
            "((q4_revenue - q4_cogs)/q4_revenue) "
            "- ((q1_revenue - q1_cogs)/q1_revenue) AS yoy_margin_change "
            f'FROM "{table_name}" ORDER BY yoy_margin_change ASC LIMIT 1. '
            "Then reply with the product name from the first row and "
            f'cite the table "{table_name}". Be concise (one sentence).'
        )

        from kaos_agents.tools.registry import AgentChatTool

        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": question,
                "session_id": session_id,
                "model": MODEL,
                "tool_filter": "kaos-tabular-query,kaos-tabular-describe,kaos-tabular-list",
                # Cap spend — the question is small.
                "max_cost_usd": 0.10,
            },
            context,
        )

        assert not result.isError, f"agent chat errored: {result.text}"
        payload = result.structuredContent
        assert payload is not None, "agent chat returned no structured payload"

        # Shape checks — the chat tool's structured output contract.
        for key in ("text", "turn_number", "intent", "tool_calls"):
            assert key in payload, f"missing {key} in payload"

        answer = (payload.get("text") or "").strip()
        assert len(answer) > 10, f"answer suspiciously short: {answer!r}"

        # Content checks — the model must actually identify Gizmos.
        # Case-insensitive so the test doesn't flake on capitalization.
        answer_lc = answer.lower()
        assert "gizmos" in answer_lc, (
            f"agent did not identify Gizmos. Answer was:\n{answer!r}\n"
            f"Tool calls: {payload['tool_calls']}"
        )

        # The agent should not also claim one of the non-winners is
        # the answer. Surface-level guard against the model returning
        # a comma-list "Gizmos and Widgets" sort of answer.
        for distractor in ("widgets", "gadgets", "doodads", "sprockets"):
            # We tolerate the distractor name appearing if it's in a
            # comparison clause ("more than Widgets"), but we reject
            # the model claiming the distractor IS the answer. A
            # lightweight check: the answer must mention Gizmos
            # *before* any distractor it does name.
            if distractor in answer_lc:
                assert answer_lc.find("gizmos") < answer_lc.find(distractor), (
                    f"agent named {distractor!r} before Gizmos — likely a "
                    f"wrong answer. Full answer:\n{answer!r}"
                )

        # The answer should cite the table coordinate.
        assert table_name.lower() in answer_lc or table_name in answer, (
            f"agent did not cite the table name {table_name!r}. Answer was:\n{answer!r}"
        )

        # The agent must have actually called the tabular query tool —
        # this is the whole point of the composition test.
        tool_names_called = [tc["tool_name"] for tc in payload["tool_calls"]]
        assert any("kaos-tabular-query" in n for n in tool_names_called), (
            f"agent answered without calling kaos-tabular-query. "
            f"Tool calls: {tool_names_called}. "
            f"This means it either hallucinated SQL output or guessed. "
            f"Answer was: {answer!r}"
        )

        # None of the tabular tool calls should have errored.
        for tc in payload["tool_calls"]:
            assert not tc["is_error"], (
                f"tool call {tc['tool_name']!r} errored: {tc['result_summary']!r}"
            )

        # Budget — sanity gate. The recorder captures the real cost in
        # the JSONL header; this is the per-turn budget check.
        assert not payload["budget_exceeded"], (
            f"agent hit the cost ceiling — bump max_cost_usd or simplify "
            f"the prompt. tool_calls={tool_names_called}"
        )


# ---------------------------------------------------------------------------
# Determinism smoke — the round-trip + SQL path is fully deterministic
# (no LLM). Runs as part of the live class so a single pytest invocation
# exercises both the deterministic + stochastic halves of the
# composition.
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestDeterministicGroundTruth:
    """The fixture pipeline (build → write → parse → SQL) must be
    fully deterministic. This is a non-LLM live test so it adds zero
    cost but catches drift in kaos-office / kaos-tabular before the
    LLM half wastes a call.
    """

    def test_fixture_pipeline_is_deterministic(
        self, pnl_workbook: tuple[Path, str], preregistered_engine: tuple[Any, str]
    ) -> None:
        engine, table_name = preregistered_engine
        ranked = _yoy_margin_compression(engine, table_name)
        # Exactly 5 product lines.
        assert len(ranked) == 5, f"expected 5 rows, got {len(ranked)}"
        # Gizmos is the unambiguous winner.
        assert ranked[0][0] == "Gizmos"
        # The gap between Gizmos and the runner-up is large enough to
        # leave no room for floating-point ambiguity.
        assert ranked[1][1] - ranked[0][1] > 0.20

    def test_workbook_round_trips_with_full_fidelity(self, pnl_workbook: tuple[Path, str]) -> None:
        from kaos_office.xlsx import parse_xlsx

        path, _ = pnl_workbook
        parsed = parse_xlsx(path)
        assert len(parsed.tables) == 1
        tbl = parsed.tables[0]
        assert tbl.row_count == len(PNL_ROWS)
        # Numeric columns survive the round trip as floats.
        for parsed_row, expected_row in zip(tbl.rows, PNL_ROWS, strict=True):
            assert parsed_row[0] == expected_row[0]
            for i in range(1, 5):
                # ``expected_row`` is statically a tuple[str | float, ...],
                # but indices 1..4 are always floats by construction.
                expected_val = float(expected_row[i])  # type: ignore[arg-type]
                parsed_val = float(parsed_row[i])
                assert abs(parsed_val - expected_val) < 1e-9, (
                    f"round-trip drift at row {expected_row[0]!r} col {i}: "
                    f"parsed={parsed_row[i]} expected={expected_row[i]}"
                )


# Quick path so a developer can run this file standalone with a
# tempdir + interactive output. Pytest is still the canonical entry.
if __name__ == "__main__":  # pragma: no cover
    with tempfile.TemporaryDirectory() as tmpd:
        wb = Path(tmpd) / "pnl.xlsx"
        name = _build_pnl_xlsx(wb)
        print(f"Wrote {wb} ({wb.stat().st_size} bytes), sheet={name}")
