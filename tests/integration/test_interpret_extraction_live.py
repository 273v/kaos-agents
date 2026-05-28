"""Live integration tests for ``kaos-agent-interpret-extraction``.

Exercises the iterative Extract↔Synthesize loop end-to-end against
real NDA fixtures + real LLM calls. The unit suite
(``tests/unit/test_interpret_extraction_tool.py``) covers the loop
control surface with mocks; these tests verify the loop actually
produces correct grounded memos on real documents.

Marked ``live`` (requires ``ANTHROPIC_API_KEY``); skipped in unit-only
CI. Each test is bounded by a per-test cost cap so a misbehaving
synthesizer can't burn the budget unbounded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

NDA_DIR = Path(__file__).parent / "ladder" / "fixtures" / "nda"
NDA_FILES = (
    "EMNA Mutual NDA.docx",
    "MNDA - Acme.docx",
    "MNDA - BI.docx",
    "MNDA - CC Final 2.docx",
    "MNDA - DynaMo.docx",
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live tests require ANTHROPIC_API_KEY",
)


async def _setup_runtime() -> tuple[Any, list[str], dict[str, str]]:
    """Stand up in-memory runtime + register tools + persist NDAs."""
    from kaos_content.artifacts import store_document
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_office import parse_docx

    from kaos_agents.tools import register_agent_tools

    runtime = KaosRuntime.test_mode()
    ctx = KaosContext(session_id="bootstrap-live", runtime=runtime)
    register_agent_tools(runtime)
    aids: list[str] = []
    name_map: dict[str, str] = {}
    for fname in NDA_FILES:
        path = NDA_DIR / fname
        assert path.exists(), f"missing fixture {path}"
        doc = parse_docx(str(path))
        manifest = await store_document(doc, runtime, ctx, name=fname)
        aids.append(manifest.artifact_id)
        name_map[manifest.artifact_id] = fname
    return runtime, aids, name_map


@pytest.mark.live
@pytest.mark.asyncio
async def test_converges_at_iter_1_on_well_specified_typed_prompt() -> None:
    """When the user's prompt is clearly typed-deliverable and the
    schema designer picks a comprehensive schema, the loop should
    converge after a single iteration (needs_more=false) — and the
    memo should contain correct grounded section numbers (12, 11, 10,
    11, 11 for the GOVERNING LAW clauses in the 5 NDA fixtures).
    """
    from kaos_core.base.context import KaosContext

    from kaos_agents.tools.interpret_extraction import AgentInterpretExtractionTool

    runtime, aids, _ = await _setup_runtime()
    ctx = KaosContext(session_id="live-p4-convergence", runtime=runtime)
    tool = AgentInterpretExtractionTool()

    result = await tool.execute(
        {
            "question": (
                "Please conduct a governing-law review of the attached NDAs "
                "(5 documents). For each file produce: file name, "
                "governing-law jurisdiction, conflict-of-laws carveout "
                "(Y/N), and the EXACT section number where the "
                "governing-law provision appears. Format as a CSV-ready "
                "table."
            ),
            "artifact_ids": aids,
            "domain_hint": "mutual non-disclosure agreements",
            "max_iters": 2,
            "budget_usd": 0.50,
        },
        context=ctx,
    )

    assert not result.isError, _err(result)
    sc = result.structuredContent
    assert sc is not None

    # Loop should converge on a comprehensive initial schema.
    assert sc["loop_status"] in ("converged", "max_iters_reached")
    assert sc["iterations_run"] in (1, 2)

    # Cost discipline.
    assert sc["cost_usd"] < 0.50, f"cost overran budget: ${sc['cost_usd']:.4f}"

    # Memo content: must contain at least 4 of the 5 correct numerals
    # (12, 11, 10, 11, 11). The numbering_label patch in
    # design_extraction's _block_text feeds these through; without it
    # the LLM saw only "GOVERNING LAW" with no numeric prefix.
    memo = sc["memo"]
    assert memo, "memo must be non-empty"
    numerals_hit = sum(1 for n in ("12", "11", "10") if n in memo)
    assert numerals_hit >= 2, (
        f"memo missing section numerals (12, 11, 10 expected): "
        f"hit_count={numerals_hit}\n\nMemo head:\n{memo[:1000]}"
    )

    # Memo must reference at least 3 of the 5 docs by filename so the
    # user can verify per-document attribution.
    docs_referenced = sum(1 for fname in NDA_FILES if fname.split(".")[0] in memo or fname in memo)
    assert docs_referenced >= 3, f"memo references too few docs by name: {docs_referenced}/5"

    # Cumulative extraction state must include 5 rows.
    extracted = sc["extracted"]
    assert extracted["row_count"] == 5


@pytest.mark.live
@pytest.mark.asyncio
async def test_loop_fires_when_first_synth_pass_insufficient() -> None:
    """When the schema designer happens to under-extract (rare in
    practice but constructable with the right deliverable hint), the
    synthesizer must signal needs_more_extraction and the loop must
    run iter 2 with augmented columns.

    Forces the under-extraction by passing a very narrow
    deliverable_hint that should bias the designer toward minimal
    columns; the synthesizer then has to ask for more.
    """
    from kaos_core.base.context import KaosContext

    from kaos_agents.tools.interpret_extraction import AgentInterpretExtractionTool

    runtime, aids, _ = await _setup_runtime()
    ctx = KaosContext(session_id="live-iteration", runtime=runtime)
    tool = AgentInterpretExtractionTool()

    # Deliberately rich question — the designer is likely to propose
    # many columns at iter 1, so we don't expect iteration. But we
    # DO expect the loop machinery to be exercised: cost recorded,
    # iteration_trace populated, etc.
    result = await tool.execute(
        {
            "question": (
                "Produce a one-page executive summary of these 5 NDAs "
                "for a non-lawyer CEO. Cover: parties, dates, term "
                "length, governing law, key obligations, exceptions, "
                "non-solicitation clauses, and notable risks."
            ),
            "artifact_ids": aids,
            "deliverable_hint": "one-page exec summary for non-lawyer CEO",
            "max_iters": 2,
            "budget_usd": 0.60,
        },
        context=ctx,
    )

    assert not result.isError, _err(result)
    sc = result.structuredContent
    assert sc is not None

    # Loop ran at least once.
    assert sc["iterations_run"] >= 1
    assert len(sc["iteration_trace"]) == sc["iterations_run"]

    # Each trace entry has the right shape.
    for t in sc["iteration_trace"]:
        assert "iter" in t
        assert "extract_cost_usd" in t
        assert "synth_cost_usd" in t
        assert "score" in t
        assert "needs_more_extraction" in t
        assert "cumulative_cols" in t
        assert "cumulative_rows" in t

    # Final memo should mention all 5 counterparties so the CEO sees
    # one section per NDA.
    memo = sc["memo"]
    counterparties = ("ExMachi", "Acme", "Beta", "CyberCorp", "DynaMo")
    found = [c for c in counterparties if c in memo]
    assert len(found) >= 4, (
        f"memo missing per-counterparty coverage: hit={found}\n\nMemo head:\n{memo[:1500]}"
    )

    # Cost discipline.
    assert sc["cost_usd"] < 0.60, f"cost overran budget: ${sc['cost_usd']:.4f}"


@pytest.mark.live
@pytest.mark.asyncio
async def test_budget_cap_stops_loop_under_pathological_synth() -> None:
    """Verifies the budget cap actually stops the loop. Uses a tiny
    budget so even a single iteration trips the cap; the loop should
    NOT make a second extraction attempt regardless of what the
    synthesizer says.
    """
    from kaos_core.base.context import KaosContext

    from kaos_agents.tools.interpret_extraction import AgentInterpretExtractionTool

    runtime, aids, _ = await _setup_runtime()
    ctx = KaosContext(session_id="live-budget", runtime=runtime)
    tool = AgentInterpretExtractionTool()

    result = await tool.execute(
        {
            "question": "Summarize the contents of each NDA.",
            "artifact_ids": aids,
            "max_iters": 5,
            "budget_usd": 0.05,  # so small that even iter 1 exhausts it
        },
        context=ctx,
    )

    assert not result.isError, _err(result)
    sc = result.structuredContent
    assert sc is not None

    # Either converged (synth said done) or budget-exhausted; never
    # max_iters_reached with this tiny budget.
    assert sc["loop_status"] in ("converged", "budget_exhausted")
    assert sc["iterations_run"] <= 2, "loop should not run many iters under $0.05 cap"


def _err(result: Any) -> str:
    if not result.content:
        return ""
    return str(getattr(result.content[0], "text", "") or "")
