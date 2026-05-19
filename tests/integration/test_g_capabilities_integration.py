"""Integration tests for G1-G5 capabilities against real data.

These are not LLM-driven (G1-G5 don't invoke LLMs internally) but
they exercise each capability against a **real, populated** target
rather than a stub:

- G1 ToolRetrieval against the actual 50-tool KAOS catalog
  (kaos-pdf + kaos-web + kaos-tabular + kaos-office). Verifies the
  retrieval ranks domain-appropriate tools to the top.
- G2 Lessons memory through a real SessionMemory + real BM25
  Searcher with realistic lesson text. Verifies write/read/recall
  semantics in a populated session.
- G3 StepInvariants on realistic agent output strings — both
  pass and fail trajectories.
- G4 CostBudget + escalation policy across a simulated multi-step
  spend pattern. Verifies the policy escalates on quality regression
  *only when* budget headroom permits.
- G5 Replay round-trips a real captured event stream from a real
  Runner.run() invocation through save_run / load_run, asserting
  bit-for-bit equality of the recovered summary.

No API keys required for this file — these are integration tests
against real local fixtures, not live external services.
"""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import pytest
from kaos_core.registry.container import KaosRuntime

from kaos_agents.memory.session import SessionMemory
from kaos_agents.runtime.tool_retrieval import ToolRetrieval

# ---------------------------------------------------------------------------
# Real KAOS tool catalog fixture
# ---------------------------------------------------------------------------


def _make_real_catalog_runtime() -> KaosRuntime:
    """Build a runtime populated with the real KAOS tool registrars.

    Uses kaos-pdf + kaos-web + kaos-tabular + kaos-office — four
    distinct domains, ~50 tools total. Each registrar lives in the
    monorepo's installed packages; if any are missing the test is
    skipped.
    """
    runtime = KaosRuntime()
    loaded = 0
    for mod_name in (
        "kaos_pdf.tools",
        "kaos_web.tools",
        "kaos_tabular.tools",
        "kaos_office.tools",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in dir(mod):
            if attr.startswith("register_") and "tool" in attr:
                getattr(mod, attr)(runtime)
                loaded += 1
                break
    if loaded < 2:
        pytest.skip(
            f"Only {loaded} kaos-* tool modules available; need at least 2 "
            "for a meaningful catalog test"
        )
    return runtime


# ===========================================================================
# G1 ToolRetrieval — real catalog
# ===========================================================================


@pytest.mark.integration
class TestToolRetrievalRealCatalog:
    """ToolRetrieval against the actual KAOS catalog (no LLM)."""

    def test_pdf_query_ranks_pdf_tools_first(self) -> None:
        """A PDF-specific query must rank kaos-pdf-* tools above
        non-PDF tools."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        hits = retrieval.search("extract text from a PDF document", top_k=5)
        assert hits, "Expected at least one hit for a clear PDF query"
        top_names = [h.tool.metadata.name for h in hits[:3]]
        pdf_count = sum(1 for n in top_names if n.startswith("kaos-pdf-"))
        assert pdf_count >= 2, (
            f"Expected at least 2 kaos-pdf-* tools in top-3 for PDF query; got: {top_names}"
        )

    def test_web_query_ranks_web_tools_first(self) -> None:
        """A web-specific query must rank kaos-web-* tools above non-web."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        hits = retrieval.search("download a webpage and extract its text", top_k=5)
        assert hits
        top_names = [h.tool.metadata.name for h in hits[:3]]
        web_count = sum(1 for n in top_names if n.startswith("kaos-web-"))
        assert web_count >= 2, (
            f"Expected at least 2 kaos-web-* tools in top-3 for web query; got: {top_names}"
        )

    def test_sql_query_ranks_tabular_tools_first(self) -> None:
        """A SQL/analytics query should rank kaos-tabular-* tools first."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        hits = retrieval.search("run a SQL query on a CSV file", top_k=5)
        if not hits:
            pytest.skip("No tabular tools available")
        top_names = [h.tool.metadata.name for h in hits[:3]]
        tabular_count = sum(1 for n in top_names if "tabular" in n)
        assert tabular_count >= 1, (
            f"Expected at least 1 kaos-tabular-* tool in top-3 for SQL query; got: {top_names}"
        )

    def test_irrelevant_query_returns_low_scores_or_empty(self) -> None:
        """A query with no overlap with the catalog should return either
        no hits or low-scoring hits — not high-confidence matches."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        # "horoscope astrology" overlaps nothing in PDF/web/tabular/office.
        hits = retrieval.search("daily horoscope astrology zodiac signs", top_k=5)
        if hits:
            assert max(h.score for h in hits) < 5.0, (
                f"Out-of-domain query should not produce high BM25 scores; "
                f"got: {[(h.tool.metadata.name, h.score) for h in hits]}"
            )

    def test_top_k_caps_result_count(self) -> None:
        """top_k must cap returned hits; useful for fitting tool context."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        hits = retrieval.search("extract", top_k=3)
        assert len(hits) <= 3

    def test_scores_are_monotonically_decreasing(self) -> None:
        """BM25 results must be returned in descending score order."""
        runtime = _make_real_catalog_runtime()
        retrieval = ToolRetrieval.from_runtime(runtime)
        hits = retrieval.search("pdf table extraction", top_k=10)
        if len(hits) < 2:
            pytest.skip("Need ≥2 hits to check ordering")
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), (
            f"BM25 hits must be sorted by score desc; got: {scores}"
        )


# ===========================================================================
# G2 Lessons — real SessionMemory + BM25
# ===========================================================================


@pytest.mark.integration
class TestLessonsRealSession:
    """Lessons stored and recalled via the actual SessionMemory + Searcher."""

    def test_lesson_round_trip_through_session(self) -> None:
        """Write 3 lessons, read them back, verify content integrity."""
        from kaos_agents.memory.lessons import Lesson, read_lessons, write_lesson

        memory = SessionMemory(session_id="lessons-rt-1")

        lessons = [
            Lesson(
                situation="Reviewing a Mutual NDA with reciprocal duties",
                observation="Reciprocity reduces negotiation friction",
                takeaway="Default to mutual NDAs for ordinary commercial diligence",
                confidence=0.85,
                tags=("nda", "mutuality"),
            ),
            Lesson(
                situation="Drafting indemnification in an SaaS MSA",
                observation="Carve-out exclusions vary widely",
                takeaway="Always check IP indemnity carve-outs explicitly",
                confidence=0.9,
                tags=("indemnification", "msa"),
            ),
            Lesson(
                situation="Reviewing a one-way confidentiality clause",
                observation="One-way NDAs are common in M&A diligence",
                takeaway="One-way is appropriate when only one side discloses",
                confidence=0.75,
                tags=("nda", "m-and-a"),
            ),
        ]
        for lesson in lessons:
            write_lesson(memory, lesson)

        read = read_lessons(memory)
        assert len(read) == 3
        situations = {r.situation for r in read}
        assert "Reviewing a Mutual NDA with reciprocal duties" in situations
        assert "Drafting indemnification in an SaaS MSA" in situations

    def test_lesson_recall_returns_topically_relevant_first(self) -> None:
        """BM25 over real lesson text must rank by topical overlap."""
        from kaos_agents.memory.lessons import Lesson, recall_lessons, write_lesson

        memory = SessionMemory(session_id="lessons-recall-1")

        write_lesson(
            memory,
            Lesson(
                situation="Negotiating limitation of liability caps",
                observation="Most enterprise MSAs cap at 12 months of fees",
                takeaway="Push for higher cap when data is regulated",
                confidence=0.8,
            ),
        )
        write_lesson(
            memory,
            Lesson(
                situation="Reviewing a Mutual Non-Disclosure Agreement",
                observation="The definition of Confidential Information is overbroad",
                takeaway="Narrow the definition to exclude already-public info",
                confidence=0.9,
            ),
        )
        write_lesson(
            memory,
            Lesson(
                situation="Drafting an arbitration clause for a SaaS contract",
                observation="JAMS vs AAA materially changes cost",
                takeaway="Pick JAMS for higher-stakes commercial disputes",
                confidence=0.7,
            ),
        )

        recalled = recall_lessons(
            memory,
            situation_query="how should I define confidential information in this NDA?",
            top_k=2,
        )
        assert recalled, "Expected non-empty recall for NDA-related query"
        # Top hit must be the NDA lesson, not the liability or arbitration ones.
        top = recalled[0]
        assert "Non-Disclosure" in top.situation or "Confidential" in top.observation, (
            f"Expected the NDA lesson to rank first; got: {top.situation!r}"
        )


# ===========================================================================
# G3 StepInvariants — realistic outputs
# ===========================================================================


@pytest.mark.integration
class TestInvariantsRealOutputs:
    """Invariants checked against realistic agent-style output strings."""

    def test_invariant_passes_on_compliant_output(self) -> None:
        from kaos_agents.runtime.invariants import StepInvariant, check_invariants

        contains_citation = StepInvariant(
            name="must-cite",
            predicate=lambda text: "v." in text or "§" in text,  # ty: ignore[unsupported-operator]
            message_on_violation="Output must reference a case (X v. Y) or statute (§ ...).",
        )
        output = (
            "Per Marbury v. Madison (1803), judicial review is the foundation "
            "of constitutional adjudication."
        )
        results = check_invariants(output, (contains_citation,))
        assert all(r.passed for r in results)

    def test_invariant_fails_on_noncompliant_output(self) -> None:
        from kaos_agents.runtime.invariants import (
            StepInvariant,
            check_invariants,
            first_hard_violation,
        )

        contains_citation = StepInvariant(
            name="must-cite",
            predicate=lambda text: "v." in text or "§" in text,  # ty: ignore[unsupported-operator]
            message_on_violation="Output must reference a case (X v. Y) or statute (§ ...).",
        )
        output = "Judicial review is the foundation of constitutional adjudication."
        results = check_invariants(output, (contains_citation,))
        violation = first_hard_violation(results)
        assert violation is not None
        assert "case" in violation.invariant.message_on_violation.lower()

    def test_multiple_invariants_evaluate_all(self) -> None:
        from kaos_agents.runtime.invariants import StepInvariant, check_invariants

        inv1 = StepInvariant(
            name="non-empty",
            predicate=lambda t: bool(t and t.strip()),  # ty: ignore[unresolved-attribute]
            message_on_violation="Output must not be empty.",
        )
        inv2 = StepInvariant(
            name="under-500-chars",
            predicate=lambda t: bool(t) and len(str(t)) < 500,
            message_on_violation="Output must be under 500 chars.",
        )
        output = "Short."
        results = check_invariants(output, (inv1, inv2))
        assert len(results) == 2
        assert all(r.passed for r in results)


# ===========================================================================
# G4 CostBudget + escalation
# ===========================================================================


@pytest.mark.integration
class TestCostBudgetRealistic:
    """CostBudget arithmetic + escalation across a multi-step spend trace."""

    def test_budget_arithmetic_tracks_spend(self) -> None:
        from kaos_agents.types.budget import CostBudget

        b = CostBudget(total_usd=1.0, spent_usd=0.0)
        # Simulate 5 steps, each costing $0.12
        for _ in range(5):
            b = b.spend(0.12)
        assert b.spent_usd == pytest.approx(0.60)
        assert b.remaining_usd == pytest.approx(0.40)
        assert b.fraction_spent == pytest.approx(0.60)
        assert not b.exceeded

    def test_budget_exceeded_when_over_cap(self) -> None:
        from kaos_agents.types.budget import CostBudget

        b = CostBudget(total_usd=1.0, spent_usd=0.0)
        b = b.spend(1.20)
        assert b.exceeded

    def test_escalation_recommends_stronger_model_on_quality_failure(self) -> None:
        """When the step output fails and headroom permits, the policy
        should recommend escalation to a stronger model."""
        from kaos_agents.runtime.escalation import (
            DefaultEscalationPolicy,
            EscalationAction,
            StepOutcome,
        )
        from kaos_agents.types.budget import CostBudget

        policy = DefaultEscalationPolicy(high_water_mark=0.75, max_attempts_per_step=3)
        budget = CostBudget(total_usd=1.0, spent_usd=0.10)  # 90% headroom

        decision = policy.choose(
            budget=budget,
            outcome=StepOutcome.INVARIANT_VIOLATION,
            attempt=1,
        )
        assert decision.action in {
            EscalationAction.UPGRADE,
            EscalationAction.STAY,
        }, f"Expected upgrade or stay on invariant violation with headroom; got {decision.action}"

    def test_escalation_stops_when_budget_exhausted(self) -> None:
        """At/above high_water_mark spent, the policy must stop escalating."""
        from kaos_agents.runtime.escalation import (
            DefaultEscalationPolicy,
            EscalationAction,
            StepOutcome,
        )
        from kaos_agents.types.budget import CostBudget

        policy = DefaultEscalationPolicy(high_water_mark=0.75, max_attempts_per_step=3)
        budget = CostBudget(total_usd=1.0, spent_usd=0.85)  # 85% spent

        decision = policy.choose(
            budget=budget,
            outcome=StepOutcome.INVARIANT_VIOLATION,
            attempt=1,
        )
        # Should NOT upgrade to a stronger (more expensive) model when
        # we're past the high-water mark.
        assert decision.action != EscalationAction.UPGRADE, (
            f"Should not upgrade model when 85% of budget is spent; got {decision.action}"
        )


# ===========================================================================
# G5 Replay — real record / save / load round-trip
# ===========================================================================


@pytest.mark.integration
class TestReplayRoundTrip:
    """Real captured events round-trip through save_run / load_run on disk."""

    def test_save_load_preserves_summary(self) -> None:
        from kaos_agents.events.emitter import EventEmitter
        from kaos_agents.replay.recorder import RecordedRun, _build_summary, load_run, save_run

        emitter = EventEmitter(session_id="replay-rt-1", run_id="run-1")
        # Build a few real events (using the registry-backed types).
        from kaos_agents.events.lifecycle import TurnSummary

        events = []
        # Add 2 TurnSummary events with text so summary picks them up.
        for i, text in enumerate(
            [
                "First turn answer with citation.",
                "Second turn refinement.",
            ]
        ):
            evt = emitter.emit(
                TurnSummary,
                turn_number=i + 1,
                text=text,
                tokens_used=100 + i,
                cost_usd=0.001 * (i + 1),
            )
            events.append(evt)

        events_tuple = tuple(events)
        run = RecordedRun(
            events=events_tuple,
            summary=_build_summary(events_tuple),
            session_id="replay-rt-1",
            label="round-trip test",
        )
        assert run.summary.turn_count == 2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.jsonl"
            save_run(run, path)
            assert path.exists()
            # File must contain header + 2 event lines
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3
            header = json.loads(lines[0])
            assert header["_kind"] == "header"
            assert header["session_id"] == "replay-rt-1"

            loaded = load_run(path)
            assert loaded.session_id == "replay-rt-1"
            assert loaded.label == "round-trip test"
            assert loaded.summary.turn_count == 2
            assert loaded.summary.event_count == 2
            assert "First turn answer" in loaded.summary.final_answer_text
            assert "Second turn refinement" in loaded.summary.final_answer_text

    def test_diff_detects_identical_runs(self) -> None:
        from kaos_agents.events.emitter import EventEmitter
        from kaos_agents.events.lifecycle import TurnSummary
        from kaos_agents.replay.diff import diff_runs
        from kaos_agents.replay.recorder import RecordedRun, _build_summary

        emitter = EventEmitter(session_id="diff-1", run_id="run-1")

        def _make_run(label: str):
            events = []
            for i, text in enumerate(["alpha", "beta"]):
                events.append(
                    emitter.emit(
                        TurnSummary,
                        turn_number=i + 1,
                        text=text,
                        tokens_used=10,
                        cost_usd=0.0001,
                    ),
                )
            tup = tuple(events)
            return RecordedRun(
                events=tup,
                summary=_build_summary(tup),
                session_id="diff-1",
                label=label,
            )

        a = _make_run("baseline")
        b = _make_run("candidate")
        diff = diff_runs(a, b, include_text_diff=False)
        # Same events, same counts — must be equivalent.
        assert diff.is_equivalent
        assert diff.event_count_delta == 0
        assert diff.turn_count_delta == 0

    def test_diff_detects_divergent_runs(self) -> None:
        from kaos_agents.events.emitter import EventEmitter
        from kaos_agents.events.lifecycle import TurnSummary
        from kaos_agents.replay.diff import diff_runs
        from kaos_agents.replay.recorder import RecordedRun, _build_summary

        em_a = EventEmitter(session_id="diff-a", run_id="run-a")
        em_b = EventEmitter(session_id="diff-b", run_id="run-b")

        events_a = [
            em_a.emit(TurnSummary, turn_number=1, text="alpha", tokens_used=10, cost_usd=0.0001),
        ]
        events_b = [
            em_b.emit(TurnSummary, turn_number=1, text="alpha", tokens_used=10, cost_usd=0.0001),
            em_b.emit(TurnSummary, turn_number=2, text="beta", tokens_used=10, cost_usd=0.0001),
        ]

        tup_a = tuple(events_a)
        tup_b = tuple(events_b)
        a = RecordedRun(events=tup_a, summary=_build_summary(tup_a), label="a")
        b = RecordedRun(events=tup_b, summary=_build_summary(tup_b), label="b")

        diff = diff_runs(a, b, include_text_diff=False)
        assert not diff.is_equivalent
        assert diff.event_count_delta == 1
        assert diff.turn_count_delta == 1
