"""Unit tests for Sprint-2 #5 — findings consistency (quality).

Covers three new pieces of the contract:

1. :func:`compute_finding_id` (= the public alias for
   ``_deterministic_finding_id``) is stable across runs and varies
   only with its three inputs.
2. ``FindingsAgent(temperature=...)`` plumbs the value all the way
   down to the inner ``Call(...)``. We monkey-patch ``Call`` to
   inspect the kwargs at construction time.
3. ``FindingsAgent(runs=N)`` runs the filter pipeline N times and
   unions surviving findings by deterministic ``finding_id`` — same
   sentence across runs collapses to one entry.

Live-LLM coverage of the same contract lives in
``tests/integration/test_findings_consistency_live.py``. The unit
tier here proves the plumbing; the live tier proves the quality bar
(5-run Jaccard >= 0.95 on a real NDA).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kaos_agents.patterns import findings as findings_mod
from kaos_agents.patterns.findings import (
    FilteredFinding,
    FindingCandidate,
    FindingsAgent,
    compute_finding_id,
    every_sentence_selector,
)

# ---------------------------------------------------------------------------
# Fake DocumentView — same shape as tests/unit/test_findings.py
# ---------------------------------------------------------------------------


class _FakeSentence:
    """Stand-in for kaos_content.views.models.SentenceView.

    Carries the same five fields the selectors read, including
    ``start`` and ``end`` (the per-paragraph char offsets the
    deterministic-id hash uses).
    """

    def __init__(
        self,
        text: str,
        paragraph_ref: str = "#/body/0",
        start: int = 0,
        end: int | None = None,
        section_ref: str | None = None,
        page: int | None = None,
    ) -> None:
        self.text = text
        self.paragraph_ref = paragraph_ref
        self.start = start
        self.end = end if end is not None else len(text)
        self.section_ref = section_ref
        self.page = page


class _FakeView:
    def __init__(self, sentences: list[_FakeSentence]) -> None:
        self.sentences = sentences

    def section_by_ref(self, _ref: str) -> None:
        return None


def _sample_view() -> _FakeView:
    return _FakeView(
        [
            _FakeSentence(
                "The cap on indemnification is $100,000 per occurrence.",
                paragraph_ref="#/body/3",
                start=0,
                end=54,
            ),
            _FakeSentence(
                "Indemnification carve-outs apply for gross negligence.",
                paragraph_ref="#/body/4",
                start=0,
                end=54,
            ),
            _FakeSentence(
                "Confidential Information includes business plans.",
                paragraph_ref="#/body/2",
                start=0,
                end=49,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 1) Deterministic finding_id
# ---------------------------------------------------------------------------


class TestDeterministicFindingId:
    """``compute_finding_id`` is stable, length-bounded, and
    distinct on the inputs we care about."""

    def test_same_input_same_id(self) -> None:
        # Run the hash twice and assert byte-identical output.
        a = compute_finding_id("#/body/3", (0, 53), "The cap on indemnification is $100,000.")
        b = compute_finding_id("#/body/3", (0, 53), "The cap on indemnification is $100,000.")
        assert a == b
        # 12 hex chars, lowercase only.
        assert len(a) == 12
        assert all(ch in "0123456789abcdef" for ch in a)

    def test_different_text_different_id(self) -> None:
        a = compute_finding_id("#/body/3", (0, 53), "Sentence A.")
        b = compute_finding_id("#/body/3", (0, 53), "Sentence B.")
        assert a != b

    def test_different_block_ref_different_id(self) -> None:
        a = compute_finding_id("#/body/3", (0, 53), "Identical text.")
        b = compute_finding_id("#/body/4", (0, 53), "Identical text.")
        assert a != b

    def test_different_char_span_different_id(self) -> None:
        a = compute_finding_id("#/body/3", (0, 53), "Identical text.")
        b = compute_finding_id("#/body/3", (10, 63), "Identical text.")
        assert a != b

    def test_none_block_ref_or_span_still_stable(self) -> None:
        # The selectors over fake views may not have either field.
        # The hash must still produce a valid id and still be stable.
        a = compute_finding_id(None, None, "Just text.")
        b = compute_finding_id(None, None, "Just text.")
        assert a == b
        assert len(a) == 12

    def test_whitespace_normalized(self) -> None:
        # Trivial whitespace drift between extraction runs (extra
        # spaces, NBSPs collapsed to regular spaces, etc.) must not
        # change the id. The hash collapses ``\\s+`` to a single
        # space and strips leading/trailing whitespace.
        a = compute_finding_id("#/body/3", (0, 10), "The   cap.")
        b = compute_finding_id("#/body/3", (0, 10), "The cap.")
        c = compute_finding_id("#/body/3", (0, 10), "  The cap.  ")
        assert a == b == c

    def test_selectors_produce_deterministic_ids(self) -> None:
        # The end-to-end path the production agent takes: same
        # selector + same view → same ids on every invocation.
        view = _sample_view()
        ids_run1 = [c.finding_id for c in every_sentence_selector(view, "q")]  # ty: ignore[invalid-argument-type]
        ids_run2 = [c.finding_id for c in every_sentence_selector(view, "q")]  # ty: ignore[invalid-argument-type]
        assert ids_run1 == ids_run2
        # And the ids are pairwise distinct (different paragraphs +
        # different texts).
        assert len(set(ids_run1)) == len(ids_run1)


# ---------------------------------------------------------------------------
# 2) temperature=0 plumbed through to the inner Call
# ---------------------------------------------------------------------------


class _StubUsage:
    cost_usd: float = 0.0


class _StubOutput:
    """The decoded ``_FilterSignature`` output.

    ``_filter_chunk`` reads ``output.survivors`` (a list of dicts)
    and looks up each survivor's ``finding_id`` in the chunk index.
    """

    def __init__(self) -> None:
        self.survivors: list[dict[str, Any]] = []
        self.answer: str = ""


class _StubInvocation:
    def __init__(self) -> None:
        self.output = _StubOutput()
        self.usage = _StubUsage()


class _SpyCallKwargs:
    """Records the kwargs every Call(...) instantiation receives.

    Replaces ``Call`` in ``kaos_llm_core.programs.call`` for the
    duration of one test so we can prove that
    ``FindingsAgent(temperature=X)`` reaches the inner
    ``Call(_FilterSignature, ..., temperature=X)``.

    The stub Call always claims that the first candidate in the
    chunk is a relevant survivor — without that, Phase 2 returns
    zero survivors and Phase 3 (synthesis) is correctly skipped,
    which would defeat the synthesis-side temperature assertion.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        signature: Any,
        *,
        model: str,
        **kwargs: Any,
    ) -> Any:
        # Record everything except the (unused-for-the-test) bound
        # signature class so the assertion error message stays tidy.
        self.calls.append({"model": model, "signature": signature.__name__, **kwargs})

        spy_self = self

        class _StubCall:
            async def invoke(self, **inputs: Any) -> _StubInvocation:
                # Pull the candidate ids out of the rendered
                # candidates blob so the inner ``_filter_chunk``
                # finds them on lookup. Pattern:
                # <untrusted_document_content finding_id="...">
                import re as _re

                rendered = inputs.get("candidates", "") or ""
                ids = _re.findall(r'finding_id="([0-9a-f]+)"', rendered)
                inv = _StubInvocation()
                # Mark the first candidate as a survivor at full
                # relevance so Phase 3 fires.
                if ids:
                    inv.output.survivors = [
                        {"finding_id": ids[0], "relevance": 0.95, "reasoning": "stubbed"}
                    ]
                spy_self.calls.append({"phase": "stub_invoke", "extracted_ids": ids})
                return inv

        return _StubCall()


class TestTemperaturePlumbing:
    """``temperature`` is forwarded into every Call construction."""

    def test_default_temperature_is_zero(self) -> None:
        agent = FindingsAgent(selector=every_sentence_selector)
        assert agent.temperature == 0.0

    def test_explicit_temperature_recorded(self) -> None:
        agent = FindingsAgent(selector=every_sentence_selector, temperature=0.4)
        assert agent.temperature == 0.4

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            FindingsAgent(selector=every_sentence_selector, temperature=-0.1)

    def test_filter_call_receives_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end plumbing: a real agent.run() with a stub Call
        records ``temperature=0.0`` (default) on every filter call.

        We can't substitute the symbol on ``findings.py`` because the
        import is deferred inside ``_filter_chunk`` / ``_synthesize``;
        we patch the resolved module instead.
        """
        spy = _SpyCallKwargs()
        # The Call import happens inside the helper functions so we
        # have to patch the *source* module, not the findings module
        # namespace.
        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod, "Call", spy)

        # Stub _synthesize so we don't pull in the real synthesis path;
        # we already prove temperature plumbing fires in _filter_chunk.
        async def _stub_synthesize(
            *, question: str, findings: tuple[Any, ...], model: str, temperature: float = 0.0
        ) -> tuple[str, float]:
            spy.calls.append({"phase": "synthesize", "temperature": temperature, "model": model})
            return "stub", 0.0

        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=10, temperature=0.0)
        asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]

        # At least one call must have been issued (the filter pass).
        assert any(c.get("signature", "").startswith("_FilterSignature") for c in spy.calls), (
            f"No filter-call recorded; spy saw: {spy.calls!r}"
        )
        for c in spy.calls:
            if c.get("signature", "").startswith("_FilterSignature"):
                assert c.get("temperature") == 0.0, (
                    f"Filter Call did not receive temperature=0.0: {c!r}"
                )
        # And the synthesis stub also saw temperature=0.0.
        synth_records = [c for c in spy.calls if c.get("phase") == "synthesize"]
        assert synth_records, "Synthesis stub was never invoked"
        for c in synth_records:
            assert c["temperature"] == 0.0

    def test_nonzero_temperature_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Caller-supplied temperature=0.7 reaches both Calls."""
        spy = _SpyCallKwargs()
        from kaos_llm_core.programs import call as call_mod

        monkeypatch.setattr(call_mod, "Call", spy)

        async def _stub_synthesize(
            *, question: str, findings: tuple[Any, ...], model: str, temperature: float = 0.0
        ) -> tuple[str, float]:
            spy.calls.append({"phase": "synthesize", "temperature": temperature, "model": model})
            return "stub", 0.0

        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, temperature=0.7)
        asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]

        filter_calls = [c for c in spy.calls if c.get("signature", "").startswith("_Filter")]
        assert filter_calls, "expected at least one filter Call"
        for c in filter_calls:
            assert c["temperature"] == 0.7
        synth_calls = [c for c in spy.calls if c.get("phase") == "synthesize"]
        for c in synth_calls:
            assert c["temperature"] == 0.7


# ---------------------------------------------------------------------------
# 3) runs=N union mode
# ---------------------------------------------------------------------------


class TestRunsUnionMode:
    """``FindingsAgent(runs=N)`` issues N filter passes, unions
    survivors by deterministic finding_id, and re-synthesizes once."""

    def test_default_runs_is_one(self) -> None:
        agent = FindingsAgent(selector=every_sentence_selector)
        assert agent.runs == 1

    def test_zero_or_negative_runs_rejected(self) -> None:
        with pytest.raises(ValueError, match="runs"):
            FindingsAgent(selector=every_sentence_selector, runs=0)
        with pytest.raises(ValueError, match="runs"):
            FindingsAgent(selector=every_sentence_selector, runs=-1)

    def test_runs_one_is_unchanged_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``runs=1`` (the default) issues exactly one filter pass
        and produces the same result the historical code path did."""
        call_count = {"filter": 0, "synth": 0}

        async def _stub_filter(
            chunk: tuple[FindingCandidate, ...], **_kw: Any
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            call_count["filter"] += 1
            survivors = tuple(
                FilteredFinding(candidate=c, relevance=0.8, reasoning="ok") for c in chunk
            )
            return survivors, 0.001

        async def _stub_synth(**_kw: Any) -> tuple[str, float]:
            call_count["synth"] += 1
            return "answer", 0.005

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synth)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=10, runs=1)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        assert call_count["filter"] == 1  # one chunk, one run
        assert call_count["synth"] == 1
        assert result.total_filtered == 3
        assert result.filter_calls == 1

    def test_runs_two_unions_overlapping_survivors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two passes that return overlapping-but-not-identical
        survivor sets must union into a single set on
        ``finding_id``."""
        view = _sample_view()
        # Pre-compute the deterministic ids the selector will emit.
        cands = list(every_sentence_selector(view, "q"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 3
        id_a, id_b, id_c = (c.finding_id for c in cands)

        # Two-stage stub: run 1 returns {a, b}, run 2 returns {b, c}.
        # Union should be {a, b, c}.
        call_count = {"filter": 0}
        per_call_survivor_sets = [
            (id_a, id_b),  # first chunk call (run 1)
            (id_b, id_c),  # second chunk call (run 2)
        ]

        async def _stub_filter(
            chunk: tuple[FindingCandidate, ...], **_kw: Any
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            keep_ids = per_call_survivor_sets[call_count["filter"]]
            call_count["filter"] += 1
            survivors = tuple(
                FilteredFinding(candidate=c, relevance=0.7, reasoning="ok")
                for c in chunk
                if c.finding_id in keep_ids
            )
            return survivors, 0.001

        async def _stub_synth(**_kw: Any) -> tuple[str, float]:
            return "synth-answer", 0.005

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synth)

        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=10, runs=2)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]

        # Two filter passes (1 chunk x 2 runs).
        assert call_count["filter"] == 2
        assert result.filter_calls == 2

        # Surviving union == {a, b, c}.
        survivor_ids = {f.candidate.finding_id for f in result.findings}
        assert survivor_ids == {id_a, id_b, id_c}, f"Expected union {{a,b,c}}, got {survivor_ids!r}"

        # Synthesis was called exactly once with the unioned set.
        # (The stub doesn't introspect args; the fact that we got an
        # answer string back is the proof Phase 3 fired once.)
        assert result.answer == "synth-answer"

        # Cost rolls up: 2 filter calls * $0.001 + $0.005 synth.
        assert result.filter_cost_usd == pytest.approx(0.002)
        assert result.synthesis_cost_usd == pytest.approx(0.005)

    def test_runs_two_preserves_deterministic_finding_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The union step must not generate new finding_ids — it
        must preserve the deterministic ids the selector emitted,
        so downstream consumers can verify cites by recomputing
        the hash."""
        view = _sample_view()
        expected_ids = {
            c.finding_id
            for c in every_sentence_selector(view, "q")  # ty: ignore[invalid-argument-type]
        }

        async def _stub_filter_all_pass(
            chunk: tuple[FindingCandidate, ...], **_kw: Any
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            return (
                tuple(FilteredFinding(candidate=c, relevance=0.9, reasoning="ok") for c in chunk),
                0.001,
            )

        async def _stub_synth(**_kw: Any) -> tuple[str, float]:
            return "ok", 0.005

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_all_pass)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synth)

        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=10, runs=3)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]

        # All three runs return the same 3 survivors; union has 3.
        actual_ids = {f.candidate.finding_id for f in result.findings}
        assert actual_ids == expected_ids
        assert len(result.findings) == 3

        # Each finding_id can be recomputed from its candidate by a
        # third-party verifier.
        for f in result.findings:
            recomputed = compute_finding_id(
                f.candidate.block_ref, f.candidate.char_span, f.candidate.text
            )
            assert recomputed == f.candidate.finding_id

    def test_runs_two_picks_max_relevance_on_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When two runs return the same sentence with different
        relevance scores, the union step keeps the higher score —
        otherwise a noisy low-score pass could drown out a
        confident one."""
        view = _sample_view()
        cands = list(every_sentence_selector(view, "q"))  # ty: ignore[invalid-argument-type]
        id_a = cands[0].finding_id

        # First call returns id_a with 0.3, second with 0.9.
        per_call_scores = [0.3, 0.9]
        call_count = {"filter": 0}

        async def _stub_filter(
            chunk: tuple[FindingCandidate, ...], **_kw: Any
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            score = per_call_scores[call_count["filter"]]
            call_count["filter"] += 1
            survivors = tuple(
                FilteredFinding(candidate=c, relevance=score, reasoning="ok")
                for c in chunk
                if c.finding_id == id_a
            )
            return survivors, 0.001

        async def _stub_synth(**_kw: Any) -> tuple[str, float]:
            return "ok", 0.005

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synth)

        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=10, runs=2)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        assert len(result.findings) == 1
        assert result.findings[0].candidate.finding_id == id_a
        assert result.findings[0].relevance == pytest.approx(0.9)
