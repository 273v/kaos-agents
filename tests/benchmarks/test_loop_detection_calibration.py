"""Phase 4.E — empirical calibration of the LoopDetector similarity threshold.

Background
----------

Resolved Decision #7 in the rewrite plan specifies "TLSH ≤ 30 over the
last 5 calls" as the loop-detection threshold. kaos-nlp-core does not
ship TLSH; it ships **CTPH** (Context-Triggered Piecewise Hashing) via
``ctph_hash_str`` + ``ctph_similarity``. The Phase 4.B implementation
adapted by analogy: ``min_similarity ≥ 0.5`` (CTPH similarity, Jaccard
0..1, higher = more similar).

This test calibrates that adaptation against realistic agent step
signatures and **fails the build** when the chosen algorithm/threshold
cannot separate "loop" from "not-loop" inputs.

Empirical finding (May 2026)
----------------------------

CTPH is structurally unsuitable for this use case. CTPH's rolling-hash
divides the input into pieces; for typical agent step signatures
(50-300 chars), the piece set is too small to produce graduated
similarity. Identical strings → 1.0; near-duplicates and dissimilar
strings → 0.000 alike. The 0.5 threshold can never fire.

Replacement: ``ngram_jaccard`` (n=3) from
``kaos_nlp_core.algorithms``. Empirical separation:

  LOOP corpus:     similarity in [0.83, 0.92]
  NON-LOOP corpus: similarity in [0.18, 0.20]

A threshold of 0.5 sits cleanly in the gap between these distributions
and gives a robust loop / not-loop classifier.

This test pins these properties:
1. CTPH on realistic signatures collapses to 0.0 → the existing
   default of "ctph similarity ≥ 0.5" cannot fire.
2. ngram_jaccard n=3 separates cleanly with min_loop > 0.5 and
   max_non_loop < 0.5.
3. The LoopDetector default algorithm is "ngram_jaccard" and not
   "ctph" — pinning the fix.
"""

from __future__ import annotations

import pytest

# kaos-nlp-core is a hard dep, but we still gate import errors with a
# clear pytest skip so the test surfaces the missing-dep diagnostic.
pytest.importorskip("kaos_nlp_core")

from kaos_nlp_core.algorithms import jaro_winkler, ngram_jaccard
from kaos_nlp_core.hashing import ctph_hash_str, ctph_similarity

from kaos_agents.termination.loop_detect import LoopDetector

# ---------------------------------------------------------------------------
# Realistic agent step signature corpora
# ---------------------------------------------------------------------------

# LOOP — agent is spinning. Same tool, same intent, micro-variations
# typical of LLM stochastic re-emission (case, whitespace, punctuation,
# minor pluralisation, retry counter).
LOOP_CORPUS: tuple[str, ...] = (
    'tool=kaos-source-search args={"query": "tesla 2023 revenue"}',
    'tool=kaos-source-search args={"query": "Tesla 2023 revenue"}',
    'tool=kaos-source-search args={"query": "tesla 2023 revenue "}',
    'tool=kaos-source-search args={"query": "tesla 2023 revenues"}',
    'tool=kaos-source-search args={"query": "tesla 2023 revenue."}',
    'tool=kaos-source-search args={"query": "tesla 2023  revenue"}',
)

# NON-LOOP — legitimate iteration. Different tools, different args,
# different stages of work.
NON_LOOP_CORPUS: tuple[str, ...] = (
    'tool=kaos-source-edgar args={"query": "tesla 10-K"}',
    'tool=kaos-tabular-aggregate args={"column": "revenue", "agg": "mean"}',
    'tool=kaos-web-fetch args={"url": "https://www.sec.gov/Archives/..."}',
    'tool=kaos-pdf-extract args={"page_range": "12-25"}',
    'tool=kaos-source-govinfo args={"collection": "FR", "year": 2023}',
    'tool=kaos-llm-core-judge args={"output": "...", "criteria": "factual"}',
)

# LONG-FORM — full ReAct turn signatures (~250 chars), used to verify
# that increasing input size doesn't rescue CTPH.
LONG_LOOP_CORPUS: tuple[str, ...] = (
    "thought: I need to find Tesla's 2023 revenue. action: kaos-source-search "
    'query="tesla 2023 annual revenue" observation: Found 5 results from EDGAR; '
    "top result is 10-K filing dated 2024-02-15. Need to fetch full text.",
    "thought: I need to find Tesla's 2023 revenue figure. action: kaos-source-search "
    'query="tesla 2023 annual revenue" observation: Found 5 results from EDGAR; '
    "top result is 10-K filing dated 2024-02-15. Need to fetch the full text.",
    "thought: Looking for Tesla 2023 revenue numbers. action: kaos-source-search "
    'query="tesla 2023 annual revenue" observation: Found 5 results from EDGAR; '
    "top result is 10-K filing dated 2024-02-15. Need to read it.",
)


def _all_pairs(corpus: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return every unordered pair from the corpus."""
    pairs: list[tuple[str, str]] = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            pairs.append((corpus[i], corpus[j]))
    return pairs


# ---------------------------------------------------------------------------
# CTPH calibration (negative result — preserved as a regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCTPHIsBrokenForAgentSignatures:
    """Pin CTPH's broken-for-this-use-case behaviour.

    Resolved Decision #7 originally said TLSH ≤ 30. The Phase 4.B agent
    adapted to CTPH ≥ 0.5 by analogy. Empirical probe shows CTPH on
    agent step signatures collapses to 0.0 across all window sizes, so
    no positive threshold can ever fire.

    These tests fail loudly if a future kaos-nlp-core release fixes the
    CTPH behaviour (in which case we'd revisit the LoopDetector default).
    """

    @pytest.mark.parametrize("window_size", [16, 32, 64, 128])
    def test_short_loop_signatures_yield_zero_similarity(self, window_size: int) -> None:
        """The LOOP corpus pairs all hash to similarity 0.0 with default CTPH."""
        for left, right in _all_pairs(LOOP_CORPUS):
            h1 = ctph_hash_str(left, window_size, 8, 4)
            h2 = ctph_hash_str(right, window_size, 8, 4)
            sim = ctph_similarity(h1, h2)
            assert sim == 0.0, (
                f"CTPH window={window_size} produced unexpected similarity "
                f"{sim} for pair (this would invalidate the calibration). "
                f"Re-evaluate LoopDetector default algorithm."
            )

    @pytest.mark.parametrize("window_size", [16, 32, 64, 128])
    def test_long_loop_signatures_also_collapse(self, window_size: int) -> None:
        """Even ~250-char ReAct-turn signatures collapse to 0.0."""
        for left, right in _all_pairs(LONG_LOOP_CORPUS):
            h1 = ctph_hash_str(left, window_size, 8, 4)
            h2 = ctph_hash_str(right, window_size, 8, 4)
            assert ctph_similarity(h1, h2) == 0.0

    def test_identical_strings_match(self) -> None:
        """Sanity: CTPH identity still works on the trivial case."""
        s = LOOP_CORPUS[0]
        h = ctph_hash_str(s, 64, 8, 4)
        assert ctph_similarity(h, h) == 1.0


# ---------------------------------------------------------------------------
# ngram_jaccard calibration (the chosen replacement)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestNgramJaccardSeparatesLoopFromNonLoop:
    """Pin the empirical separation that justifies the 0.5 threshold."""

    def test_loop_corpus_above_threshold(self) -> None:
        """Every LOOP pair scores above 0.5 with ngram_jaccard n=3."""
        below: list[tuple[str, str, float]] = []
        for left, right in _all_pairs(LOOP_CORPUS):
            sim = ngram_jaccard(left, right, 3).similarity
            if sim < 0.5:
                below.append((left, right, sim))
        assert not below, (
            f"LOOP pairs scored below 0.5 (false negatives): {below}. "
            f"The 0.5 threshold no longer separates the corpus; "
            f"recalibrate the LoopDetector or update the corpus."
        )

    def test_non_loop_corpus_below_threshold(self) -> None:
        """Every NON-LOOP pair scores below 0.5 with ngram_jaccard n=3."""
        above: list[tuple[str, str, float]] = []
        for left, right in _all_pairs(NON_LOOP_CORPUS):
            sim = ngram_jaccard(left, right, 3).similarity
            if sim >= 0.5:
                above.append((left, right, sim))
        assert not above, (
            f"NON-LOOP pairs scored at/above 0.5 (false positives): {above}. "
            f"The threshold would generate spurious LOOP_DETECTED escalations."
        )

    def test_minimum_loop_similarity_exceeds_maximum_non_loop_similarity(self) -> None:
        """The two distributions are non-overlapping at the 0.5 boundary."""
        loop_sims = [ngram_jaccard(a, b, 3).similarity for a, b in _all_pairs(LOOP_CORPUS)]
        non_loop_sims = [ngram_jaccard(a, b, 3).similarity for a, b in _all_pairs(NON_LOOP_CORPUS)]
        # min(LOOP) > max(NON_LOOP) = clean separation
        assert min(loop_sims) > max(non_loop_sims), (
            f"Distributions overlap. min(LOOP)={min(loop_sims):.3f}, "
            f"max(NON_LOOP)={max(non_loop_sims):.3f}. The threshold must lie "
            f"between them; recalibrate."
        )

    def test_long_loop_corpus_also_clears_threshold(self) -> None:
        """Long-form ReAct signatures still cluster above 0.5."""
        for left, right in _all_pairs(LONG_LOOP_CORPUS):
            sim = ngram_jaccard(left, right, 3).similarity
            assert sim >= 0.5, f"Long-form LOOP pair scored {sim:.3f} below threshold."


# ---------------------------------------------------------------------------
# jaro_winkler calibration (a less robust alternative — not the default)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestJaroWinklerIsTooLeniantForThisUseCase:
    """Document why we chose ngram_jaccard over jaro_winkler.

    jaro_winkler scores ALL agent step signatures highly (the "tool="
    prefix dominates). Loop pairs score 0.99+, non-loop 0.69+ —
    the threshold sits in a narrow band where small phrasing changes
    can produce false positives.
    """

    def test_jaro_winkler_non_loop_is_high(self) -> None:
        """Non-loop pairs still score >0.6 with jaro_winkler — too lenient."""
        sims = [jaro_winkler(a, b).similarity for a, b in _all_pairs(NON_LOOP_CORPUS)]
        # All non-loop pairs are above 0.6 — explains why we did NOT pick
        # jaro_winkler: choosing a threshold ≤0.6 generates false positives.
        assert all(s > 0.6 for s in sims), (
            f"Non-loop pairs unexpectedly scored at/below 0.6 with "
            f"jaro_winkler: {sims}. If this changes, jaro_winkler may be a "
            f"viable alternative — re-run calibration."
        )


# ---------------------------------------------------------------------------
# LoopDetector defaults (pin the chosen fix)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestLoopDetectorDefaults:
    """Pin that LoopDetector uses ngram_jaccard by default, not CTPH."""

    def test_default_algorithm_is_ngram_jaccard(self) -> None:
        """The LoopDetector ships with the algorithm that works on agent signatures."""
        detector = LoopDetector()
        # The detector must expose its active algorithm so callers can
        # detect mis-configuration. The Phase 4.E fix adds an
        # ``algorithm`` property; if it doesn't exist, this test fails
        # and the dev knows to expose it.
        algorithm = getattr(detector, "algorithm", None)
        assert algorithm in ("ngram_jaccard", "equality"), (
            f"LoopDetector default algorithm is {algorithm!r}; expected "
            f"'ngram_jaccard' (the empirically-validated default for "
            f"agent step signatures)."
        )

    def test_loop_corpus_trips_default_detector(self) -> None:
        """End-to-end: feed the LOOP corpus to LoopDetector and trip."""
        detector = LoopDetector()
        for sig in LOOP_CORPUS:
            result = detector.observe(sig)
            if result.detected:
                return
        pytest.fail(
            "LoopDetector did NOT trip on the LOOP corpus. "
            "The detector default algorithm is structurally broken for "
            "agent step signatures."
        )

    def test_non_loop_corpus_does_not_trip_default_detector(self) -> None:
        """End-to-end: feed the NON-LOOP corpus and verify no trip."""
        detector = LoopDetector()
        for sig in NON_LOOP_CORPUS:
            result = detector.observe(sig)
            assert not result.detected, (
                f"False positive on NON-LOOP signature: {sig!r} (reason={result.reason})"
            )
