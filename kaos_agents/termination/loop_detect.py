"""LoopDetector — Resolved Decision #7: similarity over the last N calls.

Detects when an agent is "spinning" — making the same tool call /
producing the same step output repeatedly.

The plan (rewrite-plan-ten-questions.md §13 Resolved #7) calls for
"TLSH ≤ 30 over the last 5 calls". kaos-nlp-core does not ship TLSH;
the Phase 4.B baseline adapted to CTPH (Context-Triggered Piecewise
Hashing) by analogy. **Phase 4.E calibration disproved that
choice**: CTPH on agent step signatures (50-300 chars typical) collapses
to similarity 0.0 across all window sizes — the rolling-hash piece
set is too small to produce graduated similarity. The 0.5 threshold
could never fire.

Phase 4.E switches the default algorithm to **n-gram Jaccard** (n=3)
from ``kaos_nlp_core.algorithms.ngram_jaccard``. Empirical separation
on hand-crafted corpora (see
``tests/benchmarks/test_loop_detection_calibration.py``):

  LOOP corpus (micro-variations on the same tool call):
    similarity in [0.83, 0.92] — a stable cluster
  NON-LOOP corpus (legitimately different tools/args):
    similarity in [0.18, 0.20] — well below the cluster

A threshold of 0.5 sits cleanly in the gap between the two
distributions and gives a robust loop / not-loop classifier.

Algorithm options (the constructor's ``algorithm`` kwarg):

  * ``"ngram_jaccard"`` (default) — character 3-gram Jaccard. Best
    for typical agent step signatures.
  * ``"ctph"`` — CTPH from ``kaos_nlp_core.hashing``. Retained for
    tests and forward-compat; **not recommended** for short
    signatures (see calibration evidence above).
  * ``"equality"`` — exact string match. Used as a final fallback
    when kaos-nlp-core isn't importable. Coarser, but catches the
    "exact-same tool call N times in a row" base case.

The detector never raises — a missing kaos-nlp-core import silently
degrades to equality. The agent must keep running even when the
loop-detection optimisation is unavailable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

_AlgorithmName = str  # "ngram_jaccard" | "ctph" | "equality"


@dataclass(frozen=True, slots=True)
class LoopDetectorResult:
    """Verdict from :meth:`LoopDetector.observe` / :meth:`LoopDetector.check`.

    ``detected`` is the headline. When ``True``, ``matching_pair``
    indexes the offending pair within the current window (0-based,
    oldest-first), ``similarity`` is the algorithm-specific similarity
    score (or ``1.0`` on the equality fallback), and ``reason`` is a
    one-line human-readable explanation suitable for the Decision's
    ``feedback`` field.
    """

    detected: bool
    reason: str = ""
    matching_pair: tuple[int, int] | None = None
    similarity: float | None = None


class LoopDetector:
    """Sliding-window loop detector over the last N call signatures.

    Constructor kwargs:

      window_size: how many calls to consider (default 5; Resolved #7).
      min_similarity: similarity threshold above which a pair is a
        "loop". Default ``0.5`` — empirically validated against the
        Phase 4.E calibration corpora for ``ngram_jaccard``.
      algorithm: one of ``"ngram_jaccard"`` (default), ``"ctph"``, or
        ``"equality"``. ``"ngram_jaccard"`` is the only one that
        produces graduated similarity on typical agent step
        signatures; the others are retained for tests and degraded
        environments.
      use_fuzzy: legacy alias kept for back-compat. ``False`` forces
        ``"equality"``; ``True`` (default) honours ``algorithm``.
    """

    _ALGORITHMS = ("ngram_jaccard", "ctph", "equality")

    def __init__(
        self,
        *,
        window_size: int = 5,
        min_similarity: float = 0.5,
        algorithm: _AlgorithmName = "ngram_jaccard",
        use_fuzzy: bool = True,
    ) -> None:
        if window_size < 2:
            window_size = 2
        self._window_size = window_size
        self._min_similarity = float(min_similarity)
        # Resolve effective algorithm honouring the legacy ``use_fuzzy``
        # kwarg: ``use_fuzzy=False`` collapses to "equality" regardless
        # of the chosen algorithm.
        if not use_fuzzy:
            chosen = "equality"
        elif algorithm in self._ALGORITHMS:
            chosen = algorithm
        else:
            chosen = "ngram_jaccard"
        # Probe and degrade to equality if the chosen algorithm's
        # backing kaos-nlp-core API is unavailable. Probe failures
        # become a silent fallback so the agent never crashes on a
        # missing optional dep — but log via the algorithm property
        # so callers can introspect.
        if chosen != "equality":
            available = self._probe(chosen)
            if not available:
                chosen = "equality"
        self._algorithm: _AlgorithmName = chosen
        self._signatures: deque[str] = deque(maxlen=window_size)

    # ---- public introspection -------------------------------------

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_similarity(self) -> float:
        return self._min_similarity

    @property
    def algorithm(self) -> _AlgorithmName:
        """The active algorithm (post-fallback resolution)."""
        return self._algorithm

    @property
    def fuzzy_available(self) -> bool:
        """Back-compat: ``True`` iff a fuzzy algorithm is active."""
        return self._algorithm != "equality"

    # ---- core API --------------------------------------------------

    def observe(self, signature: str) -> LoopDetectorResult:
        """Add a signature to the window and check for a loop."""
        self._signatures.append(signature)
        return self.check()

    def check(self) -> LoopDetectorResult:
        """Test the current window without modifying it."""
        if len(self._signatures) < 2:
            return LoopDetectorResult(detected=False)
        sigs = list(self._signatures)
        scorer = self._scorer_for(self._algorithm)
        if scorer is None:
            return self._check_equality(sigs)
        return self._check_with_scorer(sigs, scorer, self._algorithm)

    def reset(self) -> None:
        """Clear the window (e.g. after a successful replan)."""
        self._signatures.clear()

    # ---- internals -------------------------------------------------

    @staticmethod
    def _probe(algorithm: _AlgorithmName) -> bool:
        """Return True if the kaos-nlp-core API for ``algorithm`` is importable."""
        try:
            if algorithm == "ngram_jaccard":
                from kaos_nlp_core.algorithms import ngram_jaccard  # noqa: F401
            elif algorithm == "ctph":
                from kaos_nlp_core.hashing import (  # noqa: F401
                    ctph_hash_str,
                    ctph_similarity,
                )
            else:
                return True
        except (ImportError, AttributeError):
            return False
        return True

    @staticmethod
    def _scorer_for(algorithm: _AlgorithmName) -> Callable[[str, str], float] | None:
        """Return a (str, str) -> similarity callable, or None for equality."""
        if algorithm == "ngram_jaccard":
            from kaos_nlp_core.algorithms import ngram_jaccard

            def _score_ngram(a: str, b: str) -> float:
                try:
                    return float(ngram_jaccard(a, b, 3).similarity)
                except Exception:
                    return 0.0

            return _score_ngram
        if algorithm == "ctph":
            from kaos_nlp_core.hashing import ctph_hash_str, ctph_similarity

            def _score_ctph(a: str, b: str) -> float:
                try:
                    h1 = ctph_hash_str(a)
                    h2 = ctph_hash_str(b)
                    if not h1 or not h2:
                        return 0.0
                    return float(ctph_similarity(h1, h2))
                except Exception:
                    return 0.0

            return _score_ctph
        return None

    def _check_with_scorer(
        self,
        sigs: list[str],
        scorer: Callable[[str, str], float],
        label: str,
    ) -> LoopDetectorResult:
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                sim = scorer(sigs[i], sigs[j])
                if sim >= self._min_similarity:
                    return LoopDetectorResult(
                        detected=True,
                        reason=(
                            f"{label} similarity {sim:.2f} >= "
                            f"{self._min_similarity:.2f} between calls {i} and {j}"
                        ),
                        matching_pair=(i, j),
                        similarity=sim,
                    )
        return LoopDetectorResult(detected=False)

    def _check_equality(self, sigs: list[str]) -> LoopDetectorResult:
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                if sigs[i] == sigs[j]:
                    return LoopDetectorResult(
                        detected=True,
                        reason=f"Exact-match between calls {i} and {j}",
                        matching_pair=(i, j),
                        similarity=1.0,
                    )
        return LoopDetectorResult(detected=False)


__all__ = ["LoopDetector", "LoopDetectorResult"]
