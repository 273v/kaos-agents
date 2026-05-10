"""LoopDetector — Resolved Decision #7: fuzzy-hash similarity over the last N calls.

Detects when an agent is "spinning" — making the same tool call /
producing the same step output repeatedly.

The plan (rewrite-plan-ten-questions.md §13 Resolved #7) calls for
"TLSH ≤ 30 over the last 5 calls". kaos-nlp-core's fuzzy-hashing
module ships **CTPH** (Context-Triggered Piecewise Hashing) rather
than TLSH — see ``kaos_nlp_core.hashing``: ``ctph_hash_str`` +
``ctph_similarity`` (Jaccard over hash blocks). We adapt:

* "TLSH distance ≤ 30" (lower = more similar) becomes
  "CTPH similarity ≥ ``min_similarity``" (higher = more similar).
* Default ``min_similarity = 0.5`` — well above the noise floor for
  CTPH on short agent step signatures, where exact-equal pairs
  return 1.0 and unrelated pairs typically return 0.0.

Phase 4.B baseline: window of the last N calls, hash each, compare
pairwise. Any pair above the threshold trips. Fallback when
kaos-nlp-core isn't available (or its hashing surface changes
shape): simple string equality over the last N — coarser, but
still catches the "exact-same tool call N times in a row" base
case.

The detector never raises — a missing or renamed kaos-nlp-core
hashing API silently degrades to equality. The agent must keep
running even when the loop-detection optimisation is unavailable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoopDetectorResult:
    """Verdict from :meth:`LoopDetector.observe` / :meth:`LoopDetector.check`.

    ``detected`` is the headline. When ``True``, ``matching_pair``
    indexes the offending pair within the current window (0-based,
    oldest-first), ``similarity`` is the CTPH Jaccard score on the
    fuzzy path (or ``1.0`` on the equality fallback), and ``reason``
    is a one-line human-readable explanation suitable for the
    Decision's ``feedback`` field.
    """

    detected: bool
    reason: str = ""
    matching_pair: tuple[int, int] | None = None  # indices in the window
    similarity: float | None = None


class LoopDetector:
    """Sliding-window loop detector over the last N call signatures.

    Constructor kwargs:

      window_size: how many calls to consider (default 5; Resolved #7).
      min_similarity: CTPH Jaccard threshold above which a pair is
        considered "the same step". Default ``0.5`` — exact duplicates
        score ``1.0``, unrelated pairs typically score ``0.0`` on
        agent-step-sized inputs, so anything ``>= 0.5`` is a real
        signal.
      use_fuzzy: ``True`` (default) to attempt CTPH; ``False`` forces
        the string-equality fallback. Useful for tests that need
        deterministic behaviour without the kaos-nlp-core dependency.

    Usage::

        detector = LoopDetector()
        detector.observe("tool_x{arg=1}")  # signature string
        result = detector.observe("tool_x{arg=1}")
        if result.detected:
            ...

    The window only grows by one per :meth:`observe`; older entries
    age out via the deque's ``maxlen``.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        min_similarity: float = 0.5,
        use_fuzzy: bool = True,
    ) -> None:
        if window_size < 2:
            # A window of <2 can never have a pair; clamp upward so the
            # contract "observe N identical calls => detected" holds.
            window_size = 2
        self._window_size = window_size
        self._min_similarity = float(min_similarity)
        self._use_fuzzy = use_fuzzy
        self._signatures: deque[str] = deque(maxlen=window_size)
        self._fuzzy_available = self._probe_fuzzy() if use_fuzzy else False

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def fuzzy_available(self) -> bool:
        """``True`` when kaos-nlp-core CTPH is importable."""
        return self._fuzzy_available

    def observe(self, signature: str) -> LoopDetectorResult:
        """Add a signature to the window and check for a loop."""
        self._signatures.append(signature)
        return self.check()

    def check(self) -> LoopDetectorResult:
        """Test the current window without modifying it."""
        if len(self._signatures) < 2:
            return LoopDetectorResult(detected=False)
        sigs = list(self._signatures)
        if self._fuzzy_available:
            return self._check_fuzzy(sigs)
        return self._check_equality(sigs)

    def reset(self) -> None:
        """Clear the window (e.g. after a successful replan)."""
        self._signatures.clear()

    # ---- internals -------------------------------------------------

    def _probe_fuzzy(self) -> bool:
        """Return True if kaos-nlp-core CTPH is importable.

        kaos-nlp-core is the fuzzy-hashing source-of-truth in this
        monorepo. The package ships CTPH (``ctph_hash_str`` +
        ``ctph_similarity``) rather than TLSH. If the import fails for
        any reason — module not installed, API renamed, build failure
        — we silently fall back to string equality. The agent must
        keep working without the optimisation.
        """
        try:
            from kaos_nlp_core.hashing import ctph_hash_str, ctph_similarity  # noqa: F401
        except (ImportError, AttributeError):
            return False
        return True

    def _check_fuzzy(self, sigs: list[str]) -> LoopDetectorResult:
        # Late import; we already know the names exist from _probe_fuzzy.
        from kaos_nlp_core.hashing import ctph_hash_str, ctph_similarity

        try:
            hashes = [ctph_hash_str(s) for s in sigs]
        except Exception:
            # Any hashing failure (binary input quirks, etc.) → fall
            # back to equality for this check rather than crash.
            return self._check_equality(sigs)

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if not hashes[i] or not hashes[j]:
                    continue
                try:
                    sim = float(ctph_similarity(hashes[i], hashes[j]))
                except Exception:
                    continue
                if sim >= self._min_similarity:
                    return LoopDetectorResult(
                        detected=True,
                        reason=(
                            f"CTPH similarity {sim:.2f} >= {self._min_similarity:.2f} "
                            f"between calls {i} and {j}"
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
