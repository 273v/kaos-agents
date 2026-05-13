"""Unit tests for kaos_agents.termination.loop_detect — LoopDetector.

Covers the empty/single/two-call edge cases, the equality fallback,
the sliding-window behaviour, the CTPH fuzzy path (gated on
kaos-nlp-core availability), and ``reset()``.

The plan called for TLSH; kaos-nlp-core ships CTPH (Jaccard
similarity, higher = more similar). The fuzzy-path tests probe
``LoopDetector.fuzzy_available`` so they pass on machines without
kaos-nlp-core too — equality fallback is the contract floor.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kaos_agents.termination.loop_detect import LoopDetector, LoopDetectorResult


class TestEdgeCases:
    def test_empty_window_not_detected(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        result = detector.check()
        assert result.detected is False

    def test_single_observation_not_detected(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        result = detector.observe("tool_x{arg=1}")
        assert result.detected is False

    def test_two_distinct_calls_not_detected(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        detector.observe("tool_x{arg=1}")
        result = detector.observe("tool_y{arg=2}")
        assert result.detected is False


class TestEqualityFallback:
    """The fallback contract — guaranteed even without kaos-nlp-core."""

    def test_two_identical_calls_detected(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        detector.observe("tool_x{arg=1}")
        result = detector.observe("tool_x{arg=1}")
        assert result.detected is True
        assert result.matching_pair == (0, 1)
        assert result.similarity == pytest.approx(1.0)
        assert "Exact-match" in result.reason

    def test_window_slides_past_max(self) -> None:
        detector = LoopDetector(window_size=3, use_fuzzy=False)
        detector.observe("a")
        detector.observe("b")
        detector.observe("c")
        # Window is now [a, b, c] — no duplicates.
        result = detector.check()
        assert result.detected is False
        # Push "a" out by adding "d"; now window is [b, c, d].
        detector.observe("d")
        # "a" reappearing should NOT match the old "a" (it aged out).
        result = detector.observe("a")
        assert result.detected is False

    def test_duplicate_within_window_detected(self) -> None:
        detector = LoopDetector(window_size=4, use_fuzzy=False)
        detector.observe("a")
        detector.observe("b")
        detector.observe("c")
        result = detector.observe("a")
        # window is [a, b, c, a] — pair (0, 3) matches.
        assert result.detected is True
        assert result.matching_pair == (0, 3)

    def test_reset_clears_window(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        detector.observe("x")
        detector.observe("x")
        assert detector.check().detected is True
        detector.reset()
        assert detector.check().detected is False
        # And a fresh duplicate should re-trip.
        detector.observe("x")
        assert detector.observe("x").detected is True


class TestWindowClamp:
    def test_window_below_two_clamped_to_two(self) -> None:
        # A window of 1 can never have a pair; the constructor clamps.
        detector = LoopDetector(window_size=1, use_fuzzy=False)
        assert detector.window_size == 2
        detector.observe("x")
        result = detector.observe("x")
        assert result.detected is True


class TestFuzzyPath:
    """CTPH fuzzy-hash path — gated on kaos-nlp-core availability."""

    def test_fuzzy_available_flag(self) -> None:
        detector = LoopDetector()
        # The flag must be a bool either way.
        assert isinstance(detector.fuzzy_available, bool)

    def test_fuzzy_disabled_flag(self) -> None:
        detector = LoopDetector(use_fuzzy=False)
        assert detector.fuzzy_available is False

    def test_fuzzy_detects_exact_duplicate(self) -> None:
        detector = LoopDetector()
        if not detector.fuzzy_available:
            pytest.skip("kaos-nlp-core fuzzy hashing not available")
        # Long-enough payload that CTPH produces a non-empty hash.
        sig = "tool=kaos-source-fr-search args={query='climate change', year=2024} step=3"
        detector.observe(sig)
        result = detector.observe(sig)
        assert result.detected is True
        assert result.similarity is not None
        assert result.similarity >= 0.5

    def test_fuzzy_distinct_calls_not_detected(self) -> None:
        detector = LoopDetector()
        if not detector.fuzzy_available:
            pytest.skip("kaos-nlp-core fuzzy hashing not available")
        detector.observe("tool=kaos-source-fr-search args={query='climate change', year=2024}")
        result = detector.observe("tool=kaos-source-edgar-filing args={cik=320193, form=10-K}")
        assert result.detected is False


class TestResultType:
    def test_result_is_frozen_dataclass(self) -> None:
        result = LoopDetectorResult(detected=True, reason="x", similarity=0.9)
        # ``setattr`` keeps the static type checker happy while still
        # exercising the frozen-dataclass runtime contract.
        with pytest.raises(FrozenInstanceError):
            setattr(result, "detected", False)  # noqa: B010

    def test_default_fields(self) -> None:
        result = LoopDetectorResult(detected=False)
        assert result.reason == ""
        assert result.matching_pair is None
        assert result.similarity is None
