"""R1.4 — semantic stuck-detection (reliability roadmap #564).

Pre-R1.4, ``_is_stuck`` only fired on byte-identity OR substring
relationship. Agent 4's C5 case had two semantically identical refusals
with different wording — the substring check missed them and the loop
kept burning budget on near-duplicate iterations.

R1.4 adds a Jaccard 3-gram similarity check that catches the
cosmetic-rewording pattern while still tolerating substantive
refinements.
"""

from __future__ import annotations

import pytest

from kaos_agents.patterns.agentic_loop import (
    _char_3grams,
    _is_stuck,
    _jaccard_similarity,
)

pytestmark = pytest.mark.unit


class TestChar3Grams:
    def test_empty_string_yields_empty_set(self):
        assert _char_3grams("") == set()

    def test_short_string_under_three_chars_is_single_token(self):
        # Special case: text shorter than 3 chars can't form 3-grams.
        # We return the normalized text itself as a single token so a
        # comparison still degrades gracefully.
        result = _char_3grams("hi")
        assert result == {"hi"}

    def test_lowercases_and_normalizes_whitespace(self):
        """Cosmetic capitalization + whitespace differences must NOT
        change the 3-gram set, so reformatting alone can't fool the
        stuck detector."""
        a = _char_3grams("Hello World")
        b = _char_3grams("hello   world")
        c = _char_3grams("HELLO\nWORLD")
        assert a == b == c

    def test_includes_expected_3grams(self):
        result = _char_3grams("abc")
        assert "abc" in result


class TestJaccardSimilarity:
    def test_identical_sets_score_1(self):
        s = {"abc", "bcd"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets_score_0(self):
        assert _jaccard_similarity({"abc"}, {"xyz"}) == 0.0

    def test_both_empty_returns_0(self):
        assert _jaccard_similarity(set(), set()) == 0.0

    def test_partial_overlap(self):
        # intersection 1, union 3 → 1/3
        result = _jaccard_similarity({"abc", "bcd"}, {"abc", "cde"})
        assert abs(result - (1.0 / 3.0)) < 1e-9


class TestIsStuckSemantic:
    """R1.4: ``_is_stuck`` fires when two iterations produce
    semantically-identical text, even when the substring check would
    miss them."""

    def test_byte_identical_still_fires(self):
        """Pre-R1.4 contract preserved: identical text → stuck."""
        assert _is_stuck(
            last_text="I cannot find the answer to that question.",
            new_text="I cannot find the answer to that question.",
            last_tool_count=2,
            new_tool_count=2,
        )

    def test_substring_relationship_still_fires(self):
        """Pre-R1.4 contract preserved: prefix/suffix relationship → stuck."""
        assert _is_stuck(
            last_text="I cannot find.",
            new_text="I cannot find. Try again later.",
            last_tool_count=2,
            new_tool_count=2,
        )

    def test_cosmetic_rewording_fires(self):
        """R1.4 new contract: very similar text with cosmetic rewording
        → stuck. ``_SEMANTIC_STUCK_JACCARD_THRESHOLD`` = 0.85.

        Constructed inputs whose 3-gram sets overlap heavily.
        """
        a = (
            "Based on my searches I cannot find any information about "
            "the specific 2025 enforcement action you asked about."
        )
        b = (
            "Based on my searches I cannot find any information about "
            "the specific 2025 enforcement action you asked about today."
        )
        # b is almost a superset of a — Jaccard should be near 1. The
        # substring check would catch this case too, but we also want
        # to exercise the new code path.
        assert _is_stuck(
            last_text=a,
            new_text=b,
            last_tool_count=3,
            new_tool_count=3,
        )

    def test_substantive_refinement_does_not_fire(self):
        """R1.4: a genuinely different answer must NOT trip the
        semantic check. Substantive refinements typically score
        well below the 0.85 threshold."""
        a = "I searched and found nothing relevant."
        b = (
            "Section 5(b) of the Williams Act requires beneficial owners "
            "of more than 5% of a public company's stock to file Schedule "
            "13D within 10 days. The 2024 amendment shortened the deadline "
            "to 5 business days."
        )
        # The two strings share almost no 3-grams — Jaccard ≪ 0.85.
        # Even though new_tool_count==last_tool_count (no new tools), the
        # semantic check must NOT fire.
        assert not _is_stuck(
            last_text=a,
            new_text=b,
            last_tool_count=2,
            new_tool_count=2,
        )

    def test_new_tool_calls_override_text_check(self):
        """When the new iteration made progress via tools, the loop is
        NOT stuck even if the text didn't change."""
        assert not _is_stuck(
            last_text="Working on it...",
            new_text="Working on it...",
            last_tool_count=2,
            new_tool_count=5,  # 3 new tool calls = real progress
        )

    def test_empty_text_does_not_fire(self):
        """Defensive: empty inputs don't fire (matches pre-R1.4 behavior)."""
        assert not _is_stuck(
            last_text="",
            new_text="hi",
            last_tool_count=0,
            new_tool_count=0,
        )
        assert not _is_stuck(
            last_text="hi",
            new_text="",
            last_tool_count=0,
            new_tool_count=0,
        )

    def test_semantic_check_threshold_boundary(self):
        """The 0.85 threshold should fire on near-clones and not on
        moderately-similar but distinct messages."""
        # Two truly different responses with some shared vocabulary —
        # similarity should be moderate (well below 0.85).
        a = "The Federal Register published a new rule on emissions."
        b = "The Federal Register also published a different rule on data privacy."
        sim = _jaccard_similarity(_char_3grams(a), _char_3grams(b))
        # Sanity: this pair scores below threshold.
        assert sim < 0.85
        assert not _is_stuck(
            last_text=a,
            new_text=b,
            last_tool_count=1,
            new_tool_count=1,
        )
