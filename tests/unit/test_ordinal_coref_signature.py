"""Unit tests for ordinal coreference resolver (plan §Issue 8 / B1.5).

Acceptance row from plan §Issue 8:

    Ordinal coref | 12-scenario fixture (each script: 5-15 turns
    culminating in "the third one") | >=90% resolution rate

This file is the deterministic resolver layer. The live LLM
integration test that drives the 12-scenario fixture lives in
`tests/integration/test_ordinal_coref_live.py` (next iteration).
The 12-case unit fixture below is the regex / index-map layer
that the live test then sanity-checks against an LLM judge.
"""

from __future__ import annotations

import pytest

from kaos_agents.context.coreference import (
    CoreferenceResolution,
    resolve_ordinal,
)

CANDIDATES = ("doc-A", "doc-B", "doc-C", "doc-D", "doc-E")


# ── Word ordinals ────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolves_first_one() -> None:
    """The simplest case — 'the first one' binds to index 0."""
    r = resolve_ordinal("Tell me about the first one.", CANDIDATES)
    assert r is not None
    assert r.ordinal == 1
    assert r.resolved_index == 0
    assert r.resolved_candidate == "doc-A"
    assert r.confidence == 1.0


@pytest.mark.unit
def test_resolves_third_nda() -> None:
    """Plan-acceptance fixture pattern: 'the third NDA'."""
    r = resolve_ordinal("What is the governing law on the third NDA?", CANDIDATES)
    assert r is not None
    assert r.ordinal == 3
    assert r.resolved_index == 2
    assert r.resolved_candidate == "doc-C"


@pytest.mark.unit
def test_resolves_word_ordinals_in_range() -> None:
    """Sweep first-through-fifth against a 5-candidate list."""
    words = ["first", "second", "third", "fourth", "fifth"]
    for n, word in enumerate(words, start=1):
        r = resolve_ordinal(f"Look at the {word} document.", CANDIDATES)
        assert r is not None
        assert r.ordinal == n
        assert r.resolved_index == n - 1
        assert r.resolved_candidate == CANDIDATES[n - 1]


# ── Numeric ordinals ────────────────────────────────────────────────


@pytest.mark.unit
def test_resolves_numeric_ordinal() -> None:
    """'the 2nd document' → index 1."""
    r = resolve_ordinal("What about the 2nd document?", CANDIDATES)
    assert r is not None
    assert r.ordinal == 2
    assert r.resolved_index == 1


@pytest.mark.unit
def test_numeric_ordinal_with_case_insensitivity() -> None:
    """'3RD', '3rd', 'THIRD' all bind the same."""
    for phrase in ("the 3RD doc", "the 3rd doc", "the THIRD doc"):
        r = resolve_ordinal(phrase, CANDIDATES)
        assert r is not None, f"{phrase!r} should resolve"
        assert r.resolved_index == 2


# ── "the last" / "the previous" / "the next" ────────────────────────


@pytest.mark.unit
def test_resolves_the_last() -> None:
    """Plan-acceptance phrase: 'the last filing'."""
    r = resolve_ordinal("Tell me about the last filing.", CANDIDATES)
    assert r is not None
    assert r.ordinal == -1
    assert r.resolved_index == 4
    assert r.resolved_candidate == "doc-E"


@pytest.mark.unit
def test_resolves_the_previous() -> None:
    """'the previous one' binds to candidates[-1]."""
    r = resolve_ordinal("What does the previous one say?", CANDIDATES)
    assert r is not None
    assert r.resolved_index == 4
    assert r.resolved_candidate == "doc-E"


@pytest.mark.unit
def test_resolves_the_next_with_heuristic_confidence() -> None:
    """'the next one' is ambiguous — we bind to index 0 with
    confidence 0.5 so the caller can surface to the user."""
    r = resolve_ordinal("What about the next one?", CANDIDATES)
    assert r is not None
    assert r.resolved_index == 0
    assert r.confidence == 0.5


# ── Out-of-range + degenerate cases ─────────────────────────────────


@pytest.mark.unit
def test_out_of_range_returns_resolution_with_low_confidence() -> None:
    """'the 99th doc' against a 5-item list — we detected the
    phrase but couldn't bind it. Resolution reports the failure
    so the caller can flag it to the user rather than silently
    dropping the reference."""
    r = resolve_ordinal("Look at the 99th doc.", CANDIDATES)
    assert r is not None
    assert r.ordinal == 99
    assert r.resolved_index is None
    assert r.resolved_candidate is None
    assert r.confidence == 0.5


@pytest.mark.unit
def test_no_ordinal_phrase_returns_none() -> None:
    """Messages without an ordinal phrase return None — caller
    skips the tag-injection step."""
    msgs = [
        "What is the governing law?",
        "Hello",
        "Summarize the documents.",
        "I'm reading first thing in the morning",  # 'first' not as ordinal
    ]
    for m in msgs:
        assert resolve_ordinal(m, CANDIDATES) is None, f"{m!r} should not match"


@pytest.mark.unit
def test_empty_candidates_returns_none() -> None:
    """No candidates → can't resolve, return None."""
    r = resolve_ordinal("the third document", ())
    assert r is None


@pytest.mark.unit
def test_empty_message_returns_none() -> None:
    """Empty string in → None out."""
    assert resolve_ordinal("", CANDIDATES) is None


# ── 12-scenario acceptance fixture (plan §Issue 8 row) ──────────────


@pytest.mark.unit
def test_12_scenario_fixture_resolution_rate() -> None:
    """Plan acceptance: ≥90% resolution rate on a 12-scenario
    fixture. Each script ends with an ordinal phrase that should
    bind correctly. This is the deterministic-resolver layer; the
    live LLM eval in tests/integration/test_ordinal_coref_live.py
    drives the same scripts through a real worker prompt.

    Fixture shape (script, candidates, expected_index, label).
    """
    scenarios: list[tuple[str, tuple[str, ...], int]] = [
        # 1
        ("Compare the NDAs. What's the third NDA's governing law?", CANDIDATES, 2),
        # 2
        ("Show me the first one again.", CANDIDATES, 0),
        # 3
        ("Re-read the second document.", CANDIDATES, 1),
        # 4
        ("Summarize the fourth file.", CANDIDATES, 3),
        # 5
        ("What does the fifth case say about indemnity?", CANDIDATES, 4),
        # 6
        ("The last filing — what's its date?", CANDIDATES, 4),
        # 7
        ("Pull up the previous document.", CANDIDATES, 4),
        # 8
        ("Look at the 2nd one and tell me about jurisdiction.", CANDIDATES, 1),
        # 9
        ("What does the 3rd doc say?", CANDIDATES, 2),
        # 10
        ("Re-read the 1st upload.", CANDIDATES, 0),
        # 11
        ("Summarize the latest filing.", CANDIDATES, 4),
        # 12
        ("The most recent one — what's the term length?", CANDIDATES, 4),
    ]
    hits = 0
    for script, cands, expected_idx in scenarios:
        r = resolve_ordinal(script, cands)
        if r is not None and r.resolved_index == expected_idx:
            hits += 1
    # Plan bar: ≥90% (≥10.8 of 12). We pin 12/12 here because the
    # deterministic resolver should be perfect on its own scripts;
    # the ≥90% bar applies to the LIVE LLM test where the model
    # may paraphrase or skip the ordinal.
    assert hits == 12, f"expected 12/12 on deterministic layer, got {hits}/12"


# ── Confidence semantics + frozen dataclass ─────────────────────────


@pytest.mark.unit
def test_confidence_is_one_for_in_range_ordinals() -> None:
    """In-range numeric or word ordinals get confidence 1.0."""
    r = resolve_ordinal("the second", CANDIDATES)
    assert r is not None
    assert r.confidence == 1.0


@pytest.mark.unit
def test_resolution_record_exposes_audit_fields() -> None:
    """The audit-trail anchor is matched_phrase + ordinal +
    resolved_index. Pin the field set so a future refactor
    doesn't silently drop something."""
    r = resolve_ordinal("Look at the third NDA.", CANDIDATES)
    assert r is not None
    assert isinstance(r, CoreferenceResolution)
    assert r.matched_phrase == "the third NDA"
    assert r.ordinal == 3
    assert r.resolved_index == 2
    assert r.resolved_candidate == "doc-C"
    assert 0.0 <= r.confidence <= 1.0
