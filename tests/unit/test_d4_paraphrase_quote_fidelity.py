"""D4 paraphrase / quote-fidelity regression fixtures (plan §Issue 1).

Plan §Issue 1 D4 acceptance row: "uploaded doc with verbatim
clause X; user asks 'what does X say'; agent quotes a paraphrase
→ M3 paraphrase-detector flags; loop replans for verbatim quote OR
labels as paraphrase".

This file ships fixture pairs that drive a future M3-grounding
paraphrase detector. The actual detector lives in
``kaos_agents/planning/m3_grounding.py`` (per plan §Issue 1 layer
table) and ships in kaos-agents 0.1.8; this test file pins the
reference fixture pairs (verbatim vs paraphrase) the detector will
be evaluated against.

The fixtures stand alone as a regression library — even before the
detector ships, a contributor can read the file and see exactly
what shape "paraphrase that should be flagged" takes in
production legal-corpus text. When the detector lands, swap the
``_is_paraphrase`` placeholder for the real call.
"""

from __future__ import annotations

import re

import pytest

# ── Paraphrase detection contract (placeholder until shipped) ──────


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation — the
    minimum normalization the M3 detector applies before
    similarity scoring."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split().__str__()


def _is_paraphrase(source: str, quoted: str) -> bool:
    """Placeholder paraphrase-vs-verbatim check.

    Real implementation will use the M3 grounding rubric in
    ``kaos_agents/planning/m3_grounding.py`` (longest common
    substring + token-set similarity); for the fixture pin, we
    declare a paraphrase whenever the quoted text isn't a literal
    substring of the source AND has > 50% token overlap.

    The fixture-level contract this gives us: each paraphrase
    fixture below evaluates to True; each verbatim fixture
    evaluates to False. The detector ships when the same fixture
    pairs route through the rubric and produce the same labels."""
    if quoted in source:
        return False  # Verbatim — substring match.
    # Quoted is NOT a substring → paraphrase candidate. Compute
    # token-set overlap to filter out unrelated text.
    src_tokens = set(re.findall(r"\w+", source.lower()))
    q_tokens = set(re.findall(r"\w+", quoted.lower()))
    if not q_tokens:
        return False
    overlap = len(src_tokens & q_tokens) / len(q_tokens)
    return overlap > 0.5


# ── Fixture pairs ──────────────────────────────────────────────────


# Each tuple: (source_text_from_uploaded_doc, agent_quoted_text,
#              expected_is_paraphrase, label).
_FIXTURE_PAIRS: tuple[tuple[str, str, bool, str], ...] = (
    # 1. Verbatim — exact substring match.
    (
        "This Agreement shall be governed by and construed in "
        "accordance with the laws of the State of Delaware.",
        "shall be governed by and construed in accordance with the laws of the State of Delaware",
        False,
        "verbatim-governing-law-clause",
    ),
    # 2. Paraphrase — same meaning, different words.
    (
        "This Agreement shall be governed by and construed in "
        "accordance with the laws of the State of Delaware.",
        "This contract is governed by Delaware state law",
        True,
        "paraphrase-governing-law-clause",
    ),
    # 3. Verbatim — multi-sentence quote.
    (
        "Confidential Information means any information disclosed by "
        "Discloser to Recipient. The Receiving Party shall hold all "
        "Confidential Information in strict confidence.",
        "The Receiving Party shall hold all Confidential Information in strict confidence",
        False,
        "verbatim-NDA-confidentiality-clause",
    ),
    # 4. Paraphrase — semantically equivalent but rephrased.
    (
        "Confidential Information means any information disclosed by "
        "Discloser to Recipient. The Receiving Party shall hold all "
        "Confidential Information in strict confidence.",
        "The recipient must keep all disclosed information strictly confidential",
        True,
        "paraphrase-NDA-confidentiality",
    ),
    # 5. Verbatim — short fragment.
    (
        "Either party may terminate this Agreement with thirty (30) days written notice.",
        "thirty (30) days written notice",
        False,
        "verbatim-termination-fragment",
    ),
    # 6. Paraphrase — numbers preserved but phrasing changed.
    (
        "Either party may terminate this Agreement with thirty (30) days written notice.",
        "Either party can end this agreement by giving 30 days notice",
        True,
        "paraphrase-termination-clause",
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize("source, quoted, expected, label", _FIXTURE_PAIRS)
def test_paraphrase_detector_fixture_pair(
    source: str, quoted: str, expected: bool, label: str
) -> None:
    """Each fixture pair drives the paraphrase/verbatim contract.
    A regression in either direction (verbatim mis-flagged as
    paraphrase, or paraphrase silently passed as verbatim) fails
    this gate."""
    result = _is_paraphrase(source, quoted)
    assert result is expected, (
        f"Fixture {label!r}: expected is_paraphrase={expected}, "
        f"got {result}. source={source[:60]!r}... quoted={quoted!r}"
    )


# ── Fixture sanity ────────────────────────────────────────────────


@pytest.mark.unit
def test_fixture_pair_coverage_includes_both_labels() -> None:
    """The fixture set must contain BOTH verbatim and paraphrase
    cases — a regression that drops all paraphrase rows would let
    the detector pass with a degenerate "always returns False"
    implementation."""
    verbatim_count = sum(1 for _, _, e, _ in _FIXTURE_PAIRS if e is False)
    paraphrase_count = sum(1 for _, _, e, _ in _FIXTURE_PAIRS if e is True)
    assert verbatim_count >= 2, "Need at least 2 verbatim fixtures"
    assert paraphrase_count >= 2, "Need at least 2 paraphrase fixtures"


@pytest.mark.unit
def test_fixture_labels_are_unique() -> None:
    """Each fixture pair gets a unique audit label so a test
    failure points to the exact row."""
    labels = [label for _, _, _, label in _FIXTURE_PAIRS]
    assert len(set(labels)) == len(labels), f"Duplicate labels in fixture set: {labels}"


# ── Edge cases ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_empty_quote_is_not_paraphrase() -> None:
    """An empty quoted string is degenerate — not a paraphrase."""
    assert _is_paraphrase("source text", "") is False


@pytest.mark.unit
def test_unrelated_quote_is_not_paraphrase() -> None:
    """A quote with no token overlap with the source is unrelated,
    not a paraphrase. The M3 detector's role is to catch
    paraphrase-of-source, not flag every divergence."""
    source = "This Agreement is governed by Delaware law."
    quote = "Pizza is delicious"
    assert _is_paraphrase(source, quote) is False
