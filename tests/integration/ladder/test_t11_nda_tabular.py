"""Tier 11: NDA tabular review — extract 10 key fields, score vs ground truth.

The deal-room batch-extraction use case. Real DOCX fixtures (5 mutual
NDAs from a working portfolio), real parsing through kaos-office, real
LLM extraction into a structured table, then **field-by-field scoring
against a manually-built ground-truth dataset** in
``fixtures/nda_ground_truth.py``.

Ten fields per NDA — picked to mirror what a senior associate would
actually pull off a deal-room contract:

  1. counterparty
  2. counterparty_jurisdiction (state/country of formation)
  3. effective_date (ISO YYYY-MM-DD)
  4. governing_law
  5. venue
  6. term_years
  7. confidentiality_period_years
  8. mutual
  9. solicitation_period_months
 10. counterparty_signatory_name

The test asserts >=80% of (row, field) cells match ground truth (so
8 of 10 fields right per contract on average across the 5).

Cost target: ~$0.20 (one Anthropic call with ~12K tokens input).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
)
from tests.integration.ladder.fixtures.nda_ground_truth import (
    EXTRACTION_FIELDS,
    GROUND_TRUTH,
    ground_truth_by_filename,
)

pytestmark = pytest.mark.live

TIER = 11
BUDGET_USD = 0.30

NDA_DIR = Path(__file__).parent / "fixtures" / "nda"
EXPECTED_FILES = tuple(gt.filename for gt in GROUND_TRUTH)


def _load_nda_corpus() -> dict[str, str]:
    """Parse each fixture DOCX into plain text, keyed by filename."""
    from kaos_content import serialize_text
    from kaos_office import parse_docx

    out: dict[str, str] = {}
    for name in EXPECTED_FILES:
        path = NDA_DIR / name
        assert path.exists(), f"missing fixture: {path}"
        doc = parse_docx(str(path))
        out[name] = serialize_text(doc)
    return out


def _extract_json_array(text: str) -> list[dict]:
    """Find the first JSON array in ``text`` and parse it."""
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text
    bracket = re.search(r"\[.+\]", candidate, re.DOTALL)
    if bracket is None:
        raise ValueError(f"no JSON array in response: {text[:300]!r}")
    return json.loads(bracket.group(0))


def _normalize_str(value: object) -> str:
    """Lowercase + strip + collapse whitespace + drop trailing punctuation."""
    s = str(value or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".,;")


def _value_matches(predicted: object, expected: object, *, key: str) -> bool:
    """Per-field semantic equality — tolerates obvious format variation.

    Strict on values that have a single canonical form (mutual=True,
    term_years=2). Looser on free-text values (counterparty,
    governing_law) where the model can legitimately phrase things
    differently than the ground truth.
    """
    # None / null handling: ground truth None means "not specified"
    # (open-ended term, perpetual confidentiality, no non-solicit).
    # The model may return None / null / "" / "n/a" — all valid.
    none_predicates = {None, "", "null", "n/a", "none", "not specified"}
    if expected is None:
        return predicted is None or _normalize_str(predicted) in none_predicates
    if predicted is None and expected is not None:
        return False

    # Bool fields
    if isinstance(expected, bool):
        if isinstance(predicted, bool):
            return predicted == expected
        return _normalize_str(predicted) in (
            ("yes", "true", "1") if expected else ("no", "false", "0")
        )

    # Numeric fields — accept exact match OR same int after str→int
    if isinstance(expected, int):
        if predicted is None:
            return False
        try:
            return int(predicted) == expected  # ty: ignore[invalid-argument-type]
        except (TypeError, ValueError):
            return False

    # Date field — exact ISO YYYY-MM-DD
    if key == "effective_date":
        return _normalize_str(predicted) == _normalize_str(expected)

    # Canonical-form string fields (governing_law, venue) — the
    # expected value must appear in the predicted, but NOT vice versa.
    # This rejects "Michigan or Delaware" matching "Michigan" — a
    # genuine ambiguity should not score as correct. (Original
    # substring-either-direction matching let through hedged
    # answers.)
    if key in {"governing_law", "venue"}:
        p = _normalize_str(predicted)
        e = _normalize_str(expected)
        if not p or not e:
            return p == e
        if e not in p:
            return False
        # Reject hedging: predicted contains "<expected> or <other>"
        # or "<other> or <expected>" patterns that include another
        # known jurisdiction. Common hedge tokens: "or", "/", " and ".
        hedge_tokens = (" or ", "/", " and ")
        other_states = (
            "delaware",
            "michigan",
            "california",
            "nevada",
            "new york",
            "texas",
            "florida",
        )
        for hedge in hedge_tokens:
            if hedge in p:
                # Look for another known state-name on the OTHER side
                # of the hedge — that's the hedging pattern we reject.
                parts = p.split(hedge)
                other_parts = [pt for pt in parts if e not in pt]
                if any(any(s in pt for s in other_states) for pt in other_parts):
                    return False
        return True

    # Other string fields — substring match either direction (tolerates
    # "Acme Co." vs "Acme Co" vs "Acme Company").
    p = _normalize_str(predicted)
    e = _normalize_str(expected)
    if not p or not e:
        return p == e
    return e in p or p in e


@pytest.mark.asyncio
async def test_nda_tabular_extraction_vs_ground_truth(ladder_runner_factory, turn_session_id):
    """10-field extraction across 5 NDAs scored field-by-field vs ground truth."""
    docs = _load_nda_corpus()
    assert len(docs) == 5, f"expected 5 NDAs, loaded {len(docs)}"

    # Build a single-prompt context: each contract prefixed with its
    # filename. Pass the FULL text — governing-law / venue / term
    # clauses live near the end of every contract.
    context = "\n\n".join(f"=== CONTRACT: {name} ===\n{text}" for name, text in docs.items())
    fields_csv = ", ".join(f'"{f}"' for f in EXTRACTION_FIELDS)
    msg = (
        "Below are 5 mutual NDAs. For EACH contract, extract these 10 fields:\n"
        '  - "filename": the contract filename (exactly as shown above)\n'
        '  - "counterparty": the non-273-Ventures party name\n'
        '  - "counterparty_jurisdiction": state or country where the counterparty is formed\n'
        '  - "effective_date": ISO YYYY-MM-DD\n'
        '  - "governing_law": the U.S. state whose laws govern (single state name)\n'
        '  - "venue": court venue per the jurisdiction-and-dispute-resolution clause\n'
        '  - "term_years": integer years the Agreement runs (null if open-ended)\n'
        '  - "confidentiality_period_years": years confidentiality survives termination '
        "(null if perpetual / until written release)\n"
        '  - "mutual": boolean (true/false)\n'
        '  - "solicitation_period_months": months of post-termination non-solicit '
        "(null if no non-solicit clause)\n"
        '  - "counterparty_signatory_name": the named signatory of the counterparty\n\n'
        "Output ONLY a JSON array with exactly 5 objects, one per contract. "
        f"Use these exact keys: {fields_csv}. "
        "No prose, no code fences, no commentary.\n\n"
        f"{context}"
    )

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a contracts-extraction assistant. Output structured "
                "JSON exactly as requested. Be precise and conservative — use "
                "null when a field is genuinely absent from the contract."
            ),
            "model": model_for_tier(TIER),
        },
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-11")

    text = text_from_summary(events)
    rows = _extract_json_array(text)
    assert len(rows) == 5, f"expected 5 rows, got {len(rows)}"

    # Score field-by-field against ground truth
    truth_by_name = ground_truth_by_filename()
    field_keys = [k for k in EXTRACTION_FIELDS if k != "filename"]
    total_cells = 0
    correct_cells = 0
    misses: list[tuple[str, str, object, object]] = []

    for row in rows:
        # Identify which contract this row corresponds to, by filename
        # match. The model SHOULD return the exact filename; if it
        # paraphrases ("MNDA-Acme") fall back to substring match.
        row_filename = str(row.get("filename") or "")
        truth = truth_by_name.get(row_filename)
        if truth is None:
            # Substring fallback — the canonical filename appears as
            # a substring of the row's filename or vice versa.
            for name, candidate in truth_by_name.items():
                tokens = name.replace(".docx", "").lower()
                if any(t in row_filename.lower() for t in tokens.split()):
                    truth = candidate
                    break
        if truth is None:
            pytest.fail(f"row references unknown contract: {row_filename!r}")

        for key in field_keys:
            total_cells += 1
            predicted = row.get(key)
            expected = getattr(truth, key)
            if _value_matches(predicted, expected, key=key):
                correct_cells += 1
            else:
                misses.append((truth.filename, key, expected, predicted))

    accuracy = correct_cells / total_cells if total_cells else 0.0

    # Floor: 80% of cells correct (8 of 10 fields per contract on
    # average across the 5). The fixtures are intentionally varied
    # (Delaware vs Michigan governing law, open-ended vs fixed term,
    # German vs U.S. counterparty) so the model has to read carefully
    # — pure pattern-matching scores well below 80%.
    assert accuracy >= 0.80, (
        f"tier-11: accuracy {accuracy:.0%} ({correct_cells}/{total_cells}) "
        f"below 80% threshold. Misses:\n"
        + "\n".join(
            f"  {fn} | {key}: expected {exp!r}, got {pred!r}" for fn, key, exp, pred in misses
        )
    )
