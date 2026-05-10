"""S9 — NDA tabular extraction: 10 fields x 5 NDAs scored vs ground truth.

2 surfaces (API + MCP) x 2 providers = 4 tests. CLI covered by ladder T11.

The 5 NDA texts are passed inline in the message (neither API nor
MCP supports a per-request corpus today). Each row is compared to
the manually-built ground truth (fixtures/nda_ground_truth.py).
Floor: ≥80% of cells correct.
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.fixtures.nda_ground_truth import (
    EXTRACTION_FIELDS,
    GROUND_TRUTH,
    ground_truth_by_filename,
)
from tests.integration.ladder.test_t11_nda_tabular import (
    _extract_json_array,
    _load_nda_corpus,
    _value_matches,
)
from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    api_call,
    assert_no_error,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

EXPECTED_FILES = tuple(gt.filename for gt in GROUND_TRUTH)


def _build_prompt() -> str:
    """Build the same extraction prompt used by ladder T11."""
    docs = _load_nda_corpus()
    assert len(docs) == 5
    context = "\n\n".join(f"=== CONTRACT: {name} ===\n{text}" for name, text in docs.items())
    fields_csv = ", ".join(f'"{f}"' for f in EXTRACTION_FIELDS)
    return (
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


def _score_extraction(rows: list[dict], surface: str, provider: str) -> None:
    """Apply the T11 scoring rubric: ≥80% cell accuracy."""
    assert len(rows) == 5, f"{surface}/{provider}: expected 5 rows, got {len(rows)}"
    truth_by_name = ground_truth_by_filename()
    field_keys = [k for k in EXTRACTION_FIELDS if k != "filename"]
    total_cells = 0
    correct_cells = 0
    misses: list[tuple[str, str, object, object]] = []

    for row in rows:
        row_filename = str(row.get("filename") or "")
        truth = truth_by_name.get(row_filename)
        if truth is None:
            for name, candidate in truth_by_name.items():
                tokens = name.replace(".docx", "").lower()
                if any(t in row_filename.lower() for t in tokens.split()):
                    truth = candidate
                    break
        if truth is None:
            pytest.fail(f"{surface}/{provider}: row references unknown contract {row_filename!r}")

        for key in field_keys:
            total_cells += 1
            predicted = row.get(key)
            expected = getattr(truth, key)
            if _value_matches(predicted, expected, key=key):
                correct_cells += 1
            else:
                misses.append((truth.filename, key, expected, predicted))

    accuracy = correct_cells / total_cells if total_cells else 0.0
    assert accuracy >= 0.80, (
        f"{surface}/{provider}: accuracy {accuracy:.0%} ({correct_cells}/"
        f"{total_cells}) below 80%. Misses:\n"
        + "\n".join(
            f"  {fn} | {key}: expected {exp!r}, got {pred!r}" for fn, key, exp, pred in misses
        )
    )


_MAX_S9_ATTEMPTS = 2


async def _api_rows_with_retry(provider: str, *, session_id_prefix: str) -> list[dict]:
    """Retry once on truncated/unparseable JSON arrays.

    gpt-5.4-mini in JSON-structured-output mode occasionally truncates
    a long array mid-field (observed empirically: row 2 of 5 cut off
    around char ~1K). One retry brings the rate to ~0. The cost is one
    extra LLM call per retry, capped at _MAX_S9_ATTEMPTS.
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_S9_ATTEMPTS):
        result = await api_call(
            _build_prompt(),
            provider=provider,
            session_id=f"{session_id_prefix}-{attempt}",
        )
        assert_no_error(result)
        try:
            return _extract_json_array(result.text)
        except ValueError as exc:
            last_error = exc
    raise AssertionError(
        f"api/{provider}: after {_MAX_S9_ATTEMPTS} attempts, no parseable "
        f"JSON array. Last error: {last_error}"
    )


async def _mcp_rows_with_retry(provider: str, *, session_id_prefix: str) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(_MAX_S9_ATTEMPTS):
        result = await mcp_call(
            "kaos-agent-chat",
            arguments={
                "message": _build_prompt(),
                "session_id": f"{session_id_prefix}-{attempt}",
                "model": model_for(provider),
            },
            timeout=300.0,
        )
        assert_no_error(result)
        try:
            return _extract_json_array(result.text)
        except ValueError as exc:
            last_error = exc
    raise AssertionError(
        f"mcp/{provider}: after {_MAX_S9_ATTEMPTS} attempts, no parseable "
        f"JSON array. Last error: {last_error}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s9_nda_tabular_via_api(provider: str) -> None:
    rows = await _api_rows_with_retry(provider, session_id_prefix=f"s9-api-{provider}")
    _score_extraction(rows, "api", provider)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s9_nda_tabular_via_mcp(provider: str) -> None:
    rows = await _mcp_rows_with_retry(provider, session_id_prefix=f"s9-mcp-{provider}")
    _score_extraction(rows, "mcp", provider)
