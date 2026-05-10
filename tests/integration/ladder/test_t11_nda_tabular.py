"""Tier 11: NDA tabular review — extract structured fields from 5 NDAs.

The deal-room batch-extraction use case. Real DOCX fixtures (mutual
NDAs from a working portfolio), real parsing through kaos-office, real
LLM extraction into a structured table. Catches regressions in:

- kaos-office DOCX parser
- kaos-content text serialization
- The agent's ability to follow a structured-output instruction
- Multi-document context handling

The fixtures are 5 mutual NDAs (~21-22 KB each, ~10K chars of text
each after parsing), covering different counterparties (Acme, BI, CC,
DynaMo, EMNA) so the agent must distinguish them by name.

Cost target: ~$0.10 (one Anthropic call with ~12K tokens input,
~1K tokens output).
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

pytestmark = pytest.mark.live

TIER = 11
BUDGET_USD = 0.20

NDA_DIR = Path(__file__).parent / "fixtures" / "nda"
EXPECTED_FILES = (
    "EMNA Mutual NDA.docx",
    "MNDA - Acme.docx",
    "MNDA - BI.docx",
    "MNDA - CC Final 2.docx",
    "MNDA - DynaMo.docx",
)


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
    """Find the first JSON array in ``text`` and parse it.

    The model may wrap the array in code fences or surrounding prose.
    Strip fences first, then find the first ``[...]`` substring that
    parses as valid JSON.
    """
    # Strip ```json fences if present
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text
    # Find the first [...] that's a valid JSON array
    bracket = re.search(r"\[.+\]", candidate, re.DOTALL)
    if bracket is None:
        raise ValueError(f"no JSON array in response: {text[:300]!r}")
    return json.loads(bracket.group(0))


@pytest.mark.asyncio
async def test_nda_tabular_extraction(ladder_runner_factory, turn_session_id):
    """Extract a 5-row table of NDA terms from the fixture corpus."""
    docs = _load_nda_corpus()
    assert len(docs) == 5, f"expected 5 NDAs, loaded {len(docs)}"

    # Build a single-prompt context: each contract prefixed with its
    # filename. Pass the FULL text (~10KB each) — the governing-law
    # / jurisdiction clauses live near the end (chars 7000-8000) and
    # any aggressive truncation drops them. Five 10KB contracts ≈
    # ~12K input tokens; well within budget.
    context = "\n\n".join(f"=== CONTRACT: {name} ===\n{text}" for name, text in docs.items())
    msg = (
        "Below are 5 mutual NDAs. For EACH contract, extract these 5 fields:\n"
        '  - "filename": the contract filename (exactly as shown)\n'
        '  - "counterparty": the non-273-Ventures party name (string)\n'
        '  - "effective_date": the effective date (ISO YYYY-MM-DD if possible)\n'
        '  - "governing_law": jurisdiction whose law governs (e.g. "Michigan")\n'
        '  - "confidentiality_period_years": numeric — years confidential info '
        "must be protected (or null if perpetual / not stated)\n\n"
        "Output ONLY a JSON array with exactly 5 objects, one per contract. "
        "No prose, no code fences, no commentary.\n\n"
        f"{context}"
    )
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a contracts-extraction assistant. Output structured "
                "JSON exactly as requested. Be concise."
            ),
            "model": model_for_tier(TIER),
        },
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-11")

    text = text_from_summary(events)
    rows = _extract_json_array(text)
    assert len(rows) == 5, f"expected 5 rows in extraction table, got {len(rows)}"

    # Each row has the expected keys
    expected_keys = {
        "filename",
        "counterparty",
        "effective_date",
        "governing_law",
        "confidentiality_period_years",
    }
    for i, row in enumerate(rows):
        assert isinstance(row, dict), f"row {i} not a dict: {row!r}"
        missing = expected_keys - set(row.keys())
        assert not missing, f"row {i} missing keys {missing}: {row!r}"

    # The 5 distinct counterparties should appear (avoids the model
    # "all rows say 273 Ventures" failure mode). Filename mnemonics
    # don't match actual party names — these are the real parties
    # in the fixtures: Acme Co. / Beta Inc. / CyberCorp / DynaMo
    # GmbH / ExMachi Bank.
    counterparties = {str(r.get("counterparty", "")).lower() for r in rows}
    distinctive_tokens = ("acme", "beta", "cyber", "dynamo", "exmachi")
    matched = sum(1 for cp in counterparties if any(token in cp for token in distinctive_tokens))
    assert matched >= 4, (
        f"expected >=4 of the 5 distinct counterparty names in the table; "
        f"matched {matched}. Counterparties seen: {counterparties}. "
        f"Expected tokens: {distinctive_tokens}"
    )

    # Counterparties must actually be DISTINCT (not all 5 saying
    # the same thing — would mean the model collapsed rows).
    assert len(counterparties) >= 4, (
        f"expected >=4 distinct counterparty values; got {len(counterparties)} "
        f"unique: {counterparties}"
    )

    # At least one governing-law value populated (the contracts pick
    # specific jurisdictions; if they all come back null/empty the
    # extraction is broken).
    gov_laws = [str(r.get("governing_law") or "").strip() for r in rows]
    assert sum(1 for g in gov_laws if g and g.lower() != "null") >= 3, (
        f"expected >=3 contracts with a governing_law value; got: {gov_laws}"
    )
