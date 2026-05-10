"""Drift detection — asserts test-code thresholds match the baseline.

Bug pattern this catches: someone lowers a floor in a test file (e.g.
``BUDGET_USD = 0.30`` becomes ``BUDGET_USD = 1.00`` to make a flaky
test pass) without acknowledging that they're relaxing the standing
bar. The baseline file is checked-in; this test reads BOTH the
baseline AND the current test-file thresholds, and fails when they
diverge.

To intentionally change a floor, update ``ladder_baseline.json`` in
the same commit. The diff is then visible and reviewable.

Deterministic — no LLM calls.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

BASELINE_PATH = Path(__file__).parent / "fixtures" / "ladder_baseline.json"


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def _module_for_tier(tier_id: str) -> object:
    """Tier 'T01' -> module test_t01_smoke; raises if not found."""
    tier_num = int(tier_id[1:])
    # Map tier -> filename suffix (matches the actual filenames)
    suffixes = {
        1: "smoke",
        2: "single_tool",
        3: "multi_tool_react",
        4: "plan_execute_4step",
        5: "memory_continuity",
        6: "citation_extraction",
        7: "pdf_extraction",
        8: "research_delegation",
        9: "permission_gating",
        10: "budget_cap",
        11: "nda_tabular",
        12: "nda_risk_memo",
    }
    suffix = suffixes[tier_num]
    return importlib.import_module(f"tests.integration.ladder.test_t{tier_num:02d}_{suffix}")


@pytest.mark.parametrize("tier_id", [f"T{i:02d}" for i in range(1, 13)])
def test_tier_budget_matches_baseline(tier_id: str) -> None:
    """BUDGET_USD in each tier file must match the baseline JSON.

    Catches silent relaxation (someone bumping a budget to mask
    flakiness without recording the decision). To legitimately raise
    a budget, update fixtures/ladder_baseline.json in the same commit.
    """
    baseline = _load_baseline()
    baseline_budget = baseline["tiers"][tier_id]["budget_usd"]

    module = _module_for_tier(tier_id)
    actual_budget = getattr(module, "BUDGET_USD", None)
    assert actual_budget is not None, f"tier {tier_id} module has no BUDGET_USD attribute"
    assert actual_budget == baseline_budget, (
        f"{tier_id}: code BUDGET_USD={actual_budget} but baseline expects "
        f"{baseline_budget}. If this change is intentional, update "
        f"fixtures/ladder_baseline.json in the same commit so the new "
        f"floor is recorded and reviewable."
    )


def test_t11_accuracy_floor_matches_baseline() -> None:
    """T11's accuracy floor (the 0.80 in the assert) must match baseline."""
    baseline = _load_baseline()
    expected_floor = baseline["tiers"]["T11"]["accuracy_floor"]

    # Parse the floor from the test file's source — no clean attribute
    # is exposed, so we grep for the literal. If someone changes the
    # threshold, the source-grep change is visible in the same diff
    # that has to update the baseline.
    t11_path = Path(__file__).parent / "test_t11_nda_tabular.py"
    src = t11_path.read_text()
    target = f"assert accuracy >= {expected_floor}"
    assert target in src, (
        f"T11 accuracy floor in code does not match baseline "
        f"({expected_floor}). Update fixtures/ladder_baseline.json if "
        f"the change is intentional. Looked for line: {target!r}"
    )


def test_t12_thresholds_match_baseline() -> None:
    """T12's judge-score / coverage / incorrect-claims thresholds."""
    baseline = _load_baseline()
    score_floor = baseline["tiers"]["T12"]["judge_score_floor"]
    cov_floor = baseline["tiers"]["T12"]["covered_findings_floor"]
    incorrect_ceiling = baseline["tiers"]["T12"]["incorrect_claims_ceiling"]

    t12_path = Path(__file__).parent / "test_t12_nda_risk_memo.py"
    src = t12_path.read_text()
    expectations = {
        f"score >= {score_floor}": "judge score floor",
        f"len(covered) >= {cov_floor}": "covered findings floor",
        f"len(incorrect) <= {incorrect_ceiling}": "incorrect claims ceiling",
    }
    for needle, label in expectations.items():
        assert needle in src, (
            f"T12 {label} in code does not match baseline. Looked for "
            f"line: {needle!r}. Update fixtures/ladder_baseline.json if "
            f"the change is intentional."
        )


def test_baseline_lists_all_12_tiers() -> None:
    """The baseline file must enumerate every tier the ladder has."""
    baseline = _load_baseline()
    expected = {f"T{i:02d}" for i in range(1, 13)}
    actual = set(baseline["tiers"].keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"baseline tier coverage drift: missing={missing}, extra={extra}"
    )


def test_pathological_baseline_lists_all_8() -> None:
    """The pathological suite has 8 tests; baseline must cover them."""
    baseline = _load_baseline()
    expected = {f"P{i}" for i in range(1, 9)}
    actual = set(baseline["pathological_suite"].keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"pathological coverage drift: missing={missing}, extra={extra}"
    )
