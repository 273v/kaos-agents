"""Unit tests for the TOML-backed few-shot Example loader.

The loader is the single source of truth that grounds every Signature
in kaos-agents — Iteration 4 of the kaos-llm-core design alignment
audit (kaos-modules/docs/audits/2026-05-24-kaos-agents-prompt-smells.md).
A regression here silently drops grounding from every critic /
classifier downstream, so the contract is exercised directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("kaos_llm_core", reason="loader requires the [llm] extra")

from kaos_llm_core import Example

from kaos_agents._examples import _ASSETS_ROOT, load_examples, reload_examples


def _all_pool_names() -> list[str]:
    return sorted(p.stem for p in _ASSETS_ROOT.glob("*.toml"))


# ─── Contract ───────────────────────────────────────────────────────


def test_assets_dir_exists() -> None:
    assert _ASSETS_ROOT.is_dir(), (
        f"Examples asset dir missing: {_ASSETS_ROOT}. "
        "Iteration 4 stores few-shot pools as TOML under this path."
    )


def test_at_least_one_pool_shipped() -> None:
    pools = _all_pool_names()
    assert pools, "No example TOML pools present; the loader has nothing to load."


def test_each_pool_loads_as_examples() -> None:
    for name in _all_pool_names():
        pool = load_examples(name)
        assert pool, f"Pool {name!r} is empty — at least one [[example]] required."
        for ex in pool:
            assert isinstance(ex, Example), (
                f"{name}[{pool.index(ex)}] is not an Example: {type(ex)!r}"
            )
            assert isinstance(ex.inputs, dict) and ex.inputs, (
                f"{name}[{pool.index(ex)}] has empty inputs"
            )
            assert isinstance(ex.outputs, dict) and ex.outputs, (
                f"{name}[{pool.index(ex)}] has empty outputs"
            )


def test_load_examples_is_cached() -> None:
    """The same loader call returns the same list object (lru_cache)."""
    name = _all_pool_names()[0]
    first = load_examples(name)
    second = load_examples(name)
    assert first is second, "Loader is not memoised — every Signature build would re-read TOML"


def test_reload_examples_invalidates_cache() -> None:
    name = _all_pool_names()[0]
    first = load_examples(name)
    reload_examples(name)
    second = load_examples(name)
    assert first is not second, "reload_examples did not evict the cache entry"
    # Content still equal — the TOML didn't change between calls.
    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]


def test_missing_pool_raises_filenotfound() -> None:
    with pytest.raises(FileNotFoundError):
        load_examples("__definitely_not_a_real_pool__")


# ─── Per-pool semantic invariants ───────────────────────────────────


def test_goal_check_pool_covers_all_verdict_kinds() -> None:
    """GoalCheck's pool must demonstrate every kind the loop branches on."""
    pool = load_examples("goal_check")
    kinds = {ex.outputs["result"]["kind"] for ex in pool}
    assert kinds == {"satisfied", "needs_more_work", "insufficient_evidence"}, (
        f"goal_check pool missing verdicts; got {sorted(kinds)}. "
        "The critic branches on all three — drop one and the model "
        "loses a grounding anchor for that path."
    )


def test_reflexion_critique_pool_brackets_threshold() -> None:
    """ReflexionCritic must see scores above AND below its 0.7 default threshold."""
    pool = load_examples("reflexion_critique")
    scores = [float(ex.outputs["score"]) for ex in pool]
    assert any(s >= 0.7 for s in scores), (
        "No approving example in reflexion_critique pool; critic would only see negative shape."
    )
    assert any(s < 0.7 for s in scores), (
        "No rejecting example in reflexion_critique pool; critic would only see positive shape."
    )


def test_judge_pool_covers_confidence_range() -> None:
    """Generic Judge must demonstrate confident AND ambiguous verdicts."""
    pool = load_examples("judge")
    confs = [float(ex.outputs["confidence"]) for ex in pool]
    assert any(c >= 0.85 for c in confs), (
        "No high-confidence example; Judge won't learn to assert when criteria are clearly met."
    )
    assert any(c <= 0.5 for c in confs), (
        "No low-confidence example; Judge won't learn to honestly admit ambiguity."
    )
    # Every example must enumerate exactly the (label, confidence, reasoning) triple.
    for ex in pool:
        assert set(ex.outputs.keys()) == {"label", "confidence", "reasoning"}, (
            f"Judge example {ex.outputs!r} has wrong shape; the triple is load-bearing."
        )


def test_classify_intent_pool_covers_all_routing_labels() -> None:
    """ClassifyIntent classifier must learn every routing label the dispatcher branches on."""
    pool = load_examples("classify_intent")
    labels = {ex.outputs["intent"] for ex in pool}
    expected = {"respond", "tool_use", "research", "plan", "clarify"}
    assert labels == expected, (
        f"classify_intent pool missing labels; got {sorted(labels)}, expected {sorted(expected)}. "
        "Each label is a branch in the agent loop — every branch needs a positive example."
    )
    # Anti-regression: at least one example must demonstrate that a current-
    # officeholder question with a relevant tool category in the catalog routes
    # to `tool_use`, NOT `respond`. Session 01KS0R64Q744... regression.
    senator_cases = [
        ex
        for ex in pool
        if "current" in ex.inputs.get("message", "").lower()
        and "senator" in ex.inputs.get("message", "").lower()
    ]
    assert senator_cases, "No current-senator example; #449 regression risk."
    for ex in senator_cases:
        assert ex.outputs["intent"] == "tool_use", (
            "Current-officeholder example must route to tool_use, not "
            f"{ex.outputs['intent']!r}. Session 01KS0R64Q744 anti-regression."
        )


def test_tool_fitness_pool_covers_empty_picks_and_real_picks() -> None:
    """ToolFitness ranker must learn both the empty-catalog/small-talk path AND the domain pick."""
    pool = load_examples("tool_fitness")
    has_empty = any(ex.outputs["picks"] == [] for ex in pool)
    has_real = any(len(ex.outputs["picks"]) > 0 for ex in pool)
    assert has_empty, (
        "No empty-picks example; ranker won't learn to return [] for small-talk / empty catalog."
    )
    assert has_real, "No real-pick example; ranker has no positive grounding."
    # Bug-regression: at least one example must demonstrate DOCX → office-parse,
    # not pdf-extract — #581.
    docx_examples = [
        ex
        for ex in pool
        if "docx" in ex.inputs.get("query", "").lower() or ".docx" in ex.inputs.get("query", "")
    ]
    assert docx_examples, "No DOCX-routing example; #581 regression risk."
    for ex in docx_examples:
        picks = ex.outputs["picks"]
        assert "kaos-pdf-extract-parse" not in picks, (
            "DOCX example must NOT pick pdf-extract — #581 anti-regression."
        )


# ─── Source-tree hygiene ────────────────────────────────────────────


def test_pool_files_are_in_assets_dir() -> None:
    """No stray pool files outside the canonical assets directory."""
    repo_root = Path(__file__).resolve().parents[2]
    strays = [
        p
        for p in repo_root.glob("kaos_agents/**/*.toml")
        if "/_assets/examples/" not in p.as_posix()
    ]
    assert not strays, (
        f"Found TOML example pools outside _assets/examples/: {strays}. "
        "Loader convention is one canonical directory."
    )
