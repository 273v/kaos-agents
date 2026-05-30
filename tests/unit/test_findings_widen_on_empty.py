"""Unit tests for the 0.1.27 (I3) recall-safe widen-on-empty predicate.

THE bug: the retrieval planner's lexical narrowing (ngram / token /
BM25) ran irreversibly *before* the FindingsAgent semantic filter. When
the user's vocabulary had no lexical overlap with the document's
("auto-renewal" vs "shall terminate upon"; "which state's law governs"
vs "governed by the laws of ..."), the narrowing dropped the answer
before the vocabulary-robust filter could judge it, and the agent
reported "not present in the available evidence" for a clause that
demonstrably existed (NDA-matrix P4 / P8 / P10, reproduced in the SPA
2026-05-30).

``_run_findings_dispatch`` now treats a narrowed run that yields a
*recall* refusal as a signal that the speculative narrowing missed and
re-runs once on the full corpus view (``strategy=NONE``). The decision
of *whether* a refusal is a recall miss (re-run) vs a budget stop
(don't) is the pure predicate tested here. The end-to-end recovery is
proven live in the SPA re-validation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaos_agents.runtime.agent import _findings_recall_miss

pytestmark = pytest.mark.unit


@dataclass
class _FakeRefusal:
    reason: str


@dataclass
class _FakeResult:
    """Minimal stand-in for FindingsResult — only the two fields the
    predicate reads."""

    refusal: _FakeRefusal | None
    budget_exceeded: bool = False


def test_recall_refusal_is_a_miss() -> None:
    """No relevant candidates + no budget stop → widen to the recall floor."""
    r = _FakeResult(refusal=_FakeRefusal("no_relevant_candidates"), budget_exceeded=False)
    assert _findings_recall_miss(r) is True


def test_no_candidates_enumerated_is_a_miss() -> None:
    r = _FakeResult(refusal=_FakeRefusal("no_candidates_enumerated"), budget_exceeded=False)
    assert _findings_recall_miss(r) is True


def test_answer_present_is_not_a_miss() -> None:
    """A successful run (no refusal) must never trigger a widen re-run."""
    assert _findings_recall_miss(_FakeResult(refusal=None)) is False


def test_budget_refusal_is_not_a_recall_miss() -> None:
    """A budget stop means the agent did not finish looking — widening
    would only burn more budget, so it is NOT a recall miss."""
    r = _FakeResult(refusal=_FakeRefusal("budget_exceeded"), budget_exceeded=True)
    assert _findings_recall_miss(r) is False


def test_budget_flag_dominates_even_with_refusal_set() -> None:
    """Defensive: the ``budget_exceeded`` flag suppresses the widen even
    if some other refusal reason is also stamped."""
    r = _FakeResult(refusal=_FakeRefusal("no_relevant_candidates"), budget_exceeded=True)
    assert _findings_recall_miss(r) is False
