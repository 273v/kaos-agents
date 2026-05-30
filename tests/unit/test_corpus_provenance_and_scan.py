"""Corpus / context-management coverage for the 0.1.27 fixes.

These guard the two bug *classes* that escaped the previous suite (the
citation test hand-built its own merged view, so the production
provenance picker and the narrowing decision were never exercised):

* ``_decide_full_scan`` — narrow-vs-full-scan is a cost/recall tradeoff
  made against the budget, not a document/sentence count. (NDA-matrix
  P4/P8/P10 silent-miss: a ~450-sentence deal room was narrowed to 7.)
* ``_source_uri_for_item`` — finding provenance must resolve to the
  user-facing filename, not the parser's temp-file path. (NDA-matrix
  attribution miss: ``source_uri`` was ``tmp52cu95ct.docx``.)

All pure-function, no LLM — they run in the default unit gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kaos_agents.runtime.agent import (
    _bare_name,
    _decide_full_scan,
    _source_uri_for_item,
)

pytestmark = pytest.mark.unit


# ─── _decide_full_scan — the narrowing tradeoff ────────────────────


def _decide(**kw):
    base = {
        "n_docs": 8,
        "n_sentences": 400,
        "chunk_size": 20,
        "budget_usd": 0.50,
        "cost_per_call_usd": 0.004,
        "plan_floor": 5,
    }
    base.update(kw)
    return _decide_full_scan(**base)


def test_small_corpus_below_doc_floor_always_full_scans() -> None:
    # Below the doc floor we full-scan even with a zero budget — a tiny
    # corpus is never worth an LLM planner call.
    d = _decide(n_docs=3, plan_floor=5, budget_usd=0.0, n_sentences=10_000)
    assert d.full_scan is True
    assert "floor" in d.reason


def test_corpus_that_fits_budget_full_scans() -> None:
    # 5 NDAs ~= 450 sentences -> ~23 chunks x $0.004 ~= $0.09, well under
    # the $0.50 budget. THIS is the headline regression: 0.1.26 narrowed
    # this to 7 sentences and dropped the answer.
    d = _decide(n_docs=5, n_sentences=450, budget_usd=0.50)
    assert d.full_scan is True
    assert d.est_full_scan_usd == pytest.approx(23 * 0.004)


def test_corpus_that_exceeds_budget_narrows() -> None:
    # A genuinely large corpus (50k sentences ~= 2500 chunks ~= $10) does
    # NOT fit a $0.50 budget -> narrow (the one case narrowing is earned).
    d = _decide(n_docs=200, n_sentences=50_000, budget_usd=0.50)
    assert d.full_scan is False
    assert "narrow" in d.reason


def test_budget_scales_the_decision() -> None:
    # Same corpus, only the budget differs -> opposite decisions.
    corpus = {"n_docs": 40, "n_sentences": 4_000}  # ~200 chunks ~= $0.80
    assert _decide(**corpus, budget_usd=0.10).full_scan is False
    assert _decide(**corpus, budget_usd=2.00).full_scan is True


def test_model_unit_cost_scales_the_decision() -> None:
    # Same corpus + budget; a cheaper per-call model affords the full
    # scan that an expensive one does not. (gpt-mini vs opus-class.)
    corpus = {"n_docs": 40, "n_sentences": 6_000, "budget_usd": 0.50}  # 300 chunks
    assert _decide(**corpus, cost_per_call_usd=0.0005).full_scan is True  # $0.15
    assert _decide(**corpus, cost_per_call_usd=0.006).full_scan is False  # $1.80


def test_chunk_count_is_ceiling() -> None:
    # 41 sentences / 20-chunk = 3 chunks (ceil), not 2.
    d = _decide(n_docs=10, n_sentences=41, chunk_size=20, budget_usd=10.0)
    assert d.n_filter_chunks == 3


def test_zero_sentences_does_not_full_scan_via_budget() -> None:
    # Degenerate: an unresolvable/empty view (n_sentences=0) is NOT a
    # budget-justified full scan (nothing to scan) — only the doc-floor
    # path would full-scan it.
    d = _decide(n_docs=10, n_sentences=0, plan_floor=5)
    assert d.full_scan is False


# ─── _source_uri_for_item — provenance = user-facing filename ──────


def _parsed(uri=None, title=None):
    """A fake parsed ContentDocument with metadata.source.uri + title."""
    src = SimpleNamespace(uri=uri) if uri is not None else None
    meta = SimpleNamespace(source=src, title=title)
    return SimpleNamespace(metadata=meta)


def test_friendly_filename_beats_temp_path_source_uri() -> None:
    # THE attribution bug: the parser's temp path must NOT win over the
    # user-facing upload filename.
    parsed = _parsed(uri="/tmp/tmp52cu95ct.docx", title="Mutual Non-Disclosure Agreement")
    assert _source_uri_for_item(None, "MNDA - Acme.docx", parsed) == "MNDA - Acme.docx"


def test_friendly_filename_beats_generic_title() -> None:
    # Near-identical templates share a generic title that can't
    # disambiguate; the item filename must win.
    parsed = _parsed(uri=None, title="Mutual Non-Disclosure Agreement")
    assert _source_uri_for_item(None, "MNDA - BI.docx", parsed) == "MNDA - BI.docx"


def test_falls_back_to_title_when_no_item_name() -> None:
    parsed = _parsed(uri="/tmp/x.docx", title="Quarterly Report 2025")
    assert _source_uri_for_item(None, "(unnamed)", parsed) == "Quarterly Report 2025"


def test_falls_back_to_source_uri_basename_when_no_name_or_title() -> None:
    parsed = _parsed(uri="file:///docs/real-name.pdf", title=None)
    assert _source_uri_for_item(None, "", parsed) == "real-name.pdf"


def test_unnamed_when_nothing_resolvable() -> None:
    assert _source_uri_for_item(None, "", None) == "(unnamed)"
    assert _source_uri_for_item(None, "(unnamed)", _parsed()) == "(unnamed)"


def test_item_filename_path_is_stripped_to_basename() -> None:
    # A caller may pass a full path as the item filename — surface the
    # bare, human-readable name (no local-path leak into provenance).
    assert _source_uri_for_item(None, "/home/u/contracts/Deal A.docx", None) == "Deal A.docx"


def test_percent_encoded_name_is_decoded() -> None:
    assert _bare_name("MNDA%20-%20Acme.docx") == "MNDA - Acme.docx"
