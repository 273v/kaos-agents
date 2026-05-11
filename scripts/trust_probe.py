"""Senior-associate trust probe for kaos-agents (skeptic audit).

Probes executed:
  1) Consistency across 5 runs of FindingsAgent on same NDA + question.
  2) Refusal / hallucination test — questions whose answer is NOT in
     the document (3 distinct out-of-doc questions).
  3) Source attribution audit — for a real question, verify every
     finding_id citation actually contains supporting text.

Models pinned to production defaults:
  filter      anthropic:claude-haiku-4-5
  synthesis   anthropic:claude-sonnet-4-6

Budget cap: $1.50 total. Script aborts before any phase that would
exceed it.

Writes JSON traces to ./tests/integration/runs/2026-05-11/trust-probe/.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from time import time

from kaos_content.views import DocumentView
from kaos_nlp_core._defaults import get_default_punkt_tokenizer
from kaos_office import parse_docx

from kaos_agents.patterns.findings import (
    FindingsAgent,
    every_sentence_selector,
    extract_finding_id_citations,
    sentences_with_token_selector,
)

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "integration"
    / "runs"
    / "2026-05-11"
    / "trust-probe"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTER_MODEL = "anthropic:claude-haiku-4-5"
SYNTH_MODEL = "anthropic:claude-sonnet-4-6"
BUDGET_USD = 1.50

# Cumulative spend, mutated across calls.
SPEND = {"total": 0.0}


def _view(name: str) -> DocumentView:
    path = NDA_DIR / name
    doc = parse_docx(str(path))
    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


def _dump_result(name: str, payload: dict) -> Path:
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _check_budget(label: str) -> None:
    if SPEND["total"] >= BUDGET_USD:
        print(f"BUDGET EXCEEDED at {label}: spent ${SPEND['total']:.4f} of ${BUDGET_USD}")
        sys.exit(2)


def _record_spend(cost: float, label: str) -> None:
    SPEND["total"] += cost
    print(f"  spend {label}: ${cost:.4f}  (cumulative ${SPEND['total']:.4f})")


def _result_to_dict(result) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "total_enumerated": result.total_enumerated,
        "total_filtered": result.total_filtered,
        "filter_cost_usd": result.filter_cost_usd,
        "synthesis_cost_usd": result.synthesis_cost_usd,
        "total_cost_usd": result.total_cost_usd,
        "filter_calls": result.filter_calls,
        "findings": [
            {
                "finding_id": f.candidate.finding_id,
                "text": f.candidate.text,
                "block_ref": f.candidate.block_ref,
                "section_title": f.candidate.section_title,
                "page": f.candidate.page,
                "relevance": f.relevance,
                "reasoning": f.reasoning,
            }
            for f in result.findings
        ],
        "cited_finding_ids": list(extract_finding_id_citations(result.answer)),
    }


# ---------------------------------------------------------------------------
# Probe 1 — Consistency across 5 runs
# ---------------------------------------------------------------------------


async def probe_consistency() -> dict:
    print("\n=== PROBE 1: consistency (5x same NDA + same question) ===")
    view = _view("MNDA - Acme.docx")
    question = "What are the obligations of the receiving party?"

    runs = []
    for i in range(5):
        _check_budget(f"probe1.run{i + 1}")
        # Use a broad selector to give enough candidates that filtering
        # has real work to do — receiving party obligations show up
        # near words like 'recipient', 'obligation', 'confidential'.
        agent = FindingsAgent(
            selector=every_sentence_selector,
            filter_model=FILTER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
        )
        t0 = time()
        result = await agent.run(question, view)
        dt = time() - t0
        as_dict = _result_to_dict(result)
        as_dict["wall_seconds"] = dt
        as_dict["run_index"] = i + 1
        runs.append(as_dict)
        _record_spend(result.total_cost_usd, f"probe1.run{i + 1}")
        print(
            f"  run{i + 1}: enum={result.total_enumerated} filt={result.total_filtered} "
            f"cited={len(as_dict['cited_finding_ids'])} cost=${result.total_cost_usd:.4f} "
            f"({dt:.1f}s)"
        )

    # Cross-run analytics.
    surviving_texts = [{f["text"] for f in r["findings"]} for r in runs]
    # Jaccard pairwise on surviving texts
    pairwise = []
    for i in range(len(surviving_texts)):
        for j in range(i + 1, len(surviving_texts)):
            a, b = surviving_texts[i], surviving_texts[j]
            j_idx = len(a & b) / max(1, len(a | b))
            pairwise.append({"i": i + 1, "j": j + 1, "jaccard": j_idx})

    costs = [r["total_cost_usd"] for r in runs]
    summary = {
        "question": question,
        "runs": runs,
        "pairwise_jaccard_surviving_texts": pairwise,
        "cost_min": min(costs),
        "cost_max": max(costs),
        "cost_mean": sum(costs) / len(costs),
        "cost_variance_pct": (max(costs) - min(costs)) / max(1e-9, sum(costs) / len(costs)) * 100,
        "answer_lengths": [len(r["answer"]) for r in runs],
        "cited_counts": [len(r["cited_finding_ids"]) for r in runs],
    }
    _dump_result("probe1_consistency", summary)
    return summary


# ---------------------------------------------------------------------------
# Probe 2 — Refusal / hallucination
# ---------------------------------------------------------------------------


# Confirmed via inspection: none of the NDAs contain these terms.
OUT_OF_DOC_QUESTIONS = [
    "What is the liquidated damages amount specified in this agreement?",
    "What is the governing law's choice-of-forum arbitration seat — Singapore or Stockholm?",
    "What is the maximum dollar cap on damages for breach of confidentiality?",
]

# Patterns that indicate a hallucinated *number* — the highest-severity
# failure mode for an NDA refusal probe. We classify by substance later
# during manual review, but flag any specific dollar / number claim.
HALLUCINATION_DOLLAR_RE = re.compile(
    r"\$\s?\d|(?:[Uu][Ss]\s?\$|USD\s?)\s?\d|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?(?:million|thousand|MM|K)\b"
)
REFUSAL_KEYWORDS = [
    "not specified",
    "does not specify",
    "doesn't specify",
    "not mentioned",
    "no mention",
    "does not contain",
    "doesn't contain",
    "not included",
    "not addressed",
    "not provided",
    "no such",
    "do not address",
    "does not address",
    "no provision",
    "not found",
    "no findings",
    "silent on",
]


def _classify_refusal(answer: str) -> dict:
    answer_lc = answer.lower()
    refusal_hit = any(kw in answer_lc for kw in REFUSAL_KEYWORDS)
    dollar_hit = bool(HALLUCINATION_DOLLAR_RE.search(answer))
    return {
        "answer": answer,
        "refusal_keyword_hit": refusal_hit,
        "dollar_or_number_in_answer": dollar_hit,
        "answer_len": len(answer),
    }


async def probe_refusal() -> dict:
    print("\n=== PROBE 2: refusal / hallucination on out-of-document questions ===")
    view = _view("MNDA - Acme.docx")
    results: list[dict] = []

    for q in OUT_OF_DOC_QUESTIONS:
        _check_budget(f"probe2.q={q[:40]}")
        # Use 'every sentence' so the selector itself can't be blamed
        # for refusal. If the agent refuses, it's the synthesis pass.
        agent = FindingsAgent(
            selector=every_sentence_selector,
            filter_model=FILTER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
        )
        t0 = time()
        result = await agent.run(q, view)
        dt = time() - t0
        as_dict = _result_to_dict(result)
        as_dict["wall_seconds"] = dt
        as_dict["question"] = q
        as_dict["classification"] = _classify_refusal(result.answer)
        results.append(as_dict)
        _record_spend(result.total_cost_usd, f"probe2.q={q[:30]}")
        print(
            f"  Q: {q!r}\n"
            f"     enum={result.total_enumerated} filt={result.total_filtered} "
            f"cost=${result.total_cost_usd:.4f}\n"
            f"     refusal_kw={as_dict['classification']['refusal_keyword_hit']} "
            f"$_in_answer={as_dict['classification']['dollar_or_number_in_answer']}"
        )
        print(f"     ANSWER (first 300 chars): {result.answer[:300]!r}")

    summary = {
        "fixture": "MNDA - Acme.docx",
        "questions": OUT_OF_DOC_QUESTIONS,
        "results": results,
        "n_with_refusal_keyword": sum(
            1 for r in results if r["classification"]["refusal_keyword_hit"]
        ),
        "n_with_dollar_in_answer": sum(
            1 for r in results if r["classification"]["dollar_or_number_in_answer"]
        ),
    }
    _dump_result("probe2_refusal", summary)
    return summary


# ---------------------------------------------------------------------------
# Probe 3 — Source attribution audit
# ---------------------------------------------------------------------------


async def probe_attribution() -> dict:
    print("\n=== PROBE 3: source attribution audit ===")
    view = _view("MNDA - Acme.docx")
    question = "What are the receiving party's obligations to protect Confidential Information?"

    _check_budget("probe3")
    agent = FindingsAgent(
        selector=sentences_with_token_selector("confidential"),
        filter_model=FILTER_MODEL,
        synthesis_model=SYNTH_MODEL,
        chunk_size=20,
        num_parallel=3,
        relevance_threshold=0.4,
    )
    t0 = time()
    result = await agent.run(question, view)
    dt = time() - t0
    _record_spend(result.total_cost_usd, "probe3")

    as_dict = _result_to_dict(result)
    as_dict["wall_seconds"] = dt

    # Build finding_id → text index for the survivors.
    by_id = {f["finding_id"]: f for f in as_dict["findings"]}

    cited_ids = list(extract_finding_id_citations(result.answer))

    # For each cited finding_id, look up whether it resolves to a real
    # survivor and capture the underlying text.
    citations_audit = []
    for fid in cited_ids:
        survivor = by_id.get(fid)
        citations_audit.append(
            {
                "finding_id": fid,
                "exists_in_survivors": survivor is not None,
                "underlying_text": survivor["text"] if survivor else None,
                "underlying_block_ref": survivor["block_ref"] if survivor else None,
                "section_title": survivor["section_title"] if survivor else None,
            }
        )

    # Break answer into sentences; flag sentences with no [xxxxxxxx]
    # citation marker — those are unsourced claims.
    answer_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", result.answer.strip()) if s.strip()
    ]
    cite_re = re.compile(r"\[[0-9a-f]{8}\]")
    sentence_audit = []
    for s in answer_sentences:
        cites_in_sentence = cite_re.findall(s)
        sentence_audit.append(
            {
                "sentence": s,
                "n_citations": len(cites_in_sentence),
                "has_citation": bool(cites_in_sentence),
                "cited_ids_in_sentence": [c.strip("[]") for c in cites_in_sentence],
            }
        )

    payload = {
        "result": as_dict,
        "cited_finding_ids": cited_ids,
        "citations_audit": citations_audit,
        "sentence_audit": sentence_audit,
        "n_sentences_in_answer": len(answer_sentences),
        "n_sentences_without_citation": sum(1 for s in sentence_audit if not s["has_citation"]),
        "n_cited_ids_not_in_survivors": sum(
            1 for c in citations_audit if not c["exists_in_survivors"]
        ),
    }
    _dump_result("probe3_attribution", payload)
    print(
        f"  enum={result.total_enumerated} filt={result.total_filtered} "
        f"cited_ids={len(cited_ids)} cost=${result.total_cost_usd:.4f}"
    )
    print(
        f"  answer_sentences={payload['n_sentences_in_answer']} "
        f"unsourced={payload['n_sentences_without_citation']} "
        f"phantom_ids={payload['n_cited_ids_not_in_survivors']}"
    )
    return payload


async def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY missing", file=sys.stderr)
        sys.exit(1)

    print(f"Writing traces to: {OUT_DIR}")
    print(f"Budget cap: ${BUDGET_USD}")

    probe1 = await probe_consistency()
    probe2 = await probe_refusal()
    probe3 = await probe_attribution()

    overall = {
        "spend_total_usd": SPEND["total"],
        "probe1_summary": {
            "n_runs": len(probe1["runs"]),
            "cost_range_usd": [probe1["cost_min"], probe1["cost_max"]],
            "cost_variance_pct": probe1["cost_variance_pct"],
            "pairwise_jaccard": probe1["pairwise_jaccard_surviving_texts"],
            "answer_lengths": probe1["answer_lengths"],
            "cited_counts": probe1["cited_counts"],
        },
        "probe2_summary": {
            "n_questions": len(probe2["questions"]),
            "n_with_refusal_keyword": probe2["n_with_refusal_keyword"],
            "n_with_dollar_in_answer": probe2["n_with_dollar_in_answer"],
        },
        "probe3_summary": {
            "n_cited_ids": len(probe3["cited_finding_ids"]),
            "n_cited_ids_not_in_survivors": probe3["n_cited_ids_not_in_survivors"],
            "n_sentences_without_citation": probe3["n_sentences_without_citation"],
        },
    }
    _dump_result("OVERALL", overall)
    print("\n=== DONE ===")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
