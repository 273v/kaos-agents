# Multi-format E2E: chunking precision/recall tradeoff

Date: 2026-05-06

## Context

After wiring `SectionChunker.from_outline` (T4b) into both
`_load_files_to_corpus` and `_load_files_into_memory`, re-ran
`tests/benchmarks/multiformat_e2e.py` to measure impact.

## Results

| Configuration | Accuracy | Correct answers | Correct refusals | Wrong answers | Wrong refusals |
|---|---|---|---|---|---|
| Pre-chunking (10 docs whole) | 91.7% | 6/7 | 5/5 | 1 (mf06) | 0 |
| Post-chunking (222 chunks) | 91.7% | 7/7 | 4/5 | 1 (mf09) | 0 |

**Net: same 91.7% but the shape changed.**

* **Q6 fixed** (NIST AI RMF four core functions): chunking surfaces the
  relevant section as a discrete passage instead of burying it in a
  huge paragraph block. The agent now retrieves it cleanly.
* **Q9 broken** (Voyager 2 heliopause): with 222 small chunks, BM25
  finds spurious lexical matches for "voyager" / "crossed" /
  "heliopause" in unrelated chunks. Agent treats those as evidence and
  hallucinates an answer rather than refusing.

## Implication

Chunking is the right architectural move (un-buries answers in long
paragraphs, makes the 100k char budget actually bind, eliminates the
"dropped passages" warnings) but exposes a precision gap on
distractor queries that the prior whole-document retrieval was
masking.

The fix path:

* **P6 (embedding rerank)** — semantic reranking over BM25 top-K
  catches cases where lexical overlap is high but semantic relevance
  is low (Voyager doc fragments matching Voyager queries).
* **P7 (outline in system prompt)** — when the agent sees the corpus
  outline, it can reason "this corpus has no NASA content" before
  retrieval-grounded synthesis.
* **Tighter minimum BM25 score floor** — refuse to consider matches
  below an absolute relevance threshold, not just below the relative
  top-K cutoff. Currently the agent sees the top-K *of whatever's
  there*, even if none are above noise.

## Status

P6 and P7 are next in the queue. The chunking is committed but
should not be considered "complete" until P6 lands, since the
precision regression on refusals is a real downside that an end user
would notice before they noticed the recall improvement on answers.
