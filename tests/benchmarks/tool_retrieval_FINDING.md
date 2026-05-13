# Tool Retrieval Benchmark — Lexicon + Multi-Query Findings

**Date:** 2026-05-11
**Catalog:** 50 tools (kaos-pdf + kaos-web + kaos-tabular + kaos-office)
**Queries:** 25 labeled (10 direct, 7 synonym, 8 conceptual)
**Rewrite model:** anthropic:claude-haiku-4-5 (3 paraphrases per query)

## TL;DR

| Default? | Feature | Reason |
|---|---|---|
| ❌ No | Lexicon synonyms by default | Hurts H@1 (72% → 64%) and MRR (-2.3 pts) overall. Same BEIR finding pattern. |
| ❌ No | Multi-query by default | Regresses direct queries (100% → 80% H@1) and conceptual H@1 (75% → 62.5%). |
| ✅ Yes | Multi-query as opt-in agent tool | Huge win on synonym queries: H@1 +42.8 pts, H@5 +57.1 pts, MRR +110%. |

## Overall (n=25)

| Condition | H@1 | H@3 | H@5 | MRR@10 |
|---|---|---|---|---|
| A: plain BM25 | **72.0%** | 80.0% | 80.0% | 0.766 |
| B: BM25 + lexicon | 64.0% | 80.0% | 88.0% | 0.743 |
| C: BM25 + multi-query | **72.0%** | **88.0%** | **100.0%** | **0.817** |
| D: lexicon + multi-query | 56.0% | 84.0% | 100.0% | 0.705 |

## By category

### DIRECT (n=10) — user speaks the tool's language

| Condition | H@1 | H@5 | MRR |
|---|---|---|---|
| A: plain BM25 | **100.0%** | 100.0% | **1.000** |
| B: +lexicon | 90.0% | 100.0% | 0.950 |
| C: +multi-query | 80.0% | 100.0% | 0.900 |
| D: both | 90.0% | 100.0% | 0.933 |

**Plain BM25 wins outright on direct queries.** Every modification regresses H@1.
This is the "do no harm" case — if the user already used the right vocabulary,
do not paraphrase.

### SYNONYM (n=7) — user uses different vocabulary than the tool description

| Condition | H@1 | H@5 | MRR |
|---|---|---|---|
| A: plain BM25 | 28.6% | 42.9% | 0.378 |
| B: +lexicon | 28.6% | 57.1% | 0.391 |
| C: +multi-query | **71.4%** | **100.0%** | **0.798** |
| D: both | 14.3% | 100.0% | 0.410 |

**Multi-query is a clear and large win on synonym queries.** This is the
target use case: query says "download a webpage" but the tool description
says "fetch a URL". The LLM rewrites bridge the vocabulary gap.

Lexicon alone barely helps (+0 H@1, +14 H@5) — the OpenGloss synonyms are too
generic for technical tool vocabulary.

The combination (D) is the *worst* for H@1: lexicon noise compounds with
multi-query rewrites, dropping H@1 from 28.6% baseline to 14.3%.

### CONCEPTUAL (n=8) — user describes a task abstractly

| Condition | H@1 | H@3 | H@5 | MRR |
|---|---|---|---|---|
| A: plain BM25 | **75.0%** | 87.5% | 87.5% | **0.812** |
| B: +lexicon | 62.5% | 100.0% | 100.0% | 0.792 |
| C: +multi-query | 62.5% | 75.0% | 100.0% | 0.729 |
| D: both | 50.0% | 87.5% | 100.0% | 0.677 |

**Mixed.** Plain BM25 leads on H@1 and MRR. Multi-query reaches 100% H@5 but
introduces enough noise to drop H@1. Lexicon helps H@3/H@5 but again hurts
H@1.

## Why lexicon hurts

The OpenGloss lexicon is a general English synonym graph. It correctly maps:
- `fetch` → `retrieval`, `collection`, `procurement`, `data retrieval`, `data fetch`
- `search` → `quest`, `inquiry`, `investigation`, `query`, `lookup`

But applied to a *technical tool index*, this adds vocabulary that the tools
themselves don't use. A query like "search the web" gets expanded to include
`quest` and `inquiry`, which match nothing useful and dilute the BM25
scoring against the correct hit (`kaos-web-search`).

## Why multi-query helps on synonym, hurts on direct

The LLM-paraphrases are *consistently good* on synonym queries — they map
"download" → "fetch", "look up" → "search", "spreadsheet" → "Excel workbook",
etc. So the union of rankings via RRF surfaces the correct tool that the
original query missed.

But on direct queries where the original is already optimal, paraphrases
introduce *worse* queries that drag the RRF score of the correct hit down
relative to its plain-BM25 ranking.

## Recommendation

**Do not change the default.** Keep `bridge_runtime_tools` on plain BM25
with no lexicon and a single query string. This matches the existing
behavior and the BEIR cross-domain finding.

**Add an opt-in agent tool** `kaos-tool-search-broader` (or similar name)
that exposes the multi-query path. Agents can call it when their initial
plain retrieval doesn't surface a usable tool. Triage rule:

1. Try plain BM25 first (`bridge_runtime_tools` default, no LLM cost).
2. If the agent's first step fails or the top-K all have low scores (e.g.
   max score < 1.0 on a normalized scale), call `kaos-tool-search-broader`
   which:
   - Asks an LLM for 3 paraphrases (one cheap Haiku call)
   - Runs BM25 on each
   - Fuses via RRF
   - Returns the top-10 result list
3. Cost when triggered: ~1 cheap LLM call per "broader search".

This matches how the existing `RetrievalAgent` exposes 4 retrieval tools
(`kaos-retrieval-bm25`, `kaos-retrieval-synonyms`, `kaos-retrieval-hyde`,
`kaos-retrieval-evaluate`) for memory documents — let the agent decide
which strategy to invoke based on what it sees.

**Do not integrate lexicon for tool retrieval.** The numbers say no.

## Methodology notes

- Ground-truth labels are hand-curated; some queries have multiple correct
  tools (any one of which counts as a hit).
- RRF constant k=60 (Cormack et al. 2009 default).
- 3 paraphrases per query; original query is included in the RRF union.
- Hit@K = ≥1 relevant tool in top-K. MRR@10 = mean reciprocal rank, capped
  at K=10. Both standard IR metrics.
- The benchmark is reproducible: `python tests/benchmarks/tool_retrieval_bench.py --with-multi-query`.

## Open questions / follow-ups

1. **Does this generalize to a 200+ tool catalog?** Current catalog is 50.
   Re-run when kaos-source / kaos-office writers land more tools.
2. **Does `kaos-tool-search-broader` actually get called?** Needs a live
   test once exposed: give the agent a synonym query, verify it falls
   through to the broader-search tool.
3. **RRF tuning.** Cormack's k=60 is the default; we didn't sweep it. The
   IR literature suggests k=60 is robust but a sweep over [10, 100] could
   nudge the numbers.
4. **Cost-aware triggering.** Cheaper alternative: skip the LLM entirely
   and let the agent provide multiple queries directly (e.g. a tool
   signature like `kaos-tool-search(queries: list[str])`). Avoids the
   paraphrase LLM call entirely.
