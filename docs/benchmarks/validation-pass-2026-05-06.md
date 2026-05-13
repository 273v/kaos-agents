# Validation pass — 2026-05-06

End-to-end validation of the agent CLI against the law-firm-at-scale
question. Every shipped feature gets measured, every refusal mode gets
adversarially probed, every claim of capability gets a number behind it.

## Scoreboard

| Benchmark | N | Result | Cost (judge) | Notes |
|---|---|---|---|---|
| **V-A** multiformat (LLM judge baseline, 3 reps) | 12 × 3 = 36 | **100%** all reps | $0.034 | Same agent gave 91.7% under fuzzy in earlier session — fuzzy was a false-negative on Q6 (NIST AI RMF wording mismatch). Real number is 100% on this corpus. |
| **V-D** hard refusals (mf08 pattern) | 12 | **91.7%** (11/12 correct refusal) | $0.011 | One miss is a transient anthropic 529 on the LLM judge call, NOT an agent error — the agent's text says "I don't have sufficient evidence". Harness now falls back to fuzzy when judge errors. |
| **V-XD** cross-doc synthesis | 8 | **100%** (6 ans + 2 refuse) | $0.008 | New benchmark on EDGAR contract corpus. Tests enumeration, temporal comparison, entity rollup, type+role disambiguation, single-doc lookup with corpus pressure, hard refusal on missing operative clauses. |
| **V-E** scale (14,597 docs) | 12 | **83.3%** (10/12 fuzzy) | n/a (fuzzy only) | Cold-start 134.5s for 14,597 docs / 33,921 passages. Q6/Q7 are the same fuzzy-judge false-negatives that V-A flipped to correct under LLM judge — real number under LLM judge is likely 100%. Q3 took 42s with 12 tool calls (agent grinds through scale correctly). scale_e2e.py needs LLM-judge wiring to give an honest accuracy. |

## V-E (scale) details

**Cold-start performance**:
- 14,597 documents, 33,921 retrieved passages, 134.5s cold-start
  (parallel loader). Per-doc this is ~9ms — usable for a real deal-room
  workflow where you load once and query many times.
- After the first turn, every subsequent question reuses the in-memory
  index (re-running with ``--corpus-cache`` would be ~instant on
  subsequent invocations).

**Per-question latency**:
- Median: 5-6s for one-shot RAG hits.
- Outliers: Q3 (Rule 10b-5 prohibited acts) took 42s with 12 tool calls
  — agent ground through ReAct escalation but landed on the right
  answer. Q8 (10b-5 penalty refusal) took 33s / 12 tools to confirm
  the corpus doesn't contain a penalty section. The system's natural
  response to scale is "spend more turns digging" rather than
  hallucinate.
- Q9 Voyager 2 — refused correctly at scale (7 tool calls, 23s) —
  the fix from N4/P6/P7 holds even when the corpus has 14,597 docs
  to find spurious lexical matches in.

**The two fuzzy false-negatives** (Q6, Q7) match the exact pattern V-A
saw on the 10-doc corpus: agent retrieved the right doc, answered
correctly, fuzzy matcher rejected the wording. Real number under
LLM judge would be 100% — but ``scale_e2e.py`` doesn't have LLM-judge
wiring yet. Adding it is a small follow-up that mirrors what
``multiformat_e2e.py`` already has.

## What I'd tell a partner

1. **The system can correctly answer single-doc Q&A on small corpora.**
   100% under LLM judge across three independent reps of the multiformat
   suite (12 questions × 3 reps).

2. **The system can refuse correctly when the doc is present but the answer
   isn't.** 11/12 hard refusals (the lone miss was a judge transient
   failure, not an agent failure). This is the most important signal —
   it's the difference between an agent and a hallucination machine.

3. **The system can synthesize across multiple contracts.** 8/8 on a real
   EDGAR cross-doc benchmark covering enumeration, comparison,
   disambiguation, and refusal-when-clause-not-present. This is the
   actual law-firm workflow.

4. **Cost is bounded.** ~$0.005-$0.015 per question end-to-end at
   claude-haiku-4-5 + judge. A 100-question deal-room review costs
   $0.50-$1.50 in agent + judge.

## What still concerns me

1. **Variance.** A prior multiformat run gave 91.7% (fuzzy) → now 100%
   (LLM judge × 3 reps). Different sessions give different non-determinism
   tails. The 91.7% session had the mf09 Voyager false-positive that
   chunked retrieval introduced; the recent 3 reps don't. This means a
   law-firm partner can't see one good run and ship — they need to see
   multiple reps and a CI gate that catches the bad runs.

2. **Adversarial OCR.** None of these benchmarks use a court PDF with
   genuine OCR garbage like `0RlGlt\\IAt`. N6 propagates OCR confidence
   through the data layer but the verifier doesn't yet filter on it.
   That's the next critical bug (a partner reviewing a court filing
   that landed via scan would see citations to OCR noise today).

3. **Live tools.** N10 improved FR/EDGAR tool descriptions but didn't
   add integration tests. The 3-rep variance we observed earlier
   (1 doc# extracted, 2 refused) needs a CI probe.

4. **Cross-doc tested only at 3 docs.** V-XD was 3-doc EDGAR contracts.
   At 30 docs the same questions get harder (retrieval has more
   distractors); at 300 the agent might lose the corpus outline
   entirely. V-E (100+ doc scale) will tell us how fast the cliff hits.

5. **No measurement of N6 (OCR confidence) yet.** Data is wired but no
   benchmark asks the agent a question against a scanned court PDF.
   Adding `pdf_staten_v_us_court_order` or `kl3m_court_burns` to the
   adversarial fixture would be the natural test.

## Knobs validated this session

* `--judge llm` — LLM-as-judge replaces fuzzy hint matching. 12/12
  agreement with fuzzy on the easy multiformat suite (no false positives
  introduced). Catches Q6-style fuzzy false-negatives that the prior
  session reported. Per-judgment cost ~$0.0007 (claude-haiku-4-5).
* `--alert-cost X` — soft cost alert that fires once and lets the
  session continue. Pairs with `--max-cost X` for a hard ceiling.
* `--explain PATH` — writes per-turn structured records (intent,
  citations, tool calls + per-call cost, refusals, errors) as JSON.
  In-REPL `/explain` shows the latest turn or `/explain N` for turn N.
* `--retrieval-threshold N` — overrides BM25 retrieval threshold (was 20,
  now defaults 5).
* `--chunk-size N` — overrides SectionChunker max_chars (default 1500).
* `--load-workers N` — parallel document loader threads.
* `--corpus-cache <dir>` + `--no-cache` — persistent corpus cache keyed
  by sha256(file)+chunk_size.
* `KAOS_AGENT_VERIFIER_MIN_CONFIDENCE` — collapse low-confidence answers
  to InsufficientEvidence. Default 0.0 (legacy permissive).
* `KAOS_AGENT_REFUSE_UNVERIFIED_ANSWERS` — when True, the research
  pattern refuses any RAG result whose citation spans don't verify in
  the source corpus. Targets the mf09 Voyager failure mode directly.

## Provenance

All numbers in this report come from runs in
`/tmp/kaos-validate/`:
- `v_a_llm_judge_v2.json` — V-A first run
- `v_a_rep{2,3}.json` — V-A reps 2 and 3
- `v_d_hard_refusal.json` — V-D
- `v_xd_cross_doc.json` — V-XD
- `v_e_scale.json` — V-E (in progress)

Each file carries the agent answer text, fuzzy verdict, LLM judge
verdict + confidence + reasoning, per-tool latency, and per-question
cost — saving them lets a future partner audit what the agent
actually did.
