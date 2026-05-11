# kaos-agents Skeptic Value Findings

**Date:** 2026-05-11
**Author:** Skeptic eng review (live probes against MNDA fixtures + raw Anthropic SDK).
**Model:** `claude-haiku-4-5` for everything (per budget constraint).
**Total LLM spend across all probes:** ~$0.015. Budget cap was $1.50.

## Null Hypothesis

> A thin wrapper over the Anthropic SDK + a few helper functions would do 80%
> of what kaos-agents does at 20% of the complexity.

This report tests that with four side-by-side probes against the kaos-agents
runtime, the kaos-llm-core starter API, and the raw Anthropic Python SDK
(v0.101.0).

Scratch scripts (uncommitted, under `tests/scratch/`):

- `probe1_chat_overhead.py` — same question, three surfaces.
- `probe2_findings_vs_grep.py` — FindingsAgent (K6/K7) vs regex+oneshot.
- `probe3_triage_vs_bm25.py` — K5 summary-aware triage vs raw BM25.
- `probe4_plan_vs_prompt.py` — Plan-Execute pattern vs structured prompt.

Fixture: `~/projects/273v/kelvin-app/samples/docx/MNDA - Acme.docx` (and the
other 3 MNDAs for the corpus probe).

---

## Probe 1 — `AgentChatTool` vs raw Anthropic vs `kaos_llm_core.starter.text`

**Question:** `"What is 1247 * 893? Just give me the integer answer."` (no tools needed)

| Surface | Input tok | Output tok | Cost (USD) | Wall (s) | Output text |
|---|---:|---:|---:|---:|---|
| raw `anthropic.Anthropic().messages.create()` | 36 | 7 | $0.000071 | 0.79 | `1113571` |
| `kaos_llm_core.starter.text()` | 191 | 21 | $0.000237 | 0.75 | `1113571` |
| `AgentChatTool.execute()` (CHAT pattern) | 140 | 10 | $0.000152 | 2.15 | `1113571` |

Source: `kaos_agents/tools/registry.py:171` (AgentChatTool), `kaos_llm_core/starter.py:110` (text), `kaos_agents/runtime/runner.py:69` (Runner).

**Findings:**

- The raw SDK is the cheapest by token count by **3.3x vs starter, 2.1x vs agent_chat**, and the fastest by wall time by **2.7x vs agent_chat**.
- starter.text() pays a ~5x input-token tax for the typed-signature scaffolding (system prompt + InputField/OutputField wrappers) and produces identical text output.
- AgentChatTool runs **two** LLM calls per turn — one for intent classification, one for the response (visible in the `chat_agent: no tools available — falling back to simple response` log line). That explains the ~3x wall-clock vs raw.
- The agent emits a `WARNING` when no tools are registered. This is technically correct (chat dispatches to "simple response" when there's no ReAct candidate), but it suggests the chat path is over-engineered for the no-tool case.

**Verdict:** raw Anthropic wins on cost and latency for tool-free single-shot Q&A.
starter.text() wins ONLY when you also want a typed input/output Signature.
AgentChatTool's overhead (2 LLM calls, intent classification, runner state, hook dispatch, memory persistence) is **not justified for stateless single-shot questions** — but it is justified when you genuinely need session memory or tool calling.

---

## Probe 2 — `FindingsAgent` vs regex+oneshot LLM

**Question:** `"Find clauses about how long confidentiality obligations last (term length)."`
**Document:** `MNDA - Acme.docx`

Both paths use `claude-haiku-4-5` end-to-end (overriding the prod default of Sonnet 4.6 for synthesis, per budget constraint).

| Path | Wall (s) | LLM calls | Cost (USD) | Candidates | Survivors / cited | Answer found term language? |
|---|---:|---:|---:|---:|---:|---|
| FindingsAgent (`token=confidential`, chunk=20, parallel=3) | 2.86 | 3 | $0.0033 | 27 sentences | 3 | **Yes** — found `"shall remain in full force and effect at all times"` |
| Regex + oneshot Haiku (`\b(confidential\|term\|year\|month\|...)\b`) | 2.68 | 1 | $0.0022 | 23 sentences fed inline | (no filter step) | **No** — concluded "this NDA does not specify a term length" |

Source: `kaos_agents/patterns/findings.py` (FindingsAgent), `kaos_agents/tools/findings.py` (K7 MCP wrapper).

**Key result: the FindingsAgent path identified the correct answer; the naive grep+oneshot path missed it.**

The decisive sentence — `"The confidentiality provisions of this Agreement shall remain in full force and effect at all times after the effective date of this Agreement."` — was surfaced by FindingsAgent's Phase-1 selector (Punkt-segmented sentences containing the substring `confidential`) but **was not** in the naive regex splitter's candidate list. The naive regex sentence splitter (`(?<=[.!?])\s+(?=[A-Z])`) and the keyword filter happened to lose this specific sentence even though `confidentiality` matches the `confidential` token. Punkt-based segmentation in kaos-content + kaos-nlp-core is genuinely more robust than the 20-line regex.

The naive path's answer text concludes "this NDA does not specify a term length" — which is **wrong** for this contract. The FindingsAgent answer correctly notes the obligations are perpetual (`"shall remain in full force and effect"`) and cites the `finding_id` of the supporting sentence.

**Cost overhead:** FindingsAgent is ~50% more expensive ($0.0033 vs $0.0022) and ~7% slower on this single NDA. The cost gap will widen on larger documents because the filter step is O(n / chunk_size) LLM calls.

**Verdict:** FindingsAgent **wins on correctness/recall** on this real NDA. The three-phase overhead is justified for diligence questions where missing a clause matters. If you regenerate the naive baseline with the **same** Punkt segmenter, the recall gap probably closes — but at that point you've already imported `kaos-nlp-core` and you're 80% of the way to FindingsAgent. The remaining 20% (filter + synthesis with grounded finding_id citations) is what makes the answer auditable.

---

## Probe 3 — K5 `triage_corpus()` vs raw `Searcher.from_documents()` BM25

No LLM calls. Pure retrieval scaling test against 4 NDA docs replicated 1×, 4×, 8×, 16× (n = 4, 16, 32, 64 docs).

**Query:** `"confidentiality term length how long does the obligation last"`. Top-5.

| n_docs | K5 triage median (ms) | raw BM25 median (ms) | K5 used_summary_index | Top-5 Jaccard agreement |
|---:|---:|---:|---:|---:|
| 4 | 0.72 | 0.80 | True | 1.00 (identical) |
| 16 | 1.12 | 1.09 | True | 0.67 (4/5 agree) |
| 32 | 1.81 | 1.99 | True | 0.43 (3/5 agree) |
| 64 | 2.13 | 3.81 | True | 0.11 (1/5 agree) |

Source: `kaos_agents/context/triage.py:70` (`triage_corpus`), `kaos_nlp_core/search/__init__.py:116` (Searcher).

**Findings:**

- K5 is ~1.8x faster than raw BM25 at n=64 (2.1ms vs 3.8ms). The summary index is short — head_tokens + top/bottom n-grams — so BM25 runs over a much smaller surface.
- But the Jaccard agreement collapses as n grows. At n=64, K5 and raw BM25 share only **1 of 5** top results. They are ranking different things: K5 ranks summary-vector similarity; raw BM25 ranks full-text TF-IDF.
- At realistic working-set sizes (n ≤ ~20, which is the K5 default threshold), the time difference is sub-millisecond and "doesn't matter" in absolute terms. Below the threshold, K5 returns `None` and the caller falls through to plain BM25 anyway.

**Verdict:** K5's speedup is real but **only matters at corpus sizes ≥ 50 docs**. The correctness divergence at n=64 is more interesting: the two paths are doing different things, and the K5 fast-path can rank a document highly because its summary terms match the query while its body is irrelevant (or vice versa). **K5 is a different retrieval signal, not a free speedup.** Treat it as a complementary ranker, not a drop-in replacement. The current `triage_corpus()` policy (engage K5 only when **every** doc has a `summary_text`) is the right hedge, but the production calibration story still wants a cross-check against raw BM25 on a real corpus.

A 20-doc deal room: raw BM25 in <2ms, K5 in <2ms. The infrastructure value of K5 is "we have a tested summary-index hook we can swap out for embeddings later," not "BM25 was the bottleneck."

---

## Probe 4 — Plan-Execute pattern vs structured prompt

**Task:** Given the NDA text, return JSON `{parties, effective_date, term}`.

| Path | Wall (s) | Input tok | Output tok | Cost (USD) | Intent classified as | Parsed JSON? |
|---|---:|---:|---:|---:|---|---|
| Structured prompt via raw Anthropic SDK | 1.43 | 2215 | 60 | $0.0025 | n/a | **Yes** (parties + date + term=null) |
| `Runner` + `AgentPattern.PLAN` (Haiku) | 4.46 | 4403 | 236 | $0.0045 | `tool_use` (NOT `plan`) | Yes, plus prose explanation, **0 plan steps emitted** |

Source: `kaos_agents/patterns/plan_execute.py:62` (PlanExecuteAgent), `kaos_agents/runtime/runner.py:69` (Runner).

**Findings:**

- Plan-Execute pattern is **3.1x slower** and **1.8x more expensive** on a 5-sentence extraction task.
- It used **2.0x the input tokens** for the same source document. The overhead is the intent-classification call, the planning system prompt, and the memory-context assembly.
- **Most importantly: intent was classified as `tool_use`, not `plan`. The adaptive strategy decided this didn't need decomposition, so 0 plan steps were emitted.** The agent paid the planning machinery's cost without using it.
- Both surfaces returned correct `parties` and `effective_date`. The Plan-Execute answer included a paragraph of prose explaining *why* `term` is `null` (cites Section 6 by name), which is genuinely useful for an audit log. The structured prompt returned bare JSON with no rationale.

**Verdict:** For well-bounded extraction tasks where the schema is known up front, **the structured prompt wins on cost and latency by ~2x and produces equally usable structured output.** Plan-Execute pays for capabilities it doesn't use here (the planner returned 0 steps). It's the wrong tool for closed-world extraction. Plan-Execute is the right tool when (a) the goal is open-ended ("research X and tell me what you find"), (b) you genuinely need multi-step decomposition, or (c) you need the prose explanation to be cited and auditable for a downstream human reviewer.

A reasonable production rule: use `AgentChatTool` (and accept the 2-LLM-call cost) when the user types a free-form goal. Use a single structured prompt (or `Extract` from kaos-llm-core, which does exactly this with `Cited[T]`-shaped output) when the schema is known.

---

## Final Summary

> **kaos-agents wins on:** correctness/recall in diligence-style search (Probe 2 — FindingsAgent surfaced the right sentence that the regex baseline missed); auditability (the K7 MCP surface returns `finding_id` citations with `block_ref` provenance, which the raw paths don't); session memory and multi-turn conversation continuity (not directly probed, but the entire `SessionMemory` + VFS persistence story is real platform value); production observability (TurnSummary, UsageObserved, OTel-aligned spans, CostTrackingHook) — the cost-accounting and tracing infrastructure is the kind of plumbing that takes weeks to recreate.
>
> **kaos-agents loses on:** raw cost and latency for single-shot tool-free Q&A (Probe 1 — 2-3x overhead vs raw SDK); closed-world structured extraction (Probe 4 — 2x more expensive than a one-shot structured prompt, and the planner didn't actually plan); the no-tool warm path (chat agent emits a `WARNING` when no tools are registered, which is a smell that the chat pattern wants a tool-free fast path).
>
> **kaos-agents is roughly equivalent to alternatives on:** plain BM25 retrieval (Probe 3 — at realistic deal-room sizes ≤ 20 docs, K5 and raw BM25 are sub-millisecond and within noise of each other; at n=64 K5 is ~1.8x faster but ranks different documents).
>
> **The right deployment is:** kaos-agents for (1) recall-first diligence agents over real documents, where the K6/K7 FindingsAgent's three-phase pipeline pays for itself in correctness; (2) any task that needs session memory, multi-turn conversation, tool calling via ReAct, or cost-tracking across many turns; (3) the MCP wire surface for agentic IDEs (Claude Code, Cursor, etc.) that benefit from typed annotations, permission gating, and pause/resume.
>
> **The wrong deployment is:** wrapping every single LLM call behind `AgentChatTool` "to be safe." For tool-free single-shot questions, use raw Anthropic SDK or `kaos_llm_core.starter.text()` and skip the runner. For closed-world structured extraction with a known schema, use `kaos_llm_core.programs.extract.Extract` or a one-shot structured prompt — the Plan-Execute pattern is over-instrumented for this case and the adaptive strategy will downgrade it to `tool_use` anyway.

---

## Honest caveats

- This is a one-document N=1 study on each probe. The Probe 2 recall-win for FindingsAgent could be specific to the Acme MNDA's sentence boundaries. A 20-NDA repro is the responsible next step.
- We used Haiku 4.5 for the synthesis step in Probe 2 to honor the budget cap; the prod default is Sonnet 4.6, which would change both the cost and the answer quality. The recall conclusion (FindingsAgent surfaces sentences naive regex misses) does not depend on this because the sentence pool is fixed before synthesis.
- We did not exercise (a) tool-calling via ReAct, (b) multi-turn memory persistence across turns, (c) the permission/approval gate, (d) the WebSocket/SSE wire surface. Those are where kaos-agents earns most of its complexity budget and they were out of scope for this short probe set. Probes 1 and 4 are deliberately the **least flattering** to kaos-agents because they exercise the runtime overhead without exercising the features that overhead is buying.
- Cost numbers for Haiku 4.5 use `$1/MTok in, $5/MTok out`. Live measurement matches the kaos-llm-core cost model (`Probe 1: starter_text cost_usd = 0.0002368` ≈ `191 * 1e-6 + 21 * 5e-6`).

## Recommended follow-ups

1. **Add a "no-tools fast path" to ChatAgent.** When `tools=()`, skip intent classification and dispatch straight to `_simple_respond`. Probe 1 shows the current path runs 2 LLM calls when 1 would do. The warning log already flags this.
2. **Document the Probe 4 result in `kaos_agents.tools.registry.AgentPlanTool`.** Tell the agent: "for closed-world extraction with a known schema, prefer `kaos-extract-schema` over `kaos-agent-plan`." The current description doesn't steer the caller away from the over-engineered surface.
3. **Re-run Probe 2 on all 4 MNDAs + the EMNA fixture.** If FindingsAgent wins recall on 5/5, that's a publishable correctness claim. If it ties or loses on some, the case for the three-phase pipeline weakens.
4. **Add a `K5_vs_raw_bm25` benchmark to `tests/benchmarks/`.** Probe 3 shows the two paths ranking differently at n≥16. We should know whether the K5 summary-aware ranking actually helps a downstream agent or just looks faster on a stopwatch.
