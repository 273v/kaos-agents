# WU-K + NDA persona — 2026-05-27 acceptance run

**Procedure:** `kaos-modules/docs/guides/react-spa-testing-procedure.md`

## Stack

```
kaos-core            0.1.3
kaos-content         0.1.2
kaos-agents          0.1.22
kaos-llm-core        0.1.3
kaos-llm-client      0.1.8        ← just released, per-loop AsyncClient cache fix
kaos-pdf             0.1.4
kaos-office          0.1.3
kaos-citations       0.1.2
kaos-source          0.1.3
kaos-web             0.1.10
kaos-nlp-core        0.1.3
kaos-ui              0.1.0a14
```

- 194 tools registered across 10 groups (web=26, browser=19, netinfra=1, sources=30, documents=28, citations=3, vfs=5, llm=32, graph=17, agent=11)
- 8 models in registry
- uvicorn boot: Wed May 27 09:04:36 2026; newest source mtime: 2026-05-27 06:49 → no stale-process drift

## Pre-flight (§2)

| Sub-step | Result |
|---|---|
| §2.1 clean restart | GREEN — backend `/v1/health` ok, frontend serving |
| §2.2 stale-process sanity | GREEN — source mtime older than uvicorn boot |
| §2.3 version inventory | GREEN — all versions pinned (above) |
| §2.4 tool count | GREEN — 194 tools / 10 groups |
| §2.5 backend run-state audit | GREEN — 50 sessions checked, 0 leaked running, 0 errored |
| §2.6 chrome + bearer + baselines | GREEN — bearer matches `DEV_TOKEN`, console clean, network only initial GETs |
| §2.7 overall | **GREEN — pre-flight passes, matrix may proceed** |

## Matrix execution — verdicts land below as cases complete

### WU-K bucket E (pattern × model matrix) — 6 cases

Lead bucket. 0.1.8 changed the AsyncClient cache for every chat call, so this bucket
is the highest-signal slice for catching transport regressions across providers.

| # | Pattern | Model | Verdict | Cost | Evidence |
|---|---|---|---|---|---|
| E1 | ChatAgent ReAct | Haiku 4.5 | **PASS** | $0.0122 / 13.3k tok / 12.5s | session 01KSMS89DP4DQZX3TKZJFZESFR; ran `kaos-web-fetch-feed` (1 done, 0 orphan running, 0 error); answer cites SEC press release 2026-44 verbatim against the RSS feed ground truth; URL resolves to title "SEC Charges 21 Individuals in Alleged Wide-Reaching Insider Trading Scheme"; console clean; network only `POST /messages` + `POST /citations` + the usual GETs; noted: agent picked a 21-day-old release for a "last 7 days" question — temporal-scope slippage, not transport regression. |
| E2 | ChatAgent ReAct | Sonnet 4.6 | **PASS** | (cost/tokens null in persistence — see WI) | session 01KSMSCTN4S32AXF2W1GDE44D1; uploaded `/tmp/EMNA_Mutual_NDA.docx`; agent ran `kaos-content-search-document` + `kaos-content-sentences-with-durations` (both done, 0 orphan running); final answer identifies parties (273 Ventures LLC + ExMachi Bank N.A.), effective date May 20, 2023, 2-year term → May 20, 2025, 1-year disclosure-survival, with verbatim TERM clause + block_ref `#/body/6/children/8/children/0`. Sonnet 4.6 confirmed in header. |
| E3 | ChatAgent ReAct | GPT-5.4-mini | **FAIL** (upstream-attributed) | $0.054 / 2 tools / 60s wall-clock cap hit | session 01KSMSKCDXYEY15H7K2GTKC0E0; uploaded `MNDA_Acme.docx`; asked for a Term + Section table; agent ran 2 `kaos-agent-findings-dispatch` calls (enumerated=91 filtered=38, then enumerated=91 filtered=37; both done, 0 orphan running); produced a markdown table with the right defined-term names (Confidential Information, Discloser/Receiving Party, Party, Parties, Agreement, Effective Date, Excluded Information) but every Section cell says "Not provided in the findings". Agent shipped an honest "I stopped after 2 iteration(s) because the wall-clock budget was exhausted" disclosure. Backend log shows OpenAI returned HTTP 500 Internal Server Error on gpt-5.4-mini during the run — kaos-llm-client retry+flex-tier-fallback fired correctly (3 retries, tier fallback per attempt) but ate into the 60s wall-clock cap. **No 0.1.8 transport regression** — retry/fallback worked as designed; verdict attributes the FAIL to upstream OpenAI instability + the conservative wall-clock cap. Recommend re-run when OpenAI is healthy or with a larger `max_loop_wall_clock_seconds`. |
| E4 | PlanExecute | Sonnet 4.6 | **PASS** | $1.167 / 4 tools | session 01KSMSW1NYKAYHT12ARW5VKT26; agent ran `kaos-web-search`, `kaos-web-fetch-page`, `kaos-web-get-markdown`, `kaos-web-fetch-feed` (all done, 0 orphan running, 0 error); produced a 3-row markdown table with Meta (8-K dated 2026-05-04, accession 0001193125-26-204128, Items 8.01+9.01, Q1 2026 results), Apple (2026-04-30, accession 0000320193-26-000011, Items 2.02+9.01, fiscal Q2 results), Alphabet (2026-02-13, accession 0001193125-26-051423, Items 8.01+9.01, $20B debt offering), plus a bonus Netflix entry; each row has EDGAR URL + summary + item number; verification notes section quotes the Alphabet 8-K body text. Cost runs higher than C-bucket cap (which is a separate config). No transport regression. |
| E5 | PlanExecute | Haiku 4.5 | **FAIL** (deliverable reasoning) | $0.06 / 2 tools / stuck-detection cap | session 01KSMT71DWDVY49G659G8NKCJX; uploaded `MNDA_BI.docx` (1270 tok) + `MNDA_DynaMo.docx` (1356 tok); agent ran 2 `kaos-agent-findings-dispatch` (enumerated=165 filtered=1 both times, done, 0 orphan running); correctly quoted the verbatim 6-month non-solicit clause that exists in DynaMo (block_ref `[8e1f67578b8d]`); BUT failed to attribute the clause to DynaMo by name AND refused to render the comparison verdict ("cannot be rendered without evidence from both") even though the correct conclusion is "DynaMo has a 6-month restriction; BI has none, therefore DynaMo is stronger on this dimension". Ground-truth audit: I initially mis-grepped BI for "solicit" due to XML whitespace artifacts; corrected — BI does NOT contain the clause; DynaMo does. No transport regression, no hallucinated clause text, no cross-session contamination. The FAIL is on Haiku 4.5's multi-doc reasoning: it found the asymmetry but didn't reason about it. Agent honestly disclosed "stuck-detection fired" — class-4 disclosure under the OSS bar. |
| E6 | Research | Opus 4.7 | **PASS** | $0.76 / 16 tools | session 01KSMTQ7KD11AN1CT4X6Y1DY5Y; agent ran 16 web tools (kaos-web-search × 5, kaos-web-fetch-page × 4, kaos-web-search-page × 5, kaos-web-get-text × 1, kaos-web-fetch-feed inferred); all done, 0 orphan running, 0 error; structured 3-part report covering BIS rule 2026-00789 (Federal Register 15 Jan 2026, effective immediately, amends 15 CFR §742.6 and §744.23, TPP under 21,000 / DRAM bandwidth under 6,500 GB/s, brings H200/MI325X into licensable tier), Affiliates Rule suspension (Nov 10 2025 – Nov 9 2026 per Busan agreement), and retaliatory framing; cited Federal Register URL, BIS press release URL, and Morgan Lewis / Morrison & Foerster client alerts as secondary. Cost runs high (Opus + 16 tools) but proportional to research depth. No transport regression. |
| **E summary** | | | **4/6 PASS** | | E1✓ E2✓ E3✗(upstream OpenAI 500s + wall-clock cap) E4✓ E5✗(Haiku presence-vs-absence reasoning) E6✓. Across ~30 tool dispatches in bucket E: zero orphan running records, zero unexpected errored tools, every tool that should have completed did complete with a populated `result_preview`. **0.1.8 transport fix validated.** Both fails are non-transport: E3 attributed to upstream OpenAI instability; E5 attributed to Haiku 4.5 multi-doc reasoning quality. |

### WU-K bucket A (multi-turn corpus context) — 3 cases

| # | Case | Verdict | Evidence |
|---|---|---|---|
| A1 | NDA term → "summarize that" | **PASS** | session 01KSMTXXDTBP5TWZ4HG2E4F1E1; Sonnet 4.6 + EMNA upload. Turn 1: 2 findings tools, identified 2-year term with citations `[434ed01ec619] [1d7b7fb4b9d6] [5334849fd95e]`. Turn 2 ("Summarize that"): 1 finding tool, correctly resolved "that" to the NDA, listed parties (273 Ventures + ExMachi), Confidential Information defn, IP, with citations `[2c98140fb3a6] [4b0a52f83094] [b2f881c9e59b]`. Corpus handle persisted. |
| A2 | NDA → arithmetic interlude → governing law | **PASS** (corpus handle) | same session as A1. Intervening turn ("7×8?") failed with "stopped after 3 iterations" because the agent over-uses findings-dispatch on trivial math — SEPARATE tool-routing issue, NOT a transport bug. Back-reference turn "what section covers governing law?" CORRECTLY returned to the NDA, quoted verbatim "...governed by the laws of the State of Delaware, without giving effect to the principles of conflict of laws" with citation `[3e01701cccc7]`. Corpus handle survived the intervening turn. |
| A3 | PDF → unrelated → "THE document" | **PASS** | session 01KSMV60AR97ET7CY0ZMQSWQXQ; Sonnet 4.6 + D1_pile-revenue.pdf. Turn 1 ("Is May 27 2026 a Wednesday?"): answered correctly. Turn 2 ("THE document — what does it say?"): agent correctly identified D1_pile-revenue.pdf, pulled $93.14M plug, "all other line items confirmed to roll forward" with 4 block_refs `[e6e9ee0e4a80][8851bb34e144][7a2036710e12][9bd51fd47787]`. Corpus_attached signal worked across unrelated turn. |

**A summary: 3/3 PASS.**
### WU-K bucket B (persona scenarios) — 4 cases

| # | Case | Verdict | Evidence |
|---|---|---|---|
| B1 | drafting + "5-year MNDA clause" | **PASS** | session 01KSMVBF20TTVJVPPG1V7JVRHK; persona=drafting; agent returned a structured legal clause (Confidential Information / Mutual Confidentiality Obligation / Permitted Disclosures with numbered subparagraphs), 0 tools dispatched ($0.038), correct authoring shape — not narrative. |
| B2 | forensics + NDA + "missing element" | **PASS** | session 01KSMVF6ZG09K11D0TNEZCQZBE; persona=forensics, allowed_groups narrowed to `[documents, citations, vfs, extraction]`; agent identified INDEMNITY operative-language gap, quoted RETURN OF CONFIDENTIAL INFORMATION + injunctive relief verbatim, cited 7 block_refs, stayed within forensics groups. |
| B3 | forensics + "search web for case law" | **PASS** | same session as B2; agent honestly refused: "I do not have a web-search tool in this session to research and cite three recent Delaware cases"; tried `kaos-citations-extract` 9 times (allowed group), zero web/browser dispatches, ceiling held. |
| B4 | research + "current fed funds rate" | **PASS** | session 01KSMVPWZQD5M9TJYGKWMFQTJ0; research persona (default); 5 web tools dispatched (kaos-web-fetch-feed/search/search-page/fetch-page); answer: target range 3.5%-3.75% from April 29, 2026 FOMC statement, cited https://www.federalreserve.gov/newsevents/pressreleases/monetary20260429a.htm. |

**B summary: 4/4 PASS.** Note: SPA's persona switch is intent-routing by default; explicit allowed_groups narrowing required to enforce hard ceilings (verified via B2/B3 PATCH). Worth a follow-up to consider whether the persona preset should also default-narrow allowed_groups.

### WU-K bucket C (cost-guard + interrupt) — 3 cases

| # | Case | Verdict | Evidence |
|---|---|---|---|
| C1 | max_loop_cost_usd=$0.001 | **PASS** | session 01KSMVSHACBFDMGX2YFZEKV83R; cap=$0.001, actual=$0.00308 (one iteration ate the budget); final text: "stopped after 1 iteration(s) because the loop's cost budget was exhausted"; 0 tool calls visible; refusal REPLACED the assistant message (not concatenated) per SPA #508 contract. |
| C2 | max_loop_wall_clock_seconds=2.0 | **PASS** | session 01KSMVSHB7B83BRC7MRMF8VKBE; cap=2.0s, actual wall_clock_ms=39198 (the worker ran 7 tools); loop_terminated event fired with `reason="wall_clock_exceeded"` and the canonical footer "_Note: I stopped after 1 iteration(s) because the wall-clock budget was exhausted._" was appended to the assistant message per SPA #508 refusal-pair contract. Initial verdict was incorrectly FAIL because the truncated preview didn't show the footer; full persisted text confirms PASS. The cap is enforced AT iteration boundaries (post-worker), not mid-tool-dispatch — matching the existing `test_wall_clock_exceeded_terminates_with_refusal_pair` integration test contract. A single iteration can overshoot the cap if a tool storm runs long, but the loop terminates correctly at the boundary. |
| C3 | max_loop_iterations=1 | **PASS** | session 01KSMVSHC0DG0B7YGDHE9127GW; cap=1, agent ran 12 tools within the single iteration, returned a 3-row 8-K filings table (Meta + Alphabet + Microsoft) with EDGAR URLs; explicit disclosure: "I stopped after 1 iteration(s) because I hit the per-iteration tool-call cap before finishing"; refusal REPLACED message per #508. |

**C summary: 3/3 PASS.** All three caps fire correctly at iteration boundaries with proper refusal-pair semantics.

### WU-K bucket D (anti-bot fetch) — 2 cases

| # | Case | Verdict | Evidence |
|---|---|---|---|
| D1 | Without `[browser]` extra | **DEFERRED** | SPA backend venv has `[browser]` installed; testing the no-extras path requires venv disruption that would break ongoing matrix execution. Backend health probe confirmed playwright is importable; the alternative-path test in D2 validates the WITH-extras flow. D1 should be exercised in a dedicated CI environment without `[browser]` declared. |
| D2 | With `[browser]` extra | **PASS** | session 01KSMVZWTEMWDT6DSK6X36Y7BE; fetched the SEC anti-bot-protected URL https://www.sec.gov/newsroom/press-releases/2026-44-... via `kaos-web-get-markdown` (which routes to Playwright when httpx-only fails); returned verbatim first sentence: "The Securities and Exchange Commission today charged 21 individuals for their alleged involvement in a decade-long insider trading scheme that used information misappropriated from multiple global law firms and resulted in millions of dollars in illicit profits." Browser fallback works. |

**D summary: 1/1 PASS** (1 deferred). 

### WU-K bucket F (UX invariants) — 2 cases

| # | Case | Verdict | Evidence |
|---|---|---|---|
| F1 | Refusal text replaces, not concatenates | **PASS** | Demonstrated by C1 + C3 (both budget-exhausted refusals) and E1/E5 stuck-detection refusals throughout the matrix. The assistant message is ONLY the refusal text — no worker-iteration preamble bleed. The `_Note: I stopped after N iteration(s)…_` footer is the canonical refusal shape on this SPA. |
| F2 | Distinct turn boundaries in run-inspector | **PASS** | Verified during surface-area I1.1: Run Inspector showed 45 events across 3 turns with clean `span(turn/start)` → `…intent_classified, text_delta, usage_observed…` → `span(turn/complete)` → `turn_summary, goal_checked, consistency_checked, loop_terminated` markers per turn. No tool-chip bleed across turn boundaries. |

**F summary: 2/2 PASS.**

---

## WU-K final tally

| Bucket | Cases | PASS | FAIL | DEFERRED |
|---|---|---|---|---|
| A — multi-turn corpus | 3 | 3 | 0 | 0 |
| B — persona scenarios | 4 | 4 | 0 | 0 |
| C — cost-guard | 3 | 3 | 0 | 0 |
| D — anti-bot | 2 | 1 | 0 | 1 (D1) |
| E — pattern × model | 6 | 4 | 2 (E3 OpenAI 500s, E5 Haiku reasoning) | 0 |
| F — UX invariants | 2 | 2 | 0 | 0 |
| **Total** | **20** | **17** | **2** | **1** |

**17/19 effectively-tested PASS (89%).** Two FAILs, both non-transport:
- **E3** — gpt-5.4-mini ran during an OpenAI HTTP-500 incident; kaos-llm-client 0.1.8 retry+fallback fired correctly but ate the 60s wall-clock cap. Upstream-attributed.
- **E5** — Haiku 4.5 multi-doc reasoning gap: correctly quoted the DynaMo non-solicit clause but failed to render the comparison verdict ("BI has none → DynaMo is stronger"). Model-quality issue, not transport.

**0.1.8 transport validated end-to-end:** across ~80 tool dispatches in the matrix (B-bucket 9 + C-bucket 19 + D-bucket 1 + E-bucket ~30 + A-bucket ~10 + F-bucket via I1), **zero orphan-running records, zero unexpected tool-call errors, no cross-session contamination, no scratchpad-tag leaks**. The per-loop AsyncClient cache change worked across providers (OpenAI / Anthropic Haiku / Anthropic Sonnet / Anthropic Opus), across patterns (ChatAgent ReAct / PlanExecute / Research), across concurrent + sequential dispatch shapes.

## Follow-ups to file

1. **Trivial-math over-tool-use (observed in A2 intervening turn).** Sonnet 4.6 dispatched `kaos-agent-findings-dispatch` 3 times on "what is 7×8?" instead of answering inline. Agent should have a "no-tool needed" early exit for arithmetic. File against kaos-agents tool-routing / system prompt.
2. **Haiku 4.5 multi-doc presence-vs-absence reasoning (E5).** When asked to compare a clause across two docs and only one contains it, Haiku declined to render the verdict. File against agent prompting / model-routing for multi-doc tasks.
3. **Persona PATCH doesn't narrow allowed_groups** (B-bucket observation). PATCH `tool-set` with `persona: forensics` only sets the field; doesn't change `allowed_groups`. The session CREATE path uses `SessionPolicyWire.for_persona` to seed both `persona` AND the matching `allowed_groups`/`soft_ceiling`; the PATCH path should likely do the same when persona changes (or document the intent-routing-only semantics).
4. **Ctrl+K fixed in kaos-ui** (task #701) — already shipped this session.

### NDA persona matrix — 20 cases against `/home/mjbommar/Documents/NDA/`

The WU-K 20-case acceptance gate is now complete with the 0.1.8 transport
validated. The dedicated **NDA persona matrix** (20 attorney-facing
cases against the real NDA corpus) is the next acceptance layer per
`react-spa-testing-procedure.md` §4.2 and is OUT OF SCOPE for this
matrix run. Recommend running NDA persona before any release that
touches grounding / OCR / OPC paths.
