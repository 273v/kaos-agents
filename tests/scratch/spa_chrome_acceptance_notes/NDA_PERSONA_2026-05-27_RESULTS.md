# NDA persona matrix — 2026-05-27 Chrome MCP acceptance run

**Date:** 2026-05-27 (UTC late afternoon)
**Stack:**
- kaos-agents 0.1.24 (PyPI; SESSION FILES manifest in worker context)
- kaos-core 0.1.4 (PyPI; sibling-files hint in path-resolver)
- kaos-llm-client 0.1.8 (PyPI; per-loop AsyncClient cache)
- kaos-pdf 0.1.4, kaos-office 0.1.3, kaos-content 0.1.2, kaos-web 0.1.10
- kaos-ui SPA backend on `main` (build sha at run time)
- Model floor: anthropic:claude-sonnet-4-6 (default), Haiku 4.5 for cost-sensitive personas
**Procedure:** `kaos-modules/docs/guides/react-spa-testing-procedure.md` §4.2
**Persona fixture:** `kaos-modules/data/personas/nda-matrix/personas.json` (10 personas, v1.0)
**Corpus:** 5 NDAs uploaded per session — `EMNA Mutual NDA.docx`, `MNDA - Acme.docx`, `MNDA - BI.docx`, `MNDA - CC Final 2.docx`, `MNDA - DynaMo.docx`

## Ground truth (from personas.json)

| File | Governing law | Carveout | Section # |
|---|---|---|---|
| EMNA Mutual NDA.docx | **Delaware** | yes | **11** |
| MNDA - Acme.docx | **Michigan** | yes | **12** |
| MNDA - BI.docx | **Michigan** | yes | **11** |
| MNDA - CC Final 2.docx | **Michigan** | yes | **10** |
| MNDA - DynaMo.docx | **Delaware** | yes | **10** |

## Diary

Live log of pre-flight, each persona run, four-signal capture per turn (snapshot + console + network + cost), ground-truth verification per row.

### Pre-flight (§2)

- §2.1 backend `/v1/health` = `ok`, build `4e048b43db10` ✓
- §2.2 uvicorn boot `Wed May 27 16:33:04`; newest source mtime `2026-05-27 12:00:22` (older than boot) ✓
- §2.3 version inventory captured (above)
- §2.5 50 sessions audited, 0 leaked running ✓
- §2.6 bearer matches `DEV_TOKEN` ✓
- **Pre-flight GREEN — matrix may proceed**

### Persona runs

#### P1 — M&A shorthand ("GL on these 5 — table form, no fluff.")
- Session `01KSNT3N2M1D96R5MSEKW88WXP`
- Model `anthropic:claude-sonnet-4-6`, cost **$0.062**, 7.7k tok, **0 tool_calls**
- Answered from auto-generated document summaries (post-upload-parse cache)
- Final answer: clean 5-row markdown table with file / counterparty / effective_date / governing_law / forum
- Ground-truth check:
  - EMNA → Delaware ✓
  - Acme → Michigan ✓
  - BI → Michigan ✓
  - CC → Michigan ✓
  - DynaMo → Delaware ✓
- Anaphora resolution ("these 5") ✓
- Shorthand ("GL") resolved without clarification ✓
- Honest disclosure: "EMNA and DynaMo forum: summaries don't specify a dispute-resolution venue. I can pull the exact forum clauses from those two documents if you'd like."
- **Verdict: PASS** (class-4 honest+partial → class-5 correct+cited)

#### P2 — Procurement risk review (5 NDAs × 3 terms each)
- Session `01KSNT68MXJ4VPYQR3RKF6MJQY`
- Model `anthropic:claude-sonnet-4-6`, cost **$0.242**, 55.6k tok, **2 tool_calls** (search-document, context-window)
- Stopped mid-EMNA paragraph 1 of 5 with explicit stuck-disclosure footer:
  > "I stopped after 1 iteration(s) because I hit the per-iteration tool-call cap before finishing — the loop preferred returning what I have over a runaway tool-storm."
- Only 1.5 of 5 NDA paragraphs delivered. EMNA partial paragraph quotes "any and all third-party claims … attorneys' fees" — content looks accurate but deliverable contract not met.
- **Verdict: FAIL on completeness** (class-4 honest-stuck, not class-5 correct-and-complete)
- Root cause: `max_react_iterations` cap (default 10) too tight for multi-doc-multi-fact extraction. Workaround: bump cap per session. Real fix: agent should pick `kaos-agent-findings-dispatch` (one tool, multi-doc internally) over many individual `search-document` calls.

#### P3 — Comparative analysis ("Compare EMNA Mutual NDA and MNDA - Acme")
- Session (P3 run), model `anthropic:claude-sonnet-4-6`, cost **$0.116**
- Agent identified three real differences in clause text (2-year vs 5-year term, MI/Lansing vs DE venue, non-solicit scope) — clause **text quotes are correct**.
- BUT: confidently mis-attributed the files. Wrote:
  > "the EMNA Mutual NDA involves ExMachi Bank N.A. / 273 Ventures and references Michigan, while the MNDA - Acme appears to reference Delaware"
- Ground truth: **EMNA = Delaware, Acme = Michigan** — agent SWAPPED them.
- This is OSS legal-research bar **class-1: confident wrong → P0**.
- Root cause: agent's citations are content-hashes (e.g. `[72e43288d19d]`) with no filename in the citation — the LLM has to infer which file each quote came from and got it backwards. Same kaos-agents grounding limitation surfaced in earlier audits.
- **Verdict: FAIL — confident-wrong file→law attribution**

#### P4 — Single-doc deep dive ("MNDA - Acme: term + auto-renewal? Quote the section.")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.039**, 2 tool_calls (both `kaos-agent-findings-dispatch`)
- Stuck-detection fired at iteration 2; final answer: "the findings do not contain the term or auto-renewal provisions of the agreement."
- This is **wrong by omission** — P3 quoted "TERM. This Agreement shall terminate upon the second anniversary of its Effective Date." (attributed to Michigan / Lansing → Acme by ground truth). The data IS in the corpus; findings-dispatch missed it.
- **Verdict: FAIL** — class-3 honest "I don't know" but underlying retrieval failure (data exists, dispatch didn't surface it). Stuck-disclosure footer present.

#### P5 — Non-standard provisions flag report (5 NDAs × 3 dimensions for partner review)
- Model `anthropic:claude-sonnet-4-6`, cost **$0.286**, 70.2k tok, 2 tool_calls (`search-document`, `context-window`)
- Full comparison table: file / counterparty / eff. date / indemnity / IP / non-solicit / liability cap / atty fees / governing law / term
- **All 5 governing-law attributions CORRECT** (EMNA Delaware, Acme/BI/CC Michigan-Lansing, DynaMo Delaware) — NO swap this time
- BONUS: flagged a real document defect — EMNA §9 liability cap has stale "LG GROUP / SPLITCO GROUP" boilerplate truncated mid-sentence after "PROVIDED, HOWEVER, THAT THE FOREGOING". This is the kind of value-add partner work is looking for.
- Per-NDA flags called out:
  - EMNA: defective LoL clause
  - CC Final 2: broad non-solicit (covers customers too) + 5-yr term
  - DynaMo: GmbH counterparty
- **Verdict: PASS** — class-5 correct+cited+with-defect-detection. No stuck footer.

#### P6 — One-word answer ("Governing law for DynaMo NDA? One word.")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.036**, 3 tool_calls (all findings-dispatch)
- Ground truth answer: **Delaware** (P1 nailed this from summaries with 0 tools)
- P6 hit 3-iteration budget; critic kept flagging as ungrounded; final response: "I was unable to answer this within the 3-iteration budget..."
- Astonishing miss: same model, same corpus, same answer that P1 produced in 1 turn from summaries
- Root cause hypothesis: phrasing "the DynaMo NDA? One word." triggered findings-dispatch routing instead of summaries-cache lookup. Once down that path, dispatch returned nothing usable.
- **Verdict: FAIL** — class-3 honest stuck, but over-routed for trivial single-doc query

#### P7 — Negotiator precedent arsenal ("Pull the most aggressive provision from each NDA")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.308**, 75.1k tok, 1 tool_call (`search-document`)
- Excellent content: 9k-char deliverable with all 5 NDAs covered, each with quoted aggressive provision + "why it's aggressive" + "your talking point" — then a summary table at end with negotiating levers and strategic sequencing recommendation
- Highlights:
  - EMNA: injunction on *threat* of disclosure (no actual breach required)
  - Acme: perpetual survival of obligations + mandatory individual sub-NDAs
  - BI: 3-yr term + perpetual survival + 5-day signed certification
  - CC: broad non-solicit (employees + agents + consultants + contractors + **customers**)
  - DynaMo: broad CI definition (no marking required, third-party data auto-included)
- BUT: ends with stuck-detection footer: "I stopped after 1 iteration(s) because I hit the per-iteration tool-call cap before finishing..."
- The footer is a **stuck-detection FALSE POSITIVE** — the deliverable IS complete (table at end, strategic sequencing, all 5 covered).
- **Verdict (strict, per [chrome-matrix-final-text-only](memory)): FAIL** because of stuck-disclosure footer
- **Verdict (by content quality): PASS** — would have shipped if the footer weren't auto-attached
- This identifies a real kaos-agents bug: stuck-detection footer needs to be gated on whether the response is actually incomplete, not just on iter-count.

#### P8 — Two-document side-by-side ("BI vs CC: confidentiality term + carve-outs, two columns")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.019**, 1 tool_call (findings-dispatch)
- Final: "The findings do not contain sufficient content from either MNDA - BI or MNDA - CC Final 2 to answer the question. The only retrieved finding [626cca88e3e9] is a bare filename header from MNDA - CC Final 2 with no substantive text..."
- This is **wrong by omission** — P5 successfully extracted BI ("3 years") and CC ("5 years") from same corpus. P7 extracted BI's "3-yr term + perpetual survival + 5-day certification" and CC's "5-year term".
- Findings-dispatch returned only one bare filename header — retrieval picked the wrong chunk.
- **Verdict: FAIL** — class-3 honest "I don't know" hiding a retrieval failure

#### P9 — Quick scan list ("counterparty + jurisdiction for each NDA")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.152**, 2 tool_calls (both findings-dispatch)
- Counterparties (all 5 correct):
  - ExMachi Bank N.A. ✓
  - Acme Co. (Nevada corporation) ✓
  - CyberCorp Co. (California corporation) ✓
  - DynaMo GmbH (Germany) ✓
  - Beta Inc. (Delaware corporation) ✓
- Jurisdiction column marked "not determinable from findings" with honest disclosure:
  > "The governing-law clauses found (Delaware and Michigan) cannot be reliably matched to specific NDAs from the available evidence."
- Same content-hash-citation attribution gap from P3 + P9, surfaced honestly this time.
- Stuck-detection footer fires.
- **Verdict: FAIL by strict bar** (footer + incomplete jurisdiction column). Honest about the gap which is class-3 not class-1.

#### P10 — Verbatim clause text ("EMNA Mutual NDA indemnification clause, verbatim")
- Model `anthropic:claude-sonnet-4-6`, cost **$0.038**, 3 tool_calls (all findings-dispatch)
- Hit 3-iteration budget; final: "I was unable to answer this within the 3-iteration budget."
- Critic's diagnosis (preserved in agent reply): "Call the document extraction or corpus search tool directly to locate and retrieve the verbatim indemnification clause text from the EMNA Mutual NDA; do not rely on findings dispatch."
- Critic's advice is correct but the agent has no in-loop strategy switch — once committed to findings-dispatch, can't pivot to direct extraction.
- P5 successfully quoted EMNA indemnity language ("mutual; defend, indemnify, and hold harmless for third-party claims"). Same corpus, same agent — different prompt phrasing landed on the broken path.
- **Verdict: FAIL** — class-3 honest stuck

---

### Matrix scorecard

| # | Persona | Cost | Tools | Verdict | Notes |
|---|---|---|---|---|---|
| P1 | M&A GL table | $0.062 | 0 | **PASS** | summaries cache, 5/5 GL correct |
| P2 | Procurement risk (5×3) | $0.242 | 2 | FAIL | iter-cap, 1.5/5 paragraphs |
| P3 | EMNA vs Acme compare | $0.116 | 1 | **FAIL (class-1)** | confidently swapped GL attribution |
| P4 | Acme term + auto-renewal | $0.039 | 2 | FAIL | findings-dispatch retrieval miss |
| P5 | Non-standard flags | $0.286 | 2 | **PASS** | strong: 5/5 + defect flag |
| P6 | DynaMo GL one-word | $0.036 | 3 | FAIL | over-routed for trivial query |
| P7 | Negotiator precedent | $0.308 | 1 | FAIL (footer) / PASS (content) | stuck-detection false positive |
| P8 | BI vs CC side-by-side | $0.019 | 1 | FAIL | retrieval miss, 1 chunk returned |
| P9 | Counterparty + jurisdiction | $0.152 | 2 | FAIL | jurisdiction "not determinable" |
| P10 | EMNA indemnity verbatim | $0.038 | 3 | FAIL | critic-advice / strategy-switch gap |

- **Strict scorecard: 2/10 PASS** (P1, P5)
- **By-content scorecard: 3/10 PASS** (+P7 if footer were gated)
- **Class-1 confidently-wrong: 1** (P3 GL swap)
- **Total matrix spend: ~$1.20**

---

### Cross-cutting findings (kaos-agents 0.1.24)

1. **Summaries-path is the highest-quality but most fragile route.**
   - P1 (5-doc table) hit the summaries cache directly and was perfect with 0 tools.
   - P6 (1-doc one-word) did NOT hit the summaries path and went through 3 failed findings-dispatch iterations to produce "I don't know."
   - The route depends on prompt phrasing, not query intent. Need an explicit routing policy that prefers summaries-cache for FACT-LOOKUP class queries.

2. **kaos-agent-findings-dispatch silently misses on narrow / specific queries.**
   - P4 (Acme term), P8 (BI vs CC), P10 (EMNA indemnity verbatim) ALL returned empty from findings-dispatch despite P5 and P7 successfully extracting the same content from the same corpus on broader prompts.
   - Hypothesis: dispatch's per-doc filter step is over-aggressive — when the query is narrow enough, it drops everything before synthesis.
   - This is dangerous: "no findings" is class-3 (honest) but it's effectively a confident-wrong "the data isn't here" when the data IS here.

3. **Content-hash citations break file attribution.**
   - P3 swapped EMNA↔Acme governing law (class-1 confidently wrong).
   - P9 honestly disclosed: "governing-law clauses found cannot be reliably matched to specific NDAs."
   - Citations like `[72e43288d19d]` don't carry filename. The LLM has to infer file from context and can get it backwards.
   - **Architectural fix needed**: every citation in the FINDINGS section must include `source_filename` or equivalent.

4. **Stuck-detection footer fires on complete answers.**
   - P7 delivered a polished, fully-realized negotiator precedent arsenal — table at end, all 5 NDAs covered, strategic sequencing — yet the auto-footer says "I stopped after 1 iteration(s) ... please verify any specific claims."
   - The footer should be gated on whether the response is *incomplete*, not on iter-count alone. P7's response is not incomplete.

5. **Inner-loop iteration cap of 10 is too tight for multi-doc deliverables.**
   - P2 (5 NDAs × 3 terms per doc) hit cap at 1.5 of 5 paragraphs delivered.
   - Workaround: bump `KAOS_AGENT_MAX_REACT_ITERATIONS` env var for multi-doc-class workloads.
   - Real fix: agent should recognize multi-doc-batch-extraction queries and pick `kaos-agent-findings` (multi-stage pipeline, batches internally) over many `search-document` calls.

6. **The critic gives useful advice that the agent can't act on.**
   - P10 critic: "do not rely on findings dispatch" / "Call the document extraction or corpus search tool directly"
   - Agent has no in-loop mechanism to switch tool strategy in response to critic feedback. Once committed, it just retries the same broken approach.

### Recommended P0/P1 issues

- **P0**: P3-class confident-wrong file attribution → add `source_filename` to all citations
- **P0**: P4/P8/P10-class silent findings-dispatch misses → audit the per-doc filter aggressiveness; emit observable warnings when dispatch drops all candidates
- **P1**: P7-class stuck-detection footer false positives → gate footer on actual completeness signals
- **P1**: P6-class routing gap → explicit summaries-cache preference for FACT-LOOKUP queries
- **P2**: P2-class iter-cap on multi-doc deliverables → either bump the cap or improve agent tool selection
- **P2**: P10-class critic-advice gap → allow agent to pivot tool strategy mid-loop in response to critic verdict

---

## Architectural deep-dive (post-matrix, 2026-05-27 evening)

Three parallel Explore-agent investigations on the highest-priority issues. Each lands on concrete file:line fixes.

### Fix #1 (P0) — Citation filename JOIN

**Tracked as task #705.**

**Problem.** Citations emit as bare content-hashes `[72e43288d19d]` because `FindingCandidate` carries `block_ref` (JSON pointer like `#/body/3`) and `finding_id` (12-hex hash) but NOT the source filename. At emit time (`runtime/agent.py:1154-1160`), `CitationFound.source_uri` is set to `block_ref or candidate.finding_id` — the document identity is **never JOIN'd in**.

**Single emit site.** `kaos-agents/kaos_agents/runtime/agent.py:1154-1160`:
```python
yield emitter.emit(
    CitationFound,
    claim=candidate.text,
    source_uri=block_ref or candidate.finding_id,  # <-- bare hash here
    confidence=float(finding.relevance),
    verified=True,
)
```

Note: lines 1141-1145 actually build a `source_uri_for_block` dict but **never use it** at emit time. So the data is being computed and then discarded.

**Data plumbing.** At enumeration time the `DocumentView` does carry `view.document.metadata.source.uri` (and `metadata.title`). The break is in `FindingCandidate` (frozen slotted dataclass at `patterns/findings.py:269-270`) which has no `source_uri` / `document_id` field.

**Concrete fix.**

1. `kaos-agents/kaos_agents/patterns/findings.py` ~line 269 — add fields to `FindingCandidate`:
   ```python
   document_id: str | None = None
   source_uri: str | None = None
   ```

2. `kaos-agents/kaos_agents/patterns/findings.py` — at candidate enumeration inside `FindingsAgent.run()`, inject `source_uri` from the `DocumentView`:
   ```python
   source_uri = view.document.metadata.source.uri if view.document.metadata.source else view.document.metadata.title
   ```

3. `kaos-agents/kaos_agents/runtime/agent.py:1157` — use the JOIN'd uri at emit:
   ```python
   source_uri=candidate.source_uri or f"{candidate.source_uri}#{candidate.finding_id}" if candidate.source_uri else block_ref or candidate.finding_id
   ```
   (cleaner: `source_uri=f"{candidate.source_uri or 'unknown'}#{candidate.finding_id}"`)

**Expected impact.** P3-class confident-wrong attribution should vanish. Each citation in the LLM's context becomes `[EMNA Mutual NDA.docx#72e43288d19d: GOVERNING LAW. Delaware]` instead of `[72e43288d19d]`. Deterministic file→fact mapping.

**Test.** Re-run NDA persona P3 ("Compare EMNA Mutual NDA and MNDA - Acme") and P9 ("counterparty + jurisdiction list") — both should produce correct file→GL attribution.

### Fix #2 (P0) — findings-dispatch silent miss

**Tracked as task #706.**

**Problem.** The per-doc filter stage at `patterns/findings.py:2052` drops ANY candidate below a 0.5 relevance threshold with no observability:
```python
if relevance < threshold:
    continue
```

When ALL candidates from a document score < 0.5, the filter silently drops everything and the synthesis step is skipped. The agent reports "the findings do not contain..." even when the data demonstrably IS in the corpus.

**Why narrow queries fail but broad queries succeed.** Broad queries ("non-standard provisions") enumerate more candidates in Phase 1 — even at the same 0.5 threshold, more survive by sheer volume. Narrow queries ("Acme term + auto-renewal") get fewer candidates and the filter's score variance can leave a candidate scoring 0.49 — silently dropped.

**Observability gap.** Logging at `patterns/findings.py:1769-1777` is DEBUG-level and only fires when survivors > 0. When `survivors_by_id` is empty, only the aggregate refusal reason logs (line 1820). There's no per-chunk warning, no max-score-vs-threshold signal, no OTel span for filter-dropout.

**Concrete fix (combined Option A + B).**

1. **Instrumentation first** (Option A — `patterns/findings.py` ~line 2038, before survivors loop):
   ```python
   chunk_scores: dict[str, float] = {}
   ```
   Then inside the loop at ~line 2043:
   ```python
   chunk_scores[fid] = max(chunk_scores.get(fid, 0.0), relevance)
   ```
   After the loop:
   ```python
   if not survivors and chunk:
       max_chunk_score = max(chunk_scores.values()) if chunk_scores else 0.0
       logger.warning(
           "findings.filter_culled_all: chunk_size=%d max_score=%.3f threshold=%.3f question=%r",
           len(chunk), max_chunk_score, threshold, question[:80],
       )
   ```

2. **Threshold relaxation** (Option B — defaults at `patterns/findings.py:1403` and `tools/findings.py:299`):
   - Change `relevance_threshold` default from `0.5 → 0.3`
   - Trade-off: ~10-15% more filter tokens (more survivors → bigger synthesis context), better recall on narrow queries

**Test.** Re-run NDA persona P4 (Acme term/auto-renewal), P8 (BI vs CC term/carve-outs), P10 (EMNA indemnification verbatim) — all should return non-empty findings. Watch backend logs for `findings.filter_culled_all` warnings.

### Fix #3 (P1) — Stuck-detection footer false positive

**Tracked as task #707.**

**Problem.** Two footer variants both auto-append from `patterns/agentic_loop.py:988-1010` (`_build_budget_footer`):

- Variant A (templates at line 1026-1029): `"I hit the per-iteration tool-call cap before finishing"` — fired by `tool_call_cap_exceeded` budget exit
- Variant B (templates at line 1018-1020 in `_BUDGET_REASON_PHRASE`): `"the loop's stuck-detection fired"` — fired by `stuck_no_progress` budget exit

The footer attaches unconditionally in `_emit_failure_refusal()` (~line 1141) when `_should_preserve_worker_draft()` returns True. That predicate (`patterns/agentic_loop.py:1104-1138`) checks:

```python
return (
    len(_draft_for_preserve(state).strip()) >= _MIN_WORKER_DRAFT_CHARS
    and state.last_terminal_verdict in ("", "satisfied")  # <-- "satisfied" is the bug
)
```

It **never checks if the response is actually complete**. Even when a critic verdict said `satisfied`, the footer still appends.

The M4 completeness judge (~line 731-776) CAN flip `satisfied → partial` if it detects incompleteness — but it's **opt-in** (`m4_completeness_model: str | None = None` at line 292). In standard SPA sessions M4 is not enabled.

**Concrete fix (minimal — one-line change).**

`patterns/agentic_loop.py:1138`:
```python
return (
    len(_draft_for_preserve(state).strip()) >= _MIN_WORKER_DRAFT_CHARS
    and state.last_terminal_verdict == ""  # remove "satisfied" — it passed critics
)
```

Rationale: if a critic said "satisfied," the response is defensible — no caveat needed. The footer should ONLY ship when the iteration budget expires BEFORE any critic verdict.

**Optional richer fix.** Add `state.is_complete_exit: bool` flag, set True when M4 judge approves completion. Gate footer on `not state.is_complete_exit`. More principled but more code.

**Test.** Re-run NDA persona P7 ("most aggressive provision per NDA") — should ship the same complete 9k-char deliverable WITHOUT the stuck footer. Strict verdict should flip from FAIL → PASS.

---

## Recommended follow-up research & testing

### Immediate (this week)

1. **Ship fixes #1, #2, #3 as kaos-agents 0.1.25.** All three are surgical (single emit site, one threshold change, one predicate change). Combined patch is < 50 LOC.

2. **Re-run NDA persona matrix on 0.1.25.** Acceptance bar: 7/10 PASS strict, 0 class-1 confidently-wrong, 0 spurious stuck footers on complete deliverables.

3. **Add regression test for citation JOIN.** `tests/integration/test_citation_filename_join.py` — feed FindingsAgent a corpus of 2 docs with distinguishable content, assert emitted `CitationFound.source_uri` contains the source filename, not just the hash.

### Near-term (next 2 weeks)

4. **Routing gap (task #708, P1).** Audit the chat-pattern's tool-selection prompt. Add an explicit fast-path: for queries matching FACT-LOOKUP shape (named entity + named attribute, ≤ 20 tokens), prefer summaries-cache lookup over findings-dispatch. P6 ("DynaMo GL? One word.") should be a 1-turn cache hit.

5. **Critic-driven strategy switch (task #709, P2).** When critic verdict includes a named-tool recommendation ("try `kaos-content-search-document` instead"), make the next iteration prefer that tool. Today the agent just retries the same broken path.

6. **Iteration cap policy (P2-class).** P2 (5 NDAs × 3 terms) hit cap with 1.5/5 paragraphs delivered. Either bump `KAOS_AGENT_MAX_REACT_ITERATIONS` default from 10→16 for multi-doc-class workloads, OR (better) detect "N items × M attributes" intent and auto-route to `kaos-agent-findings` (which batches internally and amortizes iteration budget across docs).

### Test fixtures to add

7. **Negative-attribution fixture.** Build a 2-NDA corpus where Doc A says "Delaware" and Doc B says "Michigan." Ask "What is the governing law of Doc A?" — currently can fail with the bare-hash citation. After fix #1, should PASS deterministically.

8. **Narrow-query findings-dispatch fixture.** Single 1-page contract with a very specific clause (e.g., "termination convenience: 90 days"). Ask "What's the termination notice period?" — currently can return empty. After fix #2, should return the clause text with `findings.filter_culled_all` warning ABSENT.

9. **Complete-deliverable-no-footer fixture.** Multi-doc query that should complete cleanly. Assert response does NOT contain "I stopped after" or "stuck-detection fired" strings.

### Open questions for product / architecture discussion

10. **Should findings-dispatch be the default for multi-doc-fact extraction, or should summaries-cache always be tried first?** Today routing depends on prompt phrasing in a way that's not user-predictable. Need an explicit policy doc.

11. **Should citation rendering always JOIN at the LLM-visible layer, even for non-findings tools?** kaos-content's `search-document` and `context-window` may have the same gap. Worth a parallel audit.

12. **What's the right max iteration default for the chat pattern in a deal-room workload?** Currently 10. WU-K showed 10 is fine for 1-doc Q&A. NDA matrix shows 10 is tight for 5-doc work. Possibly should be tied to corpus size.

---

## Layered investigation — 3 parallel sub-agents (evening of 2026-05-27)

Three parallel `general-purpose` sub-agents, each owning one layer, each running real `uv run python /tmp/...` reproductions. **Two of the three diary-proposed fixes were misdiagnoses and have been corrected. One landed working code.**

### Layer 1 — Citation / grounding (FIX LANDED, uncommitted in tree)

**Sub-agent:** Layer 1 — citation/grounding (id `ac6f278...`)

**What landed:** `+231 / -22` across `kaos_agents/patterns/findings.py` and `kaos_agents/runtime/agent.py`. ~30 LOC of behavior change, rest is docstrings + the security hardening.

**The fix.**
- Added `source_uri: str | None = None` field to `FindingCandidate` (frozen slotted dataclass, last optional field for backward compat)
- `_wrap_untrusted_text` (filter render) and `_render_synthesis_findings` (synthesis render) emit the new `source_uri="..."` attribute when present
- `_resolve_corpus_view_with_document` returns 4-tuple including `block_id_to_source_uri` map — keyed by `id(block)` not positional index so it survives `apply_retrieval_plan` narrowing (which rebuilds with a SUBSET of blocks at NEW positions but reuses the same immutable block objects by reference)
- `_selector_with_source_uri` adapter wraps `every_sentence_selector` — pure selectors stay pure
- Source-uri picker prefers `parsed.metadata.source.uri`, then `metadata.title`, then upload filename — adapter at the boundary, no kaos-content internals leak in
- `flag_injection_suspected` preserves `source_uri` when rewrapping (no field drop on injection-defense path)
- `CitationFound.source_uri` emit at `agent.py:1154-1160` now uses `f"{source_uri}#{block_ref}"` composite when resolved, else falls back to legacy

**Security hardening (caught by sub-agent, not in original plan).** Switched the attribute-value escape from `xml.sax.saxutils.escape` (only handles `&/</>`) to `xml.sax.saxutils.quoteattr` (also handles `"`/`'`). Without this, a hostile `source_uri` containing `"` could break out of the attribute and inject a synthetic `</untrusted_document_content>` envelope close. Verified injection-vector closed.

**Verification — end-to-end live run.**
- `/tmp/grounding_repro_05_live_e2e.py` with `anthropic:claude-sonnet-4-6`
- 172 candidates → 7 survivors, $0.094, 28k tokens, 12 LLM calls
- Final answer: *"**Delaware law** governs the agreement in **EMNA Mutual NDA.docx** [3e01701cccc7]: ... **Michigan law** governs the agreement in **MNDA - Acme.docx** [5b9a025781a9]: ..."*
- Ground truth: `EMNA+Delaware=True, Acme+Michigan=True` ✓; `EMNA+Michigan(P3 bug)=False, Acme+Delaware(P3 bug)=False` ✓
- **P3 class-1 confidently-wrong swap eliminated.**

**Test suite:** 142/142 `tests/unit/test_findings*.py` pass; 3197 pass / 6 skipped / 0 failures full unit suite; ruff/ty clean.

**Edge cases tested.**
| Case | Behavior |
|---|---|
| Doc without source uri | `source_uri=None` → attribute omitted, backward compat with single-doc tests |
| Duplicate filenames | Both candidates get same `source_uri` — LLM can't disambiguate by filename alone. **OPEN ISSUE** below. |
| Long filename (~200 chars) | Passes verbatim, no truncation |
| Unicode filename (`Vertrag — München & Köln 中文.docx`) | Non-ASCII preserved, `&` XML-escaped. PASS |
| Hostile `source_uri` with `"` / `<` / `>` | `quoteattr` switches to single-quote wrapping when value contains `"`. Injection vector closed. PASS |

**Open issues from Layer 1.**
1. **Duplicate filenames** (`contract.pdf` × 2 from different uploads) collapse via `.rsplit('/')`. Suggested: prefix with short content hash or upload index when collision detected. Not blocking — common SPA uploads have distinct names.
2. **SPA Citations panel UI** may need a parser update to split on the first `#` of the new composite citation format and surface the filename as the human label.
3. **`id()`-keyed block lookup is fragile** — if a future change in `kaos_agents/patterns/retrieval/apply.py` clones blocks via `block.model_copy()`, the lookup silently degrades to `None` (graceful — falls back to legacy citation). Worth a regression test.
4. **Integration test** `tests/integration/test_citation_filename_join.py` NOT yet created — port `/tmp/grounding_repro_05_live_e2e.py` into the suite as a live-integration test.
5. **Parallel audit needed** for non-findings dispatch paths (`kaos-content-search-document`, `kaos-agent-findings` MCP tool with explicit per-doc views) — likely OK because they don't merge across docs, but worth a sweep.

### Layer 2 — FindingsAgent filter (DIARY FIX #2 WAS WRONG TARGET)

**Sub-agent:** Layer 2 — FindingsAgent filter (id `a71a62c...`)

**Diary correction.** The proposed Fix #2 (lower filter threshold 0.5 → 0.3) is **wrong target**. Filter LLM scores relevant clauses at **0.85-1.00** on real narrow queries — well above 0.5. Threshold sweep 0.5 → 0.1 changes nothing:

| query | thr 0.5 | thr 0.4 | thr 0.3 | thr 0.2 | thr 0.1 |
|---|---|---|---|---|---|
| P4 Acme term | 3/829 | 3/824 | 3/870 | 3/934 | 3/870 |
| P10 EMNA indemnity | 2/515 | 2/515 | 2/556 | 2/556 | 2/556 |
| P8 BI | 10/1054 | 11/1112 | 11/1112 | 11/1028 | 11/1025 |
| P8 CC | 3/766 | 4/847 | 4/852 | 4/854 | 3/836 |

Also: chat-agent actually uses **0.4** (not 0.5) at `runtime/agent.py:1394`.

**Real root cause (with data).** Two layers upstream in `apply_retrieval_plan` → `_apply_search_document`.

On the SPA path with the real LLM planner over the merged 5-NDA corpus (422 sentences):
| query | planner picked | narrowed to | enum | surv | ans_chars |
|---|---|---|---|---|---|
| P4 Acme term | **ngram** | **11 sents** | 11 | 1 | 283 (says "no term") |
| P10 EMNA indemnity | **ngram** | **11 sents** | 11 | **0** | 0 (REFUSAL) |

The planner picks NGRAM/TOKEN → `kaos_content.search.search_document(level="sentence", top_k=20)` → BM25 narrows 422 → 3-11 sentences that don't include the answer. **Reason: user vocabulary ("auto-renewal", "verbatim") doesn't lexically overlap with contract vocabulary ("terminate upon", "defend"). The filter LLM never sees the answer text.**

Per-doc dispatch (no planner narrowing) succeeds on all narrow queries:
| query | enum | surv | ans_chars |
|---|---|---|---|
| P4 Acme | 90 | 3 | 928 (correct quote of TERM clause) |
| P10 EMNA | 80 | 2 | 515 (correct quote of indemnity) |
| P8 BI | 82 | 10 | 1054 |
| P8 CC | 84 | 3 | 766 |

**Three surgical fixes (in priority order):**

- **Fix B (HIGHEST LEVERAGE, zero cost):** Update `LLMRetrievalPlanner.PlanRetrieval` signature description at `kaos_agents/patterns/retrieval/planner.py:33-67`. Current text actively biases planner toward NGRAM for "named clause / section" questions. Add explicit guidance: when (a) question names a specific doc by filename AND (b) asks for a verbatim clause → return NONE strategy. Let FindingsAgent's per-sentence filter judge relevance, not lexical BM25.
- **Fix A (defense-in-depth):** Bump `plan_floor` default in `runtime/agent.py:1274` from 5 → ~20, OR gate on total sentence count (skip planner when sentences ≤ 1000). Per-doc path works fine on small corpora.
- **Fix C (in-flight remediation):** In `kaos_agents/patterns/retrieval/apply.py:54-63`, expand the `_narrow_document_from_search_results` fallback. Currently fires only when `kept_blocks` is empty. Add: also fall back when `len(kept_blocks) < min(threshold, ~10% of original)`. 4-line change.

**Plus instrumentation (ship even though it's observability-only):** `patterns/findings.py:2038` — add `findings.filter_culled_all` warning with `max_chunk_score` when survivors empty. Would have made this 90-min investigation a 30-second log read.

### Layer 3 — Routing / agentic loop (DIARY FIX #3 WAS WRONG)

**Sub-agent:** Layer 3 — routing/agentic loop (id `a0c3648...`)

**Diary correction.** The proposed Fix #3 (remove `"satisfied"` from `_should_preserve_worker_draft` whitelist at `agentic_loop.py:1138`) is **a no-op for P7 AND breaks an existing test**.

Truth table from Layer 3 reproduction:
```
long draft, empty verdict (tool_call_cap path)   cur=True  proposed=True   no change
long draft, satisfied verdict                    cur=True  proposed=False  CHANGED (but unreachable)
long draft, needs_more_work verdict              cur=False proposed=False  no change
```

Why no-op: P7's footer is the `tool_call_cap_exceeded` variant, fires at `agentic_loop.py:495-514` BEFORE the goal-check at line 545. At that point `state.last_terminal_verdict == ""` (reset to empty on each iteration start at line 324) — NOT `"satisfied"`. So removing `"satisfied"` from the whitelist never executes the changed branch for the P7 scenario.

Why test break: `test_substantive_draft_with_satisfied_verdict_preserves` at `tests/unit/test_agentic_loop_refusal_preserves_worker_text.py:192-201` asserts the very behavior the proposed change removes.

**Real fix options (NOT shippable as one-line):**
- **(a)** Reorder so goal-check runs before cap-check on the iteration that triggered the cap. Adds 1 LLM call per cap-fire turn.
- **(b)** Drop footer for `tool_call_cap_exceeded` when `iteration == 1` — worker hasn't replanned anything, just hit a one-iteration tool envelope. The footer's "I hit the per-iteration tool-call cap before finishing" presumes incompleteness which can't be verified cheaply.
- **(c)** Gate footer behind an M4 completeness pass.

Recommend **(b)** as cheapest correct fix. **DEFERRED to 0.1.26** — needs design doc + regression test.

**The "summaries-cache" path used by P1 is ACCIDENTAL, not designed.**

P1 vs P6 event-stream diff (real captures):
- P1 classifier: `intent=clarify conf=0.88` → routes to `_handle_clarify → _simple_respond` → sees `corpus_markdown` summaries baked into system prompt → answers from summaries with 0 tool calls.
- P6 classifier: `intent=research conf=0.95` → routes to `_handle_research_streaming → _run_findings_dispatch` → 3 iters of `enumerated=4 filtered=0` → refusal.

Single load-bearing difference: an LLM classifier coin-flip between CLARIFY and RESEARCH. The corpus-attached auto-promotion at `patterns/chat.py:495` only promotes RESPOND and TOOL_USE to RESEARCH — CLARIFY passes through unmodified. That's the accidental ramp into the "summaries-cache" path.

**Where summaries live (the mechanism).** It's the system prompt, not a tool:
1. `kaos-ui/examples/single-user-chat/backend/app/services/uploads.py:431` — `_enrich_parsed_doc` calls `kaos_llm_core.starter.summarize` at upload time, stores on `FileMeta.summary`
2. `uploads.py:1035-1136` — `render_session_corpus_markdown` builds per-file metadata block including line 1131-1132 `header_lines.append(f"- summary: {meta.summary}")`
3. `stream_proxy.py:215` — `_instructions_with_corpus` appends `f"## Documents attached to this session\n\n{corpus_markdown}"` to system instructions

**Recommended Layer 3 fix:** Make summaries-aware fast path INTENTIONAL — when session has `DOCUMENTS` with `summary` metadata AND user message is short/RESPOND-class, route through a NAMED synthesis branch that grounds on summaries explicitly. Emit a typed event so it's observable. Keep `RESEARCH → findings-dispatch` as deep-extract path. **NOT a prompt hack — a new typed dispatch branch.**

**Critic-advice gap (P10) — DEFERRED to 0.1.26.** Needs a `tool_hint: str | None` channel through `WorkerCallable` signature → `BaseAgent.turn` / `ChatAgent._classify` → bypass corpus-attached auto-promotion when set. ~150-300 LOC across kaos-agents + SPA backend worker adapter. Many P10-class cases will likely resolve once Layer 2 fixes land, so defer until those land and re-measure.

---

## Final 0.1.25 plan (corrected, synthesized across 3 layers)

### Ship in 0.1.25

| # | Source | Change | File | LOC | Status |
|---|---|---|---|---|---|
| 1 | Layer 1 | Citation `source_uri` JOIN + security hardening | `patterns/findings.py`, `runtime/agent.py` | +231/-22 (~30 behavior) | **LANDED, uncommitted, 142/142 tests pass** |
| 2 | Layer 2 Fix B | Planner signature — "named doc + verbatim clause" → NONE | `patterns/retrieval/planner.py:33-67` | ~10 | pending |
| 3 | Layer 2 Fix A | Bump `plan_floor` 5→20 or sentence-count gate | `runtime/agent.py:1274` | ~5 | pending |
| 4 | Layer 2 Fix C | Looser `_narrow_document_from_search_results` fallback (< 10%) | `patterns/retrieval/apply.py:54-63` | ~4 | pending |
| 5 | Layer 2 instr | `findings.filter_culled_all` warning when survivors empty | `patterns/findings.py:2038` | ~10 | pending |
| 6 | Layer 3 | Make summaries-aware fast path INTENTIONAL (typed branch + event) | `patterns/chat.py` + `runtime/agent.py` | ~80-100 | pending |
| 7 | Layer 1 | Integration test from `/tmp/grounding_repro_05_live_e2e.py` | `tests/integration/test_citation_filename_join.py` | ~80 | pending |
| 8 | Layer 1 | Regression test: `id()`-block-lookup contract on apply_retrieval_plan | `tests/integration/test_apply_retrieval_block_identity.py` | ~50 | pending |

**Total new code:** ~240 behavior + ~210 test = ~450 LOC.

### Defer to 0.1.26

| # | Source | Reason |
|---|---|---|
| A | Layer 3 stuck-footer fix (option b — gate on iter==1) | Needs design doc + regression test; diary's one-line fix was misdiagnosis |
| B | Layer 3 critic-driven tool switch (tool_hint channel) | ~150-300 LOC across 3 layers; many cases will resolve once Layer 2 lands |
| C | Layer 1 duplicate-filename disambiguation | Not blocking; SPA uploads typically distinct names |
| D | Layer 1 SPA Citations UI parser update | Coordinate with kaos-ui release after backend ships |
| E | Parallel audit: non-findings dispatch paths | Sweep `kaos-content-search-document`, MCP findings tool with per-doc views |

### Acceptance gate for 0.1.25 release

- **Re-run NDA persona matrix on 0.1.25.** Must hit:
  - ≥7/10 strict PASS (current baseline 2/10)
  - 0 class-1 confidently-wrong attributions (current baseline 1: P3)
  - Specific per-persona checks:
    - P3 ✓ Layer 1 fix verified end-to-end; EMNA→Delaware, Acme→Michigan attributed correctly
    - P4/P8/P10 ✓ Layer 2 Fix B routes "named doc + verbatim clause" to NONE → per-doc dispatch which works
    - P6 ✓ Layer 3 typed branch routes "what's the GL for DynaMo, one word" through summaries-cache deterministically
    - P7 — stuck-footer not fixed in 0.1.25; may still trip, document
- **Unit tests:** full `pytest tests/` green
- **Integration tests:** new citation-JOIN test passes; `tests/integration/` suite green
- **Live model floor:** sonnet-4-6 or gpt-5.4-mini for the acceptance matrix re-run; haiku NOT acceptable

### Mandatory before release

- Promote NDA persona matrix to the **acceptance gate floor** for any kaos-agents corpus-Q&A release. WU-K alone is insufficient — under-samples the failure modes that matter for attorneys.
- Update [reference_kaos_agents_document_qa](memory) memory with the new findings-dispatch routing rules + citation format.
- Update CHANGELOG with all three fix sources (cite repro scripts in `/tmp/`).

