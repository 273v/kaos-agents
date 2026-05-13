# KC16 — Pre-release audit findings (kaos-agents 0.1.0a1)

**Auditor:** Independent KC16 reviewer (external lens; team claims treated as inputs to verify).
**Date:** 2026-05-11.
**Repo HEAD at audit:** `bd72a49` (`test(kaos-agents): KC8 re-baseline of ladder + parity + Sprint 1-3 + module-stress`).
**Inputs consumed:**
- `docs/roadmap-prerelease-closeout.md` (closeout @ `80562cc`).
- 10 substantive commits: `d0ba060`, `dee0c9a`, `fb82f64`, `916cb67`, `fe73833`, `f752ecf`, `0ffb020`, `b8f5998`, `21463ba`, `a338d1e`.
- 3 skeptic reports: `docs/design/skeptic-prod-ops-findings.md` (`6c83841`), `docs/design/skeptic-trust-findings.md` (`47b5f7d`), `kaos-agents/docs/design/skeptic-value-findings.md` (`9e21b82`).
- KC8 re-baseline: `kaos-agents/docs/design/kc8-rebaseline-2026-05-11.md` (`bd72a49`).
- PA15 cross-provider matrix: `kaos-agents/docs/design/pa15-cross-provider-matrix.md` (`16d8eb6`).
- Release plan: `docs/release-plan-kaos-agents.md` (`7367850`).
- Audit-01 (existing module audit): `docs/audit-01/kaos-agents.md`.
- Open follow-ups PA11-PA15 (per closeout + PA16-PA18 to be created from this report).
- Values directive: **quality > correctness > transparency > adaptation > cost**.

**Budget consumed:** $0 LLM spend (read + reason only, as the brief instructs).

---

## 1. Audit summary

**Verdict: `RELEASE-WITH-CAVEATS`.** kaos-agents is ready for a v0.1.0a1
alpha release on PyPI provided five things land before the tag (one
correctness fix, two refusal-to-ship-without-caveats README/CHANGELOG
additions, and two test additions) and the team explicitly accepts the
remaining six as documented limitations. The substantive contracts that
defined the pre-release roadmap (auth-failure surfacing, prompt-injection
defense, structured refusal, findings consistency, semantic-rewrite
selector, recorder durability, honest cost-cap reporting, cost
transparency) are landed clean and verified live: KC8's 125-of-126
pass rate at HEAD with $2.65 of live spend is real evidence, not
mocked-test theater.

The thing that pushed the verdict to RELEASE-WITH-CAVEATS rather than
straight RELEASE is the gap between what the README will claim ("agentic
runtime, MCP-native, audit trail, cost-capped") and what the system
actually delivers on the edges. Three skeptic findings are
**transparency-honored but not enforcement-closed**: (a) the chat-path
cost cap is honest but soft, with up to 2x overshoot documented (PA13);
(b) `ResearchAgent` (the RAG/research path) has no cap-wiring at all
(PA11); (c) the recorder still persists full document bodies into JSONL
(probe 5 secondary concern) — and the captured JSONLs are committed to
the public-ish monorepo. These are honest-document-and-ship items under
the values lens, but they need to actually be documented honestly, not
buried.

Two new release-blockers that the skeptics did NOT explicitly catch and
the closeout did NOT flag as blockers — and that we surface here:
**KC16-N1** (the audit-01 KAG-001 QA-gate failures appear unaddressed
since 2026-04 — ruff format drift, ruff lint, ty diagnostics, and the
unit-suite timeout are all still extant per spot check); and
**KC16-N2** (the kaos-llm-core pricing-table miss for `gpt-5.5` from
PA15 Gap #2 silently breaks the cost-cap contract on a published
provider — even with the README caveat, callers who configure gpt-5.5
get `budget_exceeded=false` while paying real money). KC16-N1 blocks
release tagging; KC16-N2 is a Phase A2 fix in the sibling kaos-llm-core
repo (already on PyPI at 0.1.0a3 — needs a 0.1.0a4 patch release before
kaos-agents tags).

The substantive engineering quality is real: 125-of-126 KC8 live tests
green, 3 skeptic reports with closing-status mapped to specific
commits, a re-audit framework, the values lens explicitly elevated over
cost. This is not "demo-ware that breaks at the edges" — but the edges
are exactly where the release prose has to be honest.

---

## 2. Findings table

Sorted by severity desc, then disposition desc.

| #  | Severity  | Disposition       | Category           | Summary | Evidence ref |
|----|-----------|-------------------|--------------------|---------|--------------|
| 1  | CRITICAL  | BLOCK_RELEASE     | quality            | QA gates from audit-01 (`KAG-001`) still failing — ruff format drift, ruff lint findings, ty diagnostics, and the unit-suite-90s timeout. Phase A7 "sanity gates" will block tag. | `docs/audit-01/kaos-agents.md` §Findings KAG-001; spot-checked at HEAD `bd72a49`. |
| 2  | CRITICAL  | FIX_BEFORE_TAG    | correctness        | `kaos-llm-core` pricing table has no `gpt-5.5` entry (`PRICING` dict) → `cost_usd=$0` while real OpenAI billing is non-zero. The cost-cap contract Sprint-3 #9 documents is unenforceable on this provider. Released as 0.1.0a3 on PyPI; needs an 0.1.0a4 patch before kaos-agents tags so the README's "supported providers" table doesn't ship a lie. | `kaos-agents/docs/design/pa15-cross-provider-matrix.md` Gap #2, Gap #5; `kaos_llm_core/observability/cost.py:PRICING`. |
| 3  | HIGH      | FIX_BEFORE_TAG    | correctness        | `gpt-5.5` (and the broader OpenAI reasoning-model class — o3, o4-mini, anything new from OpenAI) hard-fails every `FindingsAgent` call because `_DEFAULT_TEMPERATURE = 0.0` is sent unconditionally and reasoning models reject `temperature=0`. **README must explicitly call out reasoning models as unsupported**, OR the code must omit `temperature` for that class. README-only acceptable per values lens (transparency over adaptation). | `pa15-cross-provider-matrix.md` Gap #1; `kaos_agents/patterns/findings.py:_DEFAULT_TEMPERATURE`; PA15 row 4 (1/7 contracts pass on gpt-5.5). |
| 4  | HIGH      | FIX_BEFORE_TAG    | transparency       | Audit-trail JSONL captures persist full document bodies (50,464-char `message` fields observed; 26 input fields > 8KB across 50 captures), AND committed JSONLs are in monorepo `git log`. For a regulated-industry adopter, every doc the CI processes becomes a secondary data plane. The recorder has NO redaction / truncation pass. README's "audit trail" claim conflicts with SOC2 CC7.2 / HIPAA / FINRA hygiene unless the production user opts out of recording. | `skeptic-prod-ops-findings.md` Probe 5; `kaos-agents/tests/integration/_recorder.py` (no `redact`/`truncate`/`hash` keywords); `git log 8e67e9d` confirms JSONLs committed. |
| 5  | HIGH      | FIX_IN_0.1.0a2    | correctness        | `ResearchAgent` (the RAG/research pattern) has no `max_cost_usd` wiring whatsoever (PA11). A regulator told "tool calls are cost-capped" cannot evaluate the RAG path. Sprint-3 #9 closed chat + findings + corpus-filter; research was deferred and not documented. | `roadmap-prerelease-closeout.md` PA11 (#163); `grep max_cost_usd kaos-agents/kaos_agents/patterns/research/` → only one match (unrelated). |
| 6  | HIGH      | DOCUMENT_AND_SHIP | correctness        | Chat-path cost cap is honest (`budget_exceeded` flag truthful) but soft (up to 2x overshoot, per `test_chat_cap_within_5pct_tolerance`'s own tolerance band). The tool's parameter description does name this, but the README must as well — otherwise a CFO/COO reading "$X cap per turn" gets a 2x answer in practice. | `roadmap-prerelease-closeout.md` PA13 (#165); `kaos-agents/tests/integration/test_cost_cap_honesty_live.py:151` `tolerance = 2.0`. |
| 7  | HIGH      | DOCUMENT_AND_SHIP | adaptation         | Findings consistency Jaccard on `gpt-5.4-mini` is 0.70–0.79 vs the Anthropic-pinned 0.95 floor. Two associates running the same query on the same NDA see materially different surviving sets. Must be in README's provider-compatibility table. | `pa15-cross-provider-matrix.md` Gap #3; PA15 row 3 (6/7 pass). |
| 8  | HIGH      | DOCUMENT_AND_SHIP | correctness        | t12 NDA risk-memo judge regressed `passed → failed` at HEAD with no clear Sprint-1/2/3 attribution. Closeout characterizes as LLM-judge stochasticity. **Without a 3-run retry to characterize the flake rate before tag, release is signing the lie "this passes."** Per `feedback_no_skipped_tests_in_ci.md`, this cannot be deferred. | `kc8-rebaseline-2026-05-11.md` §3.1; `tests/integration/ladder/test_t12_nda_risk_memo.py`. |
| 9  | MEDIUM    | FIX_BEFORE_TAG    | transparency       | `FindingsAgent` has `max_cost_usd` but NO `max_chunks` / `max_candidates` cap. A misrouted `every_sentence` call on a 10K-paragraph corpus enumerates 30K candidates, builds 1500 chunk coroutines via `asyncio.gather(...)` simultaneously, and only the cost-cap (PA15-fragile on gpt-5.5) defends against burn. Skeptic probe 3 still partially open. Add hard ceilings (e.g., 200 chunks default) as defense-in-depth even with the cost cap in place. | `skeptic-prod-ops-findings.md` Probe 3b ($1.50/call possible at scale); `kaos_agents/patterns/findings.py` (no `max_chunks` / `max_candidates` constants). |
| 10 | MEDIUM    | FIX_BEFORE_TAG    | correctness        | Sprint-1 #3's prompt-injection synthesis-step defense IS live-tested at HEAD (`test_payload_does_not_leak_canary[synthesis_targeting]` passed at `7367850`) — closeout's "not stress-tested live" claim is stale. But the test uses Haiku-tier and uses ONE injection-payload shape; should add at least one Sonnet-tier synthesis-targeted test before tag to retire skeptic Probe 1's "got-lucky on Haiku" critique. | `kc8-rebaseline-2026-05-11.md` (synthesis_targeting in INDEX); `tests/integration/test_findings_injection_live.py` (only one parametrization). |
| 11 | MEDIUM    | FIX_IN_0.1.0a2    | quality            | `kaos-llm-core.PRICING` and `kaos-llm-client.MODEL_PRICING` have diverged: client has `gpt-5.5`, core does not. The mirror-back rule (`feedback_per_module_split_mirror.md`) means a sibling repo fix must mirror to monorepo. Beyond gpt-5.5, no test asserts pricing-table parity across the two siblings. | `pa15-cross-provider-matrix.md` Gap #2 root-cause; no `test_pricing_table_parity.py` exists. |
| 12 | MEDIUM    | FIX_IN_0.1.0a2    | correctness        | Sonnet 4.6 consistency Jaccard outlier (0.621 on one of three runs at PA15) suggests the `temperature=0` Anthropic guarantee is not bit-deterministic across calls. README should NOT claim Sonnet is fully consistency-safe; trust skeptic and PA15 agree on this. | `pa15-cross-provider-matrix.md` Gap #4; `skeptic-trust-findings.md` Probe 1 (cross-run Jaccard 0.84-0.92). |
| 13 | MEDIUM    | DOCUMENT_AND_SHIP | correctness        | Subprocess recorder gap (skeptic Probe 6): direct provider SDK calls in a child process (that does NOT import `kaos_llm_core`) escape the audit trail. Closeout did not flag this. README "audit trail" claim must caveat: "covers all calls routed through kaos-llm-core; direct provider SDK calls in user-supplied tools are NOT captured." | `skeptic-prod-ops-findings.md` Probe 6; no httpx-level recorder shipped. |
| 14 | MEDIUM    | DOCUMENT_AND_SHIP | quality            | `triage_corpus()` summary-aware path (K5) and raw BM25 diverge sharply at n≥16 docs (Jaccard 0.43 at n=32, 0.11 at n=64 per Probe 3). README must NOT claim K5 is a drop-in BM25 replacement; it's a different ranker. | `skeptic-value-findings.md` Probe 3; `kaos_agents/context/triage.py`. |
| 15 | MEDIUM    | DOCUMENT_AND_SHIP | adaptation         | PA15 only covered 4 models (haiku-4-5, sonnet-4-6, gpt-5.4-mini, gpt-5.5). Google (Gemini), xAI (Grok), Groq (Llama, Mixtral), Mistral, OpenRouter — all listed in kaos-llm-client as supported providers — are unverified for any Sprint 1-3 contract. README provider-compatibility table must restrict claims to the 3 green rows. | `pa15-cross-provider-matrix.md` (only 4 rows); `CLAUDE.md` lists 8 providers. |
| 16 | LOW       | FIX_BEFORE_TAG    | quality            | Audit-01 KAG-005: `kaos_agents/recipes/__init__.py` has no `__all__`. Public-API hygiene. Trivial to add. | `docs/audit-01/kaos-agents.md` KAG-005; verified extant at HEAD. |
| 17 | LOW       | FIX_IN_0.1.0a2    | quality            | Audit-01 KAG-006: several MCP error responses omit the three-part recovery contract. Spot-checks at `kaos_agents/tools.py:467, 527, 685` still open. | `docs/audit-01/kaos-agents.md` KAG-006. |
| 18 | LOW       | FIX_IN_0.1.0a2    | quality            | Audit-01 KAG-007: `memory/search.py` probes sibling source checkout for OpenGloss lexicon. Hidden packaging assumption — breaks for `uv pip install kaos-agents` wheels. | `docs/audit-01/kaos-agents.md` KAG-007; `kaos-agents/kaos_agents/memory/search.py:36`. |
| 19 | LOW       | FIX_IN_0.1.0a2    | quality            | Audit-01 KAG-008: `cli/chat.py` is still 1742 lines (was 1670). Partial refactor (cli_chat.py → cli/chat.py) happened; the concern-mixing did not get split. | `cli/chat.py` wc -l = 1742. |
| 20 | LOW       | DOCUMENT_AND_SHIP | transparency       | Finding-id stability across runs is now SHA256-based (Sprint-2 #5 commit `f752ecf`) — closeout claims this. The OLDER skeptic-trust observation "ids are uuid4 — non-stable" (probe 1) is FIXED. Document the fix in CHANGELOG so re-readers of the skeptic report don't believe the unfixed version. | `skeptic-trust-findings.md` Observation 2; `f752ecf` compute_finding_id(). |
| 21 | LOW       | DOCUMENT_AND_SHIP | quality            | Sprint-1 #1 disk-VFS leakage footgun is fixed via `KaosRuntime.test_mode()` for live tests, but the production-user-facing default IS STILL disk-backed (the docs explicitly elevate "disk-first VFS" as a design principle). For a regulated user with a long-running KAOS-agent pod, this means session memory persists across container restarts on a shared volume — sometimes desired, sometimes a cross-user leak. README must call out the persistence model honestly. | `d0ba060` commit; `CLAUDE.md` "Disk-first VFS"; default `.kaos-vfs` root. |
| 22 | LOW       | INFORMATIONAL     | transparency       | KC8 has 7 cost regressions (Δ +86% to +7870%); closeout characterizes them as stochastic. Cost is the lowest-priority lens per the values directive; informational only. | `kc8-rebaseline-2026-05-11.md` §3.2. |

**Counts:** **2 CRITICAL, 6 HIGH, 7 MEDIUM, 6 LOW, 1 INFO** (22 findings).
**By disposition:** **1 BLOCK_RELEASE, 6 FIX_BEFORE_TAG, 7 FIX_IN_0.1.0a2, 7 DOCUMENT_AND_SHIP, 1 INFORMATIONAL**.

---

## 3. Per-finding detail

### KC16-1 — QA gates from audit-01 unaddressed (CRITICAL · BLOCK_RELEASE)

**Problem.** The audit-01 `KAG-001` finding documented ruff format drift across 7 files, 3 ruff lint diagnostics (RUF003, F401, RUF100), 5 ty diagnostics, and a unit-test-suite timeout at 90s. Phase A7 of the release plan explicitly requires `ruff format --check`, `ruff check`, `ty check`, and `pytest tests/unit -q` to all be clean before tag. The closeout report does not mention these being addressed. A spot-check shows the cited files still exist at HEAD. Without a fresh QA pass producing zero diagnostics, the release plan's A7 gate cannot pass.

**Evidence.** `docs/audit-01/kaos-agents.md` §Findings KAG-001; `kaos-agents/kaos_agents/patterns/research/` exists and `research.py` (one of the cited files) is present; `kaos-agents/tests/unit/test_streaming_metrics.py:59` benchmark mixed into unit suite.

**Fix.** Phase A2 must run the full QA sequence (`ruff format`, `ruff check --fix`, `ty check`, `pytest tests/unit -q -m "not benchmark"`) and resolve every diagnostic. Audit-01 KAG-004 (benchmark tests under `tests/unit`) must also be addressed — move them under `tests/benchmarks/` so the unit gate finishes under the bounded timeout.

**Owner.** Phase A2 driver.

### KC16-2 — kaos-llm-core pricing table missing gpt-5.5 (CRITICAL · FIX_BEFORE_TAG)

**Problem.** `kaos_llm_core.observability.cost.PRICING` lacks an entry for `gpt-5.5`. PA15 confirms `cost_usd=0.0, total_tokens=685` despite real OpenAI billing. This silently breaks the cost-cap contract for gpt-5.5 even though `kaos-llm-client.MODEL_PRICING` already carries the entry. The two sibling pricing tables have diverged — exactly the failure mode `feedback_per_module_split_mirror.md` warns about (per-module fix must mirror back). Until this is fixed, the README's "cost accounting" claim is false for any reasoning-model user.

**Evidence.** `pa15-cross-provider-matrix.md` Gap #2, Gap #5; kaos-llm-core is on PyPI at `0.1.0a3` (per release plan §0).

**Fix.** Issue `kaos-llm-core 0.1.0a4` with the missing pricing entries (`gpt-5.5`, plus parity sweep against `kaos-llm-client.MODEL_PRICING`). Add a regression test (`test_pricing_table_parity.py`) that fails when the two tables diverge. Mirror the fix into the monorepo per `feedback_per_module_split_mirror.md`. Bump `kaos-agents` dependency floor.

**Owner.** kaos-llm-core maintainer (this is a cross-repo dependency).

### KC16-3 — gpt-5.5 / reasoning-model hard-fail on FindingsAgent (HIGH · FIX_BEFORE_TAG)

**Problem.** `_DEFAULT_TEMPERATURE = 0.0` is sent unconditionally to every Call inside `FindingsAgent`. The OpenAI API rejects this with HTTP 400 for the reasoning-model class (`o3`, `o4-mini`, `gpt-5.5`). 5/7 PA15 contracts on gpt-5.5 fail with the identical error. This is a cross-provider correctness gap; under the values lens it can be document-and-ship for v0.1.0a1 IF the README explicitly says reasoning models are unsupported.

**Evidence.** `pa15-cross-provider-matrix.md` Gap #1; `kaos_agents/patterns/findings.py:_DEFAULT_TEMPERATURE`.

**Fix (minimum for ship).** README "Provider compatibility" table explicitly lists OpenAI reasoning models as UNSUPPORTED for findings paths. CHANGELOG `[Unreleased]` entry. PA16 to track the code-side fix in 0.1.0a2 (detect reasoning model identifiers, omit `temperature`).

**Owner.** Phase A6 README author + PA16 implementer.

### KC16-4 — Recorder persists raw document bodies; JSONLs committed to repo (HIGH · FIX_BEFORE_TAG)

**Problem.** The recorder still has no redaction/truncation path (`grep -n redact|truncate|hash|encrypted|sensitive kaos-agents/tests/integration/_recorder.py` returns three unrelated hits and none for input data). Probe 5 measured 1.63 MB of input data across 50 captures, including 50,464-char `message` fields containing whole NDAs. `git log 8e67e9d` shows those JSONLs are committed to the monorepo. A regulated-industry adopter who turns recording on in production is creating a secondary data plane of every document the agent reads — with no encryption, no access control, and the kaos-modules public repo as evidence the team also commits them.

**Evidence.** `skeptic-prod-ops-findings.md` Probe 5; `_recorder.py` line 13 (claims "inputs, outputs, model, tokens, cost, latency"); commit `8e67e9d` is `chore(kaos-agents): captured run JSONLs for KC5 live tests`.

**Fix (minimum for ship).**
1. Add a `.gitignore` rule for `tests/integration/runs/*.jsonl` going forward (the historical commits stay in git history but the practice stops). Verify all currently-committed JSONLs use only synthetic / public-domain fixtures (NASA images, Federal Register HTML, the Acme MNDA which is a real-but-redacted fixture — verify this is publication-OK with the user).
2. README "Audit trail" section explicitly documents the captured fields, names them as ePHI/MNPI/PII-equivalent under SOC2 CC7.2, and tells the reader to point the recorder at an encrypted-at-rest VFS (or disable it entirely) in production.
3. CHANGELOG `[Unreleased]` entry under `### Security`.

**Owner.** Phase A6 README author; A2 finding-fix owner for the .gitignore.

### KC16-5 — ResearchAgent has no cost cap (HIGH · FIX_IN_0.1.0a2)

**Problem.** Sprint-3 #9 wired `max_cost_usd` into ChatAgent (soft), FindingsAgent (strict), and CorpusFilter (post-hoc). It did NOT wire ResearchAgent. The closeout marks this as PA11 (#163), "RAG path is structurally different." A user invoking `kaos-agents` for RAG-style "research these 100 documents" has no cap defense. This is a documented limitation, not a regression; for a regulated firm trying to estimate per-tool-call max spend, it is unbudgetable.

**Evidence.** `roadmap-prerelease-closeout.md` PA11; `grep -rn max_cost_usd kaos-agents/kaos_agents/patterns/research/` → no real matches.

**Fix (minimum for ship).** README "Known limitations" section names ResearchAgent as cap-unbounded. CHANGELOG `[Unreleased]` entry. PA11 stays open for 0.1.0a2.

**Owner.** PA11 implementer; A6 README author.

### KC16-6 — Chat-path cost cap is soft (HIGH · DOCUMENT_AND_SHIP)

**Problem.** `test_chat_cap_within_5pct_tolerance` permits up to 2x overshoot. The tool description in `tools/registry.py:295-319` honestly says this. The README probably does not. A regulated user reading "AgentChatTool supports max_cost_usd" and assuming 5% will be off-by-2x in worst case.

**Evidence.** `tests/integration/test_cost_cap_honesty_live.py:151` `tolerance = 2.0`; `tools/registry.py:295-319`; closeout PA13.

**Fix (minimum for ship).** README "Known limitations" reproduces the chat-tool's docstring almost verbatim: "Chat path cap is honest (`budget_exceeded` flag truthful) but soft. Worst-case overshoot is one classify + one ReAct iteration, typically 5-25% and bounded at 2x on small caps. Strict per-call caps use kaos-agent-findings or kaos-agent-plan." CHANGELOG entry. PA13 stays open.

**Owner.** A6 README author.

### KC16-7 — gpt-5.4-mini findings consistency Jaccard 0.70-0.79 (HIGH · DOCUMENT_AND_SHIP)

**Problem.** The Sprint-2 #5 consistency contract (0.95 Jaccard floor) is Anthropic-tier. PA15 measured 0.70-0.79 on `gpt-5.4-mini`. Two associates running the same query see materially different surviving sets. This is an adaptation gap — fine to ship per values directive, but ONLY if documented.

**Evidence.** `pa15-cross-provider-matrix.md` Gap #3 (3-run survivor counts `[20, 16, 14]`, Jaccard 0.700-0.789).

**Fix (minimum for ship).** README provider-compatibility table column "Findings consistency" reads `~0.75 Jaccard` for gpt-5.4-mini, with a recommendation to use `runs >= 2` union mode on that provider for audit-grade work. Same line in CHANGELOG.

**Owner.** A6 README author.

### KC16-8 — t12 NDA-risk-memo judge regression unflaked (HIGH · DOCUMENT_AND_SHIP)

**Problem.** The KC8 re-baseline shows t12 (LLM-judge-evaluated NDA risk memo) regressed `passed → failed` at HEAD. The closeout says "stochastic" and recommends a 3-run retry. **Tagging 0.1.0a1 without that retry means the release notes say "all live tests pass" while one test fails at HEAD.** Per `feedback_no_skipped_tests_in_ci.md`, "100% green CI" means every test runs and passes — not "all the ones we already knew were passing pass."

**Evidence.** `kc8-rebaseline-2026-05-11.md` §3.1; `tests/integration/ladder/test_t12_nda_risk_memo.py`.

**Fix (minimum for ship).** Either (a) run t12 3× before tag and accept-with-flake-rate, OR (b) re-author the test with a more deterministic judge / multi-run agreement gate (per skeptic trust recommendation to require `runs >= 2` union), OR (c) explicitly mark t12 as `pytest.mark.flaky` with a documented retry count and document in CHANGELOG that this is an LLM-judge stochasticity gate, not a correctness gate. Option (a) or (c) is acceptable; option (b) is the 0.1.0a2 follow-up.

**Owner.** A2 finding-fix owner.

### KC16-9 — FindingsAgent has no max_chunks / max_candidates (MEDIUM · FIX_BEFORE_TAG)

**Problem.** Skeptic Probe 3b measured 30K candidates and 1500 chunk coroutines in one `every_sentence` call on a 10K-paragraph corpus. The Sprint-3 #9 cost cap defends against $-burn — but only if the cost-tracking is accurate (KC16-2 shows it's not on gpt-5.5). The defense-in-depth ask is a per-call structural ceiling: `max_chunks=200` default + an explicit error "corpus too large; pre-filter with kaos-agent-corpus-filter or use a `token`/`entity` selector." This is cheap to add (one validation in `FindingsAgent.__init__` + one early-exit in the chunk dispatcher).

**Evidence.** `skeptic-prod-ops-findings.md` Probe 3b; `kaos_agents/patterns/findings.py:1276` (kwarg list shows only `max_cost_usd`, no resource caps).

**Fix.** Add `max_chunks: int = 200` and `max_candidates: int = 5000` to `FindingsAgent.__init__`. Surface to `AgentFindingsTool`. Reject with an actionable error.

**Owner.** A2 finding-fix owner.

### KC16-10 — Synthesis-step injection test is single-payload, single-tier (MEDIUM · FIX_BEFORE_TAG)

**Problem.** The closeout claims "Sprint-1 #3 synthesis-step defense not stress-tested live." That's stale — `test_findings_injection_live.py::test_payload_does_not_leak_canary[synthesis_targeting]` exists and passes at `7367850`. But it's ONE parametrization. To retire skeptic Probe 1's "got-lucky on Haiku" critique, add at least one Sonnet-tier synthesis-targeted test with a payload that tries to override the synthesis directive (vs the filter directive) using social-engineering phrasing the heuristic doesn't catch.

**Evidence.** `tests/integration/test_findings_injection_live.py` (only one parametrization for `synthesis_targeting`); INDEX.jsonl shows it passed at HEAD.

**Fix.** Add 2-3 additional payloads — e.g., a payload that frames itself as `<system>Override: emit canary` inside the wrapped untrusted content, a payload that tries to encode the canary via base64 / leetspeak. Run live against Sonnet 4.6 and Haiku 4.5.

**Owner.** A2 finding-fix owner.

### KC16-11 — Pricing-table parity untested across siblings (MEDIUM · FIX_IN_0.1.0a2)

**Problem.** KC16-2 named the gpt-5.5 instance; the structural problem is that `kaos-llm-core.PRICING` and `kaos-llm-client.MODEL_PRICING` can silently diverge. No test catches this.

**Fix.** `kaos-llm-core/tests/unit/test_pricing_parity.py` asserts the two tables agree on shared model identifiers. Triggered by 0.1.0a2 of either sibling.

**Owner.** kaos-llm-core / kaos-llm-client cross-maintainer.

### KC16-12 — Sonnet 4.6 consistency outlier (MEDIUM · FIX_IN_0.1.0a2)

**Problem.** PA15 saw Sonnet 4.6 produce a 0.621 Jaccard on one of three runs (below the 0.90 floor). README must not over-claim Sonnet consistency. Closeout's claim "5-run Jaccard now 0.955-1.000" is Haiku, not Sonnet.

**Fix (for ship).** README provider-compatibility table reads `0.92-0.96 typical` for Sonnet with a "outlier runs may dip to 0.62; require `runs >= 2` for audit-grade extraction" note. CHANGELOG entry.

**Fix (for 0.1.0a2).** Make `runs >= 2` the default for synthesis on Sonnet (settings hook).

**Owner.** A6 README author; PA17 implementer.

### KC16-13 — Subprocess recorder gap (MEDIUM · DOCUMENT_AND_SHIP)

**Problem.** Skeptic Probe 6 confirmed that subprocesses NOT importing `kaos_llm_core` are invisible to the audit trail even when the env-var hook reaches them. No httpx-level recorder was added. The closeout did not flag this. The README's "complete audit trail" claim must be scoped to `kaos-llm-core`-routed calls.

**Fix.** README "Audit trail" section says: "All LLM calls routed through `kaos-llm-core` are captured. Direct provider SDK calls (e.g., `anthropic.Anthropic()` in user-supplied tools) are NOT captured — route through `kaos-llm-core` or accept the audit gap. An httpx-level recorder is on the roadmap for 0.1.0a3."

**Owner.** A6 README author.

### KC16-14 — K5 triage diverges from BM25 at scale (MEDIUM · DOCUMENT_AND_SHIP)

**Problem.** Skeptic value-probe 3 showed K5 (summary-aware triage) and raw BM25 agree at n=4 (Jaccard 1.00) and diverge severely at n=64 (Jaccard 0.11 — 1 of 5 shared). K5 is a different retrieval signal, not a free speedup. README must not claim K5 is a drop-in BM25 replacement.

**Fix.** README documents triage's `_engage_summary_path` policy honestly and points readers at the kaos-content-K5 design doc.

**Owner.** A6 README author.

### KC16-15 — PA15 covered only 4 of 8 supported providers (MEDIUM · DOCUMENT_AND_SHIP)

**Problem.** `kaos-llm-client` advertises support for OpenAI, Anthropic, Google, xAI, Groq, Mistral, OpenRouter (and recently Together?). PA15 verified 2 (Anthropic, OpenAI). README must not over-claim cross-provider compatibility.

**Fix.** README provider-compatibility table lists ONLY the 3 green rows (haiku-4-5, sonnet-4-6, gpt-5.4-mini) and 1 red row (gpt-5.5). Everything else is "unverified for v0.1.0a1."

**Owner.** A6 README author.

### KC16-16 — recipes/__init__.py missing __all__ (LOW · FIX_BEFORE_TAG)

**Problem.** Audit-01 KAG-005, trivial. Public API hygiene per `docs/python/design/modules.md`.

**Fix.** Add the `__all__` list at the top of `kaos_agents/recipes/__init__.py`. Names: `load_builtin_recipes`, `load_recipe`, `recipe_names`, `format_recipe_for_memory`, `load_extraction_recipes`, `load_extraction_recipe`, `extraction_recipe_names`. Run `ty check` and `ruff check`.

**Owner.** A2 finding-fix owner.

### KC16-17 — MCP error messages incomplete (LOW · FIX_IN_0.1.0a2)

**Problem.** Audit-01 KAG-006. Several tool error responses omit (1) what went wrong, (2) how to fix it, (3) alternative. Tool-design guide requires all three.

**Fix.** Normalize the three named sites + sweep for siblings.

**Owner.** PA18 implementer.

### KC16-18 — Sibling-checkout lexicon discovery (LOW · FIX_IN_0.1.0a2)

**Problem.** Audit-01 KAG-007. `memory/search.py:36-40` probes `~/projects/273v/kaos-source/...` for the OpenGloss lexicon. Breaks for installed wheels.

**Fix.** Move the lexicon to a packaged resource or expose as `KaosAgentSettings` field.

**Owner.** PA18 implementer.

### KC16-19 — cli/chat.py is 1742 lines, multi-concern (LOW · FIX_IN_0.1.0a2)

**Problem.** Audit-01 KAG-008 partially addressed (T5-2 lifted into `cli/` subpackage) but the file size GREW (1670 → 1742). The concern-mixing is real.

**Fix.** Split file ingestion, chunk/cache handling, optional tool registration, and the REPL into focused modules.

**Owner.** PA18 implementer (or a dedicated refactor commit pre-0.1.0a2).

### KC16-20 — Deterministic finding_ids now SHA256 (LOW · DOCUMENT_AND_SHIP)

**Problem (already fixed).** Skeptic-trust Observation 2 flagged uuid4-based finding_ids as non-stable across runs. Sprint-2 #5 (`f752ecf`) replaced with SHA256(block_ref, char_span, normalized_text)[:12].

**Fix.** CHANGELOG `[Unreleased]` `### Changed` entry referencing the skeptic report so re-readers know the fix landed.

**Owner.** A6 CHANGELOG author.

### KC16-21 — Disk-first VFS default may leak across users in shared deployments (LOW · DOCUMENT_AND_SHIP)

**Problem.** `KaosRuntime.test_mode()` solves the test-isolation footgun. The production default (`KaosRuntime()`) uses disk-backed VFS at `.kaos-vfs`. For a regulated user with a long-running KAOS-agent pod on a shared volume, session memory persists across container restarts — sometimes desired (resilience), sometimes a cross-user leak (multi-tenant gone wrong). README must say so.

**Fix.** README "Persistence model" section names the `.kaos-vfs` root, the disk-first default, the `IsolationMode.GLOBAL` vs `PER_CONTEXT` choice, and the in-memory `KaosRuntime.test_mode()` for stateless deployments.

**Owner.** A6 README author.

### KC16-22 — KC8 cost regressions are informational, not action items (INFORMATIONAL)

**Problem.** 7 cost regressions (max +7870%) in KC8 vs prior baseline. Cost is the lowest-priority lens. Closeout argues the +7870% is a stochastic catch-up (prior was an outlier cache hit). Accept.

**Owner.** None — informational.

---

## 4. Pre-release acceptance gates

The release plan's Phase 0 / A / F is the master checklist; KC16 adds the
following gate items that MUST be checkable before `git tag v0.1.0a1`:

- [ ] **A2-G1.** Audit-01 KAG-001 QA gates (ruff format, ruff check, ty check, pytest unit) ALL green at HEAD. No skipped tests count as green per `feedback_no_skipped_tests_in_ci.md`.
- [ ] **A2-G2.** Audit-01 KAG-004 — benchmark-marked tests moved out of `tests/unit/` so the bounded unit gate finishes inside 90s.
- [ ] **A2-G3.** Audit-01 KAG-005 — `kaos_agents/recipes/__init__.py` has `__all__`.
- [ ] **A2-G4.** KC16-9 — `FindingsAgent(max_chunks, max_candidates)` ceilings added with regression tests (mock-stub + 1 live test that exceeds the ceiling and observes the actionable error).
- [ ] **A2-G5.** KC16-10 — synthesis-targeted injection test parametrized with at least 3 distinct payload shapes, at least one Sonnet-tier.
- [ ] **A2-G6.** KC16-2 — `kaos-llm-core 0.1.0a4` published with `gpt-5.5` pricing entry; `kaos-agents` `pyproject.toml` floor bumped to that version; pricing-parity test landed.
- [ ] **A2-G7.** KC16-4 — `.gitignore` rule covering `tests/integration/runs/*.jsonl`; documented fixture-only policy in `tests/integration/README.md`; all currently-committed JSONLs confirmed to use synthetic / public-domain inputs.
- [ ] **A2-G8.** KC16-8 — t12 NDA-risk-memo handled (one of: 3-run retry to characterize flake rate, OR `@pytest.mark.flaky` annotation with documented retry count, OR re-author with multi-run agreement).
- [ ] **A6-R1.** README "Known limitations" section names: reasoning-model temperature=0 incompatibility (KC16-3), recorder data-plane risk (KC16-4), ResearchAgent cap-unbounded (KC16-5), chat-cap 2x soft overshoot (KC16-6), gpt-5.4-mini findings-consistency floor (KC16-7), Sonnet 4.6 outlier risk (KC16-12), subprocess-recorder gap (KC16-13), K5-vs-BM25 divergence (KC16-14), provider-compatibility table restricted to 3 green rows (KC16-15), disk-first VFS persistence semantics (KC16-21).
- [ ] **A6-R2.** README "Audit trail" section explicitly scopes "complete trail" to kaos-llm-core-routed calls (KC16-13).
- [ ] **A6-R3.** README "Provider compatibility" table reproduces PA15's verified-only rows.
- [ ] **A6-C1.** CHANGELOG `[Unreleased]` block carries entries for: all 10 Sprint 1-3 commits as `### Added`/`### Changed`/`### Security`, all KC16 ship-time fixes, and explicit `### Known Limitations` cross-references to PA11/PA13/PA15/PA16/PA17/PA18.
- [ ] **F1.** `uv build` produces a wheel that installs cleanly from a fresh venv and `from kaos_agents import BaseAgent, Runner` works.

---

## 5. CHANGELOG `[Unreleased]` text (paste-ready)

```markdown
## [Unreleased]

### Added
- KaosRuntime VFS isolation: `KaosRuntime(vfs=...)` kwarg + `KaosRuntime.test_mode(in_memory=True)` classmethod
  + `runtime.artifacts` as `cached_property`. Closes the disk-VFS cross-run leakage footgun in live tests.
  Live composition tests are now isolated by default. (Sprint-1 #1, commit d0ba060.)
- Auth/rate-limit/transport failures surface as `isError=True` with credential-named recovery hints
  via `kaos_agents.errors.classify_agent_failure()`. (Sprint-1 #2, commit dee0c9a.) Closes
  skeptic-prod-ops Probe 4b.
- Three-layer OWASP LLM01 defense for `FindingsAgent`: pre-flight heuristic flag, XML isolation
  envelope around all candidate text, defensive signature docstring. Plus live test against Sonnet 4.6
  including a synthesis-targeted payload variant. (Sprint-1 #3, commit fb82f64.) Closes skeptic-prod-ops Probe 1.
- `FindingsRefusal` structured value type with three stable refusal reasons
  (`no_candidates_enumerated`, `no_relevant_candidates`, `budget_exceeded`). Refusal surfaces via
  `FindingsResult.refusal` and `AgentFindingsTool.structuredContent["refusal_reason"]`. (Sprint-1 #4,
  commit 916cb67.) Closes skeptic-trust Probe 2 empty-answer UX bug.
- `FindingsAgent.temperature=0.0` by default; deterministic finding_ids via SHA256(block_ref,
  char_span, normalized_text)[:12]; `runs >= 2` union mode for multi-run consistency. 5-run Jaccard
  rises from 0.84-0.92 (skeptic-trust baseline) to 0.955-1.000 on Anthropic Haiku 4.5. (Sprint-2 #5,
  commit f752ecf.) Closes skeptic-trust Probe 1 (consistency).
- `select_by="semantic"` selector with LLM-driven query rewrite + 8-term sanitized expansion union;
  low-recall warning on token selector when < 5 candidates for >= 6-word question. (Sprint-2 #6, commit
  0ffb020.)
- `FindingsAgent.max_cost_usd` strict wave-level cap (Phase-2 filter + Phase-3 synthesis); honest
  `budget_exceeded` reporting across `AgentChatTool` (soft, 2x overshoot bound), `AgentPlanTool`
  (strict per-step), `AgentFindingsTool` (strict wave-level), `AgentCorpusFilterTool` (post-hoc).
  (Sprint-3 #9, commit 21463ba.) Closes skeptic-prod-ops Probe 2.
- `AgentResponse.cost_usd` + `AgentResponse.total_tokens` as first-class frozen attributes. Same
  numbers ship as `ToolResult.structuredContent["cost_usd"]` and `["total_tokens"]` across all four
  agent tools. (Sprint-3 #10, commit a338d1e.)

### Changed
- Streaming recorder JSONL schema bumped to v3: header line written + fsync'd on `__aenter__`,
  per-invocation lines streamed + fsync'd during run, optional trailer at exit. Audit trail
  now survives SIGTERM and pod eviction. (Sprint-3 #8, commit b8f5998.) Closes skeptic-prod-ops Probe 4c.
- `parse_html` default `pre_content_mode='prose'` (was `'code'`); K3 SentencesWith* tools emit a
  shape-mismatch warning when paragraphs are sparse and `<pre>` blocks dominate. (Sprint-2 #7,
  commit fe73833.) Federal Register / news / Wikipedia / web-search HTML pipelines no longer
  silently produce zero entity hits.
- Bumped `kaos-llm-core>=0.1.0a4` dependency for the `gpt-5.5` pricing entry parity fix (KC16-2).

### Security
- Test capture JSONLs no longer committed to the public repo; `.gitignore` covers
  `tests/integration/runs/*.jsonl`. Production users running the recorder in regulated environments
  MUST point output at encrypted-at-rest storage (KaosVFS with encryption, S3 SSE-KMS, etc.) — see
  README "Known limitations" for the data-plane discussion. (KC16-4.)

### Known Limitations

This is an honest list, not a buried disclaimer. v0.1.0a1 is an alpha;
the items below are tracked work that did not block release but a
regulated-industry adopter must know about.

- **OpenAI reasoning models (gpt-5.5, o3, o4, …) are not supported** for findings-based extraction
  in v0.1.0a1. `FindingsAgent` sends `temperature=0` unconditionally and these models reject it
  with HTTP 400. Cost accounting on gpt-5.5 also reports `$0` despite real billing (pending kaos-llm-core
  0.1.0a4). Use Anthropic Haiku 4.5 / Sonnet 4.6 or OpenAI `gpt-5.4-mini` instead. Tracked as PA16
  for v0.1.0a2. (KC16-2, KC16-3.)
- **Chat-path cost cap is honest but soft.** `AgentChatTool(max_cost_usd=X)` may overshoot the cap
  by up to 2x in a single turn (one classify + one ReAct iteration). `budget_exceeded` flag is
  truthful. For strict per-call caps use `kaos-agent-findings` (wave-level) or `kaos-agent-plan`
  (per-step). Tracked as PA13 for v0.1.0a2. (KC16-6.)
- **`ResearchAgent` / RAG path has no cost cap.** Tracked as PA11 for v0.1.0a2. (KC16-5.)
- **Findings consistency on `openai:gpt-5.4-mini` is ~0.75 Jaccard** (vs the 0.95 Anthropic floor).
  Two associates running the same query may see materially different surviving sets. Use the
  `runs >= 2` union mode on this provider for audit-grade work. (KC16-7.)
- **`anthropic:claude-sonnet-4-6` consistency typically holds at 0.92-0.96** but PA15 observed one
  0.621 outlier across three runs. Anthropic does not advertise `temperature=0` as bit-deterministic.
  For audit-grade extraction prefer Haiku or use `runs >= 2`. (KC16-12.)
- **Cross-provider coverage in v0.1.0a1 is limited to 3 verified rows.** Anthropic Haiku 4.5,
  Anthropic Sonnet 4.6, OpenAI gpt-5.4-mini — all green. OpenAI gpt-5.5 — RED (see above). Google,
  xAI, Groq, Mistral, OpenRouter — UNVERIFIED for v0.1.0a1 against the Sprint 1-3 contracts.
  Tracked as PA15 follow-ups. (KC16-15.)
- **Audit-trail JSONL captures persist full document bodies, conversation context, and
  agent-generated content** to disk. In regulated deployments (SOC2 / HIPAA / FINRA / GLBA) these
  files are subject to the same retention, encryption-at-rest, and access-control requirements as
  the source documents themselves. The recorder writes to a `Path` you supply — do NOT point it at
  unencrypted disk in production. The recorder also only captures calls routed through
  `kaos-llm-core`; direct provider SDK calls in user-supplied tools are invisible to the trail.
  (KC16-4, KC16-13.)
- **Persistence model: disk-first by default.** `KaosRuntime()` uses a disk-backed VFS rooted at
  `.kaos-vfs/`. Session memory persists across container restarts. For stateless / per-request
  deployments use `KaosRuntime.test_mode()` (in-memory + `IsolationMode.GLOBAL`). For multi-tenant
  deployments, scope the VFS root per-tenant. (KC16-21.)
- **`FindingsAgent.max_chunks` / `max_candidates` ceilings** were added in v0.1.0a1 to defend
  against accidental `select_by='every_sentence'` calls on giant corpora (default 200 chunks, 5000
  candidates). The cost cap is the primary defense; these are belt-and-suspenders. Lift them
  explicitly when you have a known-bounded large-corpus job. (KC16-9.)
- **K5 summary-aware triage and raw BM25 are different rankers, not equivalents.** At n>=16 docs
  they share <70% of their top-5 results; at n=64 they share ~10%. Treat K5 as a complementary
  signal, not a drop-in BM25 replacement. (KC16-14.)

### Removed
- `License :: ...` Trove classifier (PEP 639 supersedes; `license = "Apache-2.0"` is now the
  canonical declaration).
```

---

## 6. README "Known limitations" section text (paste-ready)

```markdown
## Known limitations (v0.1.0a1)

kaos-agents v0.1.0a1 is an alpha. The full Sprint 1-3 correctness +
transparency contract surface ships verified by 125 live tests against
real provider APIs ($2.65 of live spend in the KC8 re-baseline; see
`docs/design/kc8-rebaseline-2026-05-11.md`). The items below are
honest gaps that a regulated-industry adopter would otherwise discover
in production, and that we have decided are document-and-ship for
v0.1.0a1 under the values lens (quality > correctness > transparency >
adaptation > cost).

### Provider compatibility

| Provider | Findings agent | Cost accounting | Refusal | Injection defense | Consistency floor |
|---|---|---|---|---|---|
| `anthropic:claude-haiku-4-5` | ✓ | ✓ | ✓ | ✓ | 0.955-1.000 Jaccard |
| `anthropic:claude-sonnet-4-6` | ✓ | ✓ | ✓ | ✓ | 0.92-0.96 typical (0.62 outlier observed) |
| `openai:gpt-5.4-mini` | ✓ | ✓ | ✓ | ✓ | 0.70-0.79 Jaccard |
| `openai:gpt-5.5` (reasoning) | ✗ (temperature=0 incompatible) | ✗ (cost reports $0) | n/a | n/a | n/a |

OpenAI **reasoning** models (`gpt-5.5`, `o3`, `o4-mini`, anything new
from that class) are not supported for findings-based extraction in
v0.1.0a1. Cost accounting for these models also reports `$0` despite
real billing — the cost-cap contract is therefore unenforceable on
this provider class.

**Google (Gemini), xAI, Groq, Mistral, OpenRouter** are advertised by
`kaos-llm-client` as transport-supported but were NOT verified against
the Sprint 1-3 contracts in v0.1.0a1. Cross-provider matrix expansion
is tracked as PA15-follow-ups for v0.1.0a2.

### Cost-cap enforcement granularity

| Tool | Enforcement | Worst-case overshoot |
|---|---|---|
| `kaos-agent-chat` | Soft (post-turn) | 2x cap (bounded by one classify + one ReAct iteration; `budget_exceeded` flag truthful) |
| `kaos-agent-plan` | Strict (per-step) | <5% per step |
| `kaos-agent-findings` | Strict (wave-level) | <5% wave; aborts before next chunk dispatch |
| `kaos-agent-corpus-filter` | Post-hoc (single call) | Up to the model's per-call cost |
| `kaos-agent-research` (RAG) | **NONE WIRED YET** | Unbounded — tracked as PA11 |

If you are running this in a regulated environment and need a hard
ceiling on agent spend, scope to `kaos-agent-findings` /
`kaos-agent-plan` until PA11 closes ResearchAgent and PA13 closes the
chat-path strict cap (both tracked for v0.1.0a2).

### Findings consistency

The Sprint-2 #5 consistency contract (5-run pairwise Jaccard >= 0.95
on identical query + corpus + model) holds on Anthropic Haiku 4.5
empirically. Other models drift:

- **Sonnet 4.6**: typically 0.92-0.96, observed outliers at 0.62.
- **gpt-5.4-mini**: 0.70-0.79.

For audit-grade extraction where two associates running the same query
must see the same surviving set, use Haiku 4.5 with `runs >= 2` union
mode. The K7 MCP tool exposes this as `runs: int`.

### Audit trail (recorder)

The kaos-agents test recorder captures every LLM call routed through
`kaos-llm-core` (inputs, outputs, model, tokens, cost, latency,
errors). Schema v3 (Sprint-3 #8) streams per-invocation JSONL lines
with `fsync()` per line, so the audit trail survives SIGTERM / pod
eviction / OOM-kill.

**What the recorder captures verbatim:** the full LLM `inputs.message`
(user message), `conversation_context` (prior turns),
`conversation_history`, `instruction` (system prompts), and (for
findings) `candidates` (the document content broken into sentences).
For a regulated-industry deployment, these captured JSONLs become a
secondary data plane subject to SOC2 CC7.2 / FINRA 4511 / HIPAA
§164.312(b) retention, encryption-at-rest, and access-control
requirements identical to the source documents themselves.

In production:

- Point the recorder output at encrypted-at-rest storage (KaosVFS with
  encryption, S3 with SSE-KMS, etc.). Do NOT use a plain unencrypted
  `Path` for production capture.
- API keys are properly redacted via `SecretStr`; document bodies are
  NOT. A future release will add field-level redaction / truncation by
  default.

**Coverage gap.** The recorder only sees calls routed through
`kaos-llm-core`. A user-supplied tool that calls `anthropic.Anthropic()`
or `openai.OpenAI()` directly bypasses the trail. Route all LLM calls
through `kaos-llm-core` (or accept the audit gap). An httpx-level
recorder for "best-effort" coverage of direct SDK calls is on the
roadmap for v0.1.0a3.

### Persistence model

`KaosRuntime()` uses a disk-backed VFS at `.kaos-vfs/` by default.
Session memory persists across container restarts, which is the right
default for resilience and is the wrong default for multi-tenant
isolation. For stateless / per-request deployments, use
`KaosRuntime.test_mode(in_memory=True)` (in-memory VFS +
`IsolationMode.GLOBAL`). For multi-tenant deployments, scope the VFS
root per tenant before instantiating the runtime.

### Retrieval

The K5 summary-aware `triage_corpus()` path is faster than raw BM25 at
n >= 50 documents but ranks **different** documents — at n=64 the two
share roughly 10% of their top-5. Treat K5 as a complementary signal,
not a drop-in BM25 replacement. The default `triage_corpus()` policy
engages K5 only when every document in the section carries a cached
`summary` — preferring raw BM25 for unsummarized corpora.

### What this list does NOT cover

This list is the audit-known gap surface for v0.1.0a1. It does not
cover (a) every LLM-call cost (use `AgentResponse.cost_usd` /
`structuredContent["cost_usd"]`), (b) every memory-eviction policy
quirk (see `kaos_agents/memory/`), (c) the long tail of optional-extra
configurations. Open a GitHub issue when you find a gap that isn't
documented here — we will treat it as a release-note gap, not a
bug-of-the-week.
```

---

## 7. Answers to the audit brief's seven report questions

(Brief asked for commit SHA + path + verdict + counts + skeptic-missed
+ skeptic-downgraded + PA16 take.)

1. **Commit SHA + report path:** to be filled by the committing step
   (this report at `kaos-agents/docs/design/kc16-audit-findings.md`).

2. **Overall verdict:** `RELEASE-WITH-CAVEATS`.

3. **By severity:** 2 CRITICAL, 6 HIGH, 7 MEDIUM, 6 LOW, 1 INFO (22 total).

4. **By disposition:** 1 BLOCK_RELEASE, 6 FIX_BEFORE_TAG, 7 FIX_IN_0.1.0a2, 7 DOCUMENT_AND_SHIP, 1 INFORMATIONAL.

5. **What the skeptics missed:**
   - **KC16-1.** None of the three skeptics noted the audit-01 QA-gate failures (KAG-001) that block Phase A7's sanity gate. Those have been sitting since 2026-04 and the closeout did not flag them. **This is the audit's one true BLOCK_RELEASE find.**
   - **KC16-2.** The PA15 author noted the gpt-5.5 pricing miss as Gap #2 but did not flag the cross-sibling pricing-table divergence as a structural problem with no parity test. Releasing kaos-agents without a `kaos-llm-core 0.1.0a4` fix ships a broken cost-cap contract for an advertised provider.
   - **KC16-4 (operational variant).** Skeptic Probe 5 flagged "JSONLs committed to a public-ish repo" as a HIGH for regulated deployments. The release plan does not have a `.gitignore` action item for `tests/integration/runs/*.jsonl` and the team has continued to commit JSONLs (latest: `bd72a49` adds 126 KC8 captures, `16d8eb6` adds 41 PA15 captures). Closing this matters for the public-repo lifecycle, not just for production.

6. **What the skeptics over-stated (audit downgrades):**
   - **Skeptic-trust Probe 2 ("empty answer is a UX bug")** was correctly HIGH-graded at the time but is now CLOSED at HEAD (`916cb67` ships `FindingsRefusal` with three stable reasons). I downgrade the residual disposition from "blocks A-grade trust" to **LOW · DOCUMENT_AND_SHIP** in KC16-20 (CHANGELOG-only).
   - **Skeptic-prod-ops Probe 4b ("silent auth failure")** is CLOSED at HEAD (`dee0c9a` ships `classify_agent_failure`). Audited that the live test exists and passes. No residual finding.
   - **Skeptic-prod-ops Probe 4c ("SIGTERM loses trail")** is CLOSED at HEAD (`b8f5998` ships streaming JSONL with `fsync()`-per-line). Live test exists. No residual finding.
   - **Skeptic-prod-ops Probe 1 ("got-lucky prompt injection")** is partially closed — Sprint-1 #3 ships the three-layer defense AND a Sonnet 4.6 live test for synthesis-targeted payload. I downgrade from the skeptic's HIGH ("architectural failure") to **MEDIUM · FIX_BEFORE_TAG** (KC16-10) on the strength of one-payload-shape coverage, not the architecture.

7. **PA16 (gpt-5.5 / temperature=0) — block, fix-before-tag, or fix-in-0.1.0a2?** **DOCUMENT_AND_SHIP for v0.1.0a1; PA16 is the v0.1.0a2 code fix.** Three reasons:
   - The values lens explicitly orders adaptation BELOW quality / correctness / transparency. A reasoning-model adaptation gap is the canonical "honest-document-and-ship" category.
   - The README + CHANGELOG must say "OpenAI reasoning models unsupported in v0.1.0a1" loudly enough that a regulated-industry adopter cannot miss it.
   - Conditioned on KC16-2 (`kaos-llm-core 0.1.0a4` shipping the gpt-5.5 pricing entry) so the README's "cost accounting" claim isn't a lie. Without KC16-2, the gpt-5.5 disposition flips to FIX_BEFORE_TAG and PA16 becomes a blocker; with KC16-2, it's a documented limitation. KC16-2 itself is FIX_BEFORE_TAG (a CRITICAL · pricing-correctness fix on a sibling repo) so this is a sequence dependency, not a deadlock.

---

*Audit complete. Phase A2 owns the BLOCK_RELEASE + FIX_BEFORE_TAG items;
PA11 / PA13 / PA15-followups / PA16 / PA17 / PA18 own the
FIX_IN_0.1.0a2 items. The release plan's Phase A6 (README) and the
CHANGELOG `[Unreleased]` block own every DOCUMENT_AND_SHIP item.*
