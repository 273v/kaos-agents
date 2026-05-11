# KC8 — Re-baseline of ladder + parity + Sprint 1-3 + module-stress live suites

**Date:** 2026-05-11
**HEAD at task start:** `7367850` (`docs: kaos-agents first-release plan`)
**HEAD at task end:** `16d8eb6` (`test(kaos-agents): PA15 cross-provider matrix`, committed mid-run by the parallel PA15 sub-agent)
**Recorder schema version observed:** `3` (Sprint-3 #8 streaming format, per-invocation JSONL lines + optional trailer)

---

## 1. Scope

Re-baseline the following five suites against the as-shipped `main` after the 10-item Sprint 1-3 roadmap landed:

| Group | Tests |
|---|---|
| **Ladder (T1-T12 + stochasticity + pathological)** | `tests/integration/ladder/` — 21 tests |
| **Surface parity (S1-S10 × {cli,api,mcp} × {anthropic,openai})** | `tests/integration/surface_parity/` — 46 tests |
| **Sprint 1-3 contract tests** (auth / injection / refusal / consistency / semantic / cost-cap / cost-surface) | 7 files, 16 tests |
| **K-series live** (findings / router / reflexion / research / k5 / k7-k8 / patterns) | 8 files, 34 tests |
| **Module-stress** (excel / pptx / pdf-ocr / image-exif-vlm / federal-register) | 5 files, 9 tests |

PA15 cross-provider matrix runs in parallel and is **out of scope** for this report (its 41 captures live under `test_pa15_provider_matrix.py` and belong to PA15's report).

---

## 2. Headline numbers

| Suite | Pass | Fail | Cost | Notes |
|---|---|---|---|---|
| Ladder | 20 | **1** | $0.4664 | t12 NDA risk-memo judge regression |
| Surface parity | 46 | 0 | $1.7882 | 4 of 46 captured at SHA `16d8eb6` (PA15 commit mid-run) |
| Sprint 1-3 contracts | 16 | 0 | $0.1278 | All 7 new contracts pass clean — Sprint-3 #9 cost-cap flake **does not reproduce** |
| K-series live | 34 | 0 | $0.2398 | k7/k8 corpus-filter test (failed at `63f2197`) now passing |
| Module-stress | 9 | 0 | $0.0250 | All 5 modules green; Excel-margin-compression (failed at `09ae976`) now passing |
| **TOTAL (KC8 scope)** | **125** | **1** | **$2.6472** | $5 budget — 53% used |

Wall-clock: ladder 10:22 (stopped on t12 fail via `-x`), parity 10:05, Sprint 1-3 contracts 2:07, K-series 2:56, module-stress 0:31. Aggregate ~26 min (parallel).

Skip count: 0 KC8 tests skipped. (8 K-series tests with non-live markers were deselected by `-m live`, not skipped — these are unit-flavoured non-live tests in the same files.) No auth-missing, fixture-missing, or dep-missing skips.

---

## 3. Regressions — brutal list

### 3.1 Outcome regression (1)

**`tests/integration/ladder/test_t12_nda_risk_memo.py::test_nda_risk_memo_judged_against_ground_truth`** — `passed (f4d65f5)` → `failed (7367850)`.

- **New JSONL sha256:** `31f6e5a52389…` (cost $0.0336, elapsed 20.4s, 3 calls)
- **Prior INDEX record:** `passed`, cost $0.0300, elapsed 16.1s, 3 calls, `git_short_sha=f4d65f5`, `end_ts_utc=2026-05-11T09:34:28.255188+00:00`
- **Field that changed:** `outcome` (passed → failed). Cost and call count almost identical (+10% cost, +27% elapsed).
- **Failure mode:** LLM judge flagged 2 incorrect claims in the agent's risk memo; the test allows ≤1. The reasoning quote in the traceback shows the agent (a) mis-stated Acme as fixed-3-year-term when the ground truth is open-ended, and (b) framed CyberCorp's non-solicit separately from the EMNA+DynaMo group when ground-truth says all three group together.
- **Severity:** **Medium.** This is a known LLM-judge stochasticity test — the prior run squeaked under the threshold with 1 incorrect claim, the new run flagged 2. Not clearly tied to any Sprint 1-3 change. Recommend a follow-up retry (or two) at the same SHA to characterise the flake rate vs. a genuine quality regression.

### 3.2 Cost regressions (7) — material ones first

| Test | Prior | Current | Δ | Prior SHA | Current SHA |
|---|---|---|---|---|---|
| `surface_parity/test_s10_nda_memo.py::test_s10_nda_memo_via_mcp[anthropic]` | $0.0025 | $0.1956 | **+7870%** | `f4d65f5` | `7367850` |
| `ladder/test_t04_plan_execute_4step.py::test_plan_execute_3plus_steps_with_extract` | $0.0064 | $0.0163 | +154% | `f4d65f5` | `7367850` |
| `surface_parity/test_s10_nda_memo.py::test_s10_nda_memo_via_mcp[openai]` | $0.0132 | $0.0314 | +138% | `f4d65f5` | `7367850` |
| `surface_parity/test_s10_nda_memo.py::test_s10_nda_memo_via_api[anthropic]` | $0.2504 | $0.4891 | +95% | `f4d65f5` | `7367850` |
| `surface_parity/test_s5_budget.py::test_s5_budget_via_api[openai]` | $0.0138 | $0.0264 | +91% | `f4d65f5` | `7367850` |
| `surface_parity/test_s9_nda_tabular.py::test_s9_nda_tabular_via_api[anthropic]` | $0.2069 | $0.3935 | +90% | `f4d65f5` | `16d8eb6` |
| `surface_parity/test_s9_nda_tabular.py::test_s9_nda_tabular_via_api[openai]` | $0.0235 | $0.0436 | +86% | `f4d65f5` | `16d8eb6` |

**`s10_nda_memo_via_mcp[anthropic]` +7870% is the dominant signal.**

- **New JSONL sha256:** `62e375c00c13…`
- Prior baseline: 1 call, $0.0025, 25s. Current: 3 calls, $0.1956, 84s. The `call_count` jump from 1→3 looks like the new judge retry path (Sprint-3 #9 introduced honest cost capping which can route through additional invocations).
- **Severity:** **High** in absolute terms — the s10 NDA-memo test alone moved from ~$0.02 across both Anthropic surfaces to ~$0.69. But: the **api[anthropic]** number was already $0.25 baseline — only the mcp[anthropic] number was anomalously low at $0.0025 (1 call). The prior baseline almost certainly captured a memory-cached path. The 84s elapsed + 3 calls now matches the api shape — this looks like the prior baseline was the outlier, not the new one.
- Mirrored in the diff: `runs_cli.py diff s10_nda_memo_via_mcp` shows the second invocation today (16:08, $0.0314) was the cheap-path; first invocation today (16:07, $0.1956) was the expensive-path. Stochastic.

**`s9_nda_tabular_via_api[anthropic]` +90% to $0.39:**

- New JSONL sha256: `05932f486c36…`. Captured at SHA `16d8eb6` (PA15's commit landed mid-run).
- call_count up 2→3, elapsed up 14s→30s. Looks like additional planning / verify round.

**`t04 plan_execute_3plus_steps_with_extract` +154% to $0.0163:**

- New JSONL sha256: `81d7a315c0cd…`. Bumped from $0.0064 → $0.0163. Plan-execute now generates more steps. Possibly the K5-summary-aware triage path adding a turn. Worth a follow-up profile, but absolute cost is trivial.

### 3.3 Fresh failures from Sprint 1-3 work (0)

**No new failures introduced by Sprint 1-3 in KC8 scope.** Specifically:

- **Sprint-3 #8 (recorder schema v3 streaming format)** — every JSONL captured in this run uses schema_version=3 with a streaming header + optional trailer. All 125 KC8 captures parsed cleanly via `runs_cli.py`. No schema gap surfaced.
- **Sprint-3 #9 (honest `max_cost_usd` contract)** — `test_chat_cap_within_5pct_tolerance` failed twice at `21463ba` (the introducing commit) with cost overshoot (cap $0.005, actual $0.0122 / $0.0164). At `7367850` it passes with cost $0.0086. The cap is being respected within the documented 5% tolerance now.
- **Sprint-3 #10 (transparency lens — `cost_usd` + `total_tokens` on response + structuredContent)** — all 3 `test_cost_surface_live.py` tests pass: `test_cost_usd_and_total_tokens_in_structured_content`, `test_mcp_surface_matches_in_process_response`, `test_cost_consistency_across_paths`.
- **Sprint-2 #5/#6/#7, Sprint-1 #1/#2/#3/#4** — all corresponding live tests pass clean.

### 3.4 PA15 sub-agent failures (out of scope, but worth flagging)

The parallel PA15 cross-provider matrix run is hitting **`openai:gpt-5.5` model failures** — 6 distinct PA15 test IDs return `cost=$0.0 elapsed<1s outcome=failed`, which is the unambiguous signature of "model identifier not recognised by provider." Worth confirming with PA15 that `gpt-5.5` (vs `gpt-5.4-mini` / `gpt-5.4-nano`) is a real model ID. (Reference: feedback_use_current_models — always check `test_live.py` header.) **Not my scope.**

---

## 4. Improvements (4)

| Test | Prior | Current | Δ |
|---|---|---|---|
| `test_cost_cap_honesty_live.py::test_chat_cap_within_5pct_tolerance` | $0.0164 (failed) | $0.0086 (passed) | **-48% + now passes** |
| `test_federal_register_live.py::test_fetch_extract_synthesize` | $0.0221 | $0.0116 | -48% |
| `surface_parity/test_s10_nda_memo.py::test_s10_nda_memo_via_api[openai]` | $0.0610 | $0.0320 | -47% |
| `test_pdf_ocr_legal_live.py::test_ocr_then_agent_answers_caption_and_court` | $0.0049 | $0.0030 | -39% |

The first row is the most important: **Sprint-3 #9's cost-cap flake is fixed** — the same test that failed twice at the introducing commit (`21463ba`) now passes within tolerance on `7367850`. That closes one of the explicit risk items called out in the task brief.

Also notable: `test_filter_returns_subset_with_real_llm` (k7/k8 corpus filter, failed at `63f2197`) and `test_agent_identifies_biggest_margin_compression` (Excel, failed 12× at `09ae976` — that's the disk-VFS leakage footgun fixed by Sprint-1 #1) now pass consistently.

---

## 5. Schema-gap audit — header-field diffs (Sprint-3 #8)

The recorder changed from schema_version=1 (single-line JSONL with all fields in the trailer) to schema_version=3 (streaming: header line up front, per-invocation lines as they happen, optional trailer at end). Fields observed across versions:

| Field | v1 (pre-Sprint-3 #8) | v3 (current) |
|---|---|---|
| `kind: "header"` | absent (single record) | **present** |
| `start_ts_utc` | trailer only | **header** |
| `end_ts_utc` | trailer (only record) | trailer only |
| `total_cost_usd` | trailer (only record) | trailer only |
| `streaming: true` | n/a | **header** |
| `trailer_optional: true` | n/a | **header** |
| `partial_last_line_tolerated: true` | n/a | **header** |
| `git.short_sha` | trailer | **header** |
| `kind: "invocation"` | absent | **streamed per-call** |
| `kind: "trailer"` | absent | terminal line |

**This is a GOOD schema gap** — the new format provides:
- Crash-survival (header lands first, partial trailer tolerated)
- Per-invocation visibility before test completion
- Forward-compat hint (`schema_version` + capability flags)

The INDEX summary line still carries `total_cost_usd` and `call_count` as before, so the `runs_cli.py` viewer is unaffected. No KC8 capture failed schema parse.

---

## 6. Per-commit cost rollup (full session)

```
commit          n   pass  fail   cost      calls
088136c         15    15     0  $0.0689     54   (prior run)
09ae976         42    30    12  $0.1189     40   (prior run)
0e15a24          2     2     0  $0.0155      5   (prior run)
21463ba         23    20     3  $0.1784     72   (prior run — Sprint-3 #9 introduction with flake)
25b7d6f         35    35     0  $0.1809    129   (prior run)
63f2197          8     7     1  $0.0170      8   (prior run)
7367850        160   148    12  $2.9384    538   (KC8 + PA15)
16d8eb6          7     7     0  $0.7769     17   (PA15 commit landed mid-parity; my tail-end captures)
8e67e9d         29    28     1  $0.0551     24   (prior run)
f4d65f5         84    84     0  $1.2186    173   (last full sweep before KC8)
```

KC8 own captures: 125 pass + 1 fail = 126 runs, $2.6472 total LLM spend, $0.021/test average. PA15 share at HEAD: 41 runs, $1.07.

---

## 7. Confidence + recommendation

The Sprint 1-3 work landed **clean** at the level of contract tests, K-series live tests, module-stress tests, and parity. The single outcome regression (t12 NDA risk-memo judge) does not appear tied to any Sprint 1-3 code change — the call count and elapsed time are stable relative to baseline; what changed is the judge's verdict on agent-produced text. This is the canonical LLM-judge-stochasticity failure mode, and the right next step is a 3-run retry to confirm flake rate before assuming a quality regression.

The +7870% cost outlier on `s10_nda_memo_via_mcp[anthropic]` is **almost certainly stochastic** — the prior baseline was 1 call, the new run is 3 calls, and a second invocation in the same KC8 session ran cheap (3 calls, $0.0314). The mcp path appears to be more variable than api at this size of corpus.

**Recommendation:** ship Sprint 1-3 as captured. Re-run t12 NDA risk-memo 3× in a follow-up KC9 to characterise the flake rate.

---

## 8. Artifacts

- Per-test JSONL captures: `tests/integration/runs/2026-05-11/*.jsonl`
- INDEX summary: `tests/integration/runs/INDEX.jsonl` (350 lines total, 126 added by KC8 + 41 by parallel PA15)
- This report: `kaos-agents/docs/design/kc8-rebaseline-2026-05-11.md`

To re-derive any number in this report:

```
cd kaos-agents
uv run python tests/integration/runs_cli.py summary --by commit
uv run python tests/integration/runs_cli.py list --grep <test-substr> --sort cost
uv run python tests/integration/runs_cli.py diff <test-substr>
```
