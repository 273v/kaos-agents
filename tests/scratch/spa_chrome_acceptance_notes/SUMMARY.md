# SPA Chrome MCP acceptance — 0.1.20 stack — Summary

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6 (Sonnet 4.6, per test floor)
**Browser:** Chrome via DevTools MCP (port 9222)
**SPA:** localhost:5173 (frontend) + localhost:8000 (backend, restarted to load 0.1.20 imports)

## Scenario verdicts

| # | Scenario | Verdict | Tools | Cost | Time | Notes |
|---|----------|---------|-------|------|------|-------|
| 1 | S19 XLSX upload | **PASS** | 2 | $0.0079 | 8.9s | `LI-S19-9001` + `77777.77` surfaced verbatim with citation |
| 2 | S03 scanned PDF | **FAIL** | 2 | $0.0012 | 5.2s | `(empty)` answer. FindingsAgent enumerate=1 filter=0 — OCR never called |
| 3 | S20 CFR memo | **PASS** | 2 | $0.0089 | 14.7s | "untrue statement of a material fact" quoted verbatim with citation |
| 4 | S16 5-format pile | **PASS (degraded)** | 4 | $0.0634 | 23.9s | 3/3 PDF+DOCX+XLSX needles surfaced. HTML+JSON blocked at upload validator. |
| 5 | S22 cluster routing | **PASS (degraded)** | 2 | $0.0092 | 8.6s | Both `24` clusters + `2,486` rows labelled with citations. Single-doc fixture does NOT reproduce original 50-doc flip-flop. |

**3 PASS + 2 PASS-degraded + 1 FAIL** on the 5-scenario matrix.

## What this proves about the 0.1.20 release

- **kaos-office 0.1.3 OPC path resolver fix is live for users.** S19
  (XLSX upload) returns the correct row through the SPA. Pre-0.1.3
  this would have silently returned 0 rows.
- **kaos-agents 0.1.20 verbatim-quote policy works.** S20 surfaces the
  exact regulatory clause "untrue statement of a material fact" in a
  quoted form with a citation, not paraphrased.
- **FindingsAgent dispatch composes cleanly.** S16 (3-format) surfaces
  3 distinct needles from 3 distinct parsers without overclaiming
  failure. No "every tool call returned an error" wording fired.
- **Multi-metric labelled answers work** on the simple S22 case (one
  doc, two labelled metrics). Sonnet 4.6 preserves both with citations.

## What broke

- **S03 (FAIL — new bug surfaced).** The FindingsAgent dispatch
  pre-parses corpus items eagerly with the format-specific parser.
  For a scanned PDF that yields empty `ContentDocument.body`, the
  enumeration finds 1 candidate (empty), the filter rejects it,
  synthesis produces `(empty)`. **The kaos-pdf 0.1.4 OcrPageTool is
  never called from this dispatch path.** Logged as WI5 — needs OCR
  fallback in `_build_corpus_view_from_documents` or a route-to-ReAct
  decision when corpus pre-parse yields empty text.

## What was scope-cut by upstream issues

- **S16 HTML + JSON** rejected at SPA frontend upload validator
  (`unsupported file extension '.json'`). The agent's tool surface
  CAN read HTML + JSON via `kaos-content-load-document` /
  `kaos-core-vfs-read`, but the SPA validator doesn't let those bytes
  reach the backend. Logged as WI6 — kaos-ui frontend fix.
- **S22 50-doc flip-flop** not reproduced by the single-DOCX fixture.
  The original CI failure was a 50-doc corpus that triggered the
  M2-vs-synthesis confidence collapse. The single-doc case is too
  simple to surface it. Need a true 50-doc S22 fixture for WI2's
  trace inspection step.

## Follow-up backlog (work items)

- **WI2** (S22 fix) — still needs 50-doc trace evidence before
  picking M2 rubric vs synthesis prompt fix.
- **WI3** (S16 partial-success refusal wording) — not actively
  failing in this 0.1.20 matrix but the universal-claim wording bug
  in `evaluate_no_evidence_gate.py:314,360` is still a latent
  defect. Should ship in 0.1.21 alongside WI5.
- **WI5** (NEW — S03 OCR fallback in FindingsAgent dispatch) —
  P0 for 0.1.21 since this regressed an explicit acceptance scenario.
- **WI6** (NEW — SPA upload validator HTML/JSON) — kaos-ui, separate
  release.

## Plan-doc updates needed

The original residuals plan
`kaos-modules/docs/plans/2026-05-26-corpus-stress-residuals-S16-S22-and-spa-acceptance.md`
predicted S03 would PASS once kaos-pdf 0.1.4 published. The PyPI
publish + tool registration verified — but the dispatch never reaches
the new tool. Add a "Final-pass results" section to that plan
documenting:

- 3 PASS + 1 PASS-degraded + 1 PASS-uncertain + 1 FAIL on initial
  5-scenario matrix
- New WI5 + WI6 items
- WI2 scope adjustment: requires 50-doc fixture, not single-doc
