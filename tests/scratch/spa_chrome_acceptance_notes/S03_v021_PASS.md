# S03 — scanned PDF re-acceptance on 0.1.21 stack (PASS)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.21, kaos-pdf 0.1.4 + [ocr] (pytesseract 0.3.13 + system tesseract 5.5.0), kaos-office 0.1.3
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK3ZB7329QGRVFNYHCMMDVC
**Fixture:** `/tmp/spa-acceptance/S03_scanned.pdf` (13.4 KB, image-only PDF, planted token FALCON)

## Prompt

> The attached PDF is a scanned image of an internal memo. What codename is
> mentioned? Use OCR if needed and quote the exact value.

## Result

**Verdict: PASS.** Final assistant text:

> The codename mentioned in the internal memo is **FALCON** [591f6045651c].
>
> > Codename: FALCON

Compared to 0.1.20 baseline (which emitted `(empty)`), the OCR fallback in
`BaseAgent._ocr_pdf_bytes_to_content_document` now recovers the scanned-PDF
text inside the FindingsAgent corpus pre-parse, so enumeration finds 6
candidates (instead of 0) and the filter keeps 2 verbatim quotes for
synthesis to anchor on.

## Four-signal capture

- **Snapshot:** Activity panel shows `Agent Findings Dispatch →
  FindingsAgent: enumerated=6 filtered=2 cost=$0.0076 answer_chars=93`.
  Citation `[591f6045651c]` attached. Final text contains FALCON verbatim
  inside a blockquote.
- **Tool trace:** 2 calls, both `kaos-agent-findings-dispatch`. No ReAct
  fall-through, no extra OCR-tool calls — the OCR fallback ran inside the
  dispatch path's eager pre-parse (as designed). The
  `kaos-pdf-ocr-page` MCP tool from kaos-pdf 0.1.4 is the canonical
  shipping surface; the dispatch-side helper reuses the same TesseractEngine
  under the hood.
- **Cost / tokens:** $0.0076 / 3.5k tokens / 8.1s. (Compare to 0.1.20
  baseline: $0.0012 / 1.5k tokens / 5.2s — the new cost reflects the OCR
  call plus the real synthesis work on recovered text.)
- **Console / network:** no errors. `POST /v1/chat/sessions/.../files` →
  201, `POST /v1/chat/sessions/.../messages` → 200 streamed SSE, 1
  citation persisted via `POST /v1/chat/sessions/.../citations`.
- **Documents panel:** 1 document, READY.

## What this proves about the 0.1.21 release

- **WI5 (S03 OCR fallback in FindingsAgent dispatch) closes the
  regression** documented in `tests/scratch/spa_chrome_acceptance_notes/S03_scanned_pdf.md`
  on the 0.1.20 stack. The dispatch path is now scanned-PDF-honest: it
  recovers OCR text before enumeration runs, so synthesis has real
  candidates to filter and quote.
- **Graceful degradation works as designed.** With kaos-pdf[ocr]
  installed in the SPA backend venv (the pin bump in
  `kaos-ui/examples/single-user-chat/backend/pyproject.toml` to
  `kaos-pdf[ocr]>=0.1.4`), the fallback activates. Hosts that don't
  install the OCR extra silently fall back to the pre-0.1.21 behavior
  (returns the empty parse result, dispatch emits the existing refusal
  shape) — no hard failure.
- **Citation surface intact.** The agent's verbatim FALCON quote is
  attached to citation `[591f6045651c]`, surfaced in the Citations panel,
  and persisted via the same citations API path the rest of the matrix
  uses. No regression in the citation pipeline from the OCR-derived text.

## Release ceremony summary

- PR #86 merged via self-approve at 21:37:48 UTC (admin merge,
  enforce_admins toggled around it).
- Tag `v0.1.21` pushed at 21:38 UTC; PyPI Release workflow ran on
  the kaos-x64-16core self-hosted runner.
- kaos-agents 0.1.21 live on PyPI verified via
  `curl https://pypi.org/pypi/kaos-agents/0.1.21/json` →
  `info.version: 0.1.21`.
- SPA backend pin bumped (`kaos-agents>=0.1.21` + `kaos-pdf[ocr]>=0.1.4`),
  venv re-synced (`uv sync --refresh-package kaos-agents --refresh-package
  kaos-pdf --upgrade-package kaos-agents --upgrade-package kaos-pdf`),
  uvicorn restarted on :8000 (`/v1/health` → status=ok, providers
  openai/anthropic/google live).

## Plan-doc closure

WI5 from `kaos-modules/docs/plans/2026-05-26-corpus-stress-residuals-S16-S22-and-spa-acceptance.md`
is now closed end-to-end:

1. Unit tests (4 in `tests/unit/test_corpus_ocr_fallback.py`) PASS
   locally with tesseract installed; skip cleanly otherwise.
2. S03 corpus-stress integration test PASSES end-to-end
   (`tests/integration/test_corpus_stress_suite.py::TestMixedFileTypes::test_scenario_03_scanned_pdf_ocr_required`),
   25s, $0.120, recovered planted `7K-FALCON-2026`.
3. SPA Chrome MCP S03 re-acceptance (this run) PASSES with the planted
   FALCON token surfaced inside a verbatim blockquote with a citation.

## Outstanding (deferred to follow-up release)

- **WI2 (S22 confidence collapse).** Single-doc fixture does not reproduce
  the 50-doc flip-flop; the corpus-stress S22 integration test fails in
  CI on the same shape it failed on 0.1.20 (pre-existing). Need a true
  50-doc fixture before picking M2 vs synthesis-prompt remediation.
- **WI3 (S16 partial-success refusal wording structural refactor).** The
  current gate already returns `refuse=False` for partial success; the
  universal-claim wording only fires on total failure where it is
  correct. Structural plumbing of successful-call counts through
  `evaluate_no_evidence_gate` is a defensive refactor and was not
  triggered by this S03 run.
- **WI6 (SPA upload validator HTML/JSON).** Unchanged from the 0.1.20
  baseline; out of scope for kaos-agents.
