# S16 — 5-format pile (degraded to 3 by SPA upload validator)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK18F5SADP0BFP74AJ4TGVD
**Fixtures uploaded:** `pile-revenue.pdf`, `pile-counsel.docx`,
`pile-lineitems.xlsx` (HTML + JSON rejected by SPA upload validator —
see "New SPA bug surfaced" below).

## Prompt

> Three files are attached (PDF, DOCX, XLSX). Each contains one distinctive
> fact. From each, surface the fact verbatim: (1) the PDF's revenue
> plug-figure, (2) the DOCX's counsel-of-record name, (3) the XLSX row_id
> labelled "critical line". List all three with values.

## Result

**Verdict: PASS (degraded).** All 3 PDF/DOCX/XLSX needles surfaced
verbatim. The 5-format aspiration could not be tested because the
SPA upload validator rejects `.html` + `.json` extensions even
though the underlying kaos-agents stack can read those formats.

Final assistant text contained the three values in a table:

| # | File | Fact | Value |
|---|------|------|-------|
| 1 | pile-revenue.pdf | Revenue plug-figure | **$93.14M** |
| 2 | pile-counsel.docx | Counsel of record | **Hannah Brueggeman** |
| 3 | pile-lineitems.xlsx | Row "critical line" (description) | **"Pile-test critical line"** |

Plus a "Sources & methodology" section naming each parser
(`kaos-pdf-extract-parse`, `kaos-office-parse-docx`,
`kaos-office-parse-xlsx` + `kaos-content-search-table`).

## Four-signal capture

- **Snapshot:** All 3 needles surfaced in a table. No "every tool
  call returned an error" overclaim.
- **Tool trace:** 4 tools (Expand button shows them grouped).
- **Cost / tokens:** $0.0634 / 16.8k tokens / 23.9s.
- **Documents panel:** 3 documents, all READY.

## New SPA bug surfaced — JSON / HTML upload validator

The SPA frontend rejected `.json` (and presumably `.html`) at the
upload step with `unsupported file extension '.json'` (visible
console alert dismissed). Only PDF / DOCX / PPTX / XLSX are
accepted, per the Attach button tooltip: `Attach a file (PDF,
DOCX, PPTX, XLSX)`.

This is a SPA frontend layer restriction. The underlying
kaos-agents + kaos-content + kaos-core stack can read HTML and
JSON via `kaos-content-load-document` or `kaos-core-vfs-read` —
the CLI smoke confirmed this. But SPA users can never reach those
formats through the file-attach affordance.

**Two follow-up paths:**

1. **SPA frontend fix** — widen the accepted-extensions list on
   the file input + remove the validator error. Backend already
   sniffs content type. Lives in the kaos-ui repo, separate work.
2. **Workaround** — users currently must paste HTML/JSON content
   into the chat as text. Tolerable but not the right UX.

This is logged as the "Step 3.5 sub-issue (HTML/JSON tool
routing)" in the 0.1.21 residuals plan, but the routing path the
plan envisioned is moot if the SPA never delivers the file to the
backend. The bigger fix is in the SPA.

## Why this matters

The agent CAN handle 3-format piles correctly on the 0.1.20 stack
— the S19 XLSX fix from kaos-office 0.1.3 means all three parsers
return real data, and the FindingsAgent dispatch composes them
cleanly with no "every tool failed" overclaim. The 5-format
aspiration is blocked by a SPA frontend validator that lives
above this release ceremony's scope.

## SPA bug ID (new, for follow-up)

**WI6 — SPA upload validator allows only Office/PDF, blocks HTML
+ JSON.** Lives in kaos-ui, not kaos-agents. Logged as the
remaining S16 surface but should be triaged in a kaos-ui session.
