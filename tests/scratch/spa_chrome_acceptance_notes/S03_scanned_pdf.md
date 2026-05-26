# S03 — scanned PDF (OCR required)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK0VRDKNWY0BRHKT2D8D0RP
**Fixture:** `/tmp/spa-acceptance/S03_scanned.pdf` (13.4 KB, image-only PDF, planted token FALCON)

## Prompt

> The attached PDF is a scanned image of an internal memo. What codename is
> mentioned? Use OCR if needed and quote the exact value.

## Result

**Verdict: FAIL.** Final assistant text: `(empty)`. No answer produced.

## Four-signal capture

- **Snapshot:** Activity panel shows `Agent Findings Dispatch → FindingsAgent:
  enumerated=1 filtered=0 cost=$0.0012 answer_chars=0`. Final assistant
  text is `(empty)`.
- **Tool trace:** 2 tools total. FindingsAgent dispatch ran but
  enumerated=1 filtered=0 — the corpus pre-parse yielded 1 candidate
  (the empty body of a scanned PDF) and the filter rejected it. The
  new `kaos-pdf-ocr-page` tool added in kaos-pdf 0.1.4 was **never
  called**.
- **Cost / tokens:** $0.0012 / 1.5k tokens / 5.2s.
- **Documents panel:** 1 document, READY.

## Root cause (new bug surfaced)

The FindingsAgent dispatch path (added in 0.1.19) pre-parses
DOCUMENTS-attached corpus items eagerly via the format-specific
parsers (`parse_pdf_bytes` for PDF). For a scanned PDF this yields
an empty `ContentDocument.body` because there's no text layer. The
enumeration step then has 1 candidate (the empty doc), the filter
step rejects it, and the synthesis step produces an empty answer.

**There is no OCR fallback in `_build_corpus_view_from_documents`.**
The `kaos-pdf-ocr-page` tool exists in the agent's tool surface but
the FindingsAgent dispatch never calls it — that tool lives in the
ReAct dispatch path, which the corpus-attached classifier promotion
bypasses.

## Fix needed (new work item)

Add an OCR-fallback hook in the FindingsAgent dispatch:

1. After `_build_corpus_view_from_documents` parses corpus items,
   check if any PDF-typed item has empty / near-empty body.
2. If so, call `kaos-pdf-classify-page` to confirm scanned, then
   `kaos-pdf-ocr-page` for each page, and inject the OCR'd text
   back into the corpus view before enumeration runs.

Alternative: the dispatch path could detect "empty body after
pre-parse" and fall through to ReAct so the LLM can call the OCR
tool itself.

This is a new bug surfaced by the SPA Chrome MCP run that the
0.1.21 release plan needs to add. The original plan's Work Item 1
predicted S03 would PASS — the PyPI publish + tool registration
verified that part, but the dispatch never reaches the new tool
because FindingsAgent's pre-parse short-circuits.

## Why this matters

The kaos-pdf 0.1.4 OCR tool is shipping but isn't reachable from
the default corpus-grounded dispatch surface. Either the dispatch
must learn to call it, or scanned-PDF questions must route to ReAct
instead of FindingsAgent.
