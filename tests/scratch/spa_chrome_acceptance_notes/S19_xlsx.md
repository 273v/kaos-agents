# S19 — XLSX upload (openpyxl-produced workbook)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK0PAE3E7DN7XP9G5QKQFQQ
**Fixture:** `/tmp/spa-acceptance/S19_lineitems.xlsx` (4.9 KB, openpyxl-produced, 4 rows × 3 cols)

## Prompt

> The attached spreadsheet has a row labeled "Critical line item". Tell me its
> row_id and amount_usd. Quote the values verbatim from the spreadsheet.

## Result

**Verdict: PASS** (grounded).

Final assistant text (verbatim from UI):

> Based on the retrieved findings, the row labeled "Critical line item" has the
> following verbatim values [3c891b36d263]:
>
> - row_id: LI-S19-9001
> - amount_usd: 77777.77

## Four-signal capture

- **Snapshot:** Both needles (`LI-S19-9001` and `77777.77`) surfaced in the
  assistant text with the correct citation marker.
- **Tool trace:** 2 calls, including a successful FindingsAgent dispatch
  (`Agent Findings Dispatch → FindingsAgent: enumerated=5 filtered=1
  cost=$0.0079 answer_chars=177`).
- **Cost / tokens:** $0.0079 / 3.5k tokens / 8.9s.
- **Documents panel:** 1 document, READY.

## Why this matters

This is the end-to-end proof that the kaos-office 0.1.3 OPC path resolver
fix landed at the SPA layer: an openpyxl-produced XLSX with absolute
relationship targets now round-trips through `parse_xlsx_native`, hits the
agent's tool surface, and the agent surfaces the planted row verbatim.
Pre-0.1.3 this same fixture would have returned `table_count=0, tables=[]`
silently and the agent would have refused (the historical S19 / S16
failure mode).
