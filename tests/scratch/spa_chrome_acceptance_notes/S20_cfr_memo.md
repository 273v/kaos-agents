# S20 — CFR memo (verbatim regulation quote)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK15K0Y3C5JKQQD6ZTCA33Z
**Fixture:** `/tmp/spa-acceptance/S20_cfr_memo.docx` (36.1 KB, anti-fraud memo
quoting 17 CFR § 240.10b-5(b))

## Prompt

> The attached memo cites 17 CFR § 240.10b-5. Quote the operative language of
> subsection (b) verbatim from the memo.

## Result

**Verdict: PASS** (grounded, verbatim quote, citation present).

Final assistant text:

> The operative language of 17 CFR § 240.10b-5(b) as quoted in the memo is
> [e236dbd66600]:
> "to make any untrue statement of a material fact or to omit to state a
> material fact necessary in order to make the statements made, in the light of
> the circumstances under which they were made, not misleading."

## Four-signal capture

- **Snapshot:** Operative phrase "untrue statement of a material fact" present
  verbatim. Citation marker `[e236dbd66600]` attached.
- **Tool trace:** 2 calls including `Agent Findings Dispatch → FindingsAgent:
  enumerated=9 filtered=2 cost=$0.0089 answer_chars=301`.
- **Cost / tokens:** $0.0089 / 3.9k tokens / 14.7s.
- **Citations panel:** 1 citation surfaced (from the filtered finding).

## Why this matters

The 0.1.20 verbatim-quote policy added to `_SynthesizeSignature.answer`'s
OutputField description is reaching the synthesis step at the SPA layer.
The model picked up the policy directive — the question's pattern
("Quote the operative language of subsection (b) verbatim") triggered the
verbatim-quote path, and the answer includes the exact regulatory phrase in
quotes with a citation. Pre-0.1.20 this would have returned a paraphrase
without the quoted clause.
