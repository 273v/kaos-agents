# S22 — cluster routing (multi-metric labelled response)

**Date:** 2026-05-26
**Stack:** kaos-agents 0.1.20, kaos-office 0.1.3, kaos-pdf 0.1.4
**Model:** anthropic:claude-sonnet-4-6
**Session:** 01KSK1DFDYFA57T015P8YHCKMN
**Fixture:** `/tmp/spa-acceptance/S22_cluster.docx` (36.0 KB,
single DOCX narrating "2486 total rows" + "24 distinct clusters" with
each number labelled)

## Prompt

> According to the attached report, how many routing clusters exist, and what
> is the total transaction-record count? Both numbers are policy-relevant —
> give me each value with its label.

## Result

**Verdict: PASS (on this fixture).** Agent surfaced both metrics with their
labels and citations. No flip-flop.

Final assistant text:

> According to the report:
> - **Routing clusters**: 24 distinct routing clusters (each row assigned
>   based on destination zone + product class + handler queue)
>   [3c2270b08aa6][2dd4e7cc336a]
> - **Total transaction-record count**: 2,486 rows of transaction records
>   sourced from the upstream operational warehouse
>   [1ac68ebf15be][2dd4e7cc336a]

## Four-signal capture

- **Snapshot:** Both `24` and `2,486` present with their labels. Three
  citations attached across the two values.
- **Tool trace:** 2 calls including `Agent Findings Dispatch → FindingsAgent:
  enumerated=8 filtered=3 cost=$0.0092 answer_chars=334`.
- **Cost / tokens:** $0.0092 / 3.9k tokens / 8.6s.
- **Citations panel:** 3 citations.

## Caveat — fixture scope vs original failure mode

This single-DOCX fixture does NOT reproduce the original S22 failure
mode from CI, which was a **50-document cluster routing scenario**.
That run showed the agent surfacing N=2486 early in the response then
revising to N=24 mid-response, dropping the larger value from the
headline. The 50-doc context evidently produces more candidate
findings that trigger either the M2 critic or a synthesis
"override-later-with-earlier" pattern.

**What this scenario verifies on the 0.1.20 stack:**

- The dispatch + synthesis composition CAN handle the multi-metric
  labelled case cleanly when the corpus is small enough that the
  filter doesn't flag a contradiction.
- The 0.1.20 verbatim-quote policy doesn't interfere with multi-metric
  responses — the model didn't force either number into a verbatim
  blockquote when the question wasn't asking for "the regulation".

**What's still outstanding for the 50-doc reproduction:**

- The plan's WI2 still needs a 50-doc S22 reproduction to surface the
  flip-flop. The single-doc fixture won't trigger it.
- For the trace inspection step, the right next move is to build a
  50-doc cluster-routing fixture (or pull the original corpus-stress
  S22 fixture from the kaos-agents test suite) and re-run on the SPA.

## Why this matters

The 0.1.20 stack handles the simplest version of the multi-metric
case correctly. The 0.1.21 release plan's WI2 (S22 fix) needs a
proper 50-doc fixture to surface the flip-flop pattern that the
original CI failure documented. This degraded result means we can
**defer the M2 / synthesis rubric change until we can reproduce the
flip-flop locally** — shipping a fix without that trace evidence
risks regressing S05 / S07 per the plan's risk-rollback section.
