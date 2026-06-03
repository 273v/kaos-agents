# Agentic routing + transcript grounding

**Date:** 2026-06-02
**Status:** design → implementation
**Reference:** `../kelvin-agent` (truly-agentic routing + self-critique)

## The incident (telemetry-grounded)

Session `01KT5PK4ST95065194G4VXF4NB`, NDA corpus (`MNDA - Acme/BI/CC Final 2/DynaMo.docx`):

1. **Turn 1** — "which has the longest term" → `research` → FindingsAgent. Answer
   declared `CC Final 2` (fifth anniversary = fixed 5y) "longest" while *in the same
   sentence* noting `Acme` has "no fixed end date" (indefinite). Indefinite ⊅ shorter
   than 5y — an internal logical contradiction. Every gate passed it.
2. **Turn 2** — "uh, do you hear what you just said?" → `respond` (no tools). The model
   **confabulated**: "My last reply introduced **FRCP material**." FRCP appears *nowhere*
   in turn 1. The prior turn *was* in `SessionMemory.MESSAGES`, yet the model invented a
   plausible fault under social pressure. M2 consistency critic **passed** it ("no internal
   contradiction") — it cannot see prior turns.
3. **Turn 3** — "FRCP? what??? … find the logical fallacy" → the `corpus_attached_promotion`
   heuristic matched `"find"` + docs-attached → force-promoted `respond (0.96)` →
   `research (1.0)` → FindingsAgent ran 3× on a *meta* question, `filtered=0 answer_chars=0`,
   → `wall_clock_exceeded` refusal, **$0.21** for nothing.

## Root cause: a cheap proxy substituted for attending to evidence

Both failures are the same anti-pattern — substituting a cheap proxy for *actually
reasoning over the available evidence*:

- **Routing (turn 3):** keyword match (`"find"`, `"what is"`, …) + "docs attached"
  force-overrides the LLM router. A substring can't tell "find the *value* in the contract"
  (corpus) from "find the *fallacy* in your reasoning" (meta).
- **Grounding (turn 2):** the model answered a question *about the conversation* from
  parametric guesswork instead of reading the actual transcript; no critic verifies claims
  about prior turns.

kelvin-agent does neither. `ClassifyInput` is an **LLM** deciding response-type
(`message`/`clarify`/`plan`) + complexity 1–5 — never a keyword/data trigger
(`classify.py:82-91`, `chat_agent.py:173-250`). `InterpretLastResult` is a general
self-critique that scores 1–10 and retries (`interpret_last_action.py:50-83`). Evidence
(docs *and* messages) is LLM-filtered and passed as real content, never synthesized.

## The principle

**The agent routes and grounds against its actual evidence context — the attached
documents AND the conversation transcript — never against a keyword proxy or parametric
memory.** The transcript is a first-class evidence source, exactly like documents.

## Fix A — Agentic routing (delete the keyword promotion)

`kaos_agents/patterns/chat.py`:
- **Delete** `_CORPUS_FACT_LOOKUP_TOKENS`, `_message_looks_like_corpus_lookup`, and the
  `ChatAgent._classify` promotion override entirely.

`kaos_agents/context/classify.py` — make the LLM router document- *and* meta-aware:
- Add input `documents_available` (count + filenames of `SessionMemory.DOCUMENTS`).
- Extend `ClassifyIntentSignature` rubric:
  - `research` — the user asks about the **content** of attached documents (their terms,
    values, facts, comparisons). With documents attached, route content questions here so
    the answer is grounded on bytes (this preserves the legitimate CS-B2/CS-B3 cases the
    old promotion existed for — but now via LLM reasoning, not keywords).
  - **`respond`** — a turn that reacts to or asks about *the conversation itself* (the
    assistant's prior answer/reasoning/wording: "do you hear what you just said?", "find the
    flaw in your last reply", "what did you mean", "you're wrong") is conversational, **even
    when documents are attached**. The corpus retriever cannot answer questions about the
    dialogue.

`kaos_agents/runtime/agent.py` `BaseAgent._classify` — populate `documents_available` from
memory so every document-aware agent benefits. Add `classify_intent` examples covering
"meta turn + docs → respond" and "corpus-content turn + docs → research".

## Fix B — Transcript grounding (prevention + detection)

The model *had* the prior turn yet confabulated; so this is two layers:

**B1 — Prevention (always-on, load-bearing).** A general grounding-discipline rule in the
agent's instructions (kaos-agents, not the SPA): *when the user reacts to or asks about your
prior response, re-read the actual conversation transcript before answering; never assert you
said something unless it appears there; if the user implies you erred without specifying,
identify the specific issue from the transcript or ask what they mean — do not invent a
fault.* Ensure the recent transcript is surfaced to the responder. Thread real
`recent_turns` into the loop (currently `""`).

**B2 — Detection (critic).** Generalize grounding from "tool results only" to "available
evidence". Add a transcript-grounding critic lens (M5, or extend M3) that receives the recent
`MESSAGES` transcript + the response and flags self-referential claims ("my last reply…", "I
said…", "earlier I…") not supported by the transcript as a grounding failure → forces a
correction iteration. This is the same JudgeSignature pattern as M3, with the transcript as
the evidence channel. It runs in the SPA (M-critics are already wired there).

## Why this is general, not a band-aid

- No new keyword lists; routing is LLM reasoning over real context.
- Transcript-as-evidence is the *same* grounding principle as document-as-evidence, applied
  to a channel that was previously unguarded — it catches *all* inter-turn confabulations,
  not "FRCP".
- Fixes live in kaos-agents (the runtime), so every consumer (SPA, CLI, MCP) inherits them.

## Oracle / iteration

Reproduce the 3-turn sequence against the `data/NDA_01` set end-to-end (backend + Chrome MCP
+ telemetry). Pass criteria: meta turns route `respond`; no claim about a prior turn that
isn't in the transcript; no empty-refusal cascade; and (stretch) turn 2 surfaces the *real*
indefinite-vs-fixed-term issue from the transcript instead of a fabrication.
