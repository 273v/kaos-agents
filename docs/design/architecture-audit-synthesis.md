# Architecture audit synthesis — 2026-05-06

Three independent Sonnet audits ran against `kaos-agents` looking for
places where code bypasses the kaos-llm-core typed-Signature stack
and the kaos-llm-client transport boundary. Findings are recorded in
the three sibling files:

* `architecture-audit-raw-llm-calls.md` — direct client usage
  bypasses
* `architecture-audit-prompt-templates.md` — hand-rolled prompts
  vs Signature docstrings
* `architecture-audit-structured-output.md` — Pydantic / codec
  bypasses

This file is the synthesis: convergent findings, real bugs,
prioritised refactor plan.

## Headline

**The transport boundary is clean.** Zero direct
`kaos-llm-client.chat_async` / `json_async` / `pydantic_async` calls
in application code. Zero hand-built `{"type": "object", ...}` JSON
Schema dicts. Zero manual code-fence stripping. Whatever else is
wrong, the abstraction layer between agents and the transport client
is being respected.

**The Signature/Invocation contract is partially broken.** The same
codebase has Signatures used correctly in some places and bypassed
entirely in others. Two patterns bite:

1. `await call(...)` (returns just `output`) used in 5+ places
   instead of `await call.invoke(...)` (returns the full `Invocation`
   with usage + trace). Cost telemetry and budget enforcement have a
   hole.
2. Long load-bearing prompts (`_RESEARCH_REACT_INSTRUCTION`,
   `_RETRIEVAL_INSTRUCTIONS`) live as module-level string constants
   handed to `Agent.create(instructions=...)`, completely outside any
   Signature.

## Tier 1 — real bugs to fix now

These are not "code quality" issues; they cause incorrect behaviour.

### 1.1 `__call__` vs `.invoke()` cost-attribution leak (HIGH)

**Bug.** `Call.__call__()` returns the bare validated output;
`Call.invoke()` returns the typed `Invocation` carrying
`usage.cost_usd`, `usage.input_tokens`, `usage.output_tokens`, and
the trace. Code that uses the bare-call form silently discards all
of that.

**Affected sites (5 confirmed — likely more):**

| File:line | What it does | Cost lost |
|---|---|---|
| `context/classify.py:120` | Intent classifier; fires every turn | Per-turn classification cost never reaches `TurnComplete` |
| `context/doc2query.py:91` | LLM-driven doc indexing | Per-doc indexing invisible to session budget |
| `memory/summarize.py:80` | ON_OVERFLOW + ON_TURN summarisation | Memory eviction calls untracked |
| `planning/expand.py:102` | Plan generation | Plan-expand cost lost from `PlanBudget` |
| `planning/evaluate.py:133` | Semantic evaluation | Per-step eval cost lost |

**Symptom.** `--max-cost` (and env `KAOS_AGENT_MAX_COST_USD`) is
documented as a hard session ceiling. With this leak, the actual
spend per session can exceed the ceiling by the cost of every
classifier + summariser + planner call across the run. On a
50-turn session that's 50× classifier calls (~$0.001 each) plus
maybe 5× summarisations and 5× plan expansions — silent overage on
the order of $0.10-$0.50 per session.

**Fix.** Mechanical: replace `await call(...)` with `await
call.invoke(...)`, then use `invocation.output` where the bare
output was used and `invocation.usage` where you want token/cost
data. Add a unit test that asserts the classifier's cost reaches
`TurnComplete.usage.cost_usd`.

### 1.2 `ClassifyIntent.intent: str` instead of Literal — silent misroute (HIGH)

**Bug.** `ClassifyIntent` Signature in `context/classify.py:99-142`
declares `intent: str` and the caller does:

```python
try:
    intent = IntentType(out.intent.lower())
except ValueError:
    intent = IntentType.RESPOND  # <-- silent fallback
```

The codec's structured-output path could enforce the constraint
upstream — but only if `intent: Literal["respond", "tool_use",
"plan", "clarify"]`. Today, a model that emits an unexpected token
silently routes to RESPOND, bypassing planning entirely.

**Fix.** Replace `intent: str` with `intent: Literal[...]`. Codec
will reject malformed output at decode time, triggering
`ValidationRetryExhaustedError` after retries — which is the
correct contract.

### 1.3 `patterns/research.py:683-702` — typed grounding flattened to dicts

**Bug.** Verified `Claim` instances with `Span` citations get
re-flattened to raw dict comprehensions before going into
`MemoryType.FINDINGS`. The CLAUDE.md says FINDINGS stores "Claim
instances with Span citations"; the code stores
manually-constructed dicts where:

* `tuple[int, int]` char spans become `list[int]`
* `ClaimType` enum becomes `str(claim.claim_type)`

**Fix.** Use `claim.model_dump(mode="json")` — Pydantic's
canonical JSON-safe serialisation that preserves the round-trip
shape.

### 1.4 `grounding.py:67-71` — duck-typed refusal policy

**Bug.** `apply_refusal_policy` uses
`getattr(grounded_answer, "kind", None)` to detect refusal mode.
If a wrong type with a `.kind` attribute but no `.confidence`
field reaches the function, the `getattr` default of 1.0 silently
passes the threshold check.

**Fix.** Use `isinstance(grounded_answer, (Answer, InsufficientEvidence))`
and explicit branching.

### 1.5 Judges (mine) use `getattr(usage, ...)` instead of `InvocationUsage`

**Bug (cosmetic but mine).** `kaos_agents/benchmarks/llm_judge.py`
and `rubric_judge.py` (just refactored to typed Call+Signature) read
the Invocation's `usage` via `getattr(usage, "input_tokens", 0)`
defensive accessors. The codebase already provides
`kaos_agents.usage.InvocationUsage.from_invocation(invocation)`
which does the same coercion in one place.

**Fix.** Use `InvocationUsage.from_invocation(invocation)` and read
typed fields from it.

## Tier 2 — architectural cleanup (real refactor)

### 2.1 `_RESEARCH_REACT_INSTRUCTION` (38 lines) → `RetrievalReActTask(Signature)`

`patterns/research.py:56` defines a 38-line module-level constant
that drives the entire RAG escalation loop, and at runtime gets
mutated via f-string to inject corpus outline + max_steps +
prior_failures. None of that is visible to:

- `BootstrapFewShot` (it can't bootstrap demos for an opaque string
  template)
- `MIPROv2` (it can't search over instructions if instructions
  aren't in a Signature)
- The trace tree (the dynamic content disappears into
  `Agent.create(instructions=...)`)

**Refactor.** Promote to a proper Signature where the dynamic
fields (`corpus_outline`, `prior_failures`, `max_steps`) are
`InputField`s. Move the orchestration logic into a `Program`
subclass that consumes the Signature.

### 2.2 `_RETRIEVAL_INSTRUCTIONS` (35 lines) → `RetrievalTask(Signature)`

`retrieval_agent.py:31` — same shape as 2.1. Drives the retrieval
sub-agent. Should be a Signature so:

- The retrieval strategy decisions (BM25 → synonyms → HyDE →
  evaluate) become observable in the trace
- Optimizer-driven instruction tuning becomes possible
- The strategy sequence becomes auditable per call

### 2.3 11 Signatures defined inside `async def` bodies

`audit-raw-llm-calls.md` lists 11 places where a Signature class is
defined as a local class inside an async function — recompiled every
invocation. Performance hit + breaks codec caching (the codec hashes
the schema; recompiled classes have unstable identities).

**Fix.** Move every `class FooSig(Signature):` to module level.

### 2.4 `PlanExpand.steps: list[dict[str, Any]]`

The output of plan expansion has typed step structure (id, tool,
inputs, depends_on, …) but is declared as `list[dict[str, Any]]`,
defeating codec validation of the nested shape.

**Fix.** Define a `PlanStep(BaseModel)` and type the field
`steps: list[PlanStep]`.

### 2.5 `AgentSnapshot` / `RunState` hand-rolled (de)serialization

`interrupts.py:148-244` — bespoke `to_dict`, `from_dict`, `to_json`,
`from_json` for runtime-pause/resume state. Renamed fields produce
silent wrong defaults instead of validation errors.

**Fix.** Convert to `BaseModel` with `model_dump_json()` /
`model_validate_json()`. Three+ pages of boilerplate go away.

### 2.6 Sync retrieval helpers using `ThreadPoolExecutor + asyncio.run(call(...))`

`context/retrieval.py` — `_reflect_on_coverage`,
`_generate_pseudo_document`, `_generate_llm_queries` are sync APIs
that internally bridge to async via `ThreadPoolExecutor.submit(...)`
+ `asyncio.run(call(...))`. New event loop per call, no usage
captured, no cost rollup.

**Fix.** Make them `async def`; their callers in
`context/retrieval.py` are already inside async paths.

## Tier 3 — observability + optimizer readiness

### 3.1 No Signatures ship with `examples=[Example(...)]`

The kaos-llm-core optimizers (`BootstrapFewShot`, `MIPROv2`,
`InstructionOptimizer`) operate by tuning a Signature's instructions
+ examples against labeled data. Today, every Signature in
kaos-agents ships with zero examples — optimizers have nothing to
bootstrap from.

**Action.** Add a curated `Example`-list seed to the highest-value
Signatures: `ClassifyIntent`, `RubricVerdictSignature`,
`PlanExpand`, `EvalSig`, `SummarizeMemory`. The seed examples
become the cold-start training set for the optimizer pipeline.

### 3.2 Redundant `JudgeVerdict` / `RubricVerdict` dataclasses

After my refactor, the typed Signatures (`QAJudgeSignature`,
`RubricVerdictSignature`) define the canonical output shape. The
parallel dataclasses `JudgeVerdict` / `RubricVerdict` re-declare
the same fields and add observability-only fields (`judge_model`,
`judge_cost_usd`, `judge_*_tokens`).

**Fix.** Either:
- Pare the dataclass to observability-only fields; embed the
  Signature output directly via composition; or
- Drop the dataclass entirely and pass the typed Signature output
  + a separate `Usage` to callers.

The current shape leaves callers with an untyped dict
(`to_dict()`) at the API boundary — `ty` can't see through it.

## Tier 4 — deferred

These are real but low-leverage:

* `_ExplainTurn` (cli_chat.py) — mutable dataclass with
  `list[dict[str, Any]]` citation fields; uses manual
  `_explain_to_dict()` instead of `dataclasses.asdict()`.
* `mcp_extract.py:404-410` — corpus wire-input validated with
  `isinstance` loop instead of `pydantic.RootModel`.
* `_RESEARCH_REACT_INSTRUCTION` f-string mutation —
  `max_steps`/`prior_failures` end up in the prompt but not the
  trace's typed fields.

## Refactor sequencing

| Order | Tier | Item | Effort | Risk |
|---|---|---|---|---|
| 1 | 1.1 | `__call__` → `.invoke()` everywhere | half day | LOW (mechanical) |
| 2 | 1.2 | `ClassifyIntent.intent: Literal[...]` | 30 min | LOW |
| 3 | 1.3 | research.py FINDINGS → `model_dump(mode="json")` | 30 min | LOW |
| 4 | 1.4 | `apply_refusal_policy` `isinstance` | 15 min | LOW |
| 5 | 1.5 | judges use `InvocationUsage.from_invocation()` | 30 min | LOW |
| 6 | 2.3 | Move 11 inline-async Signatures to module level | 2 hours | LOW |
| 7 | 2.4 | `PlanExpand.steps: list[PlanStep]` | 2 hours | MEDIUM (changes wire shape) |
| 8 | 3.2 | Pare `JudgeVerdict` / `RubricVerdict` to observability-only | 1 hour | LOW |
| 9 | 2.1 | `_RESEARCH_REACT_INSTRUCTION` → Signature | 1 day | MEDIUM (load-bearing path) |
| 10 | 2.2 | `_RETRIEVAL_INSTRUCTIONS` → Signature | 1 day | MEDIUM |
| 11 | 2.5 | `AgentSnapshot` / `RunState` → BaseModel | half day | MEDIUM |
| 12 | 2.6 | Sync retrieval helpers → async-native | half day | LOW |
| 13 | 3.1 | Seed `examples=[...]` on top Signatures | 1 day | LOW |

Items 1-5 land together. Items 6-8 are the next focused day. Items
9-13 are a separate workstream.

## Decision

Items 1-5 (the real bugs) get fixed in this session. Item 8 also
lands because it's small and cleans up code I just wrote. Items 6-7
are the next day's work. Items 9-13 are a separate workstream and
deserve their own planning round.
