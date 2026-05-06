# Architecture Audit: Raw LLM Call Bypass in kaos-agents

**Audit date:** 2026-05-06  
**Audited scope:** `kaos_agents/` — benchmarks, patterns, planning, context, agent.py, runner.py, retrieval_tools.py, tools.py  
**Audit methodology:** ripgrep scan for direct kaos-llm-client imports, bare `call()` vs `call.invoke()`, hand-built JSON schema dicts, constant system-prompt strings, manual JSON parsing, manual usage/cost plumbing, and `getattr(usage, ...)` duck-typing outside the designated factory boundary.

---

## Executive Summary

The codebase is **broadly compliant** with the typed abstraction stack. There is no direct `create_client`, `chat_async`, `json_async`, or `pydantic_async` usage in non-test code. All LLM calls pass through `Call(Signature, model=...)` or higher-level programs (`ReAct`, `RAG`). However, six structural patterns represent real deficiencies — two are HIGH severity because they silently discard `Invocation` and thereby lose trace, cost, and validation-retry data.

**Severity counts:** HIGH: 3, MEDIUM: 7, LOW: 3

---

## Category 1: Direct kaos-llm-client Usage

### Finding 1.1 — No direct `create_client` / `chat_async` / `json_async` in application code

**Severity: N/A (CLEAN)**

```
rg -n 'create_client|chat_async|json_async|pydantic_async' kaos_agents/ -g "*.py"
# → zero results
```

kaos-llm-client is imported only for `ToolDefinition` types (in tests and `tool_bridge.py`), and for the availability check in `_llm_imports.py`. No agent application code calls provider transport methods directly.

---

## Category 2: Bare `call(...)` Instead of `call.invoke(...)` — Invocation Discarded

These are the highest-severity findings. `await call(...)` returns only the decoded output object; `await call.invoke(...)` returns the full `Invocation` with `.usage`, `.trace`, `.error`, and validation-retry metadata. The difference matters for cost accounting, OTel tracing, and debugging retries.

### Finding 2.1 — `classify.py:120` — classifier discards Invocation

**Severity: HIGH**

```python
# kaos_agents/context/classify.py:119-120
call = Call(ClassifyIntent, model=model, instructions=_CLASSIFY_INSTRUCTION)
result = await call(message=user_message, conversation_context=context_text)
```

`classify_intent` is called at the start of every turn. Its token spend is never surfaced in `TurnComplete` (it does not return usage and the call site discards the Invocation). The classifier model cost is entirely invisible to the cost rollup. Additionally, the function performs a round-trip string parse of `result.intent` and `result.confidence` manually (lines 123–136) rather than relying on `Call`'s codec-validated output.

**Replacement:** Use `await call.invoke(...)` and return `(intent_result, InvocationUsage.from_invocation(invocation))` so the agent loop can roll classifier cost into `TurnComplete`.

---

### Finding 2.2 — `doc2query.py:91` — doc2query discards Invocation

**Severity: HIGH**

```python
# kaos_agents/context/doc2query.py:90-91
call = Call(PredictDocumentQueries, model=model or DEFAULT_MODEL)
result = await call(document_excerpt=text[:_DOC_PREVIEW_CHARS])
```

Doc2Query is called once per indexed document. It may run on dozens of documents during corpus load. None of this LLM spend is tracked — it is invisible to the session cost ceiling enforced by `cli_chat.py` (`--max-cost` flag) and the `KAOS_AGENT_PLAN_MAX_COST_USD` budget. If the session ceiling is close to exhausted, a bulk doc2query pass could silently breach it.

**Replacement:** `invocation = await call.invoke(...)` then read `invocation.output.predicted_queries`.

---

### Finding 2.3 — `memory/summarize.py:80-86` — memory summarizer discards Invocation

**Severity: HIGH**

```python
# kaos_agents/memory/summarize.py:79-86
result = await call(
    content=content,
    section_type=section_type.value,
    target_length=f"approximately {target_chars} characters ({target_tokens} tokens)",
)
summary = str(result.summary)
```

Summarization fires during ON_OVERFLOW and ON_TURN eviction. Like doc2query, this can run many times per session without contributing to the usage accounting. The `target_length` parameter is built via f-string rather than using the Signature `instructions=` field, duplicating prompt engineering logic at the call site.

**Replacement:** `invocation = await call.invoke(...)` and propagate `InvocationUsage.from_invocation(invocation)` to the caller.

---

### Finding 2.4 — `planning/expand.py:102-106` — plan expander discards Invocation

**Severity: MEDIUM**

```python
# kaos_agents/planning/expand.py:101-106
result = await call(
    goal=goal,
    tools=tool_descriptions,
    context=context[:3000],
)
raw_steps = result.steps if result.steps else []
```

The Invocation returned by `PlanExpand` is thrown away. This means plan generation cost is not surfaced in `PlanBudget` accounting or `TurnComplete`. Unlike `_act_llm` (which correctly uses `call.invoke()`), `expand` silently loses usage. The inconsistency is likely accidental.

**Replacement:** `invocation = await call.invoke(...)` and return `(steps, usage)`.

---

### Finding 2.5 — `planning/evaluate.py:133-137` — semantic evaluator discards Invocation

**Severity: MEDIUM**

```python
# kaos_agents/planning/evaluate.py:132-137
output = await call(
    result_text=str(result)[:2000],
    expected_description=expected,
    additional_context=context[:1000],
)
return Judgment(...)
```

`evaluate_semantic` is called for every plan step that needs semantic judgment. The LLM cost is recorded nowhere — not in the `PrimitiveTrace` returned by `act.py`, not in `PlanBudget`. Contrast with `act.py:145` which uses `call.invoke()` correctly.

**Replacement:** `invocation = await call.invoke(...)`, extract `invocation.output`, and thread usage into the returned `Judgment` or the caller's `PrimitiveTrace`.

---

## Category 3: Signature-Defined Inline vs. Module-Level

### Finding 3.1 — Signatures defined inside function bodies (10+ locations)

**Severity: MEDIUM**

The following Signatures are defined as local classes inside `async def` functions:

```
kaos_agents/context/classify.py:99         class ClassifyIntent(Signature)
kaos_agents/context/doc2query.py:72        class PredictDocumentQueries(Signature)
kaos_agents/context/retrieval.py:612       class ReflectOnCoverage(Signature)
kaos_agents/context/retrieval.py:827       class GeneratePseudoDocument(Signature)
kaos_agents/context/retrieval.py:886       class SuggestQueries(Signature)
kaos_agents/memory/summarize.py:61         class SummarizeMemory(Signature)
kaos_agents/planning/act.py:138            class LLMStep(Signature)
kaos_agents/planning/evaluate.py:112       class EvalSig(Signature)
kaos_agents/planning/expand.py:70          class PlanExpand(Signature)
kaos_agents/agent.py:581                   class Respond(Signature)
kaos_agents/patterns/chat.py:156           class ToolTask(Signature)
```

Local `class` definitions inside `async def` bodies are re-evaluated on every call. For `classify_intent` (every turn) and `summarize_items` (every eviction), this means a fresh class object — with a fresh Pydantic schema — is constructed each time. This has no correctness impact but does have a performance cost (Pydantic schema compilation is not free), and it prevents the signatures from being unit-testable or inspectable outside their function scope.

The two benchmarks (`llm_judge.py`, `rubric_judge.py`) correctly define their signatures at module level.

**Replacement:** Move all Signatures to module level (following the benchmark pattern). They are already type-correct — only placement is wrong.

---

## Category 4: Hand-Built JSON / Untyped `OutputField` Shapes

### Finding 4.1 — `planning/expand.py:80-88` — `PlanExpand.steps` typed as `list[dict[str, Any]]`

**Severity: MEDIUM**

```python
# kaos_agents/planning/expand.py:80-88
steps: list[dict[str, Any]] = OutputField(
    description="List of plan steps. Each step is a dict with: "
    "step_number (int, 1-indexed), "
    "description (str, what this step does), "
    ...
)
```

This `OutputField` is typed as `list[dict[str, Any]]` and its shape is documented only as a string in `description=`. The Pydantic validation machinery inside `Call` cannot enforce the nested structure — if the LLM returns steps without `step_number` or with the wrong type, the error surfaces as a `KeyError` in `_validate_raw_steps()` rather than as a `ValidationRetryExhaustedError` that would trigger Call's retry loop.

The `GeneratedStep(BaseModel)` class already exists in the file (lines 119–132) and captures the correct schema. However, because `OutputField` is typed as `list[dict]` rather than `list[GeneratedStep]`, Pydantic never validates against it.

**Replacement:** Change the OutputField type to `list[GeneratedStep]` (moving `GeneratedStep` before the Signature class definition). This lets Call's codec enforce the step shape and retry on malformed output. Delete the manual `_validate_raw_steps` function — it becomes redundant.

---

## Category 5: Hand-Rolled System Prompts Bypassing Signature Docstrings

### Finding 5.1 — `context/classify.py:34-50` — `_CLASSIFY_INSTRUCTION` constant

**Severity: MEDIUM**

```python
# kaos_agents/context/classify.py:34-50
_CLASSIFY_INSTRUCTION = """Classify the user's intent into one of these categories:
- respond: Simple conversational response...
- tool_use: The user wants to perform an action...
...
"""
# used at:
call = Call(ClassifyIntent, model=model, instructions=_CLASSIFY_INSTRUCTION)
```

The `ClassifyIntent(Signature)` docstring (line 100: `"""Classify the user's intent for routing..."""`) duplicates the intent of `_CLASSIFY_INSTRUCTION`. The module-level constant is the actual classification policy; the Signature docstring is vestigial boilerplate. The canonical approach in kaos-llm-core is to embed the full policy in the Signature docstring (which `Call` feeds as the system instruction by default via `get_instruction(signature)`) and pass `instructions=` only for runtime overrides.

**Replacement:** Move the full `_CLASSIFY_INSTRUCTION` text into the `ClassifyIntent` docstring. Remove the `instructions=_CLASSIFY_INSTRUCTION` argument from the `Call(...)` constructor. Delete the module-level constant.

---

### Finding 5.2 — `patterns/research.py:56-93` — `_RESEARCH_REACT_INSTRUCTION` string constant

**Severity: MEDIUM**

```python
# kaos_agents/patterns/research.py:56-93
_RESEARCH_REACT_INSTRUCTION = """\
You are answering a question about a document corpus using retrieval tools.
STRATEGY:
1. Search with kaos-retrieval-bm25 using key terms from the question.
...
"""
# used at:
self._instructions = (
    saved_instructions + "\n\n" if saved_instructions else ""
) + _RESEARCH_REACT_INSTRUCTION + outline_block
```

This 38-line constant is concatenated with runtime state (`outline_block`, `saved_instructions`) via string operations to construct the `ReAct` system prompt. This bypasses `Signature`-level `instructions=` and injects prompt policy through a mutable instance attribute (`self._instructions`). The mutation is protected by a try/finally save-restore (lines 461-483), which is fragile — a concurrent turn on the same agent instance (unlikely given VFS-stateless design but not impossible) would see the mutated instruction.

**Replacement:** Define a `ResearchSignature(Signature)` with `_RESEARCH_REACT_INSTRUCTION` embedded in its docstring. Pass the outline prefix as an `InputField` (e.g., `corpus_outline: str`) rather than via instruction mutation. This also makes the outline visible to the trace tree.

---

### Finding 5.3 — `planning/expand.py:90-97` — f-string instruction built at call time

**Severity: LOW**

```python
# kaos_agents/planning/expand.py:90-97
instructions = (
    "Generate a concrete, actionable plan using ONLY the listed tools. "
    ...
    f"Maximum {max_steps} steps."
    f"{failure_context}"
)
call = Call(PlanExpand, model=model, instructions=instructions)
```

The `max_steps` and `failure_context` (prior failures from Reflexion) are injected into `instructions=` via f-string, which means they are not captured in the `Invocation.trace` as structured fields — they are buried inside a freeform string. The trace tree sees `instructions="Generate a concrete ... Maximum 10 steps.\n\nPREVIOUS FAILED ATTEMPTS..."` rather than discrete `max_steps=10` and `prior_failures=...` spans.

**Replacement:** Add `max_steps: int` and `prior_failures: str` as `InputField`s on `PlanExpand`. Pass them as call arguments. This makes Reflexion feedback traceable.

---

## Category 6: Manual Usage / Cost Plumbing Outside the Designated Factory Boundary

### Finding 6.1 — `benchmarks/llm_judge.py:199-201` and `benchmarks/rubric_judge.py:252-254` — bare `getattr(usage, ...)` instead of `InvocationUsage.from_invocation()`

**Severity: LOW**

```python
# kaos_agents/benchmarks/llm_judge.py:199-201
judge_cost_usd=float(getattr(usage, "cost_usd", 0.0) or 0.0),
judge_input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
judge_output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
```

The `InvocationUsage.from_llm_usage()` factory in `kaos_agents/usage.py:53-65` exists precisely to centralize this `getattr` duck-typing pattern. Both judge files re-implement it inline rather than calling the factory. This is low-severity because the benchmark code is not in the hot path and the logic is identical — but it's an inconsistency that will diverge if `InvocationUsage` grows new fields (e.g., `cache_read_tokens`).

**Replacement:** Replace both occurrences with:
```python
usage_obj = InvocationUsage.from_invocation(invocation)
# then use usage_obj.cost_usd, usage_obj.input_tokens, usage_obj.output_tokens
```
Import `InvocationUsage` from `kaos_agents.usage`.

---

### Finding 6.2 — `kaos_agents/usage.py:62-65` — canonical `getattr` duck-typing in `from_llm_usage()`

**Severity: LOW (ACCEPTED DESIGN)**

```python
# kaos_agents/usage.py:62-65
input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
cost_usd=float(getattr(usage, "cost_usd", 0.0) or 0.0),
```

This is the intentional cross-layer boundary adapter. The `[llm]`-optional invariant (documented in `CLAUDE.md`) means kaos-agents cannot import `kaos_llm_core.programs._invocation.TokenUsage` directly at module level. The `getattr` guards handle the `Any`-typed parameter. This pattern is **correct by design** — it is the designated factory. The two benchmark files (Finding 6.1) should call this factory instead of reimplementing it.

---

## Category 7: Sync-to-Async Bridge Anti-Pattern for LLM Calls in Sync Contexts

### Finding 7.1 — `context/retrieval.py:654-661`, `857-862`, `925-931` — `asyncio.run(call(...))` in sync retrieval functions

**Severity: MEDIUM**

```python
# kaos_agents/context/retrieval.py:654-661 (_reflect_on_coverage)
if loop and loop.is_running():
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = pool.submit(
            asyncio.run,
            call(original_query=query, document_summaries=summary_text),
        ).result(timeout=_DEFAULT_LLM_TIMEOUT)
else:
    result = asyncio.run(call(original_query=query, document_summaries=summary_text))
```

The same pattern appears in `_generate_pseudo_document` (lines 854-862) and `_generate_llm_queries` (lines 921-931). These three private functions in `retrieval.py` are synchronous callers that need to make LLM calls, so they implement a `ThreadPoolExecutor` + `asyncio.run()` bridge to fire the coroutine. This is the correct workaround for calling async code from sync contexts. However:

1. All three calls use `await call(...)` (bare, not `call.invoke()`), so usage is discarded.
2. The `ThreadPoolExecutor(max_workers=None)` uses the default worker count. Each LLM call spawns a new executor with potentially many idle threads.
3. The bridge creates a new event loop in the thread via `asyncio.run()`, bypassing any `asyncio`-level connection pooling that kaos-llm-client uses.

**Replacement:** Convert `_reflect_on_coverage`, `_generate_pseudo_document`, and `_generate_llm_queries` to `async def` functions. All three callers in the same file are also sync wrappers — push the `async def` boundary up to the first async context, or use a shared `ThreadPoolExecutor` with a bounded worker count.

---

## Summary Table

| # | File:Line | Pattern | Severity |
|---|-----------|---------|----------|
| 2.1 | `context/classify.py:120` | `await call(...)` discards Invocation — classifier cost invisible to TurnComplete | HIGH |
| 2.2 | `context/doc2query.py:91` | `await call(...)` discards Invocation — doc indexing cost invisible to session budget | HIGH |
| 2.3 | `memory/summarize.py:80` | `await call(...)` discards Invocation — eviction LLM cost invisible | HIGH |
| 2.4 | `planning/expand.py:102` | `await call(...)` discards Invocation — plan generation cost not in PlanBudget | MEDIUM |
| 2.5 | `planning/evaluate.py:133` | `await call(...)` discards Invocation — semantic eval cost lost | MEDIUM |
| 3.1 | 11 sites (see above) | Signatures defined as local classes inside `async def` — re-compiled every call | MEDIUM |
| 4.1 | `planning/expand.py:80` | `steps: list[dict[str, Any]]` OutputField — codec cannot validate nested shape | MEDIUM |
| 5.1 | `context/classify.py:34,119` | `_CLASSIFY_INSTRUCTION` constant passed as `instructions=` bypasses Signature docstring | MEDIUM |
| 5.2 | `patterns/research.py:56,469` | `_RESEARCH_REACT_INSTRUCTION` built by string concat with runtime state | MEDIUM |
| 5.3 | `planning/expand.py:90` | f-string instruction hides `max_steps` and `prior_failures` from trace | LOW |
| 6.1 | `benchmarks/llm_judge.py:199` `benchmarks/rubric_judge.py:252` | `getattr(usage, ...)` re-implemented instead of using `InvocationUsage.from_invocation()` | LOW |
| 7.1 | `context/retrieval.py:654,857,925` | `asyncio.run(call(...))` sync bridge — bare call discards Invocation, new event loop per call | MEDIUM |

---

## What Is Clean (No Findings)

The following components correctly use the typed stack end-to-end:

- `benchmarks/llm_judge.py` and `benchmarks/rubric_judge.py` — module-level Signatures, `call.invoke()`, proper Invocation access (minus Finding 6.1 factory cleanup)
- `patterns/chat.py` — `ReAct.invoke()` with `InvocationUsage.from_invocation()`
- `patterns/research.py` — `RAG.invoke()` with `InvocationUsage.from_invocation()`
- `patterns/plan_execute.py` — `InvocationUsage.from_llm_usage()` via event accumulation
- `planning/act.py` — `call.invoke()` with correct `invocation.usage` extraction
- `agent.py:_simple_respond` — `call.invoke()` with `InvocationUsage.from_invocation()`
- No direct `kaos_llm_client.create_client`, `chat_async`, or `ProviderResponse` usage anywhere in application code
- No hand-built `{"type": "object", "properties": {...}}` JSON schema dicts in application code
- No manual code-fence / JSON parsing (`s.strip("` + '`' + `")` patterns) anywhere
