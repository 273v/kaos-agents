# Structured Output Bypass Audit — kaos-agents

**Date:** 2026-05-06
**Scope:** `/home/mjbommar/projects/273v/kaos-modules/kaos-agents/kaos_agents/`
**Auditor:** Claude Code (automated)

---

## Executive Summary

kaos-agents is largely well-structured with respect to the kaos-llm-core codec stack. The
Signature/Call/RAG infrastructure is used throughout, and there are no hand-rolled JSON parsers
or fence-stripping routines in the production path. The bypasses that exist fall into three
recurring patterns:

1. **Intermediate dataclass wrappers for judge verdicts** — `JudgeVerdict` and `RubricVerdict`
   duplicate the Signature output fields and force an extra `to_dict()` hop that destroys type
   information at the benchmark-harness boundary.
2. **Manual string-to-enum coercion of `intent` in the classifier** — the `ClassifyIntent`
   Signature's `intent` output field is typed `str`, not `IntentType`, so post-hoc `.lower()`,
   `IntentType(...)` conversion, and range clamping have to happen outside the codec.
3. **Dict-flattened findings storage in memory** — verified `Claim`/`Span` objects from the
   RAG pipeline are re-serialized to raw dicts before being written to `MemoryType.FINDINGS`,
   discarding the typed grounding instances.

There are no `TypedDict` usages, no manual fence-stripping, and no `json.loads` calls on LLM
response bodies. The `json.loads` calls that do appear are all wire-protocol deserializers
(SSE, JSONL, VFS persistence, MCP input parsing) — they are appropriate and not bypasses.

---

## Finding 1 — Redundant intermediate dataclasses in benchmark judges (HIGH)

### Files and lines

- `kaos_agents/benchmarks/llm_judge.py:107–120`, `llm_judge.py:192–203`
- `kaos_agents/benchmarks/rubric_judge.py:150–170`, `rubric_judge.py:244–256`

### Pattern

Both judge modules declare a frozen dataclass that exactly mirrors the Signature's output fields,
then immediately convert the typed Pydantic instance into that dataclass, then call `to_dict()`
to return a plain `dict` to callers.

```python
# llm_judge.py:107
@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    correct: bool
    confidence: float
    reasoning: str
    judge_model: str = ""
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

# ...called at llm_judge.py:192
output = invocation.output          # typed Pydantic output
verdict = JudgeVerdict(
    correct=bool(output.correct),   # bool() coercion on a bool field
    confidence=max(0.0, min(1.0, float(output.confidence))),  # float() on float field
    reasoning=str(output.reasoning).strip() or "(no reasoning supplied)",
    ...
)
return verdict.to_dict()            # destroys the dataclass back to dict
```

The `QAJudgeSignature` and `RubricVerdictSignature` already validate `correct`/`passed`
(bool), `confidence` (float with `ge=0.0, le=1.0` constraints), and `reasoning` (str).
Pydantic's codec enforces those constraints before `invocation.output` is available.
The re-coercions (`bool(output.correct)`, `float(output.confidence)`) are therefore
redundant and misleading — they imply the codec output is unvalidated.

The `JudgeVerdict`/`RubricVerdict` dataclasses add two extra fields (`judge_model`,
`judge_cost_usd`, `judge_input_tokens`, `judge_output_tokens`) that are not part of the
Signature. That portion is legitimate. But the three Signature-mirror fields (`correct`/
`passed`, `confidence`, `reasoning`) should be read directly from `invocation.output`
instead of re-packaged.

The public return type of `llm_judge()` and `rubric_judge()` is `dict`, which means every
caller receives an untyped blob. The benchmark harness in `tests/benchmarks/multiformat_e2e.py`
at line 343 has a comment acknowledging this: `# llm_judge returns dict (via JudgeVerdict.to_dict())`.

### Clean replacement

Extend `JudgeVerdict`/`RubricVerdict` to carry only the non-Signature fields (`judge_model`,
cost, token counts), change the return type to `JudgeVerdict`/`RubricVerdict`, and read the
Signature fields directly:

```python
@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Benchmark fields only — Signature output fields read directly from invocation.output."""
    judge_model: str = ""
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

async def llm_judge(...) -> JudgeVerdict:
    ...
    invocation = await judge_call.invoke(...)
    output = invocation.output       # correct: bool, confidence: float, reasoning: str
    return JudgeVerdict(
        judge_model=model,
        judge_cost_usd=float(getattr(usage, "cost_usd", 0.0) or 0.0),
        ...
    )
```

Callers that previously did `verdict["correct"]` would access `output.correct` (the Pydantic
field) and `verdict.judge_cost_usd` (the new dataclass field) instead.

### Severity: HIGH

`to_dict()` is called at every call site, converting the result to an untyped dict. The
benchmark harness and any downstream consumer lose Pydantic validation, IDE autocompletion,
and `ty` static-type checking at the internal API boundary.

---

## Finding 2 — `ClassifyIntent` Signature uses untyped `str` for `intent` output (HIGH)

### File and lines

`kaos_agents/context/classify.py:99–142`

### Pattern

The `ClassifyIntent` Signature declares `intent` as a plain `str` output field:

```python
class ClassifyIntent(Signature):
    ...
    intent: str = OutputField(description="One of: respond, tool_use, research, plan, clarify")
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0")
    reasoning: str = OutputField(description="Brief explanation of the classification")
```

After `call()` returns, the code manually coerces the string into the `IntentType` enum
with a `try/except ValueError` fallback, and clamps the confidence:

```python
raw_intent = result.intent.lower().strip()
try:
    intent_type = IntentType(raw_intent)
except ValueError:
    logger.warning(...)
    intent_type = IntentType.RESPOND

raw_confidence = float(result.confidence)
if not 0.0 <= raw_confidence <= 1.0:
    logger.debug(...)
confidence = max(0.0, min(1.0, raw_confidence))
```

The Signature's `confidence` already maps to a `float` output field — the confidence clamping
outside the codec is redundant (the codec would already have validated `float`). The `intent`
coercion is a real bypass: a Signature output declared as `str` cannot be validated against
the `IntentType` enum by the codec, so the validation moves entirely to post-hoc Python.

This means:
- A malformed intent string (e.g. `"plan_execute"`) silently becomes `IntentType.RESPOND`
  rather than triggering a codec `ValidationRetryExhaustedError` that could prompt the LLM
  to correct itself.
- The retry budget in `Call` is never consumed on malformed intents — the bug is swallowed
  locally.

### Clean replacement

Use `Literal` or a constrained string in the OutputField so the codec enforces the enum:

```python
from typing import Literal

class ClassifyIntent(Signature):
    ...
    intent: Literal["respond", "tool_use", "research", "plan", "clarify"] = OutputField(
        description="Intent category."
    )
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = OutputField(description="Brief explanation.")
```

With `ge=0.0, le=1.0` constraints on `confidence`, the post-call clamping disappears entirely.
With `Literal[...]` on `intent`, an invalid intent causes a codec validation failure and
triggers the Call's `max_retries` retry path instead of silently defaulting.

### Severity: HIGH

Malformed LLM outputs that should trigger structured-output retry instead silently map to
`RESPOND`. This hides provider issues and can cause incorrect routing (e.g. a research question
routed to simple respond) without any log evidence of a Signature validation failure.

---

## Finding 3 — Findings stored as raw dicts, discarding typed `Claim`/`Span` instances (MEDIUM)

### File and lines

`kaos_agents/patterns/research.py:683–702`

### Pattern

After RAG returns an `Answer` with typed `Claim` objects (each carrying `Span` instances
with `source_uri`, `char_span`, `quote`, `page`), the research pattern re-serializes
the grounding data to a hand-built dict before writing to `MemoryType.FINDINGS`:

```python
memory.add(
    MemoryType.FINDINGS,
    finding,
    metadata={
        "claim_type": str(claim.claim_type),
        "statement": claim.statement,
        "confidence": claim.confidence,
        "verified": result.is_verified,
        "sources": [s.source_uri for s in claim.supporting_spans],
        "spans": [
            {
                "source_uri": s.source_uri,
                "quote": s.quote,
                "char_span": list(s.char_span),
                "page": s.page,
            }
            for s in claim.supporting_spans
        ],
    },
)
```

The `Claim` and `Span` types from `kaos_llm_core.signatures.grounding` are Pydantic models
with validated fields (`char_span: tuple[int, int]`, `confidence: float`, `claim_type: ClaimType`).
Converting them to dicts via dict comprehension:

- Loses Pydantic validation on the stored data.
- Makes the `char_span` round-trip fragile: stored as `list(s.char_span)` (a `list[int]`),
  but the original type is `tuple[int, int]`. When retrieved from FINDINGS, the char_span
  is now a `list` not a `tuple`, so any code that tries to re-validate it with
  `Span.model_validate(...)` would need an extra coercion step.
- Loses the `ClaimType` enum on `claim_type` — stored as `str(claim.claim_type)`, i.e.
  the string value of the enum, not a typed instance.

The CLAUDE.md for kaos-agents notes: "Grounding integration. FINDINGS section stores `Claim`
instances with `Span` citations." The actual implementation stores manually-flattened dicts,
not `Claim` instances.

### Clean replacement

Use `claim.model_dump()` (Pydantic's own serializer, which handles enum → str and tuple →
list correctly) as the metadata dict, and deserialize with `Claim.model_validate(item.metadata)`
when reading findings:

```python
memory.add(
    MemoryType.FINDINGS,
    finding,
    metadata={
        "claim": claim.model_dump(mode="json"),  # Pydantic-serialized, round-trip safe
        "verified": result.is_verified,
    },
)
```

This preserves all field types through the round-trip and allows `Claim.model_validate(...)`
on the read side without special-casing `char_span` or `claim_type`.

### Severity: MEDIUM

The stored data is functionally correct for display purposes (the text rendering works), but
the dict shape is fragile. A `Span` deserialized from FINDINGS cannot be round-tripped back
to a typed `Span` without manual coercions, making future consumers of FINDINGS data
(e.g. citation cross-reference, retrieval by span offset) harder to write correctly.

---

## Finding 4 — `apply_refusal_policy` uses `getattr` to duck-type `Answer` instead of `isinstance` (MEDIUM)

### File and lines

`kaos_agents/grounding.py:67–71`

### Pattern

`apply_refusal_policy` detects whether the `GroundedAnswer` is an `Answer` by checking
`getattr(grounded_answer, "kind", None) == "answer"` and reading `confidence` via
`getattr`:

```python
kind = getattr(grounded_answer, "kind", None)
if kind != "answer":
    return grounded_answer, None

confidence = getattr(grounded_answer, "confidence", 1.0)
min_conf = getattr(refusal_policy, "min_confidence", 0.7)
```

This is a duck-type pattern rather than the typed `isinstance(grounded_answer, Answer)`
check that the research pattern uses correctly at line 655. The `getattr` approach silently
defaults `confidence` to `1.0` if the field is absent, which means a malformed or unexpected
type would pass the threshold check and not be collapsed — the refusal policy would appear
to work while actually doing nothing.

The reason for `getattr` is that `Answer` is a lazy import (`[llm]` optional dep). But
`kaos_llm_core` is already imported at the call site in `research.py` (the function is
called inside the `try: from kaos_llm_core...` block), so the lazy-import concern does not
apply there.

### Clean replacement

Import `Answer` at the use site (inside the same guard that already imports `InsufficientEvidence`)
and use `isinstance`:

```python
from kaos_llm_core.signatures.grounding import Answer, InsufficientEvidence

if not isinstance(grounded_answer, Answer):
    return grounded_answer, None
confidence = grounded_answer.confidence   # typed attribute access, no getattr
```

### Severity: MEDIUM

A subtle correctness issue: if an unexpected type with a `kind` attribute but no `confidence`
field reaches this function, the default `1.0` causes it to silently pass the policy check.
The `isinstance` guard eliminates this class of error.

---

## Finding 5 — `RunState` / `AgentSnapshot` use hand-rolled `to_dict()` / `from_dict()` instead of Pydantic (MEDIUM)

### File and lines

`kaos_agents/interrupts.py:148–180`, `interrupts.py:220–244`

### Pattern

`AgentSnapshot` is a `@dataclass(frozen=True, slots=True)` with a manual `to_dict()` /
`from_dict()` pair. `RunState` (also a frozen dataclass) has a hand-rolled `to_json()` /
`from_json()` that manually builds the dict, including special-casing for `pending_tool_call`
and `agent_config`:

```python
# interrupts.py:220
def to_json(self) -> str:
    data: dict[str, Any] = {
        "run_id": self.run_id,
        "session_id": self.session_id,
        "event_count": self.event_count,
        ...
    }
    if self.pending_tool_call is not None:
        data["pending_tool_call"] = {
            "call_id": self.pending_tool_call.call_id,
            "tool_name": self.pending_tool_call.tool_name,
            "arguments": [list(pair) for pair in self.pending_tool_call.arguments],
            "reason": self.pending_tool_call.reason,
        }
    ...
    if self.agent_config is not None:
        data["agent_config"] = self.agent_config.to_dict()
    return json.dumps(data, separators=(",", ":"))
```

`RunState` and `AgentSnapshot` are pure configuration snapshots — they contain no LLM-output
fields and no grounding primitives — so this is not a bypass of the codec stack per se.
However, the hand-rolled serialization duplicates what `pydantic.BaseModel.model_dump(mode="json")`
/ `model_validate_json()` would give for free, and the `from_dict` for `AgentSnapshot` uses
plain `data.get(...)` without field validation. A field that was renamed (e.g. `max_tools` →
`tool_limit`) would silently produce a wrong default rather than raising a validation error.

### Clean replacement

Convert `AgentSnapshot` and `RunState` to `pydantic.BaseModel` (or add a thin Pydantic
adapter around the frozen dataclasses). Serialization becomes `model.model_dump_json()` and
`Model.model_validate_json()`, eliminating the manual dict construction and the field-presence
`data.get(...)` calls.

### Severity: MEDIUM

Forward-compatibility risk rather than an active data loss. Renamed or added fields silently
produce wrong defaults. The immediate impact is low because `AgentSnapshot` and `RunState`
are versioned VFS artifacts whose format is controlled, but the pattern will cause silent
bugs when the schema evolves.

---

## Finding 6 — `_ExplainTurn` in `cli_chat.py` is a mutable dataclass with dict-valued fields (LOW)

### File and lines

`kaos_agents/cli_chat.py:62–106`, `cli_chat.py:138–159`

### Pattern

`_ExplainTurn` is a mutable `@dataclass` (not `frozen=True, slots=True`) with fields typed
as `list[dict[str, Any]]`:

```python
@dataclass
class _ExplainTurn:
    turn_index: int
    user_message: str
    intent: str = ""
    intent_confidence: float = 0.0
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
```

The `citations` list is populated by building dicts from `CitationFound` event fields
(cli_chat.py:1317–1326). `CitationFound` is already a typed frozen dataclass — the dict
construction discards the typed event and loses `ty` checking at the UI boundary.

Additionally, `_explain_to_dict` (line 138) is a free function that manually replicates the
dataclass fields rather than using `dataclasses.asdict(turn)`. This is a maintenance hazard:
adding a field to `_ExplainTurn` requires updating three places (the dataclass, the populate
site, and `_explain_to_dict`).

### Clean replacement

For citations and other event-derived subrecords, either keep the typed event instances in the
`_ExplainTurn` (using `list[CitationFound]` etc.) and serialize at the JSON dump boundary
with `serialize_event`, or define a small frozen Pydantic model for the explain record so
`model_dump(mode="json")` replaces `_explain_to_dict`. Given that `_ExplainTurn` is private
CLI scaffolding, converting to `frozen=True` and using `dataclasses.asdict()` directly would
be a proportionate improvement over the current free-function approach.

### Severity: LOW

`_ExplainTurn` is private, CLI-only, and never crosses a service boundary. The data loss
(typed event → dict) affects only the `/explain` command display and the `--explain <file>`
dump, not the agent's execution path or grounding integrity.

---

## Finding 7 — `mcp_extract.py` inbound schema and corpus parsing uses `json.loads` with no Pydantic validation layer (LOW)

### File and lines

`kaos_agents/mcp_extract.py:92–104`, `mcp_extract.py:404–410`, `mcp_extract.py:566–584`

### Pattern

MCP tool inputs arrive as JSON strings that must be parsed before they reach
`ExtractionSchema.from_dict()` or `Cited[Any].model_validate()`. The parse step itself
is correct (`json.loads` + error message). The three sites are:

1. **Schema input** (line 92): `spec = json.loads(schema_json)` → then passed to
   `_normalize_schema_dict()` → then `ExtractionSchema.from_dict(spec)`. The
   `ExtractionSchema.from_dict()` call is the Pydantic validation layer; the
   `_normalize_schema_dict()` function is a reasonable LLM-error-correction shim that
   fixes `"fields"` → `"columns"` etc. before Pydantic sees it.
2. **Corpus input** (line 404): `corpus = json.loads(corpus_json)` → then validated manually
   with `isinstance` checks. Could use a Pydantic `RootModel[dict[str, str]]` validator
   instead of the manual `isinstance` loop.
3. **Claim input** (line 566): `claim = json.loads(claim_json)` → `Cited[Any].model_validate(claim)`.
   This is correct: JSON parse → Pydantic model_validate.

Sites 1 and 3 are correctly connected to the codec/Pydantic layer. Site 2 (corpus validation)
has a manual isinstance loop where a `RootModel` would be cleaner, but it is functionally
equivalent.

### Severity: LOW

These are MCP wire-input deserializers, not LLM response parsers. The patterns are appropriate
for the context. Site 2's manual loop is the only sub-optimal choice, and only aesthetically.

---

## Non-Findings (False Positives Eliminated)

The following items were checked and found to be correct usage:

- **`events.py` — `json.loads` in `deserialize_event_json`**: Wire-protocol deserializer
  for the SSE/JSONL event stream. `AgentEvent` dataclasses are not LLM outputs; they are
  the agent's own serialized events. `deserialize_event` uses a typed registry, not raw dicts.

- **`interrupts.py` — `json.loads` in `RunState.from_json`**: VFS persistence
  deserializer for run-state snapshots. Not an LLM response.

- **`memory/store.py` — `json.loads` in `SessionStore.load`**: VFS persistence
  deserializer for session memory. Not an LLM response.

- **`runner.py` — `_json.loads(raw.decode())`**: VFS snapshot load. Not an LLM response.

- **`classify.py` — `Call(ClassifyIntent, ...)` usage**: The Call program is correctly
  used. The bypass is only in the output field type declaration (Finding 2).

- **`planning/expand.py` — `GeneratedStep.model_validate(raw)`**: This is correct. Raw
  LLM output (a list of dicts) is validated through a Pydantic model before conversion to
  the plan `Step` type.

- **`mcp_extract.py` — `Cited[Any].model_validate(claim)`**: Correct use of Pydantic
  model_validate for claim deserialization.

- **`patterns/research.py` — RAG invocation via `rag.invoke(...)`**: Correct. The RAG
  program's Signature/codec stack handles all output validation. `result.grounded_answer`
  is a typed `Answer | InsufficientEvidence` union, not a raw dict.

- **Benchmark scripts in `tests/benchmarks/`**: The `asdict()` calls in benchmark scripts
  (`beir_eval.py`, `scale_e2e.py`, etc.) are JSON output serializers for benchmark results
  — not internal API boundaries. These are appropriate and are excluded from the QA `ty check`
  by policy.

---

## Summary Table

| # | File | Lines | Pattern | Severity |
|---|------|-------|---------|----------|
| 1 | `benchmarks/llm_judge.py` | 107–120, 192–203 | Redundant dataclass + `to_dict()` over Signature output | HIGH |
| 1 | `benchmarks/rubric_judge.py` | 150–170, 244–256 | Same pattern, `RubricVerdict` | HIGH |
| 2 | `context/classify.py` | 99–142 | `intent: str` OutputField; manual `IntentType()` coercion post-call | HIGH |
| 3 | `patterns/research.py` | 683–702 | `Claim`/`Span` re-serialized to raw dicts in FINDINGS | MEDIUM |
| 4 | `grounding.py` | 67–71 | `getattr` duck-typing instead of `isinstance(answer, Answer)` | MEDIUM |
| 5 | `interrupts.py` | 148–180, 220–244 | Hand-rolled `to_dict`/`from_dict` for `AgentSnapshot`/`RunState` | MEDIUM |
| 6 | `cli_chat.py` | 62–106, 1317–1326 | Mutable `_ExplainTurn` with `list[dict]` typed citation fields | LOW |
| 7 | `mcp_extract.py` | 404–410 | Corpus validated with manual `isinstance` loop vs. Pydantic `RootModel` | LOW |

**By severity:** HIGH: 3 instances across 2 modules. MEDIUM: 3 instances. LOW: 2 instances.
