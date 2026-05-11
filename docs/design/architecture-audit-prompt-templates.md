# Prompt-Template Audit: kaos-agents

**Date:** 2026-05-06
**Scope:** `/kaos-agents/kaos_agents/` — all agent, pattern, planning, context, benchmark, memory, and recipe files.
**Goal:** Identify hand-rolled prompts that bypass the kaos-llm-core `Signature + Call` machinery and flag opportunities for typed Signatures, optimizer compatibility, and `Example`-based few-shot bootstrapping.

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| HIGH     | 3     | Multi-paragraph module-level prompt constants driving core inference paths |
| MEDIUM   | 7     | Prompt fragments concatenated outside the Signature docstring; `instructions=` strings on `Call()`; f-string injection into instructions at call sites |
| LOW      | 4     | Single-sentence defaults / convenience strings; test/CLI pass-throughs |

**Total offenses: 14**

The codebase has adopted Signatures broadly and correctly in the benchmark judges, planning primitives, retrieval helpers, memory summarizer, and intent classifier — those are well-formed. The remaining hand-rolled patterns fall into two clusters: (1) the three large multi-paragraph "system persona" constants that drive the ReAct escalation path and the retrieval sub-agent, and (2) scattered `instructions=` string literals passed to `Call()` that duplicate guidance already expressible in a Signature docstring or as optimizer-tunable fields.

---

## Finding 1 — `_RESEARCH_REACT_INSTRUCTION` (HIGH)

**File:line:** `kaos_agents/patterns/research.py:56–93`

**Snippet:**
```python
_RESEARCH_REACT_INSTRUCTION = """\
You are answering a question about a document corpus using retrieval tools.

STRATEGY:
1. Search with kaos-retrieval-bm25 using key terms from the question.
2. Look at the results. Check the expansion_assessment signal.
3. If results look relevant and cover the question:
   → Call kaos-retrieval-answer with the question and the relevant passage texts.
4. If results are noisy or miss the topic:
   → Try more specific terms, or use kaos-retrieval-hyde for vocabulary bridging.
5. If kaos-retrieval-answer says "insufficient evidence":
   → Use the what_would_resolve hint to search again with different terms.
6. After 2-3 search attempts, if you still can't find evidence:
   → State clearly that the corpus doesn't contain the answer.

WHEN A CITATION SEEMS INCOMPLETE OR UNCLEAR:
...
IMPORTANT:
- Always search BEFORE answering. Never answer from your training data.
...
"""
```

This 38-line constant is the primary system instruction for the RAG-escalation ReAct loop. It is baked into prose, concatenated via f-string onto `self._instructions` at line 469, and silently mutates instance state to thread it through to `ChatAgent._handle_tool_use_streaming`. Because it lives outside any Signature, it is invisible to `get_instruction()`, cannot be tuned by `InstructionOptimizer` or `MIPROv2`, and cannot accept `Example` demonstrations to bootstrap few-shot retrieval strategies.

**The Signature it should become:**
```python
class RetrievalReActTask(Signature):
    """Answer a question about a document corpus using retrieval tools.

    STRATEGY: (1) Search with kaos-retrieval-bm25. (2) Check expansion_assessment.
    (3) If relevant → call kaos-retrieval-answer. (4) If noisy → try kaos-retrieval-hyde.
    (5) If insufficient evidence → use what_would_resolve hint for another search round.
    (6) After 2-3 rounds without evidence → refuse. Never answer from training data.
    When a citation is incomplete, expand with kaos-content-context-window before answering.
    """

    question: str = InputField(description="The user's research question")
    corpus_outline: str = InputField(
        description="Table of contents / document list for the corpus (may be empty)"
    )
    conversation_context: str = InputField(description="Recent conversation history")
    answer: str = OutputField(description="Grounded answer with cited passages, or explicit refusal")
```

The `corpus_outline` currently arrives via f-string injection into the instructions string (`outline_block`). Making it an explicit `InputField` gives the optimizer a lever to learn *when* the outline is useful.

---

## Finding 2 — `_RETRIEVAL_INSTRUCTIONS` (HIGH)

**File:line:** `kaos_agents/retrieval_agent.py:31–66`

**Snippet:**
```python
_RETRIEVAL_INSTRUCTIONS = """\
You are a document retrieval specialist. Your job is to find the most
relevant documents from a corpus for a given search query.

STRATEGY:
1. Check the corpus first with kaos-retrieval-corpus-info ...
2. Run a plain BM25 keyword search (kaos-retrieval-bm25).
3. Look at the results and the expansion_assessment signal. If results
   look good and expansion is not suggested, STOP.
4. If results are missing important aspects:
   a. Use kaos-retrieval-synonyms to find alternative terminology.
...
IMPORTANT:
- Do NOT run every tool on every query. Simple queries need only BM25.
...
"""
```

35-line instruction block passed directly as `instructions=_RETRIEVAL_INSTRUCTIONS` to `Agent.create()`. The retrieval agent uses a `ChatAgent` ReAct loop internally; the instruction bypasses Signature machinery entirely. The retrieval strategies it encodes (BM25 → synonyms → HyDE → rerank → evaluate) are the exact step sequence that `Refine` or a multi-step `Program` would express typed. There is also no `Example`-based few-shot — no demonstrations of a successful BM25-only query or a successful HyDE escalation.

**The Signature it should become:**
```python
class RetrievalTask(Signature):
    """Find the most relevant documents from a corpus for a search query.

    Preferred strategy: (1) BM25 keyword search; (2) if expansion_assessment
    suggests a gap, try synonym expansion; (3) only use HyDE for large
    vocabulary mismatch; (4) rerank only when ≥10 candidates exist;
    (5) evaluate coverage; stop after 2-3 rounds. Report what was found
    and what gaps remain.
    """

    search_query: str = InputField(description="Natural-language description of what to find")
    corpus_info: str = InputField(
        description="Brief description of the corpus (document count, domain, vocabulary)"
    )
    retrieved_documents: str = OutputField(
        description=(
            "Summary of retrieved documents: count found, strategies used, "
            "key document previews (first 200 chars each), coverage gaps"
        )
    )
```

With this shape, `BootstrapFewShot` can learn from labeled retrieval sessions which tool sequence succeeds on which query type.

---

## Finding 3 — `_CLASSIFY_INSTRUCTION` separate from `ClassifyIntent` docstring (MEDIUM)

**File:line:** `kaos_agents/context/classify.py:34–50`

**Snippet:**
```python
_CLASSIFY_INSTRUCTION = """Classify the user's intent into one of these categories:

- respond: Simple conversational response, greeting, or acknowledgment. No tools needed.
- tool_use: The user wants to perform an action that requires calling tools ...
- research: The user is asking a question about loaded documents ...
- plan: The user wants a multi-step workflow ...
- clarify: The user's request is ambiguous ...

Consider the conversation history and available context when classifying.
If documents are loaded and the question relates to their content, prefer "research".
...
When in doubt between tool_use and research, prefer tool_use (it's more general).
"""
```

Used at line 119: `call = Call(ClassifyIntent, model=model, instructions=_CLASSIFY_INSTRUCTION)`. This is the correct Signature pattern *except* that the guidance is in a separate `instructions=` string rather than in the `ClassifyIntent` docstring. The Signature at line 99 has only a bare one-liner: `"""Classify the user's intent for routing to the appropriate handler."""`. The eleven-line guidance block is thus invisible to `get_instruction(ClassifyIntent)` and cannot be tuned by `InstructionOptimizer` independently of the Signature shape.

**Corrected form:** Move the entire `_CLASSIFY_INSTRUCTION` content into the `ClassifyIntent` docstring and drop the `instructions=` kwarg:
```python
class ClassifyIntent(Signature):
    """Classify the user's intent into one of: respond, tool_use, research, plan, clarify.

    - respond: simple greeting or acknowledgment; no tools needed.
    - tool_use: user wants an action requiring tool calls.
    - research: question about loaded documents requiring retrieval.
    - plan: multi-step sequential workflow.
    - clarify: request is ambiguous; more information needed.

    If documents are loaded and the question relates to their content, prefer research.
    When in doubt between tool_use and research, prefer tool_use (more general).
    """
    message: str = InputField(...)
    conversation_context: str = InputField(...)
    intent: str = OutputField(...)
    confidence: float = OutputField(...)
    reasoning: str = OutputField(...)
```

---

## Finding 4 — `_REACT_INSTRUCTION` module constant (MEDIUM)

**File:line:** `kaos_agents/patterns/chat.py:33–35`

**Snippet:**
```python
_REACT_INSTRUCTION = (
    "Complete the user's request using the available tools. Be thorough and cite your sources."
)
```

Used at lines 211–215:
```python
react_instructions = (
    f"{self._instructions}\n\n{_REACT_INSTRUCTION}"
    if self._instructions
    else _REACT_INSTRUCTION
)
react = ReAct(ToolTask, tools=tools, ..., instructions=react_instructions)
```

`ToolTask` at line 156–161 has a docstring `"""Complete the user's request using the available tools."""` — nearly identical to `_REACT_INSTRUCTION`. The constant and docstring say the same thing, but only one of them is optimizer-visible. The f-string concatenation of `self._instructions + "\n\n" + _REACT_INSTRUCTION` is also a runtime instruction-merging pattern that should instead be a Signature field (e.g., an `agent_persona: str = InputField(...)` that the `Call` or `ReAct` receives as data, not as a prompt mutation).

**Corrected form:** Move the guidance into `ToolTask`'s docstring; remove the concatenation:
```python
class ToolTask(Signature):
    """Complete the user's request using the available tools. Be thorough and cite your sources."""

    question: str = InputField(description="The user's request")
    context: str = InputField(description="Conversation context")
    agent_persona: str = InputField(
        description="The agent's role and identity (empty string for default assistant)"
    )
    answer: str = OutputField(description="Your final answer to the user")
```

---

## Finding 5 — `_DEFAULT_RESPOND_INSTRUCTION` and inline `Respond` docstring (MEDIUM)

**File:line:** `kaos_agents/agent.py:55` and `kaos_agents/agent.py:582`

**Snippet (constant):**
```python
_DEFAULT_RESPOND_INSTRUCTION = "You are a helpful assistant."
```

**Snippet (Signature):**
```python
class Respond(Signature):
    """You are a helpful assistant. Respond to the user's message."""
    message: str = InputField(...)
    conversation_history: str = InputField(...)
    response: str = OutputField(...)
```

Used at lines 601–603:
```python
instructions = self._instructions or _DEFAULT_RESPOND_INSTRUCTION
if extra_instruction:
    instructions = f"{instructions} {extra_instruction}"
call = Call(Respond, model=..., instructions=instructions)
```

Three problems: (1) the agent persona is threaded through `instructions=` rather than through a typed `InputField`, so it varies per-call but is not part of the Signature schema; (2) `extra_instruction` is an arbitrary f-string-concatenated string (e.g., `"A tool-calling attempt failed: {exc}. Respond helpfully without tools."`) that changes the system prompt dynamically; (3) `_DEFAULT_RESPOND_INSTRUCTION` is a bare string not accessible to `get_instruction()`.

**Corrected form:**
```python
class Respond(Signature):
    """Respond to the user's message helpfully and accurately.

    When extra_context is non-empty, it contains situational guidance
    (e.g., that a prior tool-call failed) that should inform the response.
    """

    message: str = InputField(description="The user's message")
    conversation_history: str = InputField(description="Recent conversation for context")
    agent_persona: str = InputField(
        description="Agent identity / role (e.g., 'legal research assistant')"
    )
    situational_guidance: str = InputField(
        description="Optional situational guidance for this specific turn (may be empty)"
    )
    response: str = OutputField(description="Your response to the user")
```

---

## Finding 6 — `expand()` instructions f-string with `failure_context` injection (MEDIUM)

**File:line:** `kaos_agents/planning/expand.py:66–97`

**Snippet:**
```python
failure_context = ""
if prior_failures:
    failure_context = f"\n\nPREVIOUS FAILED ATTEMPTS (avoid repeating these):\n{prior_failures}"

instructions = (
    "Generate a concrete, actionable plan using ONLY the listed tools. "
    "Each step must reference a specific tool by its exact name, or "
    "use 'llm' for reasoning steps. "
    "Keep plans focused — use the minimum number of steps needed. "
    f"Maximum {max_steps} steps."
    f"{failure_context}"
)
call = Call(PlanExpand, model=model, instructions=instructions)
```

The `instructions=` string is built entirely outside the Signature, including a runtime-varying `max_steps` integer and a `failure_context` block from the previous ReAct trajectory. `PlanExpand`'s docstring says only `"""Generate a structured plan to accomplish a goal."""`. The `prior_failures` text is the precise input an `Example`-based bootstrapper or `BootstrapFewShot` optimizer should learn from — but because it arrives via prompt concatenation rather than as an `InputField`, the optimizer cannot reason about it.

**Corrected form:**
```python
class PlanExpand(Signature):
    """Generate a concrete, minimum-step plan using ONLY the listed tools.

    Each step must name a specific tool by its exact name, or use 'llm'
    for a pure reasoning step. Never hallucinate tool names. Keep plans
    as short as possible while still accomplishing the goal.
    """

    goal: str = InputField(description="The goal to accomplish")
    tools: str = InputField(description="Available tools and their descriptions")
    context: str = InputField(description="Conversation context and prior knowledge")
    prior_failures: str = InputField(
        description="Previous failed plan attempts to avoid repeating (may be empty)"
    )
    max_steps: int = InputField(description="Maximum number of steps to generate")
    steps: list[dict[str, Any]] = OutputField(description="List of plan steps ...")
```

---

## Finding 7 — `evaluate_semantic()` inline `instructions=` (MEDIUM)

**File:line:** `kaos_agents/planning/evaluate.py:125–129`

**Snippet:**
```python
call = Call(
    EvalSig,
    model=model,
    instructions="Judge whether the result satisfies the expected output. "
    "Be generous — partial matches count. Extract any key facts.",
)
```

`EvalSig`'s docstring (`"""Judge whether a result satisfies an expectation."""`) is minimal. The actual guidance — "be generous", "partial matches count", "extract key facts" — lives in `instructions=` outside the Signature. This means `InstructionOptimizer` would tune the docstring but the behavior-driving guidance would remain untouched.

**Corrected form:** Move guidance into the docstring and drop `instructions=`:
```python
class EvalSig(Signature):
    """Judge whether a result satisfies an expected output description.

    Be generous: partial matches count. A result that provides most of
    what was expected, with some gaps, still matches. Extract any key
    facts found in the result, even if they don't match the expectation.
    """
    result_text: str = InputField(...)
    expected_description: str = InputField(...)
    additional_context: str = InputField(...)
    matched: bool = OutputField(...)
    confidence: float = OutputField(...)
    reasoning: str = OutputField(...)
    new_facts: list[str] = OutputField(...)
```

---

## Finding 8 — `summarize_items()` inline `instructions=` (MEDIUM)

**File:line:** `kaos_agents/memory/summarize.py:70–77`

**Snippet:**
```python
call = Call(
    SummarizeMemory,
    model=model,
    instructions=(
        "Summarize the following memory items concisely. "
        "Preserve key facts, names, dates, numbers, and decisions. "
        "Drop greetings, filler, and redundant phrasing. "
        "Use bullet points for multiple distinct facts."
    ),
)
```

`SummarizeMemory`'s docstring is `"""Summarize memory items into a concise summary preserving key facts."""`. The four-sentence instruction block carries behavioral guidance that is optimizer-invisible. The `target_length` field also arrives as a free-form string (`"approximately 800 characters (200 tokens)"`) — this is a format-baked field that could be a structured `InputField(description="Target character count")` accepting an `int`.

**Corrected form:**
```python
class SummarizeMemory(Signature):
    """Summarize memory items concisely. Preserve key facts, names, dates,
    numbers, and decisions. Drop greetings, filler, and redundant phrasing.
    Use bullet points when there are multiple distinct facts.
    """

    content: str = InputField(description="The content to summarize")
    section_type: str = InputField(description="What kind of memory this is")
    target_chars: int = InputField(description="Target summary length in characters")
    summary: str = OutputField(description="Concise summary preserving key facts")
```

---

## Finding 9 — `ReflectOnCoverage`, `GeneratePseudoDocument`, `SuggestQueries` inline Signatures (MEDIUM)

**File:line:** `kaos_agents/context/retrieval.py:612–622`, `827–833`, `886–899`

These three Signatures are defined inside function bodies (`_reflect_on_coverage`, `_generate_hyde_passage`, `_generate_llm_queries`). The definitions themselves are structurally correct, but the placement inside function bodies means:

1. They are not importable or reusable by external code.
2. `get_instruction()` cannot be called on them by an optimizer without first calling the enclosing function — which makes a live LLM call as a side effect.
3. There is no way to attach labeled `Example` instances to bootstrap few-shot behavior without modifying the function body.
4. The `SuggestQueries` docstring bakes a five-item enumeration of vocabulary-expansion strategies into prose that a fine-tuned `InstructionOptimizer` run could improve but cannot access.

**Corrected form:** Promote all three to module-level:
```python
class ReflectOnCoverage(Signature):
    """Evaluate whether retrieved documents cover all aspects of a search query.
    Identify what topics, subtopics, or perspectives are MISSING. Generate
    2-3 highly targeted queries to fill those gaps. Return an empty list
    if the current results already provide good coverage.
    """
    original_query: str = InputField(...)
    document_summaries: str = InputField(...)
    gap_queries: list[str] = OutputField(...)

class GeneratePseudoDocument(Signature):
    """Generate a hypothetical expert-vocabulary excerpt relevant to the query.
    Use precise technical jargon, acronyms, and domain terminology that would
    appear in real source documents — not the simplified language of the query.
    Write 150-250 words.
    """
    search_query: str = InputField(...)
    hypothetical_passage: str = OutputField(...)

class SuggestAlternativeQueries(Signature):
    """Generate 3-5 alternative search queries using substantially different
    vocabulary from the original. Consider: technical vs. plain language,
    acronyms vs. expanded forms, synonyms, related concepts, hypernyms/hyponyms.
    Each alternative must use different vocabulary from the original and from
    each other.
    """
    original_query: str = InputField(...)
    results_found: int = InputField(...)
    alternative_queries: list[str] = OutputField(...)
```

---

## Finding 10 — `_simple_respond` fallback `extra_instruction` f-string (MEDIUM)

**File:line:** `kaos_agents/patterns/research.py:910–914`

**Snippet:**
```python
extra_instruction=(
    f"A document Q&A attempt failed: {exc}. "
    "Answer the question using what you know, but note that you couldn't "
    "verify your answer against the source documents."
),
```

And in `agent.py:527`:
```python
extra_instruction="The user's request is ambiguous. Ask a clarifying question.",
```

These are f-strings injected into the running `instructions=` of a `Call(Respond, ...)` at line 603. The runtime exception message (`{exc}`) becomes part of the system prompt — this is potentially unsafe (exception text can contain user-controlled content that manipulates the prompt) and is not representable in a Signature. The correct fix is the typed `situational_guidance: str = InputField(...)` approach from Finding 5, with the exception message appearing in the *user-turn* context rather than the system prompt.

---

## Finding 11 — `_DEFAULT_RESPOND_INSTRUCTION` used as `Agent.instructions` default (LOW)

**File:line:** `kaos_agents/agent.py:55`, `kaos_agents/config.py:58`, `kaos_agents/interrupts.py:94`

```python
# agent.py
_DEFAULT_RESPOND_INSTRUCTION = "You are a helpful assistant."
# config.py
instructions: str = "You are a helpful assistant."
# interrupts.py
instructions: str = "You are a helpful assistant."
```

These are placeholders at the agent-configuration layer, not LLM call sites. They are intentionally minimal and user-replaceable. The `Agent.instructions` field is the correct carrier for these — they are not offenses against the Signature machinery. However, the three independent copies of `"You are a helpful assistant."` should be a single constant in `_constants.py` to avoid drift.

**Severity:** LOW — no optimizer impact; purely a DRY issue.

---

## Finding 12 — `cli_chat.py` default instructions (LOW)

**File:line:** `kaos_agents/cli_chat.py:1044`

```python
instructions=args.instructions or "You are a helpful assistant.",
```

CLI pass-through; user provides the actual instruction via `--instructions`. The fallback is a sensible default for interactive use. No optimizer impact.

---

## Finding 13 — `runner.py` placeholder (LOW)

**File:line:** `kaos_agents/runner.py:14`

```python
instructions="You are a research assistant.",
```

A hardcoded default in a test/example `Runner` construction. No production inference path is driven by this string. Should be a constant or come from `KaosAgentSettings`.

---

## Finding 14 — Recipe JSON files — prose steps with no Signature mapping (LOW)

**Files:** `kaos_agents/recipes/contract-extraction.json`, `corpus-qa.json`, `edgar-research.json`, `federal-register-research.json`, `summarization.json`

The five core recipes describe multi-step workflows as JSON arrays of `description` / `tool` / `note` objects. They map structurally to a sequence of `Signature + Call` invocations but the mapping is not realized in code — the step descriptions are prose notes for the planner LLM to interpret, not typed field contracts. The extraction recipes in `kaos_agents/recipes/extraction/` are significantly better (they carry typed `ExtractionSchema` payloads), but the core five recipes have no schema.

A minimal improvement: each recipe step should declare the `input_fields` and `output_fields` it consumes and produces so `PlanExpand` can verify tool-output compatibility at plan-time rather than discovering it at execution-time. This also enables recipe steps to serve as labeled `Example` instances for `BootstrapFewShot` on the planner.

**Severity:** LOW for the extraction recipes (already have schemas); LOW-to-MEDIUM for the five core workflow recipes (no schema at all).

---

## What is Already Done Well

The following files correctly implement Signature-first patterns and should be used as reference implementations:

- **`kaos_agents/benchmarks/llm_judge.py`** — `QAJudgeSignature` is a module-level Signature with detailed multi-paragraph docstring guidance, typed `InputField`/`OutputField` with constraints (`ge`, `le`), and usage via `Call.invoke()` with cost attribution. This is the gold standard.
- **`kaos_agents/benchmarks/rubric_judge.py`** — `RubricVerdictSignature` follows the same pattern, with numbered decision rules in the docstring.
- **`kaos_agents/planning/expand.py`** — `PlanExpand` uses Signature + `Call`; the only issue is that `instructions=` carries guidance that belongs in the docstring (Finding 6).
- **`kaos_agents/planning/evaluate.py`** — `EvalSig` uses Signature correctly; `instructions=` issue only (Finding 7).
- **`kaos_agents/planning/act.py`** — `LLMStep` is minimal but correct.
- **`kaos_agents/memory/summarize.py`** — `SummarizeMemory` is correct structurally; `instructions=` externalization is the only issue (Finding 8).
- **`kaos_agents/context/classify.py`** — `ClassifyIntent` is correct; guidance split between docstring and `instructions=` is the only issue (Finding 3).
- **`kaos_agents/context/doc2query.py`** — `PredictDocumentQueries` is module-boundary-correct and has a rich docstring.

---

## Optimizer Readiness Gap

None of the Signatures in `kaos-agents` ship with `Example` instances, and none of the Call sites pass `examples=[...]`. This means `BootstrapFewShot` and `MIPROv2` have zero labeled demonstrations to work from. The three high-severity items (Findings 1–2, partially Finding 3) are the highest-leverage targets: the retrieval strategy and RAG-escalation paths are exactly the kind of behavioral policy that few-shot demonstrations can dramatically improve. A set of 5–10 labeled `(question, corpus_outline, retrieved_context) → answer` examples for `RetrievalReActTask` and `RetrievalTask` would make those Signatures optimizer-compatible for the first time.
