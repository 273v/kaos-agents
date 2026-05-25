# kaos-agents design deviation report - 2026-05-24

## Scope

This report reviews current `kaos-agents` against the design center shared by
`kaos-agents` and `kaos-llm-core`.

The design target is:

- `kaos-agents` is the agent runtime above `kaos-llm-core`, not a parallel LLM
  programming framework. Its README places it above `kaos-llm-core`; `CLAUDE.md`
  says ReAct is the inner loop and the agent is the outer loop.
- LLM behavior should be expressed as typed `Signature` contracts, composable
  `Program` objects, codecs, invocations, traces, metrics, examples, and
  optimizers. `kaos-llm-core` treats those as public contracts.
- Prompt and message formats are contracts, but hand-maintained long prompt
  strings should be minimized. When instruction tuning is needed, it should
  have metrics, examples, recorded trials, and optimizer support.
- Dynamic context should be passed as typed input fields, not spliced into
  system instructions.
- Any LLM call path that matters for runtime behavior should use
  `.invoke()` and preserve `Invocation.usage`, trace, and error metadata.

Prior audits from 2026-05-06 are partially stale. Several earlier findings are
fixed now, especially raw provider bypasses, many bare `Call.__call__` usages,
and several structured-output gaps. The remaining deviations are more specific:
long instruction surfaces, mutable prompt state, untyped nested outputs,
manual JSON parsing around LLM output, and missing usage propagation after
some `.invoke()` calls.

## Executive Summary

The transport boundary is still mostly clean. I found no direct provider SDK
calls in `kaos_agents`; provider-facing work still routes through
`kaos-llm-core` / `kaos-llm-client`. Mentions of OpenAI, Anthropic, or Google
are comments, model names, docs, or schema-error handling.

The main design drift is one level above transport:

1. Long prompt constants still drive core agent behavior in Chat, Research, and
   RetrievalAgent.
2. Research escalation mutates `self._instructions` to inject a retrieval
   policy and corpus outline.
3. Some LLM helpers still call a program through the output-only surface,
   losing invocation usage and trace metadata.
4. Several structured outputs are `list[dict]` or JSON strings plus manual
   validation instead of nested Pydantic output models.
5. Several `Signature` classes are local to functions, which weakens stable
   identity, testability, serialization, and optimizer workflows.
6. Plan execution still constructs monolithic prompts for LLM steps and tool
   argument synthesis.
7. Critic behavior has accreted very large rubric/instruction strings instead
   of smaller typed checks, rubric specs, examples, and optimization harnesses.

## Severity Key

- P0: contract or budget behavior is wrong.
- P1: major design drift likely to cause brittle agent behavior.
- P2: maintainability or optimizer-readiness problem that will compound.
- P3: lower-risk cleanup or guardrail.

## Findings

### P1: Chat ReAct still depends on a long instruction constant

Evidence:

- `kaos_agents/patterns/chat.py:57` defines `_REACT_INSTRUCTION`, 1219 chars.
- `kaos_agents/patterns/chat.py:540-565` concatenates caller instructions with
  that constant and passes it to `ReAct(..., instructions=react_instructions)`.

Why this deviates:

The tool-use policy is a load-bearing behavior contract, but it lives as a raw
string outside the `ToolTaskSignature` schema and outside any optimizer/eval
loop. The policy mixes several concepts: source citation, stopping discipline,
aggregation behavior, and refusal behavior. Because it is appended to arbitrary
agent instructions, there is no typed boundary between persona, task inputs,
tool policy, and safety/refusal policy.

Improvement:

- Move invariant task shape into a richer `ToolTaskSignature` or a dedicated
  `ToolUseProgram`.
- Represent dynamic parts as fields, for example `task`, `context`,
  `available_tools_summary`, `retrieval_state`, `aggregation_required`, and
  `refusal_policy`.
- Keep the system instruction short and stable.
- Add labelled eval cases for stop-vs-search, aggregation, and refusal.
- Use an optimizer or at least recorded examples to tune instruction variants.

### P1: Research escalation mutates prompt state and injects corpus outline into system instructions

Evidence:

- `kaos_agents/patterns/research/agent.py:76` defines
  `_RESEARCH_REACT_INSTRUCTION`, 1992 chars.
- `kaos_agents/patterns/research/agent.py:560-565` builds `outline_block` and
  assigns a new value to `self._instructions`.
- `kaos_agents/patterns/research/agent.py:573-578` then calls the parent
  Chat dispatch and restores the saved instructions in `finally`.

Why this deviates:

This is mutable global-ish state on the agent object used as a prompt transport
mechanism. It also makes the corpus outline a system-instruction suffix instead
of typed data. The resulting behavior is hard to test, hard to optimize, and
fragile under future concurrency or nested delegation.

Improvement:

- Replace the escalation branch with a dedicated `ResearchEscalationProgram`.
- Pass `corpus_outline`, `failed_answer`, `what_would_resolve`, and
  `retrieval_history` as typed input fields.
- Keep retrieval strategy as a signature/program contract with examples.
- Make "always search before answering" an evaluated policy, not just a prompt
  sentence.

### P1: RetrievalAgent is a long-prompt sub-agent rather than an optimized retrieval program

Evidence:

- `kaos_agents/retrieval_agent.py:31-66` defines `_RETRIEVAL_INSTRUCTIONS`,
  1764 chars.
- `kaos_agents/retrieval_agent.py:94-99` creates an `Agent` with
  `instructions=_RETRIEVAL_INSTRUCTIONS`.

Why this deviates:

The local design says retrieval is a delegated sub-agent that decides whether
BM25, synonyms, HyDE, rerank, or evaluation is justified. That decision policy
is important enough to be a typed program with metrics, but it is currently a
long natural-language playbook. The prompt also says the RetrievalAgent has
several concrete tools, while `CLAUDE.md` says the default should remain plain
BM25 and advanced expansion should only be used when the agent identifies a
specific vocabulary gap.

Improvement:

- Model retrieval selection as a `RetrievalStrategySignature` or
  `RetrievalPolicyProgram`.
- Inputs should include query, corpus stats, previous result summaries,
  expansion assessment, and budget.
- Outputs should be a typed next action, rationale, stop condition, and final
  candidate set.
- Validate on the documented BEIR-style cross-domain benchmarks before making
  expansion policies default.

### P0: Adaptive retrieval helper still uses output-only calls and sync event-loop shims

Evidence:

- `kaos_agents/context/retrieval.py:694-712` constructs
  `Call(ReflectOnCoverageSignature)` and runs `call(...)` through
  `asyncio.run`, including a `ThreadPoolExecutor` branch when already inside a
  running loop.
- `kaos_agents/context/retrieval.py:880-897` does the same for
  `GeneratePseudoDocumentSignature`.
- `kaos_agents/context/retrieval.py:923-936` does the same for
  `SuggestQueriesSignature`.

Why this deviates:

These paths discard `Invocation.usage`, trace, error, and retry context. They
also bridge async code by starting new event loops in worker threads. Even if
the adaptive pipeline is no longer the default production path, it remains
importable and can silently underreport spend when used.

Improvement:

- Make the adaptive retrieval path async end-to-end.
- Replace `call(...)` with `await call.invoke(...)`.
- Return a typed value that includes generated query text plus
  `InvocationUsage`.
- Add a regression check that no production LLM helper calls `asyncio.run(call(...))`.

### P1: Semantic evaluation claims budget accounting, but Compose does not record judge usage

Evidence:

- `kaos_agents/planning/evaluate.py:176-184` uses `.invoke()` and comments that
  semantic-eval cost flows into `PlanBudget`.
- `kaos_agents/planning/evaluate.py:186-204` builds a `Judgment` but does not
  store usage or emit a usage event.
- `kaos_agents/planning/compose.py:239-242` records only `act_result.cost_usd`
  and `act_result.token_count`; semantic judge usage is not included.

Why this deviates:

The implementation uses the right invocation surface but drops the usage before
the budget can see it. This breaks the design promise that every completed
program yields real usage and that planning budgets are complete.

Improvement:

- Add usage fields to `Judgment`, or return `(Judgment, InvocationUsage)`.
- Record semantic-eval usage in `PlanBudget`.
- Emit `UsageObserved(source="evaluate")` when an emitter is available.
- Add a unit test where a step requires semantic evaluation and the plan total
  includes both Act and Evaluate cost.

### P1: Perception RAG invokes the output-only program surface

Evidence:

- `kaos_agents/perception/perceiver.py:338-353` calls
  `output = await rag(question=query.query_text)`.
- `kaos_agents/perception/rag.py:88-90` shows `PerceptionRAG.forward()` itself
  uses `self._rag.invoke(**kwargs)` internally, but `_invoke_rag()` accepts any
  RAG-like object and intentionally drives it through `__call__`.

Why this deviates:

The generic path loses invocation usage and trace data whenever the supplied
object is a normal `Program`/`RAG` rather than a wrapper that re-invokes
internally. The design standard is to use `Program.invoke()` when the runtime
needs traceable cost and error context.

Improvement:

- Prefer `await rag.invoke(question=...)` when the object exposes `invoke`.
- Fall back to `await rag(...)` only for explicit test stubs.
- Return or emit usage metadata from perception queries.

### P1: Plan expansion still asks for `list[dict[str, Any]]` and validates after decode

Evidence:

- `kaos_agents/planning/expand.py:71-85` declares
  `steps: list[dict[str, Any]] = OutputField(...)`.
- `kaos_agents/planning/expand.py:177-192` defines `GeneratedStep`, but only
  after the signature.
- `kaos_agents/planning/expand.py:195-205` post-validates raw dicts and skips
  malformed entries.

Why this deviates:

The codec cannot use the nested `GeneratedStep` schema to constrain or retry
the model output. Bad nested data becomes a partial plan rather than a
validation retry. This is exactly the kind of brittle structured-output surface
`kaos-llm-core` is meant to remove.

Improvement:

- Move `GeneratedStep` above the signature.
- Change the output to `steps: list[GeneratedStep]`.
- Treat malformed nested steps as codec validation failures where possible.
- Keep deterministic post-checks for tool names and dependency references.

### P1: Compose builds monolithic prompts for LLM steps

Evidence:

- `kaos_agents/planning/compose.py:461-491` concatenates description, input
  description, predecessor outputs, and expected output into a single prompt
  string.
- `kaos_agents/planning/act.py:24-35` has `LLMStepSignature` with only
  `instruction: str` input and `response: str` output.

Why this deviates:

Plan execution has a typed graph, typed steps, typed predecessor results, and
typed expectations, but the LLM step collapses all of that structure into one
natural-language field. That makes the model responsible for rediscovering the
schema hidden inside the prose.

Improvement:

- Replace `LLMStepSignature.instruction` with a richer step execution
  signature: `step_description`, `input_description`, `expected_output`,
  `predecessor_results`, and optional `constraints`.
- Consider a `PlanStepProgram` that can be optimized independently.
- Preserve predecessor outputs as structured artifacts where available instead
  of prompt text.

### P1: Tool argument synthesis returns JSON as a string and reparses it manually

Evidence:

- `kaos_agents/planning/compose.py:555-566` defines a local
  `_ToolArgSynthesisSignature` with `args_json: str` and the instruction
  "Output ONLY the JSON object".
- `kaos_agents/planning/compose.py:583-596` strips code fences, runs
  `json.loads()`, and returns `{}` on malformed output.

Why this deviates:

The tool argument synthesis call bypasses the native structured-output contract
and recreates a brittle prompt/parse loop. Returning `{}` on malformed output
pushes the failure downstream into a tool error instead of letting the codec
retry the schema.

Improvement:

- Generate a Pydantic model or typed dict equivalent from the tool schema where
  practical.
- If the schema is dynamic, use a core codec/tool-call facility that validates
  against JSON Schema instead of asking for a JSON string.
- Return a typed error when synthesis fails instead of `{}`.

### P2: Several `Signature` classes are local to functions

Confirmed current local signature classes:

| File | Local class |
|---|---|
| `kaos_agents/patterns/findings.py:897` | `_RewriteSignature` |
| `kaos_agents/patterns/findings.py:1969` | `_FilterSignature` |
| `kaos_agents/patterns/findings.py:2075` | `_SynthesizeSignature` |
| `kaos_agents/patterns/reflexion.py:165` | `_ReflexionCritiqueSignature` |
| `kaos_agents/patterns/router.py:294` | `_RoutingSignature` |
| `kaos_agents/planning/compose.py:555` | `_ToolArgSynthesisSignature` |
| `kaos_agents/planning/goal_check.py:295` | `_GoalCheckerSignature` |
| `kaos_agents/planning/policy.py:180` | `_TurnToolPolicySignature` |
| `kaos_agents/tools/corpus_filter.py:307` | `_CorpusFilterSig` |

Why this deviates:

Local signatures are harder to import, document, test, serialize, optimize, and
reuse. They also make signature identity less stable across program envelopes
and tracing. `goal_check.py` and `policy.py` cache lazy-built classes for the
optional `[llm]` extra; that explains the import strategy but not the very large
docstrings and local-only contract.

Improvement:

- Prefer module-level signatures when `kaos-llm-core` is already a required
  import for the module.
- Where `[llm]` optionality requires lazy imports, use a cached module-level
  factory and expose the contract through a stable public helper.
- Add a static check that flags new function-local `Signature` classes unless
  explicitly exempted.

### P1: GoalChecker is now a long incident-driven prompt instead of a compact critic contract

Evidence:

- `kaos_agents/planning/goal_check.py:295-620` defines
  `_GoalCheckerSignature` with a very long docstring.
- The docstring includes production incident references, session IDs, many
  special cases, and precedence rules.

Why this deviates:

The critic is carrying an expanding body of policy in one prompt. This makes
the most safety-critical loop brittle: each new exception competes with older
exceptions in natural language. It also makes the behavior difficult to
optimize because examples and metrics are embedded as prose rather than a
separate labelled suite.

Improvement:

- Split deterministic pre-checks out of the LLM critic wherever possible.
- Represent critic rules as a versioned `GoalCheckPolicy` data structure.
- Use a small `GoalCheckSignature` over typed inputs plus `policy_id` or
  compact policy excerpts.
- Maintain labelled regression examples outside the prompt and run them as
  judge evals.

### P2: TurnToolPolicy repeats the same prompt-growth pattern

Evidence:

- `kaos_agents/planning/policy.py:180-220` embeds group-selection shortcuts and
  corpus-kind hints in `_TurnToolPolicySignature`.
- It is cached lazily, but the behavioral policy is still prompt prose.

Why this deviates:

Tool group selection is a classifier with a small finite label set and stable
features. It should be especially amenable to typed labels, examples, and
retrieval/classification programs. A growing prose shortcut list will be harder
to validate than a labelled set.

Improvement:

- Use typed enum labels for groups and a structured `ToolGroupCandidate` input.
- Keep corpus-kind and raw group hints as data fields.
- Build an eval set from prior routing failures and use optimizer-supported
  examples.

### P2: Generic judge rubrics are useful, but the current rubrics are giant raw strings

Evidence:

- `kaos_agents/planning/m2_consistency.py:46` defines a 4335-character rubric.
- `kaos_agents/planning/m3_grounding.py:50` defines a 3169-character rubric.
- `kaos_agents/planning/m4_completeness.py:52` defines a 2446-character rubric.
- `kaos_agents/planning/judge.py` intentionally models the generic judge as
  `rubric`, `input_text`, and `context`.

Why this deviates:

The generic judge design is valid, and these rubrics are less problematic than
system-prompt mutation. However, the rubrics are still raw prompt assets with
embedded examples and decision policies. As they grow, they become hard to
version, test, and optimize.

Improvement:

- Store each rubric as a typed `RubricSpec` with `allowed_labels`,
  positive/negative criteria, examples, and version.
- Pass `allowed_labels` through the signature or enforce them with a typed
  output model.
- Keep long examples in eval fixtures, not only in the rubric prose.
- Track rubric changes with benchmark deltas.

### P2: Findings and corpus filtering still use `list[dict]` nested outputs

Evidence:

- `kaos_agents/patterns/findings.py:2005-2016` declares
  `survivors: list[dict]` with manual key descriptions.
- `kaos_agents/tools/corpus_filter.py:326-338` declares
  `kept: list[dict]` and `dropped: list[dict]`.

Why this deviates:

These are structured decisions with known fields. Describing dict keys in prose
means the codec cannot validate nested item shape as strongly as it could with
Pydantic models. The code then has to manually clamp, skip hallucinated IDs,
and recover from malformed entries.

Improvement:

- Define `FilteredFindingOutput`, `KeptArtifactOutput`, and
  `DroppedArtifactOutput` Pydantic models.
- Make the signature outputs typed lists of those models.
- Keep deterministic ID round-trip validation after decode.

### P2: Tool results are packed as text plus JSON, then recovered by string parsing

Evidence:

- `kaos_agents/actions/tool_bridge.py:262-279` returns
  `f"{text_part}\n\n{json.dumps(structured)}"` when a tool has both text and
  structured content.
- `kaos_agents/patterns/chat.py:123-160` reparses the combined string to
  recover `structured_content`.

Why this deviates:

The bridge preserves data that was previously dropped, which fixed real
behavior, but it does so by smuggling structured content through a text channel.
That is brittle when tool text itself contains JSON-like tails, and it forces
downstream code to infer structure from a rendered observation.

Improvement:

- Extend the ReAct tool observation path to carry both `text` and
  `structured_content`.
- Let the LLM see a rendered observation, but keep wire/events/memory on the
  structured object.
- Remove `_extract_structured_content()` once the trajectory preserves
  structured payloads.

### P2: Runtime instructions remain a central untyped behavior channel

Evidence:

- `kaos_agents/runtime/agent.py:997-1017` builds `instructions` and passes them
  to `Call(RespondSignature, instructions=instructions)`.
- `kaos_agents/runtime/runner.py` passes `self._agent.instructions` into
  `ResearchAgent`, `PlanExecuteAgent`, and `ChatAgent`.
- `kaos_agents/config.py` makes `instructions` part of ergonomic
  `Agent.create(...)`.
- `kaos_agents/planning/react_planner.py` explicitly documents that its
  `instructions` kwarg is forwarded to `ReAct`, not a signature attribute.

Why this deviates:

Some instruction channel is expected for persona and high-level runtime
configuration. The drift is that this channel is also used for behavior policy,
task strategy, and dynamic context. That encourages prompt sprawl and makes
optimizer support harder because behavior moves in opaque strings rather than
typed program state.

Improvement:

- Keep `instructions` for stable persona or deployment policy only.
- Move task strategy and dynamic context into signature fields.
- Add code comments or types that distinguish `persona_instructions` from
  `program_policy`.
- Consider a lint rule for new `instructions=` call sites that require a design
  note or exemption.

### P2: Hierarchical planner creates sub-agent instructions from an f-string

Evidence:

- `kaos_agents/planning/hierarchical_planner.py:511-516` creates an
  `AgentEnvelope` with `instructions=f"You are a research sub-agent. Answer
  this sub-goal: {intent.goal.statement}"`.

Why this deviates:

The sub-agent task is typed upstream as an `IntentResult` / goal, then converted
to a prompt instruction string. The code comment says this is a minimum Phase
3.C implementation, but it should not remain the durable design.

Improvement:

- Add a typed sub-agent task payload to `AgentEnvelope`.
- Use an envelope/program builder that passes `subgoal` as a field.
- Keep the instruction constant stable and short.

### P3: Broad LLM fallbacks can hide provider and schema failures

Evidence:

- `kaos_agents/context/retrieval.py` catches broad exceptions in LLM query
  expansion and returns empty strings/lists.
- `kaos_agents/context/doc2query.py` and `kaos_agents/memory/summarize.py`
  also use best-effort fallback behavior.

Why this deviates:

Fallbacks are often useful in an agent runtime, but silent fallback can hide
provider failures, auth problems, and schema regressions. That makes it harder
to evaluate whether a program works or merely degraded quietly.

Improvement:

- Keep graceful fallback for user-facing turns, but emit structured events or
  metrics for LLM helper failures.
- Distinguish "model said no useful expansion" from "LLM call failed".
- Add test coverage for failure telemetry.

## Already Improved Since The Earlier Audit

These older issues should not be re-filed as current findings:

- `context/classify.py` now uses a module-level signature, `Literal` intent
  labels, `.invoke()`, and `InvocationUsage.from_invocation(...)`.
- `context/doc2query.py` now uses a module-level signature and `.invoke()`.
- `memory/summarize.py` now uses a module-level signature and `.invoke()`.
- `planning/expand.py` and `planning/evaluate.py` now use `.invoke()` for the
  main calls, although nested schema and budget propagation issues remain.
- `patterns/research/agent.py` now stores RAG `Claim` payloads with
  `model_dump(mode="json")` instead of manual flattening.
- Benchmark judge code now uses `InvocationUsage.from_invocation(...)`.

## Recommended Remediation Sequence

1. Fix usage accounting holes first.
   - Convert adaptive retrieval helpers and perception RAG to `.invoke()`.
   - Carry semantic-evaluation usage into `PlanBudget`.

2. Remove long runtime prompts from Chat, Research escalation, and
   RetrievalAgent.
   - Introduce typed programs/signatures for tool-use policy, research
     escalation, and retrieval strategy.
   - Add eval fixtures before changing behavior.

3. Fix nested structured-output surfaces.
   - `PlanExpandSignature.steps`.
   - `FindingsAgent` filter survivors.
   - `corpus_filter` kept/dropped artifacts.
   - Tool argument synthesis.

4. Stabilize signature identity.
   - Move or expose local signatures.
   - Keep optional-extra lazy imports, but make contracts importable and
     documented.

5. Replace text-smuggled structured tool output.
   - Preserve `structured_content` through the tool trajectory and events.

6. Refactor critics into typed policy specs plus eval suites.
   - GoalChecker first, then TurnToolPolicy, then M2/M3/M4 rubrics.

## Suggested Static Guardrails

Add lightweight checks that fail on new drift:

- No `await call(...)` or `asyncio.run(call(...))` in production LLM helpers.
- No `await rag(...)` when `.invoke()` is available.
- No function-local `Signature` classes outside an allowlist.
- No new string constants over a threshold such as 800 characters with names
  matching `*_INSTRUCTION`, `*_PROMPT`, or `*_RUBRIC` unless they are declared
  as versioned prompt assets with eval coverage.
- No `OutputField` of `list[dict]` when the item schema is known.
- No LLM output field named `*_json` followed by manual `json.loads()`.
- New `instructions=` call sites require an exemption comment explaining why a
  typed input field or program policy is not appropriate.

## Validation Plan

For the remediation work, use focused checks rather than only the broad test
suite:

- Unit test that semantic evaluation cost increments `ComposeResult.total_cost_usd`.
- Unit test that perception RAG returns or emits usage when the inner RAG has
  usage.
- Unit test that adaptive retrieval helpers expose usage and do not start a
  nested event loop.
- Codec test that malformed plan step output fails validation instead of being
  silently skipped.
- Regression test for research escalation proving `self._instructions` is not
  mutated.
- Golden eval set for Chat/ReAct stop discipline, aggregation, and refusal.
- BEIR-style retrieval benchmark before changing retrieval strategy defaults.
- GoalChecker labelled eval suite with old incident cases moved out of the
  prompt and into fixtures.

## Evidence Inventory

Design baseline:

- `../kaos-llm-core/AGENTS.md:18-19`: public surface includes typed
  signatures, programs, codecs, routers, metrics, optimizers, traces, batch,
  and MCP integration.
- `../kaos-llm-core/AGENTS.md:73-85`: typed LLM programming principles treat
  signatures, programs, codecs, optimizers, invocation records, traces, and
  prompt/message formats as contracts.
- `../kaos-llm-core/README.md:13-25`: `kaos-llm-core` is the LLM programming
  layer, with Signatures as Pydantic models, Programs as composition, and
  optimizers over labelled data.
- `../kaos-llm-core/README.md:153-159`: core concepts define Signature, Call,
  Program, Codec, Optimizer, and Invocation.
- `README.md:13-17`: `kaos-agents` sits above `kaos-llm-core`.
- `CLAUDE.md:153-167`: ReAct is the inner loop; RAG is the
  `kaos-llm-core` program; real cost accounting depends on `.invoke()`.

Current source hotspots:

- Long prompt/rubric constants:
  - `kaos_agents/patterns/chat.py:57`: `_REACT_INSTRUCTION`, 1219 chars.
  - `kaos_agents/patterns/research/agent.py:76`:
    `_RESEARCH_REACT_INSTRUCTION`, 1992 chars.
  - `kaos_agents/retrieval_agent.py:31`: `_RETRIEVAL_INSTRUCTIONS`, 1764 chars.
  - `kaos_agents/planning/m2_consistency.py:46`:
    `M2_REASONING_ACTION_RUBRIC`, 4335 chars.
  - `kaos_agents/planning/m3_grounding.py:50`: `M3_GROUNDING_RUBRIC`,
    3169 chars.
  - `kaos_agents/planning/m4_completeness.py:52`: `M4_COMPLETENESS_RUBRIC`,
    2446 chars.
- Bare/output-only calls:
  - `kaos_agents/context/retrieval.py:709-712`
  - `kaos_agents/context/retrieval.py:893-897`
  - `kaos_agents/context/retrieval.py:936-941`
  - `kaos_agents/perception/perceiver.py:353`
- Prompt assembly and manual parsing:
  - `kaos_agents/planning/compose.py:461-491`
  - `kaos_agents/planning/compose.py:555-596`
  - `kaos_agents/actions/tool_bridge.py:262-279`
  - `kaos_agents/patterns/chat.py:123-160`
- Local signatures:
  - `kaos_agents/patterns/findings.py:897`
  - `kaos_agents/patterns/findings.py:1969`
  - `kaos_agents/patterns/findings.py:2075`
  - `kaos_agents/patterns/reflexion.py:165`
  - `kaos_agents/patterns/router.py:294`
  - `kaos_agents/planning/compose.py:555`
  - `kaos_agents/planning/goal_check.py:295`
  - `kaos_agents/planning/policy.py:180`
  - `kaos_agents/tools/corpus_filter.py:307`
