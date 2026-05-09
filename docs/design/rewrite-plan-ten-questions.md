# kaos-agents — ground-up rewrite plan (Ten Questions)

**Status:** draft, 2026-05-09. Supersedes `kaos-agents-improvement-plan.md` and `reflexive-agent-v2.md` for the architectural axis. Tactical bug fixes from those docs (output truncation, retrieval pipeline) remain valid and should land first or in parallel.

**Mandate:**
1. Keep external contracts verbatim — MCP tool shapes, FastAPI routes, CLI entry points, wire format, recipe JSON schema.
2. Rewrite the core agentic loop from scratch.
3. **Build everything on kaos-llm-core.** The `[llm]` optional extra is dead. kaos-llm-core is a hard dependency; every primitive that has a kaos-llm-core analog uses it directly, not a duplicate.
4. Organize the package around the **Ten Questions** of agentic system design: Triggers, Intent, Perception, Action, Memory, Planning, Termination, Escalation, Delegation, Governance.
5. Support the three adaptation mechanisms (in-context, cross-session, model-level via optimizers) and the three planning patterns (ReAct, Plan-Execute, Hierarchical) as first-class.

---

## 1. Why rewrite (in one page)

The four-Opus audit (2026-05-09) found that kaos-agents has the *names* of kaos-llm-core's discipline but not the *shapes*:

- `BaseAgent.run()` is a 275-line monolith. `KaosPattern` is a phantom ABC no concrete pattern inherits.
- No canonical per-turn bundle. `AgentResponse` is a derived view of the event log; if `TurnSummary` is missing on early failure, `tokens_used` silently becomes `0`.
- No `_active_turn_var` ContextVar. Per-turn state is mutated on instance attributes (`ResearchAgent._instructions`, `_outline_cache`, `Runner.context._config["_corpus"]`) — racy under concurrent calls.
- `LoopRunner` is reused exactly once (`output/refine.py`). Every other iteration — turn, plan-execute, research escalation, planning compose — is hand-rolled.
- `KaosHook` is a flat 14-callback ABC. ReAct/RAG built inside patterns without `hooks=`/`program_hooks=` — two parallel observability systems that never meet.
- `Span.parent_span_id` is declared but never set by any caller. Sub-agent runs discard the entire child event stream and return a 200-char string.
- Zero imports from `kaos_llm_core.optimization` / `programs.envelope` / `programs.cloning`. `PlanBudget` duplicates `BudgetTracker`. `CostTrackingHook` never calls `publish_invocation`. Recipes are free-form JSON, not envelopes.

The cumulative effect: kaos-agents looks like an agent runtime in CLAUDE.md and reads like an agent runtime in tests, but structurally it is a 2024-vintage agent loop wearing 2026 abstraction names. The Ten Questions framework gives us a sharp organizing principle to fix this in one coherent rewrite rather than a slow strangler.

---

## 2. Goals & non-goals

### Goals
- **Typed, traced, optimizable.** Every subsystem is a kaos-llm-core `Program` (or trivially built on one) — has a Signature, returns an `Invocation`, traces push through `collect_traces`, costs publish through `TrialRunner`, candidates can be optimized by `MiproV2Optimizer`.
- **Governable.** Every primitive emits `KaosEvent`s; `Runner.pause()` snapshots full state to VFS; `PermissionPolicy` is the default least-privilege gate; circuit breakers and override hooks ship in-tree.
- **Trigger-source agnostic.** MCP, HTTP, CLI, scheduled, escalation-from-sub, webhook, filesystem-watch — all produce the same typed `Trigger`. The agent loop is invariant to the source.
- **Escalation-aware.** `EscalationRequired` is a first-class event, not a `RunError`. HITL is built in, not bolted on.
- **Declarative.** Agents have a content-addressed `AgentEnvelope` (extending `ProgramEnvelope`) so runs are replayable, hashable, and cross-process.
- **Three planners as Programs.** ReAct, Plan-Execute, Hierarchical are three `Planner` classes — interchangeable, composable, individually optimizable.

### Non-goals
- **Model-level fine-tuning** (paper §6.6). Out of scope; the optimizer layer (MIPROv2) is the kaos answer to "model-level adaptation," and it already lives in kaos-llm-core.
- **A2A federation across organizations.** We ship the `AgentEnvelope` schema and the in-process protocol. Cross-org transport (auth, signing, replay protection) is a future RFC.
- **Breaking external contracts.** No MCP tool input schema, route, or CLI flag changes meaning. Additions are allowed.
- **A new memory store.** `SessionMemory` stays; we add a third tier (institutional) on top.
- **A new tracing system.** kaos-llm-core's `ExecutionTrace` is canonical; agent `Span` events become a view of it, not a parallel log.

---

## 3. External contracts (preserved verbatim)

These are the immovables. Anything that breaks them is a P0 bug in this rewrite.

| Surface | Files | Contract |
|---|---|---|
| MCP tools | `tools/registry.py`, `tools/graph.py`, `tools/extract.py`, `tools/retrieval.py` | All 12 tool names, input schemas, output shapes, `ToolAnnotations`. |
| FastAPI | `api/server.py`, `api/wire.py`, `api/serve.py` | Routes (`POST /v1/sessions/{id}/messages`, session CRUD, memory query/search), SSE/JSONL/WebSocket payloads. |
| CLI | `cli/chat.py`, `cli/extract.py` | Commands, flags (`--session`, `--message`, `--max-cost`, `--files`, `--pattern`, `--json`), exit codes. |
| Recipe JSON | `recipes/*.json`, `recipes/extraction/*.json` | Existing keys (`name`, `description`, `tools`, `steps`, `schema`, `golden_sets`) keep meaning. New `envelope` block is **additive**. |
| Settings env vars | `settings.py` | All `KAOS_AGENT_*` env vars keep their names and defaults. |

Internal additions (new event kinds, new MCP tool flags, new envelope block) are allowed if they default off. `instructions` returned by MCP servers may grow but not change meaning of existing fields.

---

## 4. The Ten Questions → Ten Subsystems

Top-level package layout becomes:

```
kaos_agents/
├── core/                  ← shared types + ABCs (canonical bundle, base ABCs)
│   ├── invocation.py      TurnInvocation, _active_turn_var, current_turn()
│   ├── plan.py            TurnPlan
│   ├── envelope.py        AgentEnvelope (extends ProgramEnvelope)
│   ├── pattern.py         KaosPattern (real ABC) + Planner protocol
│   └── events.py          re-exports of events package + helpers
├── triggers/              ← Q1: How does the agent know it has work?
├── intent/                ← Q2: How does the agent understand what's asked?
├── perception/            ← Q3: How does the agent find things out?
├── action/                ← Q4: How does the agent make things happen?
├── memory/                ← Q5: How does the agent remember things?
├── planning/              ← Q6: How does the agent break a job into steps?
├── termination/           ← Q7: How does the agent know when it's done?
├── escalation/            ← Q8: How does the agent know to ask for help?
├── delegation/            ← Q9: How does the agent work with other agents?
├── governance/            ← Q10: How do we make systems governable?
├── loop/                  ← The AgentLoop Program; the canonical outer loop
├── runtime/               ← Runner (engine), executor pool, ContextVars
├── events/                ← KaosEvent taxonomy (existing, lightly extended)
├── hooks/                 ← KaosHook + adapters to CallHooks/ProgramHooks
├── recipes/               ← workflow playbooks + envelope blocks
├── api/, cli/, tools/     ← external contract surfaces (preserved)
└── settings.py
```

Below: each subsystem, its primary contracts, and the kaos-llm-core primitive it sits on.

---

### Q1 — Triggers (`kaos_agents.triggers`)

**Question:** How does the agent know it has work to do?

**Today:** Implicit. A "turn" starts when an MCP tool call lands or `runner.run(message)` is invoked. The triggering source is opaque to the loop.

**Future:** First-class, pluggable.

```
core/events.py:           Trigger(KaosEvent)  — frozen pydantic; carries (kind, source_id, payload, occurred_at, correlation_id, metadata)
triggers/base.py:         TriggerSource(ABC)  — async iterator yielding Trigger events
triggers/mcp.py:          MCPToolTrigger      — wraps an inbound MCP tool invocation
triggers/http.py:         HTTPMessageTrigger  — wraps a FastAPI POST
triggers/cli.py:          CLIPromptTrigger    — wraps stdin / interactive REPL input
triggers/schedule.py:     ScheduledTrigger    — cron / interval (paper §2.3)
triggers/escalation.py:   EscalationTrigger   — sub-agent escalating to parent / human
triggers/webhook.py:      WebhookTrigger      — external feed (paper §2.1)
triggers/fs.py:           FileSystemTrigger   — watch a path
```

`Runner.run_trigger(trigger: Trigger) -> AsyncIterator[KaosEvent]` becomes the canonical entry. The existing `Runner.turn(message, session_id)` becomes a thin shim: `await runner.run_trigger(MCPToolTrigger.from_message(message, session_id))`.

**Why:** the paper (§2.7) explicitly names triggers as the entry point. Today MCP and HTTP are baked into `Runner.run()` via overloaded args. Lifting the trigger source out lets us add scheduled jobs, webhooks, and inter-agent escalation without touching the loop. Surface evaluation (paper §2.5) — chat, embedded, ambient, autonomous — falls out of the trigger taxonomy.

---

### Q2 — Intent (`kaos_agents.intent`)

**Question:** How does the agent understand what's being asked?

**Today:** A `BaseAgent._classify(...)` method that calls a Call returning an `IntentType` enum (`RESPOND`/`TOOL_USE`/`PLAN`/`RESEARCH`). Constraint identification, ambiguity detection, and goal extraction (paper §3.2–3.4) are absent.

**Future:** A typed `IntentExtractor` Program over a `Signature`.

```
intent/signatures.py:
  class IntentSignature(Signature):
      """Extract typed intent from a Trigger and conversation context."""
      trigger: Trigger
      conversation: list[MemoryItem]    # last N MESSAGES section items
      domain_examples: list[str]        # paper §3.5 — calibrators
      ---
      goal: Goal                        # primary objective (typed)
      constraints: list[Constraint]     # deadline, budget, jurisdiction, format, …
      ambiguities: list[Ambiguity]      # span-level pointers to unclear text
      requires_clarification: bool      # paper §3.3
      pattern: AgentPattern             # which planner to dispatch
      confidence: float

intent/extractor.py:
  class IntentExtractor(Program):
      """Composes IntentSignature with optional ChainOfThought."""

intent/types.py:        Goal, Constraint, Ambiguity (frozen pydantic)
intent/clarify.py:      ClarificationLoop — when confidence < threshold,
                        produces an EscalationTrigger(CLARIFICATION_NEEDED)
                        with the Ambiguity list as payload
```

Built on: kaos-llm-core `Signature` + `Call` + optional `ChainOfThought`. Optimizable by `InstructionOptimizer` / `MiproV2Optimizer` against a labeled intent corpus.

**Why:** paper §3 separates "the trigger arrived" from "we know what to do." Today they are conflated. Constraint identification (paper §3.4) is the foundation for budgets, termination criteria, and escalation thresholds — without it, every other subsystem operates on assumed defaults.

---

### Q3 — Perception (`kaos_agents.perception`)

**Question:** How does the agent find things out?

**Today:** Tools are bridged to ReAct ad hoc; RAG is built into `ResearchAgent`. No separation between "perceive" (read-only, idempotent) and "act" (mutating).

**Future:** A `Perceiver` Program that composes the tool registry + RAG + memory.

```
perception/perceiver.py:
  class Perceiver(Program):
      """Read-only view onto the world. Composes tools (readOnlyHint=True),
      RAG over corpus, and memory recall."""
      def forward(self, query: PerceptionQuery) -> PerceptionResult:
          # 1. consult institutional memory (KnowledgeBase)
          # 2. consult session memory (SessionMemory.search)
          # 3. fan out to read-only tools per query.required_capabilities
          # 4. RAG over corpus when query.kind == "DOCUMENT_QA"

perception/registry.py:   ToolRegistry filtered by readOnlyHint=True
perception/rag.py:        wraps kaos-llm-core RAG; emits CitationFound events
perception/types.py:      PerceptionQuery, PerceptionResult, Cited[T] passthrough
```

Built on: kaos-llm-core `RAG` Program, `Cited[T]` / `GroundedAnswer[T]`, `Span` verification. Reuses `ToolDataTypeRegistry` and `ToolGroupRegistry` from existing `tools/`.

**Why:** paper §4 names perception as a distinct capability. Today perception (search/read/RAG) and action (mutate/send/write) share one tool registry with no boundary; the LLM is trusted to honor `destructiveHint`. A separate `Perceiver` enforces read-only-by-construction at the registry level.

---

### Q4 — Action (`kaos_agents.action`)

**Question:** How does the agent make things happen?

**Today:** Tools are dispatched through ReAct with a `PermissionPolicy` overlay. Reversibility is implicit in tool annotations but unenforced; circuit breakers and rate limiters are absent.

**Future:** An `Actor` Program that wraps the ReAct loop with reversibility + approval + rate limiting.

```
action/actor.py:
  class Actor(Program):
      """Mutation surface. Composes ReAct + reversibility framework
      + approval workflow + rate limiter + circuit breaker."""

action/reversibility.py:
  class Reversibility(Enum):
      REVERSIBLE = "reversible"           # filesystem write to scratch dir
      RECOVERABLE = "recoverable"         # DB insert with rollback tx
      EXTERNALLY_VISIBLE = "externally_visible"  # email send, API call
      IRREVERSIBLE = "irreversible"       # legal filing, payment

action/approval.py:       ApprovalWorkflow — emits ToolCallApprovalRequired;
                          waits for ApprovalGranted/Denied via Runner.resume
action/rate_limit.py:     RateLimiter / CircuitBreaker — KaosHook subclasses
action/types.py:          ActionPlan, ActionResult, ActionRefusal
```

Tool annotations gain `reversibility: Reversibility` (default `IRREVERSIBLE` — fail safe). `PermissionPolicy` consults reversibility: `REVERSIBLE` auto-allow, `RECOVERABLE` log, `EXTERNALLY_VISIBLE` ask, `IRREVERSIBLE` always ask + dual-key.

Built on: kaos-llm-core `ReAct` Program (composed inside Actor), `Tool` adapters, the permission machinery (already exists, expanded).

**Why:** paper §5.3 makes reversibility the central security primitive. Today `destructiveHint=True` is the only signal — coarser than the four-tier framework the paper recommends and not enforced beyond auto-ask. The Actor wrapper makes reversibility load-bearing.

---

### Q5 — Memory (`kaos_agents.memory`)

**Question:** How does the agent remember things?

**Today:** `SessionMemory` covers session memory well (14 sections, eviction, BM25, persistence). Working memory is implicit (local vars in `run()`). Institutional/cross-session memory is absent — each session starts cold.

**Future:** Three tiers, paper §6.1 (working desk → session file → archive).

```
memory/working.py:        WorkingMemory — lives on TurnInvocation.extras;
                          garbage-collected at turn end. The "desk".
memory/session.py:        SessionMemory (kept; see audit Phase 3 fixes).
                          The "file."
memory/institutional.py:
  class KnowledgeBase(Program):
      """Across-session knowledge. RAG'd from a vector + BM25 store
      with provenance. Matter/client-isolated by namespace."""
      def forward(self, query: str, *, namespace: str) -> KBResult: ...
memory/promotion.py:      PromotionPolicy — when does a session finding
                          become institutional? Confidence threshold +
                          human approval (paper §6.6 adaptation rules)
memory/isolation.py:      MatterClientGuard — namespace enforcement
                          (paper §6.4) — every read/write requires a
                          (matter_id, client_id) pair; mixing raises
```

Cross-session adaptation (paper §6.6) becomes: `KnowledgeBase.add(finding, provenance)` after a turn ends, gated by `PromotionPolicy`. Optimizer-level adaptation (model weights) is out of scope; in-context adaptation is the existing instructions+examples flow.

Built on: kaos-llm-core `RAG`, `Cited[T]`. KnowledgeBase is a Program — replaceable with any retrieval backend that implements its Signature.

**Why:** the paper makes a sharp distinction the current code blurs. `SessionMemory` is correctly scoped; what's missing is the layer above (institutional) and below (per-turn working). Matter/client isolation is a hard requirement for legal/financial work and is currently unenforced.

---

### Q6 — Planning (`kaos_agents.planning`)

**Question:** How does the agent break a big job into steps?

**Today:** `PlanExecuteAgent` and `ResearchAgent` are runtime classes that subclass `BaseAgent` and override `_dispatch_streaming`. Three planning patterns are conflated with three runtime classes.

**Future:** Three `Planner` Programs, all over kaos-llm-core's `LoopRunner`. The runtime stays one class; the planner is a constructor parameter.

```
core/pattern.py:
  class Planner(Program, Protocol):
      """A Program that produces and executes a plan from intent."""
      async def plan(self, intent: Goal, memory: SessionMemory) -> Plan: ...
      async def execute(self, plan: Plan) -> PlanResult: ...

planning/react_planner.py:        ReActPlanner    — wraps kaos-llm-core ReAct
planning/plan_execute_planner.py: PlanExecutePlanner
                                  — produces typed PlanGraph upfront,
                                    executes via LoopRunner
                                  — variants ReWOO / LLMCompiler exposed as
                                    PlanExecutePlanner(strategy="rewoo"|"compiler")
planning/hierarchical_planner.py: HierarchicalPlanner
                                  — decomposes via delegation;
                                    sub-agents are Agent.clone_with(...)
                                  — sub-plans are AgentEnvelopes
planning/strategies/:             choice strategy (paper §7.2):
                                  task-shape → planner selection
planning/budget.py:               PlanBudget = thin wrapper over
                                  kaos_llm_core.BudgetTracker; the
                                  duplicate StopReason enum is deleted
planning/graph.py:                PlanGraph (existing, kept; backed by
                                  kaos-graph)
```

The classes `ChatAgent`, `PlanExecuteAgent`, `ResearchAgent` are deleted. Their behaviors are reproduced as:

- `ChatAgent` → `AgentLoop(planner=ReActPlanner(...))`
- `PlanExecuteAgent` → `AgentLoop(planner=PlanExecutePlanner(...))`
- `ResearchAgent` → `AgentLoop(planner=ReActPlanner(retrieval_subagent=RetrievalAgent), perceiver=Perceiver(rag=on))`

Built on: kaos-llm-core `LoopRunner`, `ReAct`, `Refine`, `BestOfN` for plan-step retry.

**Why:** the paper (§7.1) names ReAct, Plan-Execute, and Hierarchical as the three patterns. Today they are three subclasses with copy-pasted streaming-drain code. As Programs they compose — a HierarchicalPlanner can have a PlanExecutePlanner as one sub-agent and a ReActPlanner as another. Each is independently optimizable by `MiproV2Optimizer`.

---

### Q7 — Termination (`kaos_agents.termination`)

**Question:** How does the agent know when it's done?

**Today:** Termination is implicit — `LoopRunner.StopReason` for ReAct/Refine, `PlanBudget.should_stop` for plans, `Runner.run` ends when `_dispatch_streaming` returns. No goal-based termination, no quality-based termination, no loop detection.

**Future:** A `TerminationJudge` Program composing six axes (paper §8.1).

```
termination/judge.py:
  class TerminationJudge(Program):
      """Decides if a turn / plan / loop is complete."""
      def forward(self, *, intent: Goal, current: PlanResult) -> Decision:
          # 1. budget check (cost / time / iterations) — BudgetTracker
          # 2. quality check — Judge against intent.success_criteria
          # 3. failure check — RunError, EvidenceInsufficient, refusal
          # 4. loop check — has the same step fired N times?
          # 5. graceful degradation — partial result acceptable?

termination/criteria.py:    SuccessCriteria — derived from Goal at intent time
termination/loop_detect.py: LoopDetector — kaos-nlp-core fingerprinting
                            of the last N tool_call signatures
termination/degrade.py:     DegradationPolicy — paper §8.6
termination/types.py:       Decision { is_complete, kind, partial_result, reason }
```

Built on: kaos-llm-core `Judge`, `BudgetTracker`, kaos-nlp-core fuzzy hashing for loop detection.

**Why:** the paper (§8.5 "reliability cliff") names termination as where agents most often fail. Today the cliff is real: the `--max-cost` ceiling is the only enforced budget; `_assess_complexity` makes planning decisions but not termination decisions; loop detection doesn't exist. Centralizing termination as a Judge makes failure modes auditable in `KaosEvent`.

---

### Q8 — Escalation (`kaos_agents.escalation`)

**Question:** How does the agent know when to ask for help?

**Today:** Escalation is failure. `RunError` is emitted, run dies, MCP returns an error. There is no human-in-the-loop primitive.

**Future:** Escalation is a typed event with a resume contract.

```
events/escalation.py:    EscalationRequired(KaosEvent) — already partially
                         exists as ToolCallApprovalRequired; generalize
escalation/kinds.py:
  class EscalationKind(Enum):
      CLARIFICATION_NEEDED      # ambiguous intent (Q2)
      APPROVAL_REQUIRED         # destructive action (Q4)
      OUTSIDE_COMPETENCE        # agent recognizes its limits
      BUDGET_EXCEEDED           # cap hit (Q7)
      EVIDENCE_INSUFFICIENT     # RAG refused (Q3/Q5)
      LOOP_DETECTED             # Q7 detected
      DOMAIN_SPECIFIC           # extensible by recipe

escalation/policy.py:    EscalationPolicy — when to escalate vs continue
escalation/hitl.py:      HITLBridge — surfaces escalation to a channel:
                           - CLI: prints + waits on stdin
                           - HTTP: 202 + webhook callback
                           - MCP: returns approval-pending result with
                                  resume token
escalation/resume.py:    extends Runner.pause()/resume(); writes a full
                         RunState snapshot to VFS; resume hydrates
                         and continues from the escalation point
```

`Runner.pause()` and `Runner.resume()` already exist for tool approval. We generalize them: any `EscalationRequired` event triggers a checkpoint. `RunState` carries `(turn_invocation, pending_escalation, pending_tool_call?, memory_handle, envelope_hash)`. The HITL bridge writes the resume URL/token to the channel that originated the trigger.

Built on: kaos-llm-core has no analog (this is genuinely agent-specific). Uses kaos-core's VFS for snapshot persistence.

**Why:** paper §9 makes escalation the difference between "an agent" and "a hallucinating assistant." Today an agent that hits an ambiguity has no recovery path other than guessing — exactly the failure the paper warns about.

---

### Q9 — Delegation (`kaos_agents.delegation`)

**Question:** How does the agent work with other agents?

**Today:** `agent_as_tool()` wraps an Agent as a tool returning a string. The sub-agent's event stream is discarded. New `run_id` per delegation. No `AgentEnvelope`. No A2A protocol.

**Future:** delegation = `clone_with` + `AgentEnvelope` + push-based event collection.

```
core/envelope.py:
  class AgentEnvelope(KaosModel):
      """Content-addressed agent definition. Extends ProgramEnvelope."""
      pattern: AgentPattern
      instructions: str
      model: str
      perceiver: PerceiverEnvelope
      actor: ActorEnvelope
      memory_namespace: tuple[str, str]   # (matter_id, client_id)
      planner: PlannerEnvelope
      termination: TerminationEnvelope
      escalation: EscalationEnvelope
      delegated_agents: tuple[AgentEnvelope, ...]
      handoffs: tuple[AgentEnvelope, ...]
      settings_overrides: Mapping[str, Any]
      recipe_id: str | None
      ---
      def agent_hash(self) -> str: ...
      @classmethod
      def from_envelope(cls, data: dict) -> AgentEnvelope: ...

core/agent.py:           Agent — frozen+slotted (kept); add Agent.clone_with(),
                         Agent.from_envelope(), Agent.to_envelope()

delegation/router.py:    DelegationRouter — selects sub-agent by capability
delegation/a2a.py:       A2A protocol = pickling AgentEnvelope + message
                         over HTTP/WS to another Runner (in-process for v1;
                         cross-process via existing FastAPI for v2)
delegation/merge.py:     event_stream merge — sub-agent events flow into
                         parent.children via collect_events() ContextVar;
                         parent_span_id threaded
```

The `RetrievalAgent` (today wired imperatively in `Runner._build_internal_agent`) becomes a declared `delegated_agents=(RETRIEVAL_AGENT_ENVELOPE,)` on the relevant agent envelopes.

Built on: kaos-llm-core `clone_call` (analog), `ProgramEnvelope` (extended), `program_hash` (pattern reused).

**Why:** paper §10 names A2A as the multi-agent protocol. Today there is no protocol — just an in-process function pointer. AgentEnvelope makes delegation declarable, hashable, replayable, and (eventually) cross-process.

---

### Q10 — Governance (`kaos_agents.governance`)

**Question:** How do we design systems that can be governed?

**Today:** `LoggingHook`, `AuditHook`, `CostTrackingHook`, `OTelHook` exist but don't compose with kaos-llm-core's hook layers. No state snapshots beyond `RunState` for tool approval. No override mechanism. Least privilege exists (`SessionToolSet`) but isn't the default.

**Future:** five governance primitives, all hook-based, all event-emitting.

```
governance/logging.py:        LoggingArchitecture — KaosEvent IS the audit log;
                              durable JSONL append to VFS at every event
governance/snapshot.py:       StateSnapshot — periodic full RunState write to
                              VFS; on-demand via Runner.snapshot()
governance/override.py:       OverrideHook — admin can inject events
                              (force-stop, clear-section, replay-from-snapshot)
governance/circuit.py:        CircuitBreaker — trips on N consecutive errors,
                              cost spike, or external rate-limit signal;
                              emits CircuitOpened event
governance/least_priv.py:     LeastPrivilege — SessionToolSet is default;
                              tools must be explicitly granted, not
                              explicitly denied
governance/hooks_adapter.py:  KaosHook → CallHooks/ProgramHooks adapter
                              so a single OTelHook sees both layers
```

Built on: kaos-llm-core `CallHooks`/`ProgramHooks` (adapter pattern). VFS for snapshot persistence.

**Why:** paper §11 is explicit: governance is a design-time concern, not a runtime overlay. Every primitive must emit auditable events (we already do via `KaosEvent`), checkpoints must be cheap (we have VFS), overrides must exist (today they don't). Least-privilege-by-default flips a 1-line setting and closes the largest open security gap.

---

## 5. The core object model

Six new types form the spine. All are frozen, slotted, pydantic where wire-facing.

### `Trigger` (event)

```python
class Trigger(KaosEvent):
    kind: TriggerKind                  # MCP / HTTP / CLI / SCHEDULED / ESCALATION / WEBHOOK / FS
    source_id: str                     # provenance (session_id, request_id, cron_name)
    payload: TriggerPayload            # discriminated union by kind
    occurred_at: datetime
    correlation_id: str | None         # for request tracing
    metadata: Mapping[str, str]
```

### `TurnPlan` (composition surface, mirrors `CallPlan`)

```python
@dataclass(frozen=True, slots=True)
class TurnPlan:
    trigger: Trigger
    intent: IntentResult                  # output of IntentExtractor
    memory: SessionMemory                  # hydrated
    working_memory: dict[str, Any]         # per-turn scratch
    perceiver: Perceiver
    actor: Actor
    planner: Planner
    termination_judge: TerminationJudge
    escalation_policy: EscalationPolicy
    permission_policy: PermissionPolicy
    emitter: EventEmitter
    run_id: str
    turn_number: int
    parent_span_id: str | None
```

`AgentLoop.prepare_turn(trigger) -> TurnPlan` is the public composition API. Delegation, MCP wrappers, evaluation harnesses, and FastAPI routes all consume `TurnPlan`. Nothing reaches into `BaseAgent` private fields.

### `TurnInvocation` (canonical bundle, mirrors `Invocation`)

```python
@dataclass(slots=True)
class TurnInvocation:
    id: str                                       # uuid4 hex
    session_id: str
    run_id: str
    turn_number: int
    trigger: Trigger                              # what kicked this off
    intent: IntentResult | None
    plan: PlanResult | None
    output: str = ""
    tool_executions: tuple[ToolExecution, ...] = ()
    events: tuple[KaosEvent, ...] = ()
    usage: InvocationUsage = ZERO_USAGE
    cost_usd: Decimal = Decimal(0)
    children: tuple[TurnInvocation, ...] = ()     # nested delegations
    escalations: tuple[EscalationRequired, ...] = ()
    error: BaseException | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    agent_envelope_hash: str
    started_at: datetime
    finished_at: datetime | None = None
```

`AgentResponse` becomes `AgentResponse.from_turn(invocation) -> AgentResponse` — a derived view, not the source of truth. The wire surfaces (SSE/JSONL/WebSocket) are populated from `TurnInvocation.events`.

`exc.turn_invocation = invocation` is set on the raise path so `except` blocks can recover the partial trace + usage.

### `AgentEnvelope` (declarative, extends `ProgramEnvelope`)

See Q9. Content-addressed via `agent_hash()`. Serializable to JSON. Loadable via `Agent.from_envelope()`.

### `AgentLoop` (Program — the canonical outer loop)

A single Program. No subclasses. Variation lives in injected components (Planner, Perceiver, Actor, TerminationJudge, EscalationPolicy).

```python
class AgentLoop(Program):
    def __init__(
        self,
        intent_extractor: IntentExtractor,
        perceiver: Perceiver,
        actor: Actor,
        planner: Planner,
        memory: SessionMemory,
        termination_judge: TerminationJudge,
        escalation_policy: EscalationPolicy,
        delegation_router: DelegationRouter,
        governance: GovernanceRecorder,
        permission_policy: PermissionPolicy,
        hooks: tuple[KaosHook, ...] = (),
    ) -> None: ...

    async def forward(self, trigger: Trigger) -> TurnInvocation: ...

    async def prepare_turn(self, trigger: Trigger) -> TurnPlan: ...

    # streaming companion (preserves existing wire contract)
    async def stream(self, trigger: Trigger) -> AsyncIterator[KaosEvent]: ...
```

`AgentLoop.invoke(trigger)` returns `TurnInvocation` (the runtime contract). `AgentLoop(trigger)` returns `invocation.output` (ergonomic). `AgentLoop.stream(trigger)` yields events live for SSE.

### `KaosPattern` reborn → `Planner`

`KaosPattern` is renamed `Planner` and made real (Q6). The old empty ABC is deleted. Any imports of `KaosPattern` get a `DeprecationWarning` shim that re-exports `Planner`.

---

## 6. The outer loop, step-by-step

```python
class AgentLoop(Program):
    async def forward(self, trigger: Trigger) -> TurnInvocation:
        plan = await self.prepare_turn(trigger)         # (1) intent + context

        invocation = TurnInvocation.start(plan)
        token = _active_turn_var.set(invocation)
        try:
            with collect_events() as events_collector:
                # (2) governance: record turn-start
                self.governance.record_turn_start(invocation)

                # (3) escalate early if intent is ambiguous
                if plan.intent.requires_clarification:
                    return invocation.with_escalation(
                        EscalationRequired(kind=CLARIFICATION_NEEDED, payload=plan.intent.ambiguities)
                    )

                # (4) plan + execute
                plan_result = await self.planner.plan(plan.intent, plan.memory)
                exec_result = await self.planner.execute(
                    plan_result,
                    perceiver=plan.perceiver,
                    actor=plan.actor,
                )

                # (5) termination judge — done? continue? escalate?
                decision = await self.termination_judge.invoke(
                    intent=plan.intent, current=exec_result
                )
                if decision.kind == DecisionKind.ESCALATE:
                    return invocation.with_escalation(decision.escalation)

                if not decision.is_complete and decision.allows_replan:
                    # bounded by plan.budget — TerminationJudge enforces the cap
                    plan_result = await self.planner.replan(
                        plan_result, decision.feedback
                    )
                    exec_result = await self.planner.execute(plan_result, ...)

                # (6) memory: persist findings, promote candidates to KB
                await self.memory.persist(exec_result, plan.intent)
                await self.knowledge_base.maybe_promote(
                    exec_result, plan.intent,
                    namespace=plan.intent.matter_client,
                )

                # (7) finalize
                invocation.events = tuple(events_collector.collected)
                invocation.output = exec_result.text
                invocation.tool_executions = exec_result.tool_executions
                invocation.usage = self._sum_usage(events_collector)
                invocation.cost_usd = self._sum_cost(events_collector)
                invocation.finished_at = utcnow()

                self.governance.record_turn_complete(invocation)
                return invocation
        except BaseException as exc:
            invocation.error = exc
            exc.turn_invocation = invocation
            self.governance.record_turn_error(invocation, exc)
            raise
        finally:
            _active_turn_var.reset(token)
```

Streaming: `stream()` runs `forward()` in a task and yields events from the collector live. `Runner.run_trigger()` is the wire-facing entry that calls `stream()`.

Each numbered step is overridable via subclassing `AgentLoop` and replacing the corresponding underscore method (`_resolve_planner`, `_finalize_invocation`, etc.) — the kaos-llm-core `Call` step-method discipline applied at the agent layer.

---

## 7. Three planners, three patterns

Per paper §7.1 and the kaos-llm-core convention that programs compose:

### `ReActPlanner` (paper §7.1)

```python
class ReActPlanner(Planner):
    """Wraps kaos_llm_core.programs.ReAct directly."""
    def __init__(self, *, max_iterations: int, instructions: str, ...):
        self._react = ReAct(
            ToolTaskSignature, tools=[], instructions=instructions,
            max_iterations=max_iterations,
        )
    async def plan(self, intent, memory) -> Plan:
        return Plan(strategy="react", root=ReActStep(goal=intent.goal))
    async def execute(self, plan, *, perceiver, actor) -> PlanResult:
        # tools come from perceiver + actor at execution time
        configured = self._react.with_tools(perceiver.tools + actor.tools)
        invocation = await configured.invoke(task=plan.root.goal)
        return PlanResult.from_react(invocation)
```

### `PlanExecutePlanner` (paper §7.1)

```python
class PlanExecutePlanner(Planner):
    """Produces a typed PlanGraph upfront, executes via LoopRunner.
    Variants: vanilla / ReWOO / LLMCompiler."""
    async def plan(self, intent, memory) -> Plan:
        # Call: emit a PlanGraph from intent + corpus triage
        plan_graph = await self._planner_call.invoke(
            goal=intent.goal, constraints=intent.constraints,
            corpus_summary=memory.get_summary(MemoryType.DOCUMENTS),
        )
        return Plan(strategy="plan_execute", graph=plan_graph.output)

    async def execute(self, plan, *, perceiver, actor) -> PlanResult:
        loop = LoopRunner(
            step=self._execute_step,
            build_result=PlanResult.from_steps,
            stop=self._stop_condition,
            on_step_error=self._on_step_error,
        )
        return await loop.run(plan.graph)
```

### `HierarchicalPlanner` (paper §7.1, paper §10)

```python
class HierarchicalPlanner(Planner):
    """Decomposes via delegation. Sub-agents are AgentEnvelopes;
    each is run via Agent.clone_with(envelope) on the parent runtime."""
    async def plan(self, intent, memory) -> Plan:
        # Call: produce a delegation tree of (subagent_envelope, sub_goal)
        ...

    async def execute(self, plan, *, perceiver, actor) -> PlanResult:
        children: list[TurnInvocation] = []
        for sub in plan.subagents:
            sub_agent = Agent.from_envelope(sub.envelope)
            sub_loop = AgentLoop.from_agent(sub_agent, runtime=self.runtime)
            children.append(await sub_loop.invoke(
                trigger=Trigger.delegation(sub.goal, parent=current_turn().id)
            ))
        return PlanResult.from_subagents(children)
```

All three are individually optimizable by `MiproV2Optimizer` against an `evaluate_agent(...)` harness.

---

## 8. Runtime + cross-cutting plumbing

These are the audit Phase 1–3 items, restated as part of this rewrite (not a separate effort).

```python
# runtime/_invocation.py
_active_turn_var: ContextVar[TurnInvocation | None] = ContextVar(
    "kaos_agents_active_turn", default=None,
)

def current_turn() -> TurnInvocation | None:
    return _active_turn_var.get()

# events/collector.py
_event_collector_var: ContextVar[EventCollector | None] = ContextVar(
    "kaos_agents_event_collector", default=None,
)

@contextmanager
def collect_events() -> Iterator[EventCollector]:
    """Mirror of kaos_llm_core.observability.collectors.collect_traces."""
    coll = EventCollector()
    token = _event_collector_var.set(coll)
    try:
        yield coll
    finally:
        _event_collector_var.reset(token)

def push_event(event: KaosEvent) -> None:
    coll = _event_collector_var.get()
    if coll is not None:
        coll.append(event)

# events/emitter.py — every emit() calls push_event(event) after construction
# spans synthesize parent_span_id from the collector's current span stack
```

Sub-agent runs open a child collector that inherits the parent's top span_id. Sub-agent events flow into the parent collector via the same `push_event` — so a `HierarchicalPlanner` sees its children's events natively, and `Runner.delegate()` no longer needs to manually splice streams.

Hooks adapter:

```python
# hooks/adapter.py
class _CallHookAdapter(CallHooks):
    """Forwards kaos-llm-core CallHooks events into the active turn's emitter."""
    def __init__(self, kaos_hooks: tuple[KaosHook, ...]) -> None: ...
    async def on_call_start(self, call, inputs, *, context=None) -> None: ...
    # ... etc

class _ProgramHookAdapter(ProgramHooks): ...

def adapt_hooks(kaos_hooks: tuple[KaosHook, ...]) -> tuple[CallHooks, ProgramHooks]: ...
```

`AgentLoop` constructs adapted hooks at startup; planners thread them into every `ReAct(...)` / `RAG(...)` / `Refine(...)` they construct.

`CostTrackingHook.on_usage_observed` synthesizes an Invocation shim and calls `kaos_llm_core.optimization.trial_runner.publish_invocation(...)` so an agent run inside `with TrialRunner().trial("eval"):` charges to the trial automatically.

---

## 9. External contract preservation, mechanically

For each preserved surface, here is how it stays bit-identical:

### MCP tools

`tools/registry.py` keeps the 12 `KaosTool` subclasses. Their `execute()` methods now delegate to `AgentLoop.invoke(trigger)` instead of `BaseAgent.run`. Input schema unchanged. Output schema unchanged. Annotations unchanged.

```python
class AgentChatTool(KaosTool):
    async def execute(self, message: str, session_id: str | None = None,
                      envelope_id: str | None = None,            # NEW, optional
                      **kwargs) -> ToolResult:
        agent = (
            self._runtime.agent_envelopes.load(envelope_id)
            if envelope_id else self._default_agent
        )
        loop = AgentLoop.from_agent(agent, runtime=self._runtime)
        trigger = Trigger.mcp(message=message, session_id=session_id)
        invocation = await loop.invoke(trigger)
        return ToolResult.create_success(
            result=AgentResponse.from_turn(invocation).model_dump()
        )
```

### FastAPI

`api/server.py` route handlers get the same treatment. `POST /v1/sessions/{id}/messages` → `AgentLoop.stream(trigger)` → SSE.

### CLI

`kaos-agent chat` constructs a `CLIPromptTrigger`, runs `AgentLoop.invoke()`, prints `AgentResponse.from_turn()`. `--max-cost` consumes `invocation.cost_usd`. Exit codes preserved.

### Recipes

`recipes/*.json` adds an optional `envelope` block. Loader prefers `envelope` if present; falls back to the legacy `{tools, steps}` planning-context shape. No existing recipe stops working.

---

## 10. Migration strategy — strangler fig, six phases

The rewrite is incremental, not a big-bang. Each phase is shippable, reversible, and gated on green tests.

### Phase 0 — Foundations (1 week, low risk)
- `core/invocation.py` — `TurnInvocation`, `_active_turn_var`, `current_turn`.
- `core/plan.py` — `TurnPlan`.
- `events/collector.py` — `collect_events`, `push_event`.
- `Span` emitters wired to thread `parent_span_id` from the collector stack.
- Hooks adapter (`hooks/adapter.py`).
- `Agent.clone_with()`, `Agent.to_envelope()`, `Agent.from_envelope()`.
- Tests: every existing unit test passes unchanged. New unit tests for the new types.

### Phase 1 — IntentExtractor + Perceiver + Actor (1.5 weeks)
- New subsystems built; not yet integrated into the main path.
- Each ships with a dedicated benchmark (intent extraction accuracy on a labeled dataset; perceiver recall on BEIR; actor dry-run safety on synthetic destructive-tool tests).
- Existing `BaseAgent.run` is unchanged.

### Phase 2 — AgentLoop alongside BaseAgent (1.5 weeks)
- `loop/agent_loop.py` — `AgentLoop` Program.
- `triggers/*.py` — Trigger types.
- `Runner.run_trigger(trigger)` added; old `Runner.run(message, ...)` kept as a thin shim that constructs an `MCPToolTrigger`.
- Behind a feature flag `KAOS_AGENT_LOOP=v2`. Default off.
- Live tier: run all existing integration tests with `KAOS_AGENT_LOOP=v2` and confirm parity.

### Phase 3 — Three Planners (1.5 weeks)
- `planning/react_planner.py` — replaces `ChatAgent`'s tool-use handling.
- `planning/plan_execute_planner.py` — replaces `PlanExecuteAgent`.
- `planning/hierarchical_planner.py` — new; replaces ad-hoc delegation in `Runner._build_internal_agent`.
- Old pattern classes deprecated (re-export shims raising `DeprecationWarning`).

### Phase 4 — Memory (1 week) + Termination (1 week) + Escalation (1 week, can parallelize)
- `memory/working.py`, `memory/institutional.py`, `memory/promotion.py`, `memory/isolation.py`.
- `termination/judge.py` + loop detection + degradation policy.
- `escalation/*` — `EscalationRequired` event generalized; `Runner.pause/resume` extended; HITL bridge.
- The existing `RetrievalAgent` declared as an envelope in recipes, removed from imperative runner code.

### Phase 5 — Governance + cleanup (1 week)
- `governance/*` — logging architecture, snapshots, override hooks, circuit breaker, least-privilege default.
- `PlanBudget` deleted; replaced with `kaos_llm_core.BudgetTracker` re-export + thin agent-specific extensions.
- Duplicate `StopReason` deleted.
- `CostTrackingHook` calls `publish_invocation`.
- `evaluate_agent(agent_envelope, examples, metric) -> EvalResult`.

### Phase 6 — Cutover (1 week)
- Flip `KAOS_AGENT_LOOP=v2` to default on.
- Two-week soak with telemetry comparing old vs new on the live benchmark suite (Harvey CoC, BEIR cross-domain, CUAD).
- Delete `BaseAgent.run`, `ChatAgent`, `PlanExecuteAgent`, `ResearchAgent`, the duplicated `_handle_*` shims, the imperative `_build_internal_agent` body.

**Total realistic effort: 8–10 engineer-weeks.** Half of which is testing, not coding.

---

## 11. Testing strategy

The audit was unflattering partly because tests passed against the wrong shapes. The rewrite tightens this.

### Per-subsystem evals (one Ten-Question harness each)

| Subsystem | Eval | Dataset |
|---|---|---|
| Triggers | trigger-source latency + delivery semantics | synthetic |
| Intent | goal-extraction accuracy, ambiguity recall | labeled corpus (existing recipes' golden_sets) |
| Perception | retrieval NDCG@10 | BEIR (NFCorpus + SciFact + FiQA — already used) |
| Action | reversibility-honoring safety on destructive tools | synthetic (must refuse without approval) |
| Memory | session-recall + matter-isolation | synthetic |
| Planning | plan-quality vs Harvey CoC | existing benchmark |
| Termination | budget-honoring + loop-detection | synthetic adversarial |
| Escalation | clarification-recall, HITL round-trip | synthetic |
| Delegation | sub-agent event-merge, cost roll-up | synthetic |
| Governance | snapshot/restore round-trip, override semantics | synthetic |

Each eval is an `evaluate_agent(envelope, examples, metric)` call. All run under `--include-live --include-network` in `validate-platform.sh` per CLAUDE.md.

### No shortcut to "green"

Per the project's testing standard: live API tests are the quality bar. Mocked unit tests are supplementary. The acceptance gate is:

```bash
./scripts/validate-platform.sh --profile ubuntu-26.04 --include-network --include-live
```

Plus the BEIR cross-domain run (3 datasets) before any retrieval-touching change.

### Regression guard

Phase 6 cutover requires no regression on:
- Harvey CoC benchmark (latest baseline)
- BEIR NDCG@10 (NFCorpus, SciFact, FiQA)
- CUAD extraction calibration

Plus latency parity within ±10% on the live tier.

---

## 12. File-by-file impact

### Delete
- `kaos_agents/runtime/agent.py::BaseAgent` — replaced by `AgentLoop`.
- `kaos_agents/patterns/chat.py::ChatAgent` — replaced by `ReActPlanner`.
- `kaos_agents/patterns/plan_execute.py::PlanExecuteAgent` — replaced by `PlanExecutePlanner`.
- `kaos_agents/patterns/research/agent.py::ResearchAgent` — replaced by `AgentLoop(planner=ReActPlanner(...), perceiver=Perceiver(rag=...))`.
- `kaos_agents/types/plan.py::PlanBudget`, `StopReason` — replaced by re-exports from kaos-llm-core.
- `kaos_agents/runtime/runner.py::_build_internal_agent` — replaced by `AgentLoop.from_agent(envelope)`.
- `kaos_agents/base/pattern.py::KaosPattern` (the empty ABC) — replaced by `core/pattern.py::Planner`.

### Rewrite (same file path, new contents)
- `kaos_agents/runtime/runner.py` — becomes a thin engine: trigger dispatch, hook fan-out, ContextVar setup. ~150 LOC, was ~900.
- `kaos_agents/runtime/delegation.py` — `agent_as_tool` keeps its name; body uses `clone_with` + `AgentEnvelope`.
- `kaos_agents/types/response.py` — `AgentResponse` becomes a `from_turn(invocation)` projection.
- `kaos_agents/runtime/events_to_response.py` — degrades to a fallback-only path used by `Runner.resume`.
- `kaos_agents/hooks/builtin.py::CostTrackingHook` — gains `publish_invocation` call.

### Keep, mostly unchanged
- `kaos_agents/memory/session.py::SessionMemory` — already correct. Minor: Phase 3 ContextVar threading.
- `kaos_agents/events/*` — extended with `Trigger`, `EscalationRequired` (generalized), `CircuitOpened`, `OverrideApplied`.
- `kaos_agents/types/*` — frozen value types, mostly untouched.
- `kaos_agents/registry/*` — auto-registration patterns; extended for `AgentEnvelopeRegistry`.
- `kaos_agents/recipes/*.json` — extended with `envelope` block, legacy keys preserved.

### New
- `kaos_agents/core/` — invocation, plan, envelope, pattern.
- `kaos_agents/triggers/` — full subsystem.
- `kaos_agents/intent/` — full subsystem.
- `kaos_agents/perception/` — full subsystem.
- `kaos_agents/action/` — full subsystem.
- `kaos_agents/memory/{working,institutional,promotion,isolation}.py` — new tiers.
- `kaos_agents/planning/{react,plan_execute,hierarchical}_planner.py` — new planners.
- `kaos_agents/termination/`, `kaos_agents/escalation/`, `kaos_agents/governance/` — full subsystems.
- `kaos_agents/loop/agent_loop.py` — the canonical Program.
- `kaos_agents/hooks/adapter.py` — kaos-llm-core hook bridge.

---

## 13. Open decisions

> **Resolved 2026-05-09:**
> - **#1 `[llm]` extra removal** — confirmed dead. kaos-llm-core is a hard dep.
> - **#3 Pattern selection** — classifier picks at runtime. `IntentExtractor` returns `intent.pattern`; `AgentLoop` uses it to dispatch to the right `Planner`. One AgentLoop instance can fan out to ReAct/PlanExecute/Hierarchical based on the classified intent. Unblocks Phase 3.
> - **#4 Sub-agent event-merge default** — collapse stream-deltas (`TextDelta`/`ThinkingDelta`/`ToolCallArgsDelta` not forwarded), propagate value events (`UsageObserved`/`CitationFound`/`EvidenceInsufficient`/`Span(SUBAGENT/STEP/...)`/`TurnSummary` forwarded). Configurable per-Hierarchical via `stream_mode='full'|'value-only'|'summary-only'`.
> - **#5 Memory promotion threshold** — confidence ≥ 0.85 + grounding-verified, no human review. Auto-promote when both gates pass. Trusts the LLM's confidence + the `Cited[T]` verifier; right answer for high-volume use. Phase 4 work.
> - **#7 Loop detection sensitivity** — TLSH distance ≤ 30 over the last 5 calls. Validate against synthetic adversarial corpus before locking. Phase 4 work.
> - **#8 Streaming hierarchy** — `AgentLoop.forward()` is blocking; `AgentLoop.stream()` is a thin wrapper that runs `forward()` in an `asyncio.Task` and yields from a Queue the collector pushes into. Symmetric API: every Program has `invoke()` (blocking, returns Invocation/TurnInvocation) and `stream()` (yields events live).

The remaining decisions need a maintainer call before the relevant phase starts.

1. **`[llm]` extra removal.** This plan deletes the optional-dep stance (kaos-llm-core becomes hard). Confirm: are there real consumers running kaos-agents without kaos-llm-core? If yes, the duplicate `InvocationUsage` and the import-guarding stay.

2. **A2A wire format.** `AgentEnvelope` is the in-process artifact. Cross-process, do we ship it as JSON over the existing FastAPI POST, as a custom MCP resource, or follow Google's A2A protocol? The paper suggests A2A; we should evaluate whether to align with that spec or roll our own.

3. **Pattern selection: classifier vs constructor.** Today `IntentResult.pattern` is an output of the intent classifier. Alternative: the AgentLoop is constructed with a fixed planner per agent envelope, and the classifier picks a sub-agent (which has a fixed planner). The latter is simpler; the former is more dynamic. Pick one before Phase 3.

4. **Sub-agent event merge: include or filter?** When a `HierarchicalPlanner` runs three sub-agents, do all of their `TextDelta`s flow into the parent's stream by default, or are they collapsed to `Span(SUBAGENT, COMPLETE)` summaries? Default proposal: collapse stream-deltas, propagate value events (`UsageObserved`, `CitationFound`, `EvidenceInsufficient`). Configurable via `Hierarchical.stream_mode`.

5. **Memory promotion threshold.** When does a session finding become institutional? Confidence-only (≥0.9), human-approval-required, or hybrid? The paper (§6.6) recommends hybrid; default proposal: confidence ≥0.85 + grounding-verified + matter/client tag, plus an opt-in human review queue.

6. **`Runner.run(message)` shim lifetime.** Phase 2 keeps it. Phase 6 deletes it. Two-month deprecation window or one? CLI/HTTP/MCP wrappers all switch over in Phase 2, so the only consumers are external scripts. Survey usage.

7. **Loop detection sensitivity.** kaos-nlp-core fuzzy hashing of the last-N tool-call signatures has a tunable threshold. Too tight → false-positive escalations on legitimate retries; too loose → real loops slip through. Default proposal: TLSH distance ≤ 30 over the last 5 calls = loop. Validate against synthetic adversarial corpus before locking.

8. **Streaming hierarchy.** kaos-llm-core has no streaming surface — every Program is fully blocking. The agent layer commits to streaming via SSE/JSONL/WS. Phase 2 introduces `AgentLoop.stream()` that runs `forward()` in a background task and yields from the collector live. This adds a Task and a Queue per turn. Acceptable? Alternative: stream at the planner boundary only (each planner exposes `astream()`).

---

## 14. Success criteria for this rewrite

We declare success when, simultaneously:

- All 12 MCP tool input/output schemas pass the existing integration suite unchanged.
- All FastAPI route contracts pass `tests/integration/test_api_*.py` unchanged.
- All CLI golden-file tests pass unchanged.
- BEIR cross-domain NDCG@10 within ±2% of pre-rewrite baseline on NFCorpus / SciFact / FiQA.
- Harvey CoC benchmark within ±5% of pre-rewrite baseline.
- CUAD extraction calibration within ±2% of pre-rewrite baseline.
- A new test, `tests/integration/test_ten_questions_e2e.py`, exercises every subsystem end-to-end on the live tier.
- `BaseAgent`, `ChatAgent`, `PlanExecuteAgent`, `ResearchAgent`, `PlanBudget`, the duplicate `StopReason`, and `_build_internal_agent` are deleted from the codebase.
- Every concrete subsystem has a corresponding `kaos_llm_core.programs.X` ancestor in its MRO or composition graph (i.e., no agent subsystem is built from scratch where a kaos-llm-core primitive exists).

That is the bar.
