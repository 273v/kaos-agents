# kaos-agents Development Notes

## Required Checklists

Apply these checklist sources to every change in this module.

Python:
- `../docs/python/checklists/index.md`
- `../docs/python/checklists/01-research.md`
- `../docs/python/checklists/02-design.md`
- `../docs/python/checklists/03-implement.md`
- `../docs/python/checklists/04-test.md`
- `../docs/python/checklists/05-quality.md`
- `../docs/python/checklists/06-review.md`
- `../docs/python/checklists/07-commit.md`
- `../docs/python/checklists/08-debug.md`
- `../docs/python/checklists/09-optimize.md`
- `../docs/python/checklists/10-document.md`
- `../docs/python/checklists/11-retrieval-and-evaluation.md`
- `../docs/python/checklists/12-benchmarking.md`
- `../docs/python/checklists/13-kaos-agent-retrieval.md`

Rust-adjacent:
- `../kaos-nlp-core/docs/FUZZY_HASHING_PLAN.md` (`QA Checklist`) for Rust, PyO3, native bindings, and performance-critical boundary work
- `../kaos-nlp-core/docs/todo/API_IMPROVEMENTS_TODO.md` for Rust-adjacent backlog and API-shape guidance

## Purpose

Agentic runtime with persistent, section-based memory for KAOS. Sits above kaos-llm-core (which provides the LLM programming primitives) and below applications (which use the agent as a multi-turn, tool-using assistant).

The agent is **stateless** — reconstructed per MCP call. All persistent state lives in `SessionMemory`, which hydrates from VFS on each call and persists at the end. This fits the MCP stateless-request model.

## Architecture

```
Application / MCP Client
    ↓
kaos-agents (mirrors kaos-core layout: base/ + types/ + registry/ + decorators/ + runtime/)
    ├── base/                — ABCs
    │                          ├── agent.py        KaosAgent (run/turn + metadata)
    │                          ├── pattern.py      KaosPattern (dispatch ABC)
    │                          ├── event.py        KaosEvent (frozen pydantic ABC)
    │                          └── hook.py         KaosHook + HookAction
    ├── types/               — frozen value types (intents, response, tool_call, plan, memory,
    │                          providers, permissions, usage, metadata)
    ├── events/              — 15 KaosEvent subclasses + Span (universal phase boundary)
    │                          ├── stream.py        TextDelta, ThinkingDelta, ToolCallArgsDelta
    │                          ├── spans.py         Span + SpanSubject + SpanPhase
    │                          ├── lifecycle.py     TurnSummary, IntentClassified, UsageObserved, RunError
    │                          ├── tools.py         ToolCallSummary + ToolCallApprovalRequired
    │                          ├── plan.py          PlanProposed + PlanStepSummary
    │                          ├── research.py      CitationFound, EvidenceInsufficient, GroundingRefusalTriggered
    │                          ├── memory.py        MemoryEvent + MemoryEventKind
    │                          ├── budget.py        BudgetExceeded
    │                          ├── emitter.py       EventEmitter (with span_start/complete/error helpers)
    │                          └── serde.py         serialize_event / deserialize_event (registry-backed)
    ├── hooks/               — KaosHook ABC + dispatch + 4 built-in hooks
    │                          ├── base.py          KaosHook ABC + HookAction enum + identity hooks
    │                          ├── dispatch.py      dispatch_hook() — Runner-side fan-out
    │                          ├── builtin.py       LoggingHook, AuditHook, CostTrackingHook
    │                          └── otel.py          OTelHook (optional [otel] extra)
    ├── registry/            — catalogues (mirror kaos-core/registry)
    │                          ├── event_registry.py    type-string → KaosEvent class (auto-populated)
    │                          ├── hook_registry.py     name → KaosHook instance
    │                          └── pattern_registry.py  name → KaosAgent class (chat/plan/research)
    ├── runtime/             — concrete agent implementations
    │                          ├── agent.py             BaseAgent (canonical KaosAgent impl, 8-step turn loop)
    │                          ├── runner.py            Runner (execution engine)
    │                          ├── delegation.py        DelegatedAgent + agent_as_tool
    │                          ├── interrupts.py        PendingToolCall + RunState
    │                          ├── permissions.py       PermissionPolicy engine (value types in types/)
    │                          └── events_to_response.py  events stream → AgentResponse
    ├── decorators/          — Tier-1 on-ramp (mirror kaos-core/decorators)
    │                          └── hook.py          @hook wraps async fn → FunctionHook + auto-register
    ├── tools/               — MCP tool implementations (Track 5 T5-1 consolidation)
    │                          ├── registry.py      6 agent tools + register_agent_tools (was tools.py)
    │                          ├── graph.py         3 graph tools (was tools_graph.py)
    │                          ├── extract.py       3 extraction tools (was mcp_extract.py)
    │                          └── retrieval.py     7 retrieval tools (was retrieval_tools.py)
    ├── cli/                 — User-facing CLIs (Track 5 T5-2 consolidation)
    │                          ├── chat.py          kaos-agent CLI (was cli_chat.py)
    │                          └── extract.py       kaos-extract CLI (was cli_extract.py)
    ├── api/                 — HTTP / wire surface (Track 5 T5-3 consolidation)
    │                          ├── server.py        FastAPI create_app (was api.py)
    │                          ├── serve.py         kaos-agents-serve CLI
    │                          └── wire.py          SSE / JSONL / WebSocket serialisers
    │
    ├── Agent + Runner    — Agent is frozen config (instructions, model, tools, pattern)
    │                       Runner is the execution engine (runtime, context, VFS, hooks)
    │                       run() → AsyncIterator[KaosEvent], turn() → AgentResponse
    ├── KaosEvent stream  — 15 typed events (3 stream deltas, 1 universal Span,
    │                       6 value events, 1 TurnSummary aggregate, 1 MemoryEvent,
    │                       2 errors, 1 control-flow). OTel-aligned via Span(subject, phase).
    │                       Auto-registered in default_event_registry on subclass creation.
    │                       Frozen pydantic KaosModel; mirrors kaos-core value-type convention.
    ├── KaosHook system   — KaosHook ABC with no-op typed callbacks per event family.
    │                       Tier-1: @hook decorator wraps async fn → FunctionHook
    │                       Tier-2: subclass KaosHook + override on_*
    │                       Tier-3: custom HookRegistry + manual register
    │                       HookAction (CONTINUE/SKIP/REQUIRE_APPROVAL) for tool-call gating
    │                       Built-ins: LoggingHook, AuditHook, CostTrackingHook, OTelHook
    │                       ProviderConfig with ModelRole (classify/respond/plan/research/evaluate)
    │                       FAST/BALANCED/STRONG presets
    ├── SessionMemory     — 14-section context management with budgets, eviction, BM25 search, persistence
    │                       Section 14 is MemoryType.GRAPH — per-session RDF graph (PROV-O + CiTO + kaos:)
    │                       populated by emit_from_event hook on the run loop; persisted as Turtle
    ├── KnowledgeGraph    — Per-session kaos_graph.Graph (B1-B4):
    │                       B1: SessionMemory.graph lazy-init + Turtle save/load
    │                       B2: emit_from_event(KaosEvent → triples) for tool calls / steps / citations
    │                       B3: 3 MCP tools (graph-walk / graph-sparql / graph-projection)
    │                       B4: assemble_context auto-injects 1-hop graph context for retrieved findings
    ├── ToolBridge        — wraps KaosTool → kaos-llm-core Tool for ReAct
    ├── AgentLoop         — 8-step turn: add message → assemble context → classify → dispatch → update memory
    ├── Permissions        — PermissionRule (glob pattern + allow/deny/ask) + PermissionPolicy
    │                       Auto-allow readOnlyHint, auto-ask destructiveHint
    │                       RunState for durable pause/resume across process restarts
    ├── Delegation         — agent_as_tool() wraps an Agent as a callable sub-agent
    │                       Agent.delegated_agents are auto-injected as ReAct tools
    │                       Agent.handoffs become handoff_to_<name> tools — emit Span(HANDOFF, ...)
    │                       ContextVar depth tracking prevents infinite recursion
    │                       Runner.delegate() / Runner.handoff() yield Span(SUBAGENT/HANDOFF) events
    │                       Runner.resume() continues a paused run (with VFS-restored memory)
    ├── Wire + API        — SSE, JSONL, WebSocket serializers + FastAPI REST API with streaming
    │                       POST /v1/sessions/{id}/messages → SSE event stream
    │                       Session CRUD, memory query/search endpoints
    ├── MCP Tools         — 12 tools across 3 groups (Track 4):
    │                       agent: chat, plan, memory-{query,search,clear}, recipe-list
    │                       extraction: extract-{schema,corpus,verify}
    │                       graph: graph-{walk,sparql,projection}  (Track 3 B3)
    ├── Tool Taxonomy     — Track 4: discovery + composition surface for tools:
    │                       T4-1 DataType enum (TEXT/MARKDOWN/JSON/JSONL/HTML/CSV/TABLE/RDF/BINARY)
    │                       T4-1 ToolDataTypeRegistry — type-driven discovery
    │                            (tools_by_input_type / tools_by_output_type)
    │                       T4-2 ToolGroup + ToolGroupRegistry — 2-level discovery
    │                            (groups → tools_in / groups_for_tool)
    │                       T4-3 FieldSet (SMALL/MEDIUM/LARGE/ALL) + project() — per-tier
    │                            metadata projection for prompt context
    │                       T4-4 render_tool_catalog(mode=flat|grouped|compact|full) — prompt-ready
    │                            catalogue rendering composing T4-1/T4-2/T4-3
    │                       T4-5 SessionToolSet + filter_tools — per-session allow/deny rules
    ├── Recipes           — 5 built-in workflow playbooks auto-loaded into PLAN_EXAMPLES
    └── Planning          — 7 primitives, 4 strategies, PlanGraph (kaos-graph backed)
    ↓
kaos-llm-core            — Call, ReAct, Refine, RAG, Budget, traces
    ↓
kaos-llm-client          — HTTP transport, provider normalization
    ↓
kaos-core                — runtime, VFS, artifacts, settings, tools
```

## Key Design Decisions

- **ReAct is the inner loop; the agent is the outer loop.** The agent decides which Program to invoke (Call, ReAct, RAG, plan-execute). The chosen Program drives the LLM.
- **SessionMemory sections map to Program inputs.** Each action declares which sections it needs. Memory assembles context with per-section token budgets.
- **VFS-backed persistence.** STREAMING sections (messages, actions, findings) append to JSONL. SNAPSHOT sections (role, playbooks) checkpoint as JSON.
- **Token budgeting.** Each section has a `budget_tokens` limit. Total assembly budget is a secondary cap. Priority ordering determines what gets trimmed first.
- **Eviction policies per section.** FIFO, LRU, LFU, PRIORITY, REFUSE, NONE — each section declares its policy.
- **Grounding integration.** FINDINGS section stores `Claim` instances with `Span` citations from kaos-llm-core grounding.
- **RAG integration.** Document Q&A via `ResearchAgent` dispatches to kaos-llm-core's `RAG` program. RAG handles retrieve → reason → verify → retry. Verified claims (Answer/Claim/Span) are stored in the FINDINGS section with provenance metadata. Insufficient evidence results in explicit refusal (InsufficientEvidence).
- **Retrieval-augmented context assembly.** `assemble_context()` replaces bare FIFO `get_sections()` in the turn loop. When any memory section has >= `retrieval_threshold` items (default 20), plain BM25 search via `search_memory(..., expand_relations=[])` selects the most relevant items instead of FIFO-oldest. Below the threshold, FIFO returns all documents. The DOCUMENTS section is unbounded (`budget_tokens=0`, `eviction_policy=NONE`) so it can hold a full deal room corpus (1000+ docs); the retrieval layer handles context window management.
- **Plain BM25 is the production default.** `_build_corpus_bm25()` in `research.py` uses `search_memory(..., expand_relations=[])` for large corpora. The adaptive retrieval pipeline (`adaptive_retrieve()`) is deprecated — it is still importable but is **not** the default path. Cross-domain BEIR benchmarks proved it scores worse than plain BM25 (0.231 vs 0.296 NDCG@10 on NFCorpus). Lexicon expansion and pseudo-relevance feedback also hurt cross-domain performance (lexicon: -18% to -22% NDCG@10; PRF: -6% to -12% NDCG@10 across 3 BEIR datasets). These techniques should only be used when the agent identifies a specific vocabulary gap.
- **Retrieval is a delegated sub-agent, not a pipeline function.** The `RetrievalAgent` (auto-injected by Runner for RESEARCH pattern agents) has 4 tools: `kaos-retrieval-bm25`, `kaos-retrieval-synonyms`, `kaos-retrieval-hyde`, `kaos-retrieval-evaluate`. The agent decides which retrieval strategies to use based on what it finds — synonym expansion and HyDE are available when plain BM25 is insufficient, but the agent must justify their use from observed results.
- **Cross-domain benchmark requirement.** Any change to the retrieval pipeline must be validated on at least 3 BEIR datasets (e.g., NFCorpus, SciFact, FiQA) before shipping. Cherry-picked improvements on 1-2 queries are not evidence of correctness. See `tests/benchmarks/beir_eval.py` and `tests/benchmarks/cross_domain_benchmark.py`.
- **Corpus triage.** `triage_corpus()` runs BM25 on the DOCUMENTS section before plan expansion to narrow a large corpus to the relevant subset. The triage summary is injected into planning context: "Selected 15 of 1000 documents — plan over these only."
- **Query-aware planning.** `_assess_complexity()` now considers corpus size (100+ docs → decompose) and intent confidence (low confidence → more planning). Composes with the adaptive strategy's existing word-count heuristic.
- **OpenTelemetry tracing.** `OTelHook(KaosHook)` emits OTel spans for turns, tool calls, plan steps (optional `[otel]` extra). Maps directly onto our internal `Span(subject, phase)` events.
- **Real token/cost accounting (Phase 5.0).** `_simple_respond` and every pattern's streaming handler call `.invoke()` (not bare `__call__`) so `Invocation.usage` is available. Each completed Program yields a `UsageObserved` event (`input_tokens`/`output_tokens`/`total_tokens`/`cost_usd`/`source`). `BaseAgent.run()` aggregates these into the `TurnSummary` event's token+cost fields (the typed turn-end aggregate that ships alongside `Span(TURN, COMPLETE)`). The value type is `kaos_agents.types.usage.InvocationUsage` (frozen, slotted, pointwise-addable, with `ZERO_USAGE` identity + `from_invocation`/`from_llm_usage` factories). Pre-5.0 the turn loop hard-coded `tokens_used=0` / `cost_usd=0.0` — now downstream consumers (OTel spans, CostTrackingHook session totals, planning's `Budget`, MCP tool responses, SSE/JSONL wire streams) all see real numbers. Duplicating the four-field value type in kaos-agents preserves the `[llm]`-optional invariant (agents that never call an LLM still run without the kaos-llm-core dep).
- **Transparency lens (Sprint-3 #10) — `cost_usd` + `total_tokens` are one attribute access away.** Pre-Sprint-3 #10, in-process consumers had to drain the event stream looking for `TurnSummary.cost_usd`; MCP consumers had to walk an inconsistent set of per-stage fields. Now:
  - `AgentResponse.cost_usd` (float, USD) and `AgentResponse.total_tokens` (int) are first-class frozen attributes on the response value type. Same numbers also mirrored into `metadata` for backward compat with callers that walk `dict(response.metadata)`.
  - Every MCP tool wrapper (`AgentChatTool`, `AgentPlanTool`, `AgentFindingsTool`, `AgentCorpusFilterTool`) carries `cost_usd: float` and `total_tokens: int` as top-level fields in `ToolResult.structuredContent`. The numbers match `AgentResponse.cost_usd` / `AgentResponse.total_tokens` for the same turn.
  - **Naming convention across the four tools:**
    | Tool | Single-call surface | Multi-stage surface |
    |---|---|---|
    | `kaos-agent-chat` | `cost_usd` / `total_tokens` | — |
    | `kaos-agent-plan` | `cost_usd` / `total_tokens` | — |
    | `kaos-agent-corpus-filter` | `cost_usd` / `total_tokens` | — |
    | `kaos-agent-findings` | `cost_usd` / `total_tokens` (headline) | `total_cost_usd` (mirrors `cost_usd`); `filter_cost_usd` / `synthesis_cost_usd` / `semantic_rewrite_cost_usd`; `filter_tokens` / `synthesis_tokens` / `semantic_rewrite_tokens` |

    Single-call tools (chat / plan / corpus_filter) only need the canonical headline. Findings (the only multi-stage pipeline) keeps the per-stage breakdown for accounting but ALSO emits the canonical `cost_usd` headline so a transparency-aware consumer can read one number regardless of which tool it called.
  - **From Python:** `response = await runner.turn(...)`, then `response.cost_usd` and `response.total_tokens` — done.
  - **From over MCP:** `result.structuredContent["cost_usd"]` and `result.structuredContent["total_tokens"]` — same numbers, same shape.
- **Event taxonomy: 1 Span + typed value events.** Phase boundaries (turn start/complete, step start/complete, tool call start/complete, sub-agent lifecycle, handoffs, run lifecycle) all become `Span(subject, phase)` events with span_id / parent_span_id / duration_ms / attributes. Concrete value events keep typed classes for facts that carry payload beyond a phase: `IntentClassified`, `PlanProposed`, `CitationFound`, `UsageObserved`, `EvidenceInsufficient`, `GroundingRefusalTriggered`, `TurnSummary`. Memory mutations are a single `MemoryEvent(kind, ...)` with a `MemoryEventKind` enum (ADDED / EVICTED / SUMMARIZED / HYDRATED / PERSISTED / SEARCHED). Errors split into `RunError` (generic terminal) and `BudgetExceeded` (typed subtype). `ToolCallApprovalRequired` is the one tool-related event that isn't a Span — it's a control-flow signal with persistence semantics. 15 events total. OTel-aligned via Span; consumers pattern-match on `(event.subject, event.phase)` for boundary events and `isinstance(event, X)` for value events.

## Dependencies

- **kaos-core** — KaosRuntime, VFS, ArtifactStore, KaosContext, ModuleSettings, KaosCoreError
- **kaos-nlp-core** — tokenizer for token counting, BM25 for memory search
- **kaos-llm-client** (optional [llm]) — LLM transport for agent actions
- **kaos-llm-core** (optional [llm]) — Call, ReAct, RAG, Tool, Budget, traces
- **kaos-mcp** (optional [mcp]) — MCP server bridge for agent-serve

No new external dependencies.

## MCP Tools

14 tools registered via `register_agent_tools(runtime)`:

| Tool | Name | Purpose |
|------|------|---------|
| AgentChatTool | `kaos-agent-chat` | Single conversational turn with optional ReAct tool calling |
| AgentPlanTool | `kaos-agent-plan` | Multi-step plan-execute for complex goals |
| AgentMemoryQueryTool | `kaos-agent-memory-query` | Read session memory contents (read-only) |
| AgentMemorySearchTool | `kaos-agent-memory-search` | BM25 search across memory sections |
| AgentMemoryClearTool | `kaos-agent-memory-clear` | Clear session memory (destructive) |
| AgentRecipeListTool | `kaos-agent-recipe-list` | List available workflow recipes |
| ExtractSchemaTool | `kaos-extract-schema` | WS-TR.PR-4 — schema-driven structured extraction on a single document (read-only, closed-world). Pass `schema_json` or `recipe_name` |
| ExtractCorpusTool | `kaos-extract-corpus` | WS-TR.PR-4 — resumable corpus fan-out (composes `extract_corpus` + `batch_run`) |
| ExtractVerifyTool | `kaos-extract-verify` | WS-TR.PR-4 — verify a `Cited[T]`-shaped claim's spans against source text (no LLM needed) |
| AgentGraphWalkTool | `kaos-agent-graph-walk` | Track 3 B3 — N-hop ego subgraph from a starting IRI in the session knowledge graph |
| AgentGraphSparqlTool | `kaos-agent-graph-sparql` | Track 3 B3 — SPARQL SELECT/ASK over the session graph (requires `kaos-graph[rdf]`) |
| AgentGraphProjectionTool | `kaos-agent-graph-projection` | Track 3 B3 — pre-built typed views (findings_with_citations, tool_calls_by_step, step_timeline, all_nodes) |
| AgentFindingsTool | `kaos-agent-findings` | K7 (0.1.0a6) — run a FindingsAgent over a corpus: per-doc candidate extraction → multi-doc filter → optional synthesis. Returns the surviving findings with `block_ref` citations and a per-stage cost breakdown. Composes with `kaos-agent-corpus-filter` upstream and `kaos-extract-verify` downstream |
| AgentCorpusFilterTool | `kaos-agent-corpus-filter` | K8 (0.1.0a6) — LLM-aided scope tightener: given a corpus + an intent string, score each document for relevance and drop the long tail. Cheaper than running a full pattern over the whole corpus and complements the BM25 path in `triage_corpus()` |

## K-series surfaces (0.1.0a6 — pre-release scope)

Three new agent-side surfaces landed alongside the kaos-content K-series:

- **K5: summary-aware `triage_corpus()`**. When a document in the
  DOCUMENTS section carries a cached `ContentDocument.summary`
  (K1), `triage_corpus()` uses the summary's top n-grams to widen
  the BM25 query before scoring. Falls back to the pure-BM25 path
  when no summary is present, so existing pipelines are unchanged
  unless they opt in by precomputing summaries.
- **K6: `FindingsAgent` wrapper pattern**. Lives in
  `kaos_agents/patterns/findings.py`. Wraps any inner agent with a
  three-stage extract → filter → synthesize pipeline. Value types
  are frozen slotted dataclasses: `FindingCandidate`,
  `FilteredFinding`, `FindingsResult`. The filter step routes
  candidates through `Call.invoke()` (Phase 5.0 cost accounting,
  see KC9), so `FindingsResult.cost_usd` reflects the real LLM
  spend across all chunks. Selectors for "every sentence" /
  "candidate sentences" / "candidate paragraphs" let callers
  choose the recall ↔ cost tradeoff.
- **K7 / K8 MCP wrappers** above expose the agent-side surface to
  remote callers. Both honor the agent permission policy and the
  per-request `_meta.kaos_config` override hook.

## Recipe Library

5 built-in recipes in `kaos_agents/recipes/` — auto-loaded into PLAN_EXAMPLES on session creation:

| Recipe | Description |
|--------|-------------|
| `contract-extraction` | Extract key terms with Cited[T] provenance |
| `corpus-qa` | RAG-backed document Q&A with grounded citations |
| `federal-register-research` | FR + eCFR regulatory research workflow |
| `edgar-research` | SEC filing analysis with EDGAR tools |
| `summarization` | Configurable style document summarization |

API: `load_builtin_recipes()`, `load_recipe(name)`, `recipe_names()`, `format_recipe_for_memory()`

### Extraction recipes (WS-TR.PR-4)

5 schema-bundled extraction recipes in `kaos_agents/recipes/extraction/` — mirrors Harvey's published Workflow library with recall floors as the competitive baseline:

| Recipe | Schema ID | Cols | Harvey recall floor |
|--------|-----------|------|---------------------|
| `merger-agreement` | `merger-agreement-v2` | 27 | 99.66% |
| `spa-deal-points` | `spa-deal-points-v2` | 32 | 98.13% |
| `lease` | `lease-v1` | 24 | 97.20% |
| `lpa` | `lpa-v2` | 27 | 99.14% |
| `court-opinion` | `court-opinion-v2` | 16 | 96.49% |

Each recipe JSON has `name`, `description`, `harvey_recall_floor`, `schema` (an `ExtractionSchema.from_dict` payload), `notes`, and `golden_sets` (named eval suites from Atticus/LegalBench). Consumed by `kaos-extract-schema` and `kaos-extract-corpus` MCP tools via `recipe_name="..."`.

API: `load_extraction_recipes()`, `load_extraction_recipe(name)`, `extraction_recipe_names()`.

## Memory Search

BM25 search across searchable memory sections (MESSAGES, ACTIONS, DOCUMENTS, FINDINGS) using kaos-nlp-core Searcher:

```python
from kaos_agents.memory.search import search_memory
results = search_memory(memory, "filing fee", top_k=5)
```

Returns ranked `MemorySearchResult` with content, section, score, item_id.

### MCP Serve

```bash
# stdio (for Claude Code)
kaos-agents-serve

# with additional tool modules available to the agent
kaos-agents-serve --with-source --with-web --with-pdf

# streamable HTTP
kaos-agents-serve --http --port 8000
```

### CLI: `kaos-agent chat`

Interactive REPL or one-shot non-interactive turn (the latter lets
scripts, CI, and course runnables drive the agent without a TTY).
Entry point: `kaos_agents.cli_chat:main`.

```bash
# Interactive REPL (default)
kaos-agent chat
kaos-agent chat --session my-session --verbose
kaos-agent chat --files "contracts/*.pdf" --pattern research

# One-shot: --message (or --message - for stdin)
kaos-agent chat --message "What is 2+2?"
echo "summarize this corpus" | kaos-agent chat --message -

# Session cost ceiling: --max-cost $USD (env fallback
# KAOS_AGENT_MAX_COST_USD). The current turn completes, then further
# turns are refused; non-interactive mode exits with code 2 (distinct
# from code 1 for real errors) so scripts can branch on the outcome.
kaos-agent chat --message "..." --max-cost 0.05
KAOS_AGENT_MAX_COST_USD=0.10 kaos-agent chat
```

The ``_SessionState`` dataclass is the single source of truth for
"how much did this session spend" and "should we refuse the next
turn". ``_resolve_max_cost()`` is the CLI > env > None precedence
resolver. Both are tested in ``tests/unit/test_cli_chat.py``.

## Settings — KaosAgentSettings

`KaosAgentSettings(ModuleSettings)` with `env_prefix="KAOS_AGENT_"`.

| Variable | Default | Description |
|----------|---------|-------------|
| `KAOS_AGENT_DEFAULT_CONTEXT_BUDGET_TOKENS` | 16,000 | Total token budget for context assembly |
| `KAOS_AGENT_SNAPSHOT_INTERVAL_TURNS` | 1 | SNAPSHOT persistence frequency |
| `KAOS_AGENT_MAX_SESSION_AGE_HOURS` | 168 | Session TTL (7 days) |
| `KAOS_AGENT_CHARS_PER_TOKEN` | 4.0 | Token estimation ratio |
| `KAOS_AGENT_DEFAULT_LLM_MODEL` | anthropic:claude-haiku-4-5 | Default model for classify, respond, evaluate |
| `KAOS_AGENT_PLANNING_LLM_MODEL` | anthropic:claude-haiku-4-5 | Model for plan expansion |
| `KAOS_AGENT_MAX_TOOLS` | 30 | Max tools bridged for ReAct |
| `KAOS_AGENT_MAX_REACT_ITERATIONS` | 10 | Max ReAct loop iterations |
| `KAOS_AGENT_TOOL_TIMEOUT_SECONDS` | 60.0 | Tool invocation timeout |
| `KAOS_AGENT_RETRIEVAL_THRESHOLD` | 20 | Section item count that triggers BM25 retrieval vs FIFO |
| `KAOS_AGENT_CONFIDENCE_THRESHOLD` | 0.5 | Below this, Route triggers REPLAN |
| `KAOS_AGENT_DEEPEN_THRESHOLD` | 0.3 | Below this, Route triggers DEEPEN |
| `KAOS_AGENT_PLAN_MAX_STEPS` | 20 | Max steps in plan execution |
| `KAOS_AGENT_PLAN_MAX_REPLANS` | 3 | Max replan attempts |
| `KAOS_AGENT_PLAN_MAX_COST_USD` | 1.0 | Max cost per plan |
| `KAOS_AGENT_PLAN_MAX_WALL_CLOCK_SECONDS` | 120.0 | Max wall-clock time per plan |
| `KAOS_AGENT_MAX_COST_USD` | — | `kaos-agent chat` session cost ceiling (USD). CLI flag `--max-cost` takes precedence. Disable with `0`. |
| `KAOS_AGENT_OCR_VLM_ESCALATION` | False | Escalate low-confidence / garbled Tesseract pages in the scanned-PDF OCR fallback to a vision model (`kaos_llm_core.vision.ocr_page`). Off by default — makes live, cost-incurring vision calls (~$0.005/page on Haiku). Requires the `[vision]` extra + a provider API key; degrades to Tesseract-only when absent. |
| `KAOS_AGENT_OCR_VLM_MODEL` | — | Vision model for OCR escalation (`provider:model`). None uses the kaos-llm-core default (Claude Haiku). |
| `KAOS_AGENT_OCR_VLM_MAX_PAGES` | — | Per-document cap on pages escalated to the vision model. None means unbounded. |

### Scanned-PDF OCR escalation (`kaos_agents.runtime.ocr_engines`)

The agent's scanned-PDF fallback (`BaseAgent._ocr_pdf_bytes_to_content_document`)
renders each page and runs Tesseract. When `ocr_vlm_escalation` is enabled, it
wraps Tesseract in `TieredOCREngine`, which escalates a page to `VlmOcrEngine`
(a vision model via `kaos_llm_core.vision.ocr_page`) only when the Tesseract
output is low-confidence OR a garbled layer (`kaos_pdf.is_low_quality_layer` —
the stronger gate, since Tesseract is over-confident on hard scans). These
engines implement kaos-pdf's `OCREngine` ABC and live here (not in kaos-pdf)
because the VLM path depends on kaos-llm-core, which kaos-pdf must not import
(extraction → LLM is one-directional). `VlmOcrEngine.extract_sync` is safe to
call from within the async runtime (it offloads to a worker thread when a loop
is already running).

## QA Sequence (mandatory)

```bash
ruff format kaos_agents/ tests/
ruff check --fix kaos_agents/ tests/
ty check kaos_agents/ tests/
pytest tests/ -v
```

## Isolation patterns for live tests

> **The hazard.** The default ``KaosRuntime()`` uses a disk-backed
> VFS rooted at ``.kaos-vfs`` (`StorageBackend.DISK` + per-context
> isolation). Session memory persists across pytest invocations.
> A live composition test that asks the agent "what's the largest
> Q4-Q1 margin compression in PnL2024?" will record the right answer
> in session memory on the first run, and on the second run the
> agent will happily answer from memory **without ever calling
> `kaos-tabular-query`** — the composition contract silently
> false-greens. The Excel sub-agent saw a 25-40% flake rate that
> collapsed to 8/8 stable after switching to an in-memory VFS.

The canonical pytest fixture is one line:

```python
@pytest.fixture
def runtime() -> KaosRuntime:
    # In-memory + GLOBAL isolation — no cross-run leakage.
    rt = KaosRuntime.test_mode()
    register_my_tools(rt)
    return rt
```

`KaosRuntime.test_mode()` (kaos-core ≥ 0.1.0a5):

- installs a fresh in-memory `VirtualFileSystem` with
  `IsolationMode.GLOBAL`
- keeps `runtime.artifacts` lazily bound to the new VFS (the
  `cached_property` invalidates when `runtime.vfs` is reassigned),
  so the historical "rebuild ArtifactStore with all 5 keyword
  args" boilerplate is no longer needed

Use `KaosRuntime.test_mode()` for **every** integration test that:

1. Hard-codes a `session_id` (it will collide with a prior run), or
2. Persists anything into `SessionMemory`, or
3. Has the agent dispatch tool calls and asserts that those tool
   calls actually happened (memory of a previous run's answer
   defeats the assertion).

Tests that pass `vfs=VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))`
directly to `Runner(..., vfs=...)` are already isolated — the
`Runner`-supplied VFS is what `SessionMemory` persists to, and the
`runtime.vfs` is just the artifact-store anchor. You can leave
those alone; `test_mode()` is a stricter, simpler default.

For tests that need a disk backend (e.g. exercising the disk-only
artifact persistence path) but still want to drop the per-context
isolation, use `KaosRuntime.test_mode(in_memory=False)`.

## Rules

- **Never add AGPL/GPL dependencies.** This is a proprietary codebase.
- Use `get_logger()` from kaos-core for all logging. Never use `logging.getLogger(__name__)`.
- Error messages must include (1) what went wrong, (2) how to fix it, (3) alternative approach.
- **Never duplicate kaos-llm-core functionality.** If it does tool calling, tracing, retry — call it.
- **Never duplicate kaos-core functionality.** If it does VFS, artifacts, settings — call it.
- All tests must be live integration tests where possible. Mocked tests are supplementary only.
- `@dataclass(frozen=True, slots=True)` for all value/result types. Mutable only for builders/accumulators.
- Token counting uses `len(text) / settings.chars_per_token` as the estimation heuristic. No external tokenizer dependency (tiktoken, etc.) — the word tokenizer in kaos-nlp-core is for search, not for LLM token estimation.
