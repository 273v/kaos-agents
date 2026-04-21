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
kaos-agents
    ├── Agent + Runner    — Agent is frozen config (instructions, model, tools, pattern)
    │                       Runner is the execution engine (runtime, context, VFS, hooks)
    │                       run() → AsyncIterator[AgentEvent], turn() → AgentResponse
    ├── AgentEvent        — 19 typed streaming events (TextDelta, ToolCallStart, StepComplete, etc.)
    │                       Two-level: stream deltas (real-time) + lifecycle events (semantic)
    │                       Serialize/deserialize with type discriminator, EventEmitter helper
    ├── SessionMemory     — 13-section context management with budgets, eviction, BM25 search, persistence
    ├── ToolBridge        — wraps KaosTool → kaos-llm-core Tool for ReAct
    ├── AgentLoop         — 8-step turn: add message → assemble context → classify → dispatch → update memory
    ├── Hooks + Providers — BaseHook with lifecycle callbacks (on_turn_start, on_tool_call_start, etc.)
    │                       HookAction (CONTINUE/SKIP/REQUIRE_APPROVAL), LoggingHook, OTelHook built-in
    │                       ProviderConfig with ModelRole (classify/respond/plan/research/evaluate)
    │                       FAST/BALANCED/STRONG presets
    ├── Permissions        — PermissionRule (glob pattern + allow/deny/ask) + PermissionPolicy
    │                       Auto-allow readOnlyHint, auto-ask destructiveHint
    │                       RunState for durable pause/resume across process restarts
    ├── Delegation         — agent_as_tool() wraps an Agent as a callable sub-agent
    │                       Agent.delegated_agents are auto-injected as ReAct tools
    │                       Agent.handoffs become handoff_to_<name> tools
    │                       ContextVar depth tracking prevents infinite recursion
    │                       Runner.delegate() / Runner.handoff() yield Subagent/Handoff events
    │                       Runner.resume() continues a paused run (with VFS-restored memory)
    ├── Wire + API        — SSE, JSONL, WebSocket serializers + FastAPI REST API with streaming
    │                       POST /v1/sessions/{id}/messages → SSE event stream
    │                       Session CRUD, memory query/search endpoints
    ├── MCP Tools         — 6 tools (chat, plan, memory-query, memory-search, memory-clear, recipe-list)
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
- **OpenTelemetry tracing.** `OTelHook(BaseHook)` emits spans for turns, tool calls, plan steps (optional `[otel]` extra).
- **Real token/cost accounting (Phase 5.0).** `_simple_respond` and every pattern's streaming handler call `.invoke()` (not bare `__call__`) so `Invocation.usage` is available. Each completed Program yields a `UsageObserved` event (`input_tokens`/`output_tokens`/`total_tokens`/`cost_usd`/`source`). `BaseAgent.run()` aggregates these into the `TurnComplete` event's token+cost fields. The value type is `kaos_agents.usage.InvocationUsage` (frozen, slotted, pointwise-addable, with `ZERO_USAGE` identity + `from_invocation`/`from_llm_usage` factories). Pre-5.0 the turn loop hard-coded `tokens_used=0` / `cost_usd=0.0` — now downstream consumers (OTel spans, UsageHook session totals, planning's `Budget`, MCP tool responses, SSE/JSONL wire streams) all see real numbers. Duplicating the four-field value type in kaos-agents preserves the `[llm]`-optional invariant (agents that never call an LLM still run without the kaos-llm-core dep).

## Dependencies

- **kaos-core** — KaosRuntime, VFS, ArtifactStore, KaosContext, ModuleSettings, KaosCoreError
- **kaos-nlp-core** — tokenizer for token counting, BM25 for memory search
- **kaos-llm-client** (optional [llm]) — LLM transport for agent actions
- **kaos-llm-core** (optional [llm]) — Call, ReAct, RAG, Tool, Budget, traces
- **kaos-mcp** (optional [mcp]) — MCP server bridge for agent-serve

No new external dependencies.

## MCP Tools

9 tools registered via `register_agent_tools(runtime)`:

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

## QA Sequence (mandatory)

```bash
ruff format kaos_agents/ tests/
ruff check --fix kaos_agents/ tests/
ty check kaos_agents/ tests/
pytest tests/ -v
```

## Rules

- **Never add AGPL/GPL dependencies.** This is a proprietary codebase.
- Use `get_logger()` from kaos-core for all logging. Never use `logging.getLogger(__name__)`.
- Error messages must include (1) what went wrong, (2) how to fix it, (3) alternative approach.
- **Never duplicate kaos-llm-core functionality.** If it does tool calling, tracing, retry — call it.
- **Never duplicate kaos-core functionality.** If it does VFS, artifacts, settings — call it.
- All tests must be live integration tests where possible. Mocked tests are supplementary only.
- `@dataclass(frozen=True, slots=True)` for all value/result types. Mutable only for builders/accumulators.
- Token counting uses `len(text) / settings.chars_per_token` as the estimation heuristic. No external tokenizer dependency (tiktoken, etc.) — the word tokenizer in kaos-nlp-core is for search, not for LLM token estimation.
