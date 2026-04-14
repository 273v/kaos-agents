# kaos-agents Development Notes

## Purpose

Agentic runtime with persistent, section-based memory for KAOS. Sits above kaos-llm-core (which provides the LLM programming primitives) and below applications (which use the agent as a multi-turn, tool-using assistant).

The agent is **stateless** — reconstructed per MCP call. All persistent state lives in `SessionMemory`, which hydrates from VFS on each call and persists at the end. This fits the MCP stateless-request model.

## Architecture

```
Application / MCP Client
    ↓
kaos-agents
    ├── SessionMemory     — 13-section context management with budgets, eviction, BM25 search, persistence
    ├── ToolBridge        — wraps KaosTool → kaos-llm-core Tool for ReAct
    ├── AgentLoop         — 8-step turn: add message → assemble context → classify → dispatch → update memory
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

## Dependencies

- **kaos-core** — KaosRuntime, VFS, ArtifactStore, KaosContext, ModuleSettings, KaosCoreError
- **kaos-nlp-core** — tokenizer for token counting, BM25 for memory search
- **kaos-llm-client** (optional [llm]) — LLM transport for agent actions
- **kaos-llm-core** (optional [llm]) — Call, ReAct, RAG, Tool, Budget, traces
- **kaos-mcp** (optional [mcp]) — MCP server bridge for agent-serve

No new external dependencies.

## MCP Tools

6 tools registered via `register_agent_tools(runtime)`:

| Tool | Name | Purpose |
|------|------|---------|
| AgentChatTool | `kaos-agent-chat` | Single conversational turn with optional ReAct tool calling |
| AgentPlanTool | `kaos-agent-plan` | Multi-step plan-execute for complex goals |
| AgentMemoryQueryTool | `kaos-agent-memory-query` | Read session memory contents (read-only) |
| AgentMemorySearchTool | `kaos-agent-memory-search` | BM25 search across memory sections |
| AgentMemoryClearTool | `kaos-agent-memory-clear` | Clear session memory (destructive) |
| AgentRecipeListTool | `kaos-agent-recipe-list` | List available workflow recipes |

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
| `KAOS_AGENT_CONFIDENCE_THRESHOLD` | 0.5 | Below this, Route triggers REPLAN |
| `KAOS_AGENT_DEEPEN_THRESHOLD` | 0.3 | Below this, Route triggers DEEPEN |
| `KAOS_AGENT_PLAN_MAX_STEPS` | 20 | Max steps in plan execution |
| `KAOS_AGENT_PLAN_MAX_REPLANS` | 3 | Max replan attempts |
| `KAOS_AGENT_PLAN_MAX_COST_USD` | 1.0 | Max cost per plan |
| `KAOS_AGENT_PLAN_MAX_WALL_CLOCK_SECONDS` | 120.0 | Max wall-clock time per plan |

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
