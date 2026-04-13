# kaos-agents Development Notes

## Purpose

Agentic runtime with persistent, section-based memory for KAOS. Sits above kaos-llm-core (which provides the LLM programming primitives) and below applications (which use the agent as a multi-turn, tool-using assistant).

The agent is **stateless** — reconstructed per MCP call. All persistent state lives in `SessionMemory`, which hydrates from VFS on each call and persists at the end. This fits the MCP stateless-request model.

## Architecture

```
Application / MCP Client
    ↓
kaos-agents
    ├── SessionMemory     — 15-section context management with budgets, eviction, persistence
    ├── ToolBridge        — wraps KaosTool → kaos-llm-core Tool for ReAct
    ├── AgentLoop         — 8-step turn: add message → assemble context → classify → dispatch → update memory
    └── MCP Tools         — kaos-agent-run, kaos-agent-chat, kaos-agent-plan, kaos-agent-memory-query
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
- **RAG integration.** Document Q&A dispatches to `RAG` program (not raw ReAct). RAG handles retrieve → reason → verify → retry.

## Dependencies

- **kaos-core** — KaosRuntime, VFS, ArtifactStore, KaosContext, ModuleSettings, KaosCoreError
- **kaos-nlp-core** — tokenizer for token counting, BM25 for memory search
- **kaos-llm-client** (optional [llm]) — LLM transport for agent actions
- **kaos-llm-core** (optional [llm]) — Call, ReAct, RAG, Tool, Budget, traces
- **kaos-mcp** (optional [mcp]) — MCP server bridge for agent-serve

No new external dependencies.

## Settings — KaosAgentSettings

`KaosAgentSettings(ModuleSettings)` with `env_prefix="KAOS_AGENT_"`.

| Variable | Default | Description |
|----------|---------|-------------|
| `KAOS_AGENT_DEFAULT_CONTEXT_BUDGET_TOKENS` | 16,000 | Total token budget for context assembly |
| `KAOS_AGENT_SNAPSHOT_INTERVAL_TURNS` | 1 | SNAPSHOT persistence frequency |
| `KAOS_AGENT_MAX_SESSION_AGE_HOURS` | 168 | Session TTL (7 days) |
| `KAOS_AGENT_CHARS_PER_TOKEN` | 4.0 | Token estimation ratio |

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
