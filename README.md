# kaos-agents

Agentic runtime with persistent, section-based memory and composable planning for KAOS.

## Installation

```bash
# Core (memory, planning primitives, graph-based plans)
uv add kaos-agents

# With LLM support (intent classification, plan generation, tool calling)
uv add "kaos-agents[llm]"

# With MCP server support
uv add "kaos-agents[mcp]"
```

## Features

- **SessionMemory**: 13-section memory model with per-section token budgets, 7 eviction policies, LLM summarization, BM25 search, VFS persistence
- **Planning Primitives**: 7 composable building blocks (Recall, Evaluate, Expand, Act, Compose, Route, Graph Ops)
- **Planning Strategies**: Adaptive (ADaPT), direct execution, hierarchical decomposition, rolling horizon
- **PlanGraph**: kaos-graph backed plan DAG with topological execution, parallel levels, subplan insertion
- **Agent Patterns**: BaseAgent (8-step turn loop), ChatAgent (ReAct tool calling), PlanExecuteAgent (adaptive planning), ResearchAgent (RAG-backed document Q&A)
- **Intent Classification**: LLM-based with heuristic fallback (respond, tool_use, research, plan, clarify)
- **Recipe Library**: 5 built-in workflow playbooks (contract extraction, corpus Q&A, FR/EDGAR research, summarization) auto-loaded into planning memory
- **6 MCP Tools**: chat, plan, memory-query, memory-search, memory-clear, recipe-list

## Quick Start

```python
from kaos_agents import SessionMemory, MemoryType

# Create a session
memory = SessionMemory("my-session")
memory.add(MemoryType.MESSAGES, "user: Hello")
memory.add(MemoryType.MESSAGES, "assistant: How can I help?")

# Assemble context for an LLM call
context = memory.get_sections(
    [MemoryType.MESSAGES, MemoryType.FINDINGS],
    total_budget_tokens=8000,
)
```

## Status

Phase 0 + 0.5 complete. ~6,500 LOC source, 360 tests (291 unit + 69 integration).
