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

- **SessionMemory**: 13-section memory model with per-section token budgets, 7 eviction policies, VFS persistence
- **Planning Primitives**: 7 composable building blocks (Recall, Evaluate, Expand, Act, Compose, Route, Graph Ops)
- **Planning Strategies**: Adaptive (ADaPT), direct execution, hierarchical decomposition, rolling horizon
- **PlanGraph**: kaos-graph backed plan DAG with topological execution, parallel levels, subplan insertion
- **Agent Patterns**: BaseAgent (8-step turn loop), ChatAgent (ReAct tool calling), PlanExecuteAgent (adaptive planning)
- **Intent Classification**: LLM-based with heuristic fallback (respond, tool_use, research, plan, clarify)

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

Phase 1 complete: SessionMemory + planning primitives + strategies + agent patterns.
247 tests (216 unit + 31 live/integration).
