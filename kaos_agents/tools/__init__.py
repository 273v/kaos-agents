"""kaos-agents MCP tools — Track 5 chunk T5-1 consolidation.

Subpackage layout:

- :mod:`kaos_agents.tools.registry` — agent / chat / plan / memory /
  recipe tools and the ``register_agent_tools`` entrypoint
- :mod:`kaos_agents.tools.graph` — Track 3 B3 graph tools
  (walk / sparql / projection)
- :mod:`kaos_agents.tools.extract` — WS-TR.PR-4 schema-driven
  extraction tools (schema / corpus / verify)
- :mod:`kaos_agents.tools.retrieval` — adaptive retrieval tools
  (BM25 / HyDE / synonym / rerank / corpus info / grounded answer)
  plus ``register_retrieval_tools``

Re-exports the most commonly imported names so existing call sites
stay stable: ``from kaos_agents.tools import register_agent_tools``,
``AgentChatTool``, ``ExtractSchemaTool``, ``AgentGraphWalkTool``,
etc. all keep working without per-caller updates.

Pre-T5 modules (``kaos_agents.tools_graph``, ``kaos_agents.mcp_extract``,
``kaos_agents.retrieval_tools``) are gone — callers must import via
``kaos_agents.tools.<submodule>`` or use the re-exports here.
"""

from __future__ import annotations

# Agent / memory / recipe tools (was tools.py)
from kaos_agents.tools.extract import (
    ExtractCorpusTool,
    ExtractSchemaTool,
    ExtractVerifyTool,
)

# Graph tools (was tools_graph.py)
from kaos_agents.tools.graph import (
    AgentGraphProjectionTool,
    AgentGraphSparqlTool,
    AgentGraphWalkTool,
)
from kaos_agents.tools.registry import (
    AgentChatTool,
    AgentMemoryClearTool,
    AgentMemoryQueryTool,
    AgentMemorySearchTool,
    AgentPlanTool,
    AgentRecipeListTool,
    register_agent_tools,
)

# Retrieval tools (was retrieval_tools.py)
from kaos_agents.tools.retrieval import (
    BM25SearchTool,
    CorpusInfoTool,
    EvaluateCoverageTool,
    GroundedAnswerTool,
    HyDESearchTool,
    RerankTool,
    SynonymSearchTool,
    register_retrieval_tools,
)

__all__ = [
    "AgentChatTool",
    "AgentGraphProjectionTool",
    "AgentGraphSparqlTool",
    "AgentGraphWalkTool",
    "AgentMemoryClearTool",
    "AgentMemoryQueryTool",
    "AgentMemorySearchTool",
    "AgentPlanTool",
    "AgentRecipeListTool",
    "BM25SearchTool",
    "CorpusInfoTool",
    "EvaluateCoverageTool",
    "ExtractCorpusTool",
    "ExtractSchemaTool",
    "ExtractVerifyTool",
    "GroundedAnswerTool",
    "HyDESearchTool",
    "RerankTool",
    "SynonymSearchTool",
    "register_agent_tools",
    "register_retrieval_tools",
]
