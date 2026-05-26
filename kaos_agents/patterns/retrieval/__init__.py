"""LLM-driven retrieval planner + applier for corpus-grounded dispatch.

Composition layer over ``kaos_content.search.search_document`` and
``kaos_agents.context.triage_corpus``. The planner is an LLM
Signature + ``Call`` wrapper that mirrors the
``kaos_llm_core.programs.query_expander.LLMQueryExpander`` template —
shape parity matters because Phase 2 of the planner roadmap lifts the
primitive into ``kaos-llm-core`` so RAG and any future retrieval
program can compose it via the same Protocol.

See
``../kaos-modules/docs/plans/2026-05-26-retrieval-planner-and-findings-dispatch.md``
for the full design + lineage.
"""

from __future__ import annotations

from kaos_agents.patterns.retrieval.apply import (
    RetrievalApplyResult,
    apply_retrieval_plan,
)
from kaos_agents.patterns.retrieval.planner import (
    LLMRetrievalPlanner,
    PlanRetrieval,
    RetrievalPlanner,
)
from kaos_agents.patterns.retrieval.types import (
    RetrievalPlanResult,
    RetrievalStrategy,
)

__all__ = [
    "LLMRetrievalPlanner",
    "PlanRetrieval",
    "RetrievalApplyResult",
    "RetrievalPlanResult",
    "RetrievalPlanner",
    "RetrievalStrategy",
    "apply_retrieval_plan",
]
