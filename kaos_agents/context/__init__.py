"""Context assembly, intent classification, and corpus triage."""

from __future__ import annotations

from kaos_agents.context.assemble import assemble_context
from kaos_agents.context.classify import classify_intent
from kaos_agents.context.doc2query import expand_document_with_queries
from kaos_agents.context.graph_expand import expand_with_graph
from kaos_agents.context.sections_to_prompt import bind_sections
from kaos_agents.context.tool_catalog import CatalogMode, render_tool_catalog
from kaos_agents.context.tool_filter import filter_tools
from kaos_agents.context.triage import TriageResult, triage_corpus

__all__ = [
    "CatalogMode",
    "TriageResult",
    "assemble_context",
    "bind_sections",
    "classify_intent",
    "expand_document_with_queries",
    "expand_with_graph",
    "filter_tools",
    "render_tool_catalog",
    "triage_corpus",
]

# Deprecated — still importable from kaos_agents.context.retrieval directly
# but removed from the public API. See retrieval.py docstring for rationale.
