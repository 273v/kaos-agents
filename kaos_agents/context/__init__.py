"""Context assembly, intent classification, and corpus triage."""

from __future__ import annotations

from kaos_agents.context.assemble import assemble_context
from kaos_agents.context.classify import classify_intent
from kaos_agents.context.doc2query import expand_document_with_queries
from kaos_agents.context.triage import TriageResult, triage_corpus

__all__ = [
    "TriageResult",
    "assemble_context",
    "classify_intent",
    "expand_document_with_queries",
    "triage_corpus",
]

# Deprecated — still importable from kaos_agents.context.retrieval directly
# but removed from the public API. See retrieval.py docstring for rationale.
