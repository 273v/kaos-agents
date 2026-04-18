"""Context assembly, intent classification, corpus triage, and adaptive retrieval."""

from __future__ import annotations

from kaos_agents.context.assemble import assemble_context
from kaos_agents.context.classify import classify_intent
from kaos_agents.context.doc2query import expand_document_with_queries
from kaos_agents.context.retrieval import (
    AdaptiveRetrievalResult,
    adaptive_retrieve,
    precompute_embeddings,
)
from kaos_agents.context.triage import TriageResult, triage_corpus

__all__ = [
    "AdaptiveRetrievalResult",
    "TriageResult",
    "adaptive_retrieve",
    "assemble_context",
    "classify_intent",
    "expand_document_with_queries",
    "precompute_embeddings",
    "triage_corpus",
]
