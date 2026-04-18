"""Doc2Query — expand documents at index time with predicted queries.

For each document, generates 5-10 predicted queries that a user might
ask about it. These queries are appended to the document text so that
BM25 matches future user query vocabulary at index time.

Uses kaos-llm-core's Call to generate queries via LLM. One-time cost
per document (~$0.001 per doc on Haiku).

Usage::

    from kaos_agents.context.doc2query import expand_document_with_queries

    expanded_text = await expand_document_with_queries(document_text)
    # Original text + "\\n\\n[PREDICTED QUERIES]\\n- <predicted question 1>\\n- <predicted question 2>\\n..."
"""

from __future__ import annotations

from kaos_core.logging import get_logger

logger = get_logger(__name__)

_DOC_PREVIEW_CHARS = 3000  # Chars of document text sent to LLM for query prediction
_MAX_PREDICTED_QUERIES = 7  # Max queries to generate per document


async def expand_document_with_queries(
    text: str,
    *,
    model: str | None = None,
    max_queries: int = _MAX_PREDICTED_QUERIES,
) -> str:
    """Generate predicted queries and append them to document text.

    Args:
        text: The document text.
        model: LLM model for query generation. Defaults to settings default.
        max_queries: Maximum queries to generate.

    Returns:
        Original text with predicted queries appended as a tagged block.
    """
    queries = await generate_document_queries(text, model=model, max_queries=max_queries)
    if not queries:
        return text

    query_block = "\n".join(f"- {q}" for q in queries)
    return f"{text}\n\n[PREDICTED QUERIES]\n{query_block}"


async def generate_document_queries(
    text: str,
    *,
    model: str | None = None,
    max_queries: int = _MAX_PREDICTED_QUERIES,
) -> list[str]:
    """Generate predicted search queries for a document.

    Args:
        text: The document text (truncated to first 3000 chars for LLM).
        model: LLM model. Defaults to settings default.
        max_queries: Maximum queries to generate.

    Returns:
        List of predicted query strings.
    """
    try:
        from kaos_llm_core import Call, InputField, OutputField, Signature

        class PredictDocumentQueries(Signature):
            """Given a document excerpt, predict 5-7 specific questions that
            a researcher or analyst might ask about this document. Focus on:
            (a) key findings, claims, or provisions, (b) entities and their
            relationships, (c) dates, quantities, and measurements,
            (d) conditions, exceptions, and caveats, (e) implications and
            significance.

            Use both the document's own terminology AND simpler paraphrases
            that a non-expert might search for."""

            document_excerpt: str = InputField(description="First ~3000 characters of the document")
            predicted_queries: list[str] = OutputField(
                description="5-7 specific questions someone might search for about this document"
            )

        from kaos_agents.settings import DEFAULT_MODEL

        call = Call(PredictDocumentQueries, model=model or DEFAULT_MODEL)
        result = await call(document_excerpt=text[:_DOC_PREVIEW_CHARS])

        queries = result.predicted_queries if hasattr(result, "predicted_queries") else []
        logger.debug(
            "doc2query.generated: doc_len=%d queries=%d",
            len(text),
            len(queries),
        )
        return queries[:max_queries]
    except Exception as exc:
        logger.debug("doc2query.failed: %s", exc)
        return []
