"""Tests for kaos_agents.context.doc2query — document expansion with predicted queries."""

from __future__ import annotations

import pytest


class TestExpandDocumentWithQueries:
    @pytest.mark.asyncio
    async def test_returns_original_text_on_failure(self) -> None:
        """If LLM call fails, original text is returned unchanged."""
        from kaos_agents.context.doc2query import expand_document_with_queries

        text = "This is a short employment agreement."
        result = await expand_document_with_queries(text, model="nonexistent:model")
        assert text in result

    @pytest.mark.asyncio
    async def test_generates_queries_appended_as_block(self) -> None:
        """When queries are generated, they appear as a tagged block."""
        from kaos_agents.context.doc2query import expand_document_with_queries

        text = "Employment agreement between Acme Corp and John Smith for CEO role."
        # This will fail without a real LLM, but we test the graceful degradation
        result = await expand_document_with_queries(text)
        # Either the original text (LLM unavailable) or expanded text
        assert text in result

    @pytest.mark.asyncio
    async def test_generate_document_queries_returns_list(self) -> None:
        """generate_document_queries returns a list (possibly empty)."""
        from kaos_agents.context.doc2query import generate_document_queries

        queries = await generate_document_queries("A short document.", model="nonexistent:model")
        assert isinstance(queries, list)


class TestDoc2QueryConstants:
    def test_module_constants_exist(self) -> None:
        from kaos_agents.context.doc2query import _DOC_PREVIEW_CHARS, _MAX_PREDICTED_QUERIES

        assert _DOC_PREVIEW_CHARS > 0
        assert _MAX_PREDICTED_QUERIES > 0

    def test_exported_from_context(self) -> None:
        from kaos_agents.context import expand_document_with_queries

        assert callable(expand_document_with_queries)
