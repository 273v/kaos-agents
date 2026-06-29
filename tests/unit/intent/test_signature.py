"""Tests for :class:`kaos_agents.intent.signature.IntentSignature`.

These tests pin the wire shape of the IntentSignature — particularly
the 0.1.0a17 ``available_tool_groups`` InputField that drives the
catalog-aware factual-external-entity bias (rule 8). The Signature
docstring is the LLM-facing instruction surface, so we also assert
that the docstring references the new field and does NOT leak any
hardcoded tool / category names (the brief's hard constraint).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage

from kaos_agents.config import AgentPattern
from kaos_agents.intent.extractor import IntentExtractor
from kaos_agents.intent.signature import IntentSignature
from kaos_agents.intent.types import Goal
from kaos_agents.types.intents import IntentType


def _make_signature(**overrides: Any) -> IntentSignature:
    fields: dict[str, Any] = {
        "message": overrides.pop("message", "test message"),
        "recent_messages": overrides.pop("recent_messages", ""),
        "domain_examples": overrides.pop("domain_examples", ""),
        "goal": overrides.pop(
            "goal",
            Goal(statement="Do X.", intent_type=IntentType.RESPOND),
        ),
    }
    fields.update(overrides)
    return IntentSignature(**fields)


def _make_invocation(output: Any) -> Invocation:
    return Invocation(
        client=None,
        model="anthropic:claude-haiku-4-5",
        context=None,
        output=output,
        trace=None,
        usage=TokenUsage(),
    )


class TestIntentSignatureCatalogField:
    """0.1.0a17 — dynamic tool-catalog awareness on IntentSignature."""

    def test_available_tool_groups_field_exists(self) -> None:
        assert "available_tool_groups" in IntentSignature.model_fields

    def test_available_tool_groups_default_is_empty_string(self) -> None:
        field = IntentSignature.model_fields["available_tool_groups"]
        assert field.default == ""

    def test_available_tool_groups_description_mentions_catalog(self) -> None:
        field = IntentSignature.model_fields["available_tool_groups"]
        desc = (field.description or "").lower()
        # Description must reference the catalog concept without
        # naming any specific tool / group (the rule is abstract).
        assert "group" in desc
        assert "registered" in desc or "catalog" in desc

    def test_docstring_references_available_tool_groups(self) -> None:
        """The Signature docstring is the LLM-facing instruction. It
        must teach the model to consult the new input or the field is
        dead weight at runtime."""
        doc = IntentSignature.__doc__ or ""
        assert "available_tool_groups" in doc

    def test_docstring_avoids_hardcoded_tool_category_names(self) -> None:
        """No hardcoded tool / category names in the docstring. The
        rule must be abstract — see the 0.1.0a17 brief's hard
        constraint."""
        doc = (IntentSignature.__doc__ or "").lower()
        # Pre-0.1.0a17 rule 8 listed "FR / eCFR / EDGAR / GovInfo /
        # web-search" verbatim. Those references must be gone.
        for banned in (
            "fr / ecfr",
            "edgar",
            "govinfo",
            "web-search",
            "ecfr",
            "fr-search",
        ):
            assert banned not in doc, f"docstring leaked a hardcoded category reference: {banned!r}"


class TestIntentExtractorThreadsCatalog:
    """The extractor must thread ``available_tool_groups`` through to
    the inner Call. Without this, the InputField is set on the
    Signature but never populated at invocation time."""

    @pytest.mark.asyncio
    async def test_forward_passes_available_tool_groups_to_call(self) -> None:
        ex = IntentExtractor()
        sig_out = _make_signature(pattern=AgentPattern.CHAT)
        invocation = _make_invocation(sig_out)
        mock = AsyncMock(return_value=invocation)
        ex._call.invoke = mock
        catalog = (
            "search: tools for querying current real-world facts\n"
            "documents: tools for reading loaded documents"
        )
        await ex.forward(
            message="who is the current senator for X?",
            available_tool_groups=catalog,
        )
        mock.assert_awaited_once()
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["available_tool_groups"] == catalog

    @pytest.mark.asyncio
    async def test_forward_defaults_available_tool_groups_to_empty(self) -> None:
        """Backward compatibility: callers that don't pass the new
        kwarg still resolve, and the inner Call sees ``""``."""
        ex = IntentExtractor()
        sig_out = _make_signature()
        invocation = _make_invocation(sig_out)
        mock = AsyncMock(return_value=invocation)
        ex._call.invoke = mock
        await ex.forward(message="hi")
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["available_tool_groups"] == ""
