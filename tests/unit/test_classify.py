"""Tests for heuristic intent classification."""

from __future__ import annotations

from typing import Any

import pytest

from kaos_agents._constants import (
    HEURISTIC_CONFIDENCE_DEFAULT,
    HEURISTIC_CONFIDENCE_GREETING,
    HEURISTIC_CONFIDENCE_PLAN,
    HEURISTIC_CONFIDENCE_RESEARCH,
    HEURISTIC_CONFIDENCE_TOOL_USE,
)
from kaos_agents.context.classify import (
    ClassifyIntentSignature,
    _classify_heuristic,
    classify_intent,
)
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types import IntentType
from kaos_agents.types.memory import MemoryType


class TestHeuristicClassifier:
    def test_greeting_is_respond(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("hello", mem)
        assert result.intent == IntentType.RESPOND

    def test_heuristic_confidences_come_from_named_constants(self):
        """The fallback classifier must source its confidences from the
        HEURISTIC_CONFIDENCE_* constants, not re-inline magic numbers."""
        mem = SessionMemory("test")
        assert _classify_heuristic("hello", mem).confidence == HEURISTIC_CONFIDENCE_GREETING
        assert (
            _classify_heuristic("the thing with the stuff", mem).confidence
            == HEURISTIC_CONFIDENCE_DEFAULT
        )
        assert (
            _classify_heuristic("first do X, then do Y", mem).confidence
            == HEURISTIC_CONFIDENCE_PLAN
        )
        assert (
            _classify_heuristic("search for SEC filings", mem).confidence
            == HEURISTIC_CONFIDENCE_TOOL_USE
        )
        docs_mem = SessionMemory("test-docs")
        docs_mem.add(MemoryType.DOCUMENTS, "contract.pdf (15 pages)")
        assert (
            _classify_heuristic("what are the key dates?", docs_mem).confidence
            == HEURISTIC_CONFIDENCE_RESEARCH
        )

    def test_hi_is_respond(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("hi", mem)
        assert result.intent == IntentType.RESPOND

    def test_thanks_is_respond(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("thanks", mem)
        assert result.intent == IntentType.RESPOND

    def test_action_word_is_tool_use(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("extract dates from the contract", mem)
        assert result.intent == IntentType.TOOL_USE

    def test_search_is_tool_use(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("search for SEC filings", mem)
        assert result.intent == IntentType.TOOL_USE

    def test_question_with_docs_is_research(self):
        mem = SessionMemory("test")
        # Add a document reference to trigger research intent
        mem.add(MemoryType.DOCUMENTS, "contract.pdf (15 pages)")
        result = _classify_heuristic("what are the key dates in this contract?", mem)
        assert result.intent == IntentType.RESEARCH

    def test_question_without_docs_is_tool_use(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("what is the current stock price?", mem)
        # Without docs loaded, questions default to tool_use (not research)
        assert result.intent == IntentType.TOOL_USE

    def test_multistep_is_plan(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("first analyze the document, then extract the dates", mem)
        assert result.intent == IntentType.PLAN

    def test_step_by_step_is_plan(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("give me step by step instructions", mem)
        assert result.intent == IntentType.PLAN

    def test_ambiguous_defaults_to_tool_use(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("the thing with the stuff", mem)
        assert result.intent == IntentType.TOOL_USE
        assert result.confidence <= 0.5

    def test_confidence_is_bounded(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("hello", mem)
        assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_is_populated(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("hello", mem)
        assert "heuristic" in result.reasoning.lower()


class TestClassifyIntentSignatureCatalogField:
    """0.1.0a17 — dynamic tool-catalog awareness on the classifier.

    The new ``available_tool_categories`` InputField is the seam that
    makes the catalog-aware bullets in the docstring something the
    classifier can actually act on. Default ``""`` preserves the
    pre-fix routing path (callers that don't populate the input see
    no behavior change).
    """

    def test_available_tool_categories_field_exists(self):
        assert "available_tool_categories" in ClassifyIntentSignature.model_fields

    def test_available_tool_categories_defaults_to_empty_string(self):
        field = ClassifyIntentSignature.model_fields["available_tool_categories"]
        assert field.default == ""

    def test_available_tool_categories_description_mentions_categories(self):
        field = ClassifyIntentSignature.model_fields["available_tool_categories"]
        # Description must reference the catalog concept so an
        # InstructionOptimizer or human auditor can find it; we do NOT
        # assert specific tool / category names here (the rule is
        # abstract — see the brief's hard constraint).
        desc = (field.description or "").lower()
        assert "categor" in desc or "catalog" in desc or "registered" in desc

    def test_docstring_references_available_tool_categories(self):
        """The Signature docstring is what reaches the LLM. It must
        instruct the model to consult the new input — otherwise the
        InputField is dead weight."""
        doc = (ClassifyIntentSignature.__doc__ or "").lower()
        assert "available_tool_categories" in doc

    def test_docstring_avoids_hardcoded_tool_category_names(self):
        """The brief's hard constraint: no hardcoded tool / category
        names in the Signature docstring. The rule must be abstract."""
        doc = (ClassifyIntentSignature.__doc__ or "").lower()
        for banned in ("fr-search", "fetch-url", "edgar", "ecfr", "web-search"):
            assert banned not in doc, f"docstring leaked a hardcoded category reference: {banned!r}"


class TestClassifyIntentBackwardCompat:
    """``classify_intent`` accepts the new ``available_tool_categories``
    kwarg but the pre-0.1.0a17 calling convention still works.

    Both paths are exercised against a stubbed ``_classify_with_llm``
    so no LLM is hit and no live model is required.
    """

    @pytest.mark.asyncio
    async def test_no_catalog_kwarg_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-0.1.0a17 callers (no kwarg) still resolve correctly and
        the underlying classifier sees ``available_tool_categories=""``."""
        seen: dict[str, Any] = {}

        async def fake_classify(
            user_message: str,
            memory: SessionMemory,
            *,
            model: str,
            context_text: str = "",
            available_tool_categories: str = "",
            documents_available: str = "",
        ):
            seen["available_tool_categories"] = available_tool_categories
            from kaos_agents.types import IntentResult

            return IntentResult(
                intent=IntentType.RESPOND,
                confidence=0.7,
                reasoning="stub",
            )

        monkeypatch.setattr(
            "kaos_agents.context.classify._classify_with_llm",
            fake_classify,
        )
        mem = SessionMemory("test")
        result = await classify_intent("hello", mem)
        assert seen["available_tool_categories"] == ""
        assert result.intent == IntentType.RESPOND

    @pytest.mark.asyncio
    async def test_catalog_kwarg_threads_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        catalog = "web: search the internet\nresearch: query loaded docs"

        async def fake_classify(
            user_message: str,
            memory: SessionMemory,
            *,
            model: str,
            context_text: str = "",
            available_tool_categories: str = "",
            documents_available: str = "",
        ):
            seen["available_tool_categories"] = available_tool_categories
            from kaos_agents.types import IntentResult

            return IntentResult(
                intent=IntentType.TOOL_USE,
                confidence=0.9,
                reasoning="stub",
            )

        monkeypatch.setattr(
            "kaos_agents.context.classify._classify_with_llm",
            fake_classify,
        )
        mem = SessionMemory("test")
        result = await classify_intent(
            "who is the current US senator for Lansing Michigan?",
            mem,
            available_tool_categories=catalog,
        )
        assert seen["available_tool_categories"] == catalog
        assert result.intent == IntentType.TOOL_USE


class TestRenderToolCategoriesForClassifier:
    """The pure helper that converts a runtime to the catalog string.

    Property: with no runtime / no tools the helper returns the empty
    string. That's the regression guard that pre-fix call sites
    continue to see no behavior change.
    """

    def test_none_runtime_returns_empty_string(self) -> None:
        from kaos_agents.context.tool_catalog import (
            render_tool_categories_for_classifier,
        )

        assert render_tool_categories_for_classifier(None) == ""

    def test_runtime_without_tools_attr_returns_empty_string(self) -> None:
        from kaos_agents.context.tool_catalog import (
            render_tool_categories_for_classifier,
        )

        class _Bare:
            pass

        assert render_tool_categories_for_classifier(_Bare()) == ""

    def test_runtime_with_empty_tools_returns_empty_string(self) -> None:
        from kaos_agents.context.tool_catalog import (
            render_tool_categories_for_classifier,
        )

        class _ToolsRegistry:
            def list_tool_objects(self) -> list[Any]:
                return []

            def list_tools(self) -> list[str]:
                return []

        class _Runtime:
            tools = _ToolsRegistry()

        assert render_tool_categories_for_classifier(_Runtime()) == ""

    def test_runtime_with_tools_and_groups_renders_one_line_per_group(self) -> None:
        from kaos_agents.context.tool_catalog import (
            render_tool_categories_for_classifier,
        )
        from kaos_agents.registry import ToolGroupRegistry
        from kaos_agents.types import ToolGroup

        class _Meta:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description

        class _Tool:
            def __init__(self, name: str, description: str) -> None:
                self.metadata = _Meta(name, description)

        class _ToolsRegistry:
            def __init__(self, names: list[tuple[str, str]]) -> None:
                self._objs = [_Tool(n, d) for n, d in names]

            def list_tool_objects(self) -> list[Any]:
                return list(self._objs)

            def list_tools(self) -> list[str]:
                return [t.metadata.name for t in self._objs]

        class _Runtime:
            def __init__(self, names: list[tuple[str, str]]) -> None:
                self.tools = _ToolsRegistry(names)

        registry = ToolGroupRegistry()
        registry.register(
            ToolGroup(
                name="search",
                description="Tools for querying current real-world facts.",
                tool_names=("alpha-search", "beta-search"),
            )
        )
        registry.register(
            ToolGroup(
                name="documents",
                description="Tools for reading loaded documents.",
                tool_names=("docs-query",),
            )
        )

        runtime = _Runtime(
            [
                ("alpha-search", "search alpha"),
                ("beta-search", "search beta"),
                ("docs-query", "query docs"),
            ]
        )
        out = render_tool_categories_for_classifier(runtime, group_registry=registry)
        lines = out.splitlines()
        # One line per group present in the catalog. Order is
        # registration order (search before documents).
        assert lines == [
            "search: Tools for querying current real-world facts.",
            "documents: Tools for reading loaded documents.",
        ]

    def test_ungrouped_tools_appear_on_their_own_lines(self) -> None:
        from kaos_agents.context.tool_catalog import (
            render_tool_categories_for_classifier,
        )
        from kaos_agents.registry import ToolGroupRegistry

        class _Meta:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description

        class _Tool:
            def __init__(self, name: str, description: str) -> None:
                self.metadata = _Meta(name, description)

        class _ToolsRegistry:
            def __init__(self, objs: list[_Tool]) -> None:
                self._objs = objs

            def list_tool_objects(self) -> list[Any]:
                return list(self._objs)

            def list_tools(self) -> list[str]:
                return [t.metadata.name for t in self._objs]

        class _Runtime:
            def __init__(self, objs: list[_Tool]) -> None:
                self.tools = _ToolsRegistry(objs)

        runtime = _Runtime(
            [
                _Tool("loose-tool", "an ungrouped tool"),
            ]
        )
        out = render_tool_categories_for_classifier(runtime, group_registry=ToolGroupRegistry())
        assert out == "loose-tool: an ungrouped tool"


class TestClassifyWithLLMPropagatesCatalog:
    """Threads the catalog input through the ``Call.invoke`` boundary.

    Uses an AsyncMock to substitute the inner Call without hitting an
    LLM. Confirms the kwarg lands on the inner ``call.invoke`` call.
    """

    @pytest.mark.asyncio
    async def test_call_invoke_receives_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kaos_llm_core import Call
        from kaos_llm_core.programs._invocation import Invocation, TokenUsage

        captured: dict[str, Any] = {}

        async def fake_invoke(self: Call, **kwargs: Any) -> Invocation:
            captured.update(kwargs)
            return Invocation(
                client=None,
                model="anthropic:claude-haiku-4-5",
                context=None,
                output=ClassifyIntentSignature(
                    message=kwargs["message"],
                    conversation_context=kwargs.get("conversation_context", ""),
                    available_tool_categories=kwargs.get("available_tool_categories", ""),
                    intent="tool_use",
                    confidence=0.85,
                    reasoning="catalog-aware",
                ),
                trace=None,
                usage=TokenUsage(),
            )

        monkeypatch.setattr(Call, "invoke", fake_invoke)
        mem = SessionMemory("test")
        catalog = "search: current real-world facts"
        result = await classify_intent(
            "what is X today?",
            mem,
            model="anthropic:claude-haiku-4-5",
            available_tool_categories=catalog,
        )
        assert captured["available_tool_categories"] == catalog
        assert result.intent == IntentType.TOOL_USE
