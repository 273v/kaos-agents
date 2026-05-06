"""Tests for heuristic intent classification."""

from __future__ import annotations

from kaos_agents.context.classify import _classify_heuristic
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types import IntentType
from kaos_agents.types.memory import MemoryType


class TestHeuristicClassifier:
    def test_greeting_is_respond(self):
        mem = SessionMemory("test")
        result = _classify_heuristic("hello", mem)
        assert result.intent == IntentType.RESPOND

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
