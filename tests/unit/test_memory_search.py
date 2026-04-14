"""Unit tests for BM25 memory search."""

from __future__ import annotations

from kaos_agents.memory.search import MemorySearchResult, search_memory
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.types import MemoryType


class TestMemorySearch:
    def test_search_messages(self):
        mem = SessionMemory("test", chars_per_token=4.0)
        mem.add(MemoryType.MESSAGES, "user: What is the filing fee in Delaware?")
        mem.add(MemoryType.MESSAGES, "assistant: The filing fee is $89.")
        mem.add(MemoryType.MESSAGES, "user: What about New York?")
        mem.add(MemoryType.MESSAGES, "assistant: New York charges $125.")

        results = search_memory(mem, "filing fee")
        assert len(results) > 0
        assert isinstance(results[0], MemorySearchResult)
        assert results[0].section == MemoryType.MESSAGES
        # Most relevant result should mention filing fee
        assert "filing fee" in results[0].content.lower() or "89" in results[0].content

    def test_search_across_sections(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "What is the EPA rule?")
        mem.add(MemoryType.FINDINGS, "EPA emission standard: 50 ppm for particulates")
        mem.add(MemoryType.ACTIONS, "Tool: kaos-source-fr-search(query=EPA) → 10 results")

        results = search_memory(mem, "EPA emission")
        assert len(results) > 0
        # FINDINGS should rank high for this query
        findings_results = [r for r in results if r.section == MemoryType.FINDINGS]
        assert len(findings_results) > 0

    def test_search_empty_memory(self):
        mem = SessionMemory("test")
        results = search_memory(mem, "anything")
        assert results == []

    def test_search_specific_sections(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "message about contracts")
        mem.add(MemoryType.FINDINGS, "finding about contracts")

        # Search only FINDINGS
        results = search_memory(mem, "contracts", sections=[MemoryType.FINDINGS])
        for r in results:
            assert r.section == MemoryType.FINDINGS

    def test_search_top_k(self):
        mem = SessionMemory("test")
        for i in range(20):
            mem.add(MemoryType.MESSAGES, f"Message {i} about Delaware law")

        results = search_memory(mem, "Delaware", top_k=5)
        assert len(results) <= 5

    def test_search_result_has_score(self):
        mem = SessionMemory("test")
        mem.add(MemoryType.MESSAGES, "The filing fee is $89 in Delaware")
        results = search_memory(mem, "filing fee Delaware")
        assert len(results) > 0
        assert results[0].score > 0
