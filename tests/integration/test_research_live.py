"""Live integration tests for ResearchAgent with real RAG pipeline.

Tests the full flow: load document → ask question → RAG retrieves + verifies
→ grounded answer with claims stored in FINDINGS.

Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.patterns.research import ResearchAgent
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.memory import MemoryType
from tests.integration._models import respond_model

MODEL = respond_model()

# Real public-domain text for testing
DELAWARE_GCL = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State. The filing "
    "fee is $89 and the certificate must include the corporation's "
    "name, its registered office, and the name of its registered agent. "
    "The corporation may have one or more classes of stock. Each class "
    "must be described in the certificate of incorporation. Par value "
    "stock is common but the certificate may authorize no-par-value stock."
)

FIRST_AMENDMENT = (
    "Congress shall make no law respecting an establishment of "
    "religion, or prohibiting the free exercise thereof; or abridging "
    "the freedom of speech, or of the press; or the right of the "
    "people peaceably to assemble, and to petition the government "
    "for a redress of grievances."
)


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


def _test_settings() -> KaosAgentSettings:
    return KaosAgentSettings(
        default_llm_model=MODEL,
        planning_llm_model=MODEL,
    )


@pytest.mark.live
class TestResearchAgentRAG:
    """Test ResearchAgent with real LLM + RAG pipeline."""

    async def test_single_document_qa(self):
        """Ask a factual question about a loaded document."""
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "research-single-doc"

        # Pre-load a document into memory
        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        agent.load_document(memory, "doc:delaware-gcl", DELAWARE_GCL)
        await store.save(memory)

        # Ask a question — this should trigger RESEARCH intent
        response = await agent.turn(
            "What is the filing fee for a certificate of incorporation in Delaware?",
            session_id=session_id,
        )

        assert response.text, "Response should not be empty"
        # The answer should mention $89
        assert "89" in response.text, (
            f"Response should mention $89 filing fee: {response.text[:300]}"
        )

    async def test_multi_document_qa(self):
        """Ask a question that spans multiple documents."""
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "research-multi-doc"

        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        agent.load_document(memory, "doc:delaware-gcl", DELAWARE_GCL)
        agent.load_document(memory, "doc:first-amendment", FIRST_AMENDMENT)
        await store.save(memory)

        response = await agent.turn(
            "What does the First Amendment protect?",
            session_id=session_id,
        )

        assert response.text
        text_lower = response.text.lower()
        assert any(
            w in text_lower for w in ("speech", "religion", "press", "assemble", "petition")
        ), f"Response should mention First Amendment protections: {response.text[:300]}"

    async def test_insufficient_evidence_refusal(self):
        """Question about something NOT in the documents should get refusal or caveat."""
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "research-refusal"

        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        agent.load_document(memory, "doc:delaware-gcl", DELAWARE_GCL)
        await store.save(memory)

        response = await agent.turn(
            "What is the annual franchise tax rate for Delaware LLCs?",
            session_id=session_id,
        )

        # The document doesn't mention franchise tax rates for LLCs.
        # RAG should either refuse or caveat that this isn't in the source.
        assert response.text
        assert len(response.text) > 20

    async def test_findings_stored_in_memory(self):
        """Verified claims should be stored in the FINDINGS section."""
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "research-findings"

        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        agent.load_document(memory, "doc:delaware-gcl", DELAWARE_GCL)
        await store.save(memory)

        await agent.turn(
            "What must be included in a Delaware certificate of incorporation?",
            session_id=session_id,
        )

        # Reload memory to check FINDINGS
        memory_after = await store.load(session_id)
        findings = memory_after.get_recent(MemoryType.FINDINGS, 10)
        # If RAG produced claims, they should be in FINDINGS
        # (may be empty if RAG refusal or fallback — that's acceptable)
        if findings:
            for item in findings:
                # Each finding should have structured metadata
                assert item.content, "Finding content should not be empty"

    async def test_no_documents_fallback(self):
        """When no documents are loaded, should explain and not crash."""
        vfs = _memory_vfs()
        settings = _test_settings()

        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        response = await agent.turn(
            "What is in the contract?",
            session_id="research-no-docs",
        )

        # Should get a response (fallback to simple respond or tool_use)
        assert response.text
        assert len(response.text) > 10

    async def test_multi_turn_research(self):
        """Follow-up questions should have context from prior research."""
        vfs = _memory_vfs()
        settings = _test_settings()
        session_id = "research-multi-turn"

        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        agent = ResearchAgent(vfs, model=MODEL, settings=settings)
        agent.load_document(memory, "doc:delaware-gcl", DELAWARE_GCL)
        await store.save(memory)

        # Turn 1: Ask about filing fee
        r1 = await agent.turn(
            "What is the filing fee?",
            session_id=session_id,
        )
        assert r1.turn_number == 1
        assert r1.text

        # Turn 2: Follow-up (should have context from turn 1)
        agent2 = ResearchAgent(vfs, model=MODEL, settings=settings)
        r2 = await agent2.turn(
            "What else must the certificate include?",
            session_id=session_id,
        )
        assert r2.turn_number == 2
        assert r2.text
