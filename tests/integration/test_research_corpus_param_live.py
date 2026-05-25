"""Live E2E: ResearchAgent(corpus=...) bypasses the memory-based path.

Proves WS-3.6: a ResearchAgent built with a pre-indexed
``kaos_ml_core.CorpusIndex`` never consults ``MemoryType.DOCUMENTS``
and still returns a verified ``Answer`` with citations. This is the
"folder → index → persist → load → Q&A" flow the roadmap sketches.

Requires ``ANTHROPIC_API_KEY`` and ``kaos-ml-core`` + ``kaos-content``
installed (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import pathlib

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

_has_ml_core = importlib.util.find_spec("kaos_ml_core") is not None
_has_content = importlib.util.find_spec("kaos_content") is not None

MODEL = respond_model()


def _build_corpus_index():
    """Construct a 3-passage Corpus + CorpusIndex over public-domain text."""
    from kaos_content.model.attr import Provenance, SourceRef
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.model.metadata import DocumentMetadata
    from kaos_ml_core.corpus import Corpus
    from kaos_ml_core.index import CorpusIndex

    def _para(text: str, page: int = 1, source: SourceRef | None = None) -> Paragraph:
        prov = Provenance(source=source, page=page) if source else Provenance(page=page)
        return Paragraph(children=(Text(value=text),), provenance=prov)

    delaware_source = SourceRef(uri="doc:delaware-gcl")
    delaware_doc = ContentDocument(
        metadata=DocumentMetadata(title="Delaware GCL", source=delaware_source),
        body=(
            _para(
                "The Delaware General Corporation Law requires that a certificate of "
                "incorporation be filed with the Secretary of State.",
                source=delaware_source,
            ),
            _para(
                "The filing fee is $89 and the certificate must include the corporation's "
                "name, its registered office, and the name of its registered agent.",
                source=delaware_source,
            ),
            _para(
                "Mergers require majority approval of the outstanding shares entitled to vote.",
                source=delaware_source,
            ),
        ),
    )
    first_amendment_source = SourceRef(uri="doc:first-amendment")
    first_amendment_doc = ContentDocument(
        metadata=DocumentMetadata(title="First Amendment", source=first_amendment_source),
        body=(
            _para(
                "Congress shall make no law respecting an establishment of religion, or "
                "prohibiting the free exercise thereof.",
                source=first_amendment_source,
            ),
            _para(
                "Or abridging the freedom of speech, or of the press; or the right of the "
                "people peaceably to assemble.",
                source=first_amendment_source,
            ),
        ),
    )
    corpus = Corpus.from_documents([delaware_doc, first_amendment_doc], level="paragraph")
    return CorpusIndex(corpus)


def _memory_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        )
    )


@pytest.mark.live
@pytest.mark.skipif(not _has_ml_core, reason="kaos-ml-core not installed")
@pytest.mark.skipif(not _has_content, reason="kaos-content not installed")
class TestResearchAgentCorpusParamLive:
    async def test_corpus_index_bypasses_memory(self) -> None:
        """ResearchAgent with a pre-built CorpusIndex answers without load_document."""
        index = _build_corpus_index()
        vfs = _memory_vfs()
        session_id = "ws-3-6-corpus-param"

        # Create an EMPTY session — no documents in memory.
        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        await store.save(memory)

        agent = ResearchAgent(
            vfs,
            model=MODEL,
            corpus=index,  # CorpusIndex → unwrapped to its Corpus
            settings=KaosAgentSettings(default_llm_model=MODEL, planning_llm_model=MODEL),
        )

        assert agent.corpus is index.corpus, (
            "ResearchAgent.corpus should be the unwrapped Corpus Protocol instance"
        )

        response = await agent.turn(
            "What is the Delaware filing fee for a certificate of incorporation?",
            session_id=session_id,
        )

        assert response.text, "ResearchAgent must yield text when corpus is pre-bound"
        assert "89" in response.text, (
            f"Answer should mention the $89 fee from the CorpusIndex: {response.text[:300]}"
        )

        # Confirm the DOCUMENTS section was NEVER touched — this proves the
        # agent routed through self._corpus, not _build_corpus(memory).
        reloaded = await store.load(session_id)
        docs = (
            reloaded.get(MemoryType.DOCUMENTS) if reloaded.has_section(MemoryType.DOCUMENTS) else []
        )
        assert len(docs) == 0, (
            "DOCUMENTS section was populated — ResearchAgent fell back to the "
            "memory path instead of using the pre-built corpus"
        )

    async def test_corpus_index_round_trip_then_query(self, tmp_path: pathlib.Path) -> None:
        """Save a CorpusIndex, reload it in a fresh process-like scope,
        hand it to ResearchAgent, answer a question.

        Models the real persistent-agent flow: index once, reuse across
        restarts.
        """
        from kaos_ml_core.index import CorpusIndex

        index_a = _build_corpus_index()
        index_a.save(tmp_path)

        index_b = CorpusIndex.load(tmp_path)
        assert index_b.size == index_a.size

        vfs = _memory_vfs()
        session_id = "ws-3-6-roundtrip"
        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        await store.save(memory)

        agent = ResearchAgent(
            vfs,
            model=MODEL,
            corpus=index_b,
            settings=KaosAgentSettings(default_llm_model=MODEL, planning_llm_model=MODEL),
        )
        response = await agent.turn(
            "What does the First Amendment say about religion?",
            session_id=session_id,
        )
        text_lower = response.text.lower()
        assert any(kw in text_lower for kw in ("religion", "establishment", "free exercise")), (
            f"Reloaded-index agent must answer First Amendment Q: {response.text[:300]}"
        )
