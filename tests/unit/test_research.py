"""Unit tests for ResearchAgent — corpus building and document loading."""

from __future__ import annotations

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.types import MemoryType
from kaos_agents.patterns.research import ResearchAgent, _build_corpus


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


@pytest.mark.unit
class TestBuildCorpus:
    """Test the _build_corpus helper that extracts documents from memory."""

    def test_empty_memory(self):
        memory = SessionMemory("test-empty")
        corpus = _build_corpus(memory)
        assert corpus == {}

    def test_single_document_with_uri_metadata(self):
        memory = SessionMemory("test-single")
        memory.add(
            MemoryType.DOCUMENTS,
            "URI: doc:contract-1\nThis is a contract about services.",
            metadata={"uri": "doc:contract-1"},
        )
        corpus = _build_corpus(memory)
        assert len(corpus) == 1
        assert "doc:contract-1" in corpus
        assert "This is a contract about services." in corpus["doc:contract-1"]

    def test_multiple_documents(self):
        memory = SessionMemory("test-multi")
        memory.add(
            MemoryType.DOCUMENTS,
            "URI: doc:a\nDocument A content.",
            metadata={"uri": "doc:a"},
        )
        memory.add(
            MemoryType.DOCUMENTS,
            "URI: doc:b\nDocument B content.",
            metadata={"uri": "doc:b"},
        )
        corpus = _build_corpus(memory)
        assert len(corpus) == 2
        assert "Document A content." in corpus["doc:a"]
        assert "Document B content." in corpus["doc:b"]

    def test_uri_parsed_from_content_without_metadata(self):
        memory = SessionMemory("test-parse-uri")
        memory.add(MemoryType.DOCUMENTS, "URI: doc:parsed\nParsed from content prefix.")
        corpus = _build_corpus(memory)
        assert "doc:parsed" in corpus

    def test_fallback_to_item_id(self):
        memory = SessionMemory("test-fallback")
        memory.add(MemoryType.DOCUMENTS, "Just some text without a URI prefix.")
        corpus = _build_corpus(memory)
        assert len(corpus) == 1
        key = next(iter(corpus))
        assert key.startswith("mem:")
        assert "Just some text" in corpus[key]


@pytest.mark.unit
class TestLoadDocument:
    """Test ResearchAgent.load_document helper."""

    def test_load_stores_in_documents_section(self):
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs)
        memory = SessionMemory("test-load")

        agent.load_document(memory, "doc:test-1", "This is test document content.")

        assert memory.section_item_count(MemoryType.DOCUMENTS) == 1
        items = memory.get_recent(MemoryType.DOCUMENTS, 1)
        assert items[0].metadata["uri"] == "doc:test-1"
        assert "This is test document content." in items[0].content

    def test_load_multiple_documents(self):
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs)
        memory = SessionMemory("test-load-multi")

        agent.load_document(memory, "doc:a", "Content A")
        agent.load_document(memory, "doc:b", "Content B")

        corpus = _build_corpus(memory)
        assert len(corpus) == 2

    def test_corpus_round_trip(self):
        """load_document → _build_corpus should recover the original text."""
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs)
        memory = SessionMemory("test-roundtrip")

        original_text = (
            "The filing fee is $89 and the certificate must include the corporation's name."
        )
        agent.load_document(memory, "doc:delaware-gcl", original_text)

        corpus = _build_corpus(memory)
        assert corpus["doc:delaware-gcl"] == original_text
