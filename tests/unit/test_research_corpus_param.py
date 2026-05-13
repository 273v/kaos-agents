"""Unit tests for the WS-3.6 ``ResearchAgent(corpus=...)`` parameter.

Covers:
- ``corpus`` accessor returns what was passed.
- CorpusIndex-shaped args are unwrapped to their ``.corpus``.
- The ``_unwrap_corpus_arg`` helper is tolerant of duck-typed inputs.
- None ``corpus`` preserves the legacy memory-based flow.

Live end-to-end coverage lives in ``tests/integration/test_research_corpus_live.py``.
"""

from __future__ import annotations

import pytest
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.patterns.research import ResearchAgent, _unwrap_corpus_arg


class _FakeCorpus:
    """A stand-in that satisfies the Corpus Protocol's public shape."""

    def __init__(self, passages: list[str]) -> None:
        self._passages = tuple(passages)

    def iter_passages(self):
        return iter(self._passages)

    def get_passage(self, row: int):
        return self._passages[row]

    @property
    def size(self) -> int:
        return len(self._passages)


class CorpusIndex:
    """A stand-in shaped + named like kaos_ml_core.CorpusIndex.

    Class name must be literally ``CorpusIndex`` because
    ``_unwrap_corpus_arg`` keys the unwrap branch on the type name to
    avoid a hard import of kaos-ml-core.
    """

    def __init__(self, corpus: _FakeCorpus) -> None:
        self.corpus = corpus


@pytest.fixture()
def vfs() -> VirtualFileSystem:
    return VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))


@pytest.mark.unit
class TestUnwrapCorpusArg:
    def test_none_passes_through(self) -> None:
        assert _unwrap_corpus_arg(None) is None

    def test_corpus_protocol_passes_through(self) -> None:
        corpus = _FakeCorpus(["alpha", "beta"])
        assert _unwrap_corpus_arg(corpus) is corpus

    def test_corpus_index_is_unwrapped(self) -> None:
        corpus = _FakeCorpus(["x", "y"])
        index = CorpusIndex(corpus)
        unwrapped = _unwrap_corpus_arg(index)
        assert unwrapped is corpus

    def test_non_corpus_index_with_corpus_attr_not_unwrapped(self) -> None:
        """Only objects literally named ``CorpusIndex`` get unwrapped —
        prevents accidental unwrap of unrelated types that happen to
        carry a ``.corpus`` attribute."""

        class _HasCorpusAttr:
            corpus = _FakeCorpus(["p"])

        obj = _HasCorpusAttr()
        assert _unwrap_corpus_arg(obj) is obj


@pytest.mark.unit
class TestResearchAgentCorpusAccessor:
    def test_default_corpus_is_none(self, vfs: VirtualFileSystem) -> None:
        agent = ResearchAgent(vfs=vfs)
        assert agent.corpus is None

    def test_accepts_protocol_instance(self, vfs: VirtualFileSystem) -> None:
        corpus = _FakeCorpus(["alpha", "beta"])
        agent = ResearchAgent(vfs=vfs, corpus=corpus)
        assert agent.corpus is corpus

    def test_unwraps_corpus_index_in_ctor(self, vfs: VirtualFileSystem) -> None:
        corpus = _FakeCorpus(["alpha"])
        index = CorpusIndex(corpus)
        agent = ResearchAgent(vfs=vfs, corpus=index)
        assert agent.corpus is corpus, (
            "ResearchAgent must unwrap a CorpusIndex to its underlying Corpus "
            "so RAG.query(documents=agent.corpus) receives a Protocol instance"
        )
