"""KnowledgeBase unit tests — Phase 4.A."""

from __future__ import annotations

import pytest

from kaos_agents.memory.institutional import KBEntry, KBQuery, KBResult, KnowledgeBase
from kaos_agents.memory.isolation import MatterClientGuard, MatterIsolationError


def _entry(
    eid: str,
    statement: str,
    *,
    matter_client: tuple[str, str] = ("m1", "c1"),
    confidence: float = 0.9,
    grounding_verified: bool = True,
) -> KBEntry:
    return KBEntry(
        id=eid,
        statement=statement,
        matter_client=matter_client,
        confidence=confidence,
        grounding_verified=grounding_verified,
    )


class TestValueTypeDefaults:
    def test_kb_entry_defaults(self) -> None:
        e = _entry("e1", "hello world")
        assert e.id == "e1"
        assert e.statement == "hello world"
        assert e.matter_client == ("m1", "c1")
        assert e.confidence == 0.9
        assert e.grounding_verified is True
        assert e.provenance == ()
        assert e.metadata == {}
        assert e.created_at > 0.0

    def test_kb_query_defaults(self) -> None:
        q = KBQuery(query_text="hello", matter_client=("m1", "c1"))
        assert q.top_k == 10
        assert q.min_confidence == 0.0

    def test_kb_result_default_method(self) -> None:
        q = KBQuery(query_text="x", matter_client=("m1", "c1"))
        res = KBResult(entries=(), query=q)
        assert res.method == "bm25"


class TestAddAndQuery:
    def test_add_and_query_returns_namespaced_entries(self) -> None:
        kb = KnowledgeBase()
        kb.add(_entry("e1", "patent infringement claim"))
        kb.add(_entry("e2", "trademark dispute"))
        res = kb.query(KBQuery(query_text="patent", matter_client=("m1", "c1")))
        assert isinstance(res, KBResult)
        assert len(res.entries) == 2  # both within namespace, ranked by score
        # The "patent" match should rank first.
        assert res.entries[0].id == "e1"

    def test_query_filters_by_namespace(self) -> None:
        kb = KnowledgeBase()
        kb.add(_entry("e1", "matter-1 statement", matter_client=("m1", "c1")))
        kb.add(_entry("e2", "matter-2 statement", matter_client=("m2", "c2")))
        res = kb.query(KBQuery(query_text="statement", matter_client=("m1", "c1")))
        assert len(res.entries) == 1
        assert res.entries[0].id == "e1"

    def test_min_confidence_filter(self) -> None:
        kb = KnowledgeBase()
        kb.add(_entry("low", "low confidence claim", confidence=0.4))
        kb.add(_entry("high", "high confidence claim", confidence=0.95))
        res = kb.query(KBQuery(query_text="claim", matter_client=("m1", "c1"), min_confidence=0.85))
        ids = [e.id for e in res.entries]
        assert "high" in ids
        assert "low" not in ids

    def test_top_k_caps_results(self) -> None:
        kb = KnowledgeBase()
        for i in range(5):
            kb.add(_entry(f"e{i}", f"matching term {i}"))
        res = kb.query(KBQuery(query_text="matching", matter_client=("m1", "c1"), top_k=2))
        assert len(res.entries) == 2


class TestNamespaceIsolation:
    def test_cross_namespace_add_raises_when_guard_bound(self) -> None:
        guard = MatterClientGuard(bound=("m1", "c1"))
        kb = KnowledgeBase(guard=guard)
        with pytest.raises(MatterIsolationError):
            kb.add(_entry("rogue", "stuff", matter_client=("m2", "c2")))

    def test_cross_namespace_query_raises_when_guard_bound(self) -> None:
        guard = MatterClientGuard(bound=("m1", "c1"))
        kb = KnowledgeBase(guard=guard)
        # Adding to bound namespace works
        kb.add(_entry("e1", "ok"))
        with pytest.raises(MatterIsolationError):
            kb.query(KBQuery(query_text="ok", matter_client=("m2", "c2")))


class TestProgramContract:
    @pytest.mark.asyncio
    async def test_forward_invokes_query(self) -> None:
        kb = KnowledgeBase()
        kb.add(_entry("e1", "a hopeful finding"))
        q = KBQuery(query_text="hopeful", matter_client=("m1", "c1"))
        res = await kb.forward(query=q)
        assert isinstance(res, KBResult)
        assert len(res.entries) == 1
        assert res.entries[0].id == "e1"
