"""Unit tests for :class:`kaos_agents.perception.rag.PerceptionRAG`.

Stubs the underlying RAG via a fake ``invoke()`` method that returns
synthetic outputs containing :class:`Span` objects. Verifies:

- Construction: rejects mutually-exclusive ``rag_program`` / ``**kwargs``.
- ``forward()`` calls the underlying RAG and pushes ``CitationFound``
  events when a collector is active.
- No-op when no collector is active.
- Walks nested pydantic models for Span instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from kaos_llm_core.signatures.grounding import Answer, Claim, Span

from kaos_agents.events.collector import collect_events
from kaos_agents.events.research import CitationFound
from kaos_agents.perception.rag import PerceptionRAG

# --- Stub Invocation / RAG -------------------------------------------


@dataclass
class _StubInvocation:
    output: Any


class _StubRAG:
    """Minimal stub that exposes the same async ``invoke`` surface RAG uses."""

    def __init__(self, output: Any) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, **kwargs: Any) -> _StubInvocation:
        self.calls.append(kwargs)
        return _StubInvocation(output=self._output)


def _make_span(uri: str, quote: str) -> Span:
    return Span(source_uri=uri, char_span=(0, len(quote)), quote=quote)


def _make_answer(spans: list[Span]) -> Answer[str]:
    claim = Claim(statement="x", supporting_spans=spans, confidence=1.0)
    return Answer(value="answer text", claims=[claim], confidence=1.0)


# --- Tests ------------------------------------------------------------


def test_perception_rag_rejects_program_plus_kwargs() -> None:
    stub = _StubRAG(output=_make_answer([_make_span("doc:a", "abc")]))
    with pytest.raises(ValueError, match="rag_program OR rag_kwargs"):
        PerceptionRAG(rag_program=stub, model="anthropic:claude-haiku-4-5")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_forward_pushes_citation_found_when_collector_active() -> None:
    spans = [_make_span("doc:a", "first quote"), _make_span("doc:b", "second quote")]
    stub = _StubRAG(output=_make_answer(spans))
    rag = PerceptionRAG(rag_program=stub)  # type: ignore[arg-type]

    with collect_events() as events:
        output = await rag(question="hello", documents={"doc:a": "first quote second quote"})

    assert output is stub._output
    citation_events = [e for e in events if isinstance(e, CitationFound)]
    assert len(citation_events) == 2
    assert {e.source_uri for e in citation_events} == {"doc:a", "doc:b"}
    assert {e.claim for e in citation_events} == {"first quote", "second quote"}
    assert all(e.verified for e in citation_events)


@pytest.mark.asyncio
async def test_forward_no_op_when_no_collector_active() -> None:
    """No collector context → silent — backward-compat for direct callers."""
    spans = [_make_span("doc:a", "quoted")]
    stub = _StubRAG(output=_make_answer(spans))
    rag = PerceptionRAG(rag_program=stub)  # type: ignore[arg-type]

    output = await rag(question="hello")

    assert output is stub._output
    assert stub.calls == [{"question": "hello"}]


@pytest.mark.asyncio
async def test_forward_walks_nested_pydantic_for_spans() -> None:
    """Spans nested deep inside a pydantic model are still found."""

    spans = [_make_span("doc:nested", "deep quote")]
    answer = _make_answer(spans)
    stub = _StubRAG(output=answer)
    rag = PerceptionRAG(rag_program=stub)  # type: ignore[arg-type]

    with collect_events() as events:
        await rag(question="x")

    citation_events = [e for e in events if isinstance(e, CitationFound)]
    assert len(citation_events) == 1
    assert citation_events[0].source_uri == "doc:nested"


@pytest.mark.asyncio
async def test_forward_handles_output_with_no_spans() -> None:
    """Output without any Span instances → zero events, no crash."""
    stub = _StubRAG(output={"plain": "dict"})
    rag = PerceptionRAG(rag_program=stub)  # type: ignore[arg-type]

    with collect_events() as events:
        output = await rag(question="x")

    assert output == {"plain": "dict"}
    assert [e for e in events if isinstance(e, CitationFound)] == []


@pytest.mark.asyncio
async def test_forward_rag_property_exposes_inner_program() -> None:
    """The ``rag`` property gives test code visibility into the inner program."""
    stub = _StubRAG(output=_make_answer([_make_span("doc:x", "q")]))
    wrapped = PerceptionRAG(rag_program=stub)  # type: ignore[arg-type]
    assert wrapped.rag is stub
