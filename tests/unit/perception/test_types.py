"""Unit tests for :mod:`kaos_agents.perception.types`.

Covers construction defaults, frozen-ness, JSON round-trip, the
PerceptionRefusal default-None invariant on PerceptionResult, and
the score / confidence range checks.
"""

from __future__ import annotations

import pytest
from kaos_llm_core.signatures.grounding import Span
from pydantic import ValidationError

from kaos_agents.perception.types import (
    PerceptionItem,
    PerceptionQuery,
    PerceptionQueryKind,
    PerceptionRefusal,
    PerceptionResult,
)


def test_perception_query_kind_string_values() -> None:
    """The StrEnum values are stable and lowercase-snake."""
    assert PerceptionQueryKind.DOCUMENT_QA.value == "document_qa"
    assert PerceptionQueryKind.FACT_LOOKUP.value == "fact_lookup"
    assert PerceptionQueryKind.GENERAL_RECALL.value == "general_recall"
    assert PerceptionQueryKind.TOOL_QUERY.value == "tool_query"


def test_perception_query_defaults() -> None:
    q = PerceptionQuery(query_text="who signed the contract?")
    assert q.kind is PerceptionQueryKind.GENERAL_RECALL
    assert q.required_capabilities == ()
    assert q.max_results == 10
    assert q.matter_client is None


def test_perception_query_frozen() -> None:
    q = PerceptionQuery(query_text="x")
    with pytest.raises(ValidationError):
        q.query_text = "y"  # type: ignore[misc]


def test_perception_query_max_results_bounds() -> None:
    with pytest.raises(ValidationError):
        PerceptionQuery(query_text="x", max_results=0)
    with pytest.raises(ValidationError):
        PerceptionQuery(query_text="x", max_results=201)
    # Boundary values accepted.
    PerceptionQuery(query_text="x", max_results=1)
    PerceptionQuery(query_text="x", max_results=200)


def test_perception_item_construction() -> None:
    span = Span(
        source_uri="doc:a",
        char_span=(0, 3),
        quote="abc",
    )
    item = PerceptionItem(
        content="hello",
        source="session_memory",
        citations=(span,),
        score=0.75,
        metadata={"k": "v"},
    )
    assert item.citations == (span,)
    assert item.score == 0.75
    assert item.metadata == {"k": "v"}


def test_perception_item_extra_forbidden() -> None:
    """``extra="forbid"`` on the model rejects unknown fields at parse time."""
    with pytest.raises(ValidationError):
        PerceptionItem.model_validate({"content": "x", "source": "y", "unknown": 1})


def test_perception_refusal_default_kind() -> None:
    r = PerceptionRefusal(reason="not enough evidence")
    assert r.kind == "evidence_insufficient"
    assert r.suggested_alternatives == ()


def test_perception_result_refusal_defaults_none() -> None:
    """Default PerceptionResult has refusal=None — happy path is opt-out."""
    r = PerceptionResult()
    assert r.refusal is None
    assert r.items == ()
    assert r.confidence == 1.0
    assert r.sources_consulted == ()


def test_perception_result_confidence_range() -> None:
    with pytest.raises(ValidationError):
        PerceptionResult(confidence=-0.1)
    with pytest.raises(ValidationError):
        PerceptionResult(confidence=1.01)
    PerceptionResult(confidence=0.0)
    PerceptionResult(confidence=1.0)


def test_perception_result_with_refusal() -> None:
    refusal = PerceptionRefusal(reason="empty")
    r = PerceptionResult(refusal=refusal, sources_consulted=("kb",))
    assert r.refusal is refusal
    assert r.sources_consulted == ("kb",)


def test_perception_query_json_round_trip() -> None:
    """JSON round-trip through pydantic preserves all fields."""
    q = PerceptionQuery(
        query_text="hello",
        kind=PerceptionQueryKind.DOCUMENT_QA,
        required_capabilities=("kaos-source-fr-*",),
        max_results=5,
        matter_client=("matter-1", "client-2"),
    )
    payload = q.model_dump_json()
    restored = PerceptionQuery.model_validate_json(payload)
    assert restored == q


def test_perception_result_json_round_trip() -> None:
    item = PerceptionItem(content="hi", source="session_memory", score=0.5)
    refusal = PerceptionRefusal(reason="x", suggested_alternatives=("a",))
    r = PerceptionResult(
        items=(item,),
        confidence=0.5,
        refusal=refusal,
        sources_consulted=("session", "kb"),
    )
    payload = r.model_dump_json()
    restored = PerceptionResult.model_validate_json(payload)
    assert restored == r
