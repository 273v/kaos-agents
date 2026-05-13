"""Unit tests for Sprint-3 #10 — transparency lens.

Covers the new headline cost / token surface that lifts ``cost_usd`` +
``total_tokens`` out of the event stream and onto:

1. :class:`AgentResponse` — first-class frozen attributes plus
   backward-compat mirror entries in ``metadata``.
2. :func:`kaos_agents.runtime.events_to_response.events_to_response`
   — pulls ``cost_usd`` from ``TurnSummary`` and surfaces both numbers.
3. The four MCP tool wrappers — ``AgentChatTool``, ``AgentPlanTool``,
   ``AgentFindingsTool``, ``AgentCorpusFilterTool`` — each must
   include ``cost_usd`` and ``total_tokens`` at the top level of
   ``structuredContent``.

Mocked tests only. Real-LLM verification lives in
``tests/integration/test_cost_surface_live.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_agents.base.event import KaosEvent
from kaos_agents.events import (
    IntentClassified,
    Span,
    SpanPhase,
    SpanSubject,
    TurnSummary,
)
from kaos_agents.patterns import findings as findings_mod
from kaos_agents.patterns.findings import (
    FilteredFinding,
    FindingCandidate,
)
from kaos_agents.runtime.events_to_response import events_to_response
from kaos_agents.types import AgentResponse, IntentResult, IntentType

# ---------------------------------------------------------------------------
# Event-construction helpers — KaosEvent base requires timestamp /
# sequence / session_id / run_id, plus Span requires span_id. Centralized
# here so the tests stay focused on the cost / token surface.
# ---------------------------------------------------------------------------

_FAKE_SID = "s-test"
_FAKE_RID = "r-test"


def _span(
    *,
    sequence: int,
    subject: SpanSubject,
    phase: SpanPhase,
    attributes: dict[str, Any] | None = None,
) -> Span:
    return Span(
        timestamp=0.0,
        sequence=sequence,
        session_id=_FAKE_SID,
        run_id=_FAKE_RID,
        subject=subject,
        phase=phase,
        span_id=f"span{sequence:04d}",
        attributes=attributes or {},
    )


def _intent_event(sequence: int) -> IntentClassified:
    return IntentClassified(
        timestamp=0.0,
        sequence=sequence,
        session_id=_FAKE_SID,
        run_id=_FAKE_RID,
        intent="respond",
        confidence=0.9,
        reasoning="x",
    )


def _turn_summary(
    *,
    sequence: int,
    text: str = "ok",
    tokens_used: int = 0,
    cost_usd: float = 0.0,
) -> TurnSummary:
    return TurnSummary(
        timestamp=0.0,
        sequence=sequence,
        session_id=_FAKE_SID,
        run_id=_FAKE_RID,
        text=text,
        intent="respond",
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


# ---------------------------------------------------------------------------
# 1) AgentResponse — first-class attributes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentResponseSurface:
    def test_cost_usd_and_total_tokens_as_attributes(self) -> None:
        """The headline figures are addressable via attribute access."""
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse(
            text="hi",
            intent=intent,
            cost_usd=0.0042,
            total_tokens=523,
        )
        assert response.cost_usd == pytest.approx(0.0042)
        assert response.total_tokens == 523

    def test_create_factory_threads_cost_and_tokens(self) -> None:
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse.create(
            text="hi",
            intent=intent,
            cost_usd=0.0010,
            total_tokens=250,
        )
        assert response.cost_usd == pytest.approx(0.0010)
        assert response.total_tokens == 250

    def test_backward_compat_metadata_mirror(self) -> None:
        """Sprint-3 #10 mirrors the headline fields into ``metadata`` so
        legacy callers walking ``dict(metadata)`` still see the numbers."""
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse.create(
            text="hi",
            intent=intent,
            cost_usd=0.005,
            total_tokens=999,
            metadata={"session_id": "s"},
        )
        meta = dict(response.metadata)
        assert meta.get("cost_usd") == pytest.approx(0.005)
        assert meta.get("total_tokens") == 999
        # Caller-provided keys still come through.
        assert meta.get("session_id") == "s"

    def test_total_tokens_defaults_to_tokens_used(self) -> None:
        """When ``total_tokens`` is omitted, the factory mirrors
        ``tokens_used`` (they're semantically the same number for a turn)."""
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse.create(text="hi", intent=intent, tokens_used=42)
        assert response.total_tokens == 42

    def test_defaults_zero_when_omitted(self) -> None:
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse(text="", intent=intent)
        assert response.cost_usd == 0.0
        assert response.total_tokens == 0


# ---------------------------------------------------------------------------
# 2) events_to_response — TurnSummary → AgentResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventsToResponseCostSurface:
    def test_turn_summary_cost_flows_to_response(self) -> None:
        """A TurnSummary with cost+tokens populates AgentResponse fields."""
        events: list[KaosEvent] = [
            _span(
                sequence=1,
                subject=SpanSubject.TURN,
                phase=SpanPhase.START,
                attributes={"turn_number": 1},
            ),
            _intent_event(sequence=2),
            _turn_summary(
                sequence=3,
                text="ok",
                tokens_used=523,
                cost_usd=0.0042,
            ),
            _span(sequence=4, subject=SpanSubject.TURN, phase=SpanPhase.COMPLETE),
        ]
        response = events_to_response(events, session_id="s")
        assert response.text == "ok"
        assert response.cost_usd == pytest.approx(0.0042)
        assert response.total_tokens == 523
        # tokens_used is the same number as total_tokens at the turn level.
        assert response.tokens_used == 523
        # Backward-compat mirror in metadata.
        meta = dict(response.metadata)
        assert meta["cost_usd"] == pytest.approx(0.0042)
        assert meta["total_tokens"] == 523

    def test_missing_turn_summary_yields_zeros(self) -> None:
        """When the turn errored before TurnSummary fired, surface zeros."""
        events: list[KaosEvent] = [
            _span(sequence=1, subject=SpanSubject.TURN, phase=SpanPhase.START),
        ]
        response = events_to_response(events, session_id="s")
        assert response.cost_usd == 0.0
        assert response.total_tokens == 0


# ---------------------------------------------------------------------------
# 3) AgentChatTool / AgentPlanTool — ToolResult.structuredContent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatPlanToolStructuredContentSurface:
    @pytest.mark.asyncio
    async def test_chat_tool_emits_cost_usd_and_total_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drain ``_run_turn_with_status`` with a fake event stream
        carrying a TurnSummary — assert the chat tool's structuredContent
        carries ``cost_usd`` and ``total_tokens`` at the top level."""
        from kaos_agents.tools.registry import AgentChatTool, _run_turn_with_status

        # Fake Runner: ``run()`` yields a canonical Turn -> TurnSummary
        # event stream so _run_turn_with_status produces a real response.
        class _FakeRunner:
            async def run(self, _message: str, _session_id: str) -> Any:  # async generator
                yield _span(
                    sequence=1,
                    subject=SpanSubject.TURN,
                    phase=SpanPhase.START,
                    attributes={"turn_number": 1},
                )
                yield _intent_event(sequence=2)
                yield _turn_summary(
                    sequence=3,
                    text="hello",
                    tokens_used=314,
                    cost_usd=0.0021,
                )
                yield _span(sequence=4, subject=SpanSubject.TURN, phase=SpanPhase.COMPLETE)

        response, status = await _run_turn_with_status(_FakeRunner(), "hi", "s")
        # AgentResponse first-class attributes.
        assert response.cost_usd == pytest.approx(0.0021)
        assert response.total_tokens == 314
        # status dict carries the same numbers for the tool surface to
        # pick up.
        assert status.get("cost_usd") == pytest.approx(0.0021)
        assert status.get("total_tokens") == 314

        # Now drive AgentChatTool.execute() with a stubbed _run_turn_with_status
        # so we observe what lands in structuredContent.
        async def _stub_run_turn(_runner: Any, _message: str, _session_id: str) -> Any:
            return response, status

        monkeypatch.setattr(
            "kaos_agents.tools.registry._run_turn_with_status",
            _stub_run_turn,
        )

        # AgentChatTool imports Runner inside execute(); patch the source
        # module so the no-op stand-in is picked up. We don't actually
        # call the Runner — it's just constructed and passed through to
        # the stubbed _run_turn_with_status.
        class _NoopRunner:
            def __init__(self, *a: Any, **kw: Any) -> None: ...

        monkeypatch.setattr("kaos_agents.runtime.runner.Runner", _NoopRunner)

        tool = AgentChatTool()
        result = await tool.execute({"message": "hi", "session_id": "s"})
        payload = result.structuredContent
        assert payload is not None
        assert "cost_usd" in payload, "cost_usd missing from chat structuredContent"
        assert "total_tokens" in payload, "total_tokens missing from chat structuredContent"
        assert payload["cost_usd"] == pytest.approx(0.0021)
        assert payload["total_tokens"] == 314

    @pytest.mark.asyncio
    async def test_plan_tool_emits_cost_usd_and_total_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same shape as chat — the plan tool must also carry both fields."""
        from kaos_agents.tools.registry import AgentPlanTool

        # Build an AgentResponse + status by hand; bypass the runner drain.
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="x")
        response = AgentResponse.create(
            text="plan done",
            intent=intent,
            tokens_used=412,
            cost_usd=0.0033,
            total_tokens=412,
            metadata={"session_id": "s"},
        )
        status: dict[str, Any] = {
            "budget_exceeded": False,
            "budget_kind": None,
            "paused_for_approval": False,
            "pending_tool_name": None,
            "run_state_ref": None,
            "run_error_event": None,
            "cost_usd": 0.0033,
            "total_tokens": 412,
        }

        async def _stub_run_turn(_runner: Any, _message: str, _session_id: str) -> Any:
            return response, status

        monkeypatch.setattr(
            "kaos_agents.tools.registry._run_turn_with_status",
            _stub_run_turn,
        )

        class _NoopRunner:
            def __init__(self, *a: Any, **kw: Any) -> None: ...

        monkeypatch.setattr("kaos_agents.runtime.runner.Runner", _NoopRunner)

        tool = AgentPlanTool()
        result = await tool.execute({"message": "goal", "session_id": "s"})
        payload = result.structuredContent
        assert payload is not None
        assert payload.get("cost_usd") == pytest.approx(0.0033)
        assert payload.get("total_tokens") == 412


# ---------------------------------------------------------------------------
# 4) AgentFindingsTool — structuredContent token surface
# ---------------------------------------------------------------------------


async def _stub_filter_keep_all_with_tokens(
    chunk: tuple[FindingCandidate, ...],
    **_kwargs: Any,
) -> tuple[tuple[FilteredFinding, ...], float, int]:
    """Sprint-3 #10 stub matching the new 3-tuple shape."""
    survivors = tuple(FilteredFinding(candidate=c, relevance=0.9, reasoning="ok") for c in chunk)
    # $0.001 + 80 tokens per chunk.
    return survivors, 0.001, 80


async def _stub_synthesize_with_tokens(**kwargs: Any) -> tuple[str, float, int]:
    """Sprint-3 #10 stub matching the new 3-tuple shape."""
    findings = kwargs["findings"]
    cited = " ".join(f"[{f.candidate.finding_id}]" for f in findings)
    # $0.005 + 200 tokens.
    return f"Synthesized: {cited}", 0.005, 200


@pytest.mark.unit
class TestFindingsToolTokenSurface:
    @pytest.mark.asyncio
    async def test_findings_tool_emits_total_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end via the tool surface — stubs return 3-tuples
        carrying tokens; the tool must surface ``total_tokens`` and the
        per-stage ``filter_tokens`` / ``synthesis_tokens`` keys."""
        from kaos_content.artifacts import store_document
        from kaos_content.model.document import ContentDocument
        from kaos_content.shortcuts import paragraph
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.tools.findings import AgentFindingsTool

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_keep_all_with_tokens)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize_with_tokens)

        runtime = KaosRuntime.test_mode()
        ctx = KaosContext.create(session_id="s-tokens", runtime=runtime)
        doc = ContentDocument(
            body=(
                paragraph("Indemnification carve-outs apply for gross negligence."),
                paragraph("The Term is 24 months from the Effective Date."),
            ),
        )
        manifest = await store_document(doc, runtime, ctx, name="cost-surface-test")

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": manifest.artifact_id,
                "question": "What are the indemnification terms?",
                "select_by": "every_sentence",
                "chunk_size": 5,
                "num_parallel": 1,
            },
            context=ctx,
        )
        payload = result.structuredContent
        assert payload is not None
        # Per-stage tokens.
        assert payload.get("filter_tokens") is not None
        assert payload.get("synthesis_tokens") == 200
        # Headline aggregate.
        assert payload.get("total_tokens") is not None
        assert payload["total_tokens"] >= payload["synthesis_tokens"]
        # Headline cost present.
        assert "cost_usd" in payload
        assert payload["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_findings_tool_backward_compat_two_tuple_stubs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy 2-tuple stubs still produce a valid result — the
        helper-result unpacker keeps them working."""
        from kaos_content.artifacts import store_document
        from kaos_content.model.document import ContentDocument
        from kaos_content.shortcuts import paragraph
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.tools.findings import AgentFindingsTool

        async def _legacy_filter(
            chunk: tuple[FindingCandidate, ...], **_kwargs: Any
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            survivors = tuple(
                FilteredFinding(candidate=c, relevance=0.9, reasoning="ok") for c in chunk
            )
            return survivors, 0.001

        async def _legacy_synthesize(**kwargs: Any) -> tuple[str, float]:
            findings = kwargs["findings"]
            cited = " ".join(f"[{f.candidate.finding_id}]" for f in findings)
            return f"Synthesized: {cited}", 0.005

        monkeypatch.setattr(findings_mod, "_filter_chunk", _legacy_filter)
        monkeypatch.setattr(findings_mod, "_synthesize", _legacy_synthesize)

        runtime = KaosRuntime.test_mode()
        ctx = KaosContext.create(session_id="s-legacy", runtime=runtime)
        doc = ContentDocument(body=(paragraph("Indemnification clause text."),))
        manifest = await store_document(doc, runtime, ctx, name="legacy-stub")

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": manifest.artifact_id,
                "question": "What's the clause?",
                "select_by": "every_sentence",
                "chunk_size": 5,
                "num_parallel": 1,
            },
            context=ctx,
        )
        payload = result.structuredContent
        assert payload is not None
        # Token field still present (defaults to 0 when stubs don't surface).
        assert payload.get("total_tokens") == 0
        assert payload.get("filter_tokens") == 0
        assert payload.get("synthesis_tokens") == 0
        # Cost still works.
        assert payload["cost_usd"] > 0


# ---------------------------------------------------------------------------
# 5) AgentCorpusFilterTool — structuredContent token surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorpusFilterToolTokenSurface:
    @pytest.mark.asyncio
    async def test_corpus_filter_emits_total_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the K8 helper; assert ``total_tokens`` is in
        structuredContent at the top level."""
        from kaos_content.artifacts import store_document
        from kaos_content.model.document import ContentDocument
        from kaos_content.shortcuts import paragraph
        from kaos_core.base.context import KaosContext
        from kaos_core.registry.container import KaosRuntime

        from kaos_agents.tools import corpus_filter as cf_mod
        from kaos_agents.tools.corpus_filter import AgentCorpusFilterTool

        async def _stub_filter_llm(
            *, intent: str, artifacts: list[Any], max_keep: int, model: str
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
            kept = [
                {
                    "artifact_id": artifacts[0]["id"],
                    "relevance": 0.9,
                    "reasoning": "stub",
                }
            ]
            return kept, [], 0.0015, 150

        monkeypatch.setattr(cf_mod, "_run_corpus_filter_llm", _stub_filter_llm)

        runtime = KaosRuntime.test_mode()
        ctx = KaosContext.create(session_id="s-cf", runtime=runtime)
        doc = ContentDocument(body=(paragraph("Some artifact text"),))
        manifest = await store_document(doc, runtime, ctx, name="cf-test")

        tool = AgentCorpusFilterTool()
        result = await tool.execute(
            {
                "intent": "find the artifact",
                "artifact_ids": [manifest.artifact_id],
                "max_keep": 5,
            },
            context=ctx,
        )
        payload = result.structuredContent
        assert payload is not None
        assert payload.get("cost_usd") == pytest.approx(0.0015)
        assert payload.get("total_tokens") == 150
