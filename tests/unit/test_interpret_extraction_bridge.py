"""Unit tests for ``_try_interpret_extraction`` + the bridge wiring.

The bridge (in ``BaseAgent._handle_research_streaming``) tries the new
``AgentInterpretExtractionTool`` for corpus + RESEARCH prompts BEFORE
falling through to the legacy ``_run_findings_dispatch`` path. These
tests pin the gating contract:

- Corpus too small (<2 docs) → bridge declines (returns None)
- Runtime not attached (BaseAgent) → bridge declines
- Tool errors at runtime → bridge declines (graceful degradation)
- Tool produces empty memo → bridge declines (no value to surface)
- Tool succeeds → bridge returns the ToolResult for the caller to emit

The event-emission helper (``_emit_interpret_extraction_events``) is
covered by ad-hoc assertion on the emitted stream shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.memory.session import SessionMemory
from kaos_agents.runtime.agent import BaseAgent
from kaos_agents.types.memory import MemoryType


def _memory_with_docs(*, n: int, artifact_ids: list[str] | None = None) -> SessionMemory:
    """Build a SessionMemory with N DOCUMENT items, each carrying a
    distinct ``hydrated_artifact_id``."""
    memory = SessionMemory("test-session")
    ids = artifact_ids or [f"doc-{i}" for i in range(n)]
    for aid in ids:
        memory.add(
            MemoryType.DOCUMENTS,
            content=f"document {aid} summary",
            metadata={"hydrated_artifact_id": aid, "filename": f"{aid}.docx"},
        )
    return memory


def _make_agent_with_runtime() -> BaseAgent:
    """BaseAgent + attached _runtime so the bridge's gating allows it."""
    from kaos_core.registry.container import KaosRuntime

    runtime = KaosRuntime.test_mode()
    agent = BaseAgent(
        runtime.vfs,
        instructions="test",
        model="anthropic:claude-sonnet-4-6",
    )
    # ``_runtime`` is a subclass-attached attribute (ChatAgent /
    # ResearchAgent / PlanExecuteAgent set it); BaseAgent itself has
    # no such field, and ty correctly flags the assignment as
    # ``unresolved-attribute``. We're explicitly attaching it for the
    # bridge gating test — the bridge uses ``getattr(self, "_runtime",
    # None)`` so the dynamic attribute is the contract.
    agent._runtime = runtime  # ty: ignore[unresolved-attribute]
    return agent


def _fake_tool_result(
    *, memo: str = "Result memo", cost_usd: float = 0.10, is_error: bool = False
) -> Any:
    from kaos_core.types.results import ToolResult

    if is_error:
        return ToolResult.create_error("simulated tool failure")
    return ToolResult.create_success(
        output={
            "memo": memo,
            "score": 9,
            "loop_status": "converged",
            "converged_at_iter": 1,
            "iterations_run": 1,
            "iteration_trace": [{"iter": 1, "score": 9, "needs_more_extraction": False}],
            "extract_cost_usd": cost_usd * 0.8,
            "synth_cost_usd": cost_usd * 0.2,
            "cost_usd": cost_usd,
            "total_tokens": 1000,
            "extracted": {
                "columns": [{"id": "x", "description": "x"}],
                "rows": [
                    {
                        "artifact_id": "doc-0",
                        "cells": {
                            "x": {
                                "value": "v",
                                "spans": [{"source_uri": "doc-0", "quote": "snippet from doc-0"}],
                            }
                        },
                    },
                    {
                        "artifact_id": "doc-1",
                        "cells": {
                            "x": {
                                "value": "v2",
                                "spans": [{"source_uri": "doc-1", "quote": "snippet from doc-1"}],
                            }
                        },
                    },
                ],
                "row_count": 2,
                "col_count": 1,
            },
        },
    )


class TestGating:
    """The bridge must DECLINE (return None) in clearly-inappropriate cases
    so callers safely fall through to findings-dispatch."""

    @pytest.mark.asyncio
    async def test_no_runtime_returns_none(self) -> None:
        """BaseAgent has no _runtime by default — the bridge cannot
        invoke a tool without one, so it must decline rather than crash."""
        from kaos_core.registry.container import KaosRuntime

        runtime = KaosRuntime.test_mode()
        agent = BaseAgent(runtime.vfs, instructions="test", model="x")
        # Note: we DON'T set agent._runtime — that's the gate we're testing.
        memory = _memory_with_docs(n=5)
        result = await agent._try_interpret_extraction("a question", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_document_returns_none(self) -> None:
        """interpret_extraction is designed for per-document typed
        deliverables. Single-doc sessions fall through to findings-
        dispatch which has better prose-shape heuristics for one doc."""
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=1)
        result = await agent._try_interpret_extraction("q", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_documents_returns_none(self) -> None:
        agent = _make_agent_with_runtime()
        memory = SessionMemory("empty")
        result = await agent._try_interpret_extraction("q", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_duplicate_artifact_ids_collapse(self) -> None:
        """When the corpus has 3 items but only 1 unique artifact_id
        (e.g. the same doc was added twice), the bridge must dedupe
        and decline (single resolvable doc)."""
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=3, artifact_ids=["dup", "dup", "dup"])
        result = await agent._try_interpret_extraction("q", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_artifact_id_metadata_returns_none(self) -> None:
        """When DOCUMENTS items have no hydrated_artifact_id / artifact_id /
        kaos:// URI in metadata, the bridge can't pass anything to the
        tool — must decline."""
        memory = SessionMemory("test")
        memory.add(MemoryType.DOCUMENTS, content="x", metadata={"filename": "x.docx"})
        memory.add(MemoryType.DOCUMENTS, content="y", metadata={"filename": "y.docx"})
        agent = _make_agent_with_runtime()
        result = await agent._try_interpret_extraction("q", memory)
        assert result is None


class TestSuccess:
    """When gating passes and the tool succeeds, the bridge returns
    the ToolResult for the caller to emit."""

    @pytest.mark.asyncio
    async def test_returns_tool_result_on_success(self) -> None:
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=3)
        fake = _fake_tool_result(memo="The answer.", cost_usd=0.12)

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=AsyncMock(return_value=fake),
        ):
            result = await agent._try_interpret_extraction("question", memory)

        assert result is not None
        assert not result.isError
        sc = result.structuredContent
        assert sc is not None
        assert sc["memo"] == "The answer."

    @pytest.mark.asyncio
    async def test_extracts_artifact_ids_from_hydrated_marker(self) -> None:
        """The bridge prefers ``hydrated_artifact_id`` over other keys."""
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=2, artifact_ids=["alpha", "beta"])

        seen_inputs: dict[str, Any] = {}

        async def capture(self_, inputs: dict[str, Any], *, context: Any = None) -> Any:
            seen_inputs.update(inputs)
            return _fake_tool_result()

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=capture,
        ):
            await agent._try_interpret_extraction("q", memory)

        assert seen_inputs.get("artifact_ids") == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_extracts_artifact_id_from_kaos_uri(self) -> None:
        """Fallback path — when items only have ``uri`` field with the
        ``kaos://artifacts/<id>/...`` shape, the bridge strips the
        prefix to recover the artifact_id."""
        memory = SessionMemory("uri-test")
        memory.add(
            MemoryType.DOCUMENTS,
            content="A",
            metadata={"uri": "kaos://artifacts/from-uri-A/body"},
        )
        memory.add(
            MemoryType.DOCUMENTS,
            content="B",
            metadata={"uri": "kaos://artifacts/from-uri-B/body"},
        )
        agent = _make_agent_with_runtime()

        seen_inputs: dict[str, Any] = {}

        async def capture(self_, inputs: dict[str, Any], *, context: Any = None) -> Any:
            seen_inputs.update(inputs)
            return _fake_tool_result()

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=capture,
        ):
            await agent._try_interpret_extraction("q", memory)

        assert seen_inputs.get("artifact_ids") == ["from-uri-A", "from-uri-B"]

    @pytest.mark.asyncio
    async def test_passes_conservative_loop_bounds(self) -> None:
        """The bridge uses tighter ``max_iters`` + ``budget_usd`` than
        the tool's defaults so an autonomous misbehavior cannot burn
        the full standalone budget by accident."""
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=2)

        seen_inputs: dict[str, Any] = {}

        async def capture(self_, inputs: dict[str, Any], *, context: Any = None) -> Any:
            seen_inputs.update(inputs)
            return _fake_tool_result()

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=capture,
        ):
            await agent._try_interpret_extraction("q", memory)

        assert seen_inputs["max_iters"] == 2
        assert seen_inputs["budget_usd"] == 0.75


class TestGracefulDegradation:
    """When the tool errors or produces an empty memo, the bridge
    returns None so the caller falls through to findings-dispatch."""

    @pytest.mark.asyncio
    async def test_tool_isError_returns_none(self) -> None:
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=3)
        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=AsyncMock(return_value=_fake_tool_result(is_error=True)),
        ):
            result = await agent._try_interpret_extraction("q", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_raises_returns_none(self) -> None:
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=3)

        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated crash")

        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=boom,
        ):
            result = await agent._try_interpret_extraction("q", memory)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_memo_returns_none(self) -> None:
        agent = _make_agent_with_runtime()
        memory = _memory_with_docs(n=3)
        with patch(
            "kaos_agents.tools.interpret_extraction.AgentInterpretExtractionTool.execute",
            new=AsyncMock(return_value=_fake_tool_result(memo="")),
        ):
            result = await agent._try_interpret_extraction("q", memory)
        assert result is None


class TestEventEmission:
    """The emission helper streams a synthetic TOOL_CALL span, one
    CitationFound per cited cell, the memo as TextDelta, UsageObserved,
    and the SUBAGENT span_complete."""

    @pytest.mark.asyncio
    async def test_emits_expected_event_sequence(self) -> None:
        from kaos_agents.events import (
            CitationFound,
            SpanPhase,
            SpanSubject,
            TextDelta,
            TurnSummary,  # noqa: F401  (imported to ensure the registry is hot)
            UsageObserved,
        )
        from kaos_agents.events.emitter import EventEmitter
        from kaos_agents.events.spans import Span

        agent = _make_agent_with_runtime()
        emitter = EventEmitter(session_id="t", run_id="r")
        research_span = emitter.span_start(
            SpanSubject.SUBAGENT,
            name="research.findings_dispatch",
            attributes={"path": "test"},
        )
        result = _fake_tool_result(memo="The answer.")
        events = [
            ev
            async for ev in agent._emit_interpret_extraction_events(
                result, research_span_id=research_span.span_id, emitter=emitter
            )
        ]

        # Expected event types in order
        types = [type(ev).__name__ for ev in events]
        # tool_call span_start, 2 CitationFound (one per row), tool_call
        # span_complete, TextDelta, UsageObserved, subagent span_complete.
        assert "Span" in types
        assert types.count("CitationFound") == 2
        assert "TextDelta" in types
        assert "UsageObserved" in types

        # Memo content emitted via TextDelta
        text_deltas = [ev for ev in events if isinstance(ev, TextDelta)]
        assert len(text_deltas) == 1
        assert text_deltas[0].content == "The answer."

        # CitationFound carries the correct source_uri (stamped from row)
        citations = [ev for ev in events if isinstance(ev, CitationFound)]
        assert {c.source_uri for c in citations} == {"doc-0", "doc-1"}

        # Subagent span_complete closes the original research_span
        completes = [
            ev
            for ev in events
            if isinstance(ev, Span)
            and ev.phase == SpanPhase.COMPLETE
            and ev.subject == SpanSubject.SUBAGENT
        ]
        assert len(completes) == 1
        assert completes[0].span_id == research_span.span_id

        # Usage carries the cost
        usages = [ev for ev in events if isinstance(ev, UsageObserved)]
        assert len(usages) == 1
        assert usages[0].cost_usd == pytest.approx(0.10)
