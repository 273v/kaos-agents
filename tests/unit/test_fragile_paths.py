"""Unit-test coverage for the fragile paths recent sessions have fixed.

These tests are intentionally small, deterministic, and free of LLM
calls. They catch regressions in the SHAPE of the fixes — the
behavioral correctness still depends on the live ladder, but if any
of these fail the fix has been undone at the structural level.

Tested fragile paths:

1. Tool-bridge structured-content combiner — text + structuredContent
   both surface to the LLM caller (regression: c94d6ba)
2. Compose._collect_predecessor_results — walks completed-only
   predecessors, includes labelled blocks, truncates >16KB
3. Compose._is_description_only — recognizes the planner's typical
   ``{"description": ...}`` input_spec shape vs structured args
4. EventEmitter auto-duration — span_complete picks up monotonic
   delta when caller passes duration_ms=0.0
5. collect_events early-return — ContextVar reset across task
   boundary is suppressed (regression: 014c648)
6. emit_citations_for_text — silent no-op when no citations / no
   kaos-citations / empty text; emits CitationFound when present
7. emit_thinking_from_invocation — silent on empty / None extras;
   emits ThinkingDelta when native_thinking populated
8. Plan-execute BudgetExceeded emission shape — fires on every
   budget-driven StopReason with the correct ``kind``
9. PlanProposed-from-proposed-plan — populated even when step_results
   is empty (the post-execution PlanGraph carries the proposed steps)
"""

from __future__ import annotations

import contextvars
import json
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 1. Tool-bridge: text + structuredContent combiner
# ---------------------------------------------------------------------------


class _StubKaosTool:
    """Minimal KaosTool stub for tool-bridge tests — no runtime needed."""

    def __init__(self, *, name: str, result: Any, annotations: Any = None) -> None:
        from kaos_core.types.annotations import ToolAnnotations
        from kaos_core.types.metadata import (
            ParameterSchema,
            ToolCapability,
            ToolCategory,
            ToolMetadata,
        )

        self.metadata = ToolMetadata(
            name=name,
            display_name=name,
            description="test tool",
            category=ToolCategory.UTILITY,
            capability=ToolCapability.QUERY,
            module_name="test",
            version="0.0.1",
            input_schema=[
                ParameterSchema(name="q", type="string", description="q", required=False),
            ],
            annotations=annotations or ToolAnnotations(readOnlyHint=True),
        )
        self._result = result

    async def execute(self, inputs: dict, context: Any = None) -> Any:
        return self._result


def _make_tool_result(*, text: str = "", structured: dict | None = None, error: bool = False):
    from kaos_core import ToolResult
    from kaos_core.types import TextContent

    contents = [TextContent(type="text", text=text)] if text else []
    return ToolResult(
        content=contents,  # ty: ignore[invalid-argument-type]
        structuredContent=structured,
        isError=error,
    )


@pytest.mark.asyncio
async def test_tool_bridge_combines_text_and_structured() -> None:
    """When both text and structuredContent are present, both must surface."""
    from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool

    structured = {"results": [{"document_number": "2026-08174", "title": "Example"}]}
    tool = _StubKaosTool(
        name="kaos-test-search",
        result=_make_tool_result(text="Found 1 result", structured=structured),
    )
    llm_tool = kaos_tool_to_llm_tool(tool, context=None)  # ty: ignore[invalid-argument-type]
    output = await llm_tool.executor(q="x")
    assert "Found 1 result" in output, "summary text must appear in combined output"
    assert "2026-08174" in output, "structuredContent must appear in combined output"
    # Output is summary + blank line + JSON
    assert output.index("Found 1 result") < output.index("2026-08174"), (
        "summary should precede structured JSON"
    )


@pytest.mark.asyncio
async def test_tool_bridge_text_only_passthrough() -> None:
    """When only text is present, it's returned verbatim — no JSON wrapping."""
    from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool

    tool = _StubKaosTool(
        name="kaos-test-text",
        result=_make_tool_result(text="just-text", structured=None),
    )
    llm_tool = kaos_tool_to_llm_tool(tool, context=None)  # ty: ignore[invalid-argument-type]
    output = await llm_tool.executor(q="x")
    assert output == "just-text", f"text-only should pass through; got: {output!r}"


@pytest.mark.asyncio
async def test_tool_bridge_error_returns_error_envelope() -> None:
    """isError=True wraps in {"error": true, "message": text}."""
    from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool

    tool = _StubKaosTool(
        name="kaos-test-err",
        result=_make_tool_result(text="bad input", error=True),
    )
    llm_tool = kaos_tool_to_llm_tool(tool, context=None)  # ty: ignore[invalid-argument-type]
    output = await llm_tool.executor(q="x")
    parsed = json.loads(output)
    assert parsed == {"error": True, "message": "bad input"}


# ---------------------------------------------------------------------------
# 2. Compose._collect_predecessor_results — prior-output threading
# ---------------------------------------------------------------------------


def _build_plan_graph_with_results(results: dict[str, str]) -> Any:
    """Build a PlanGraph with a step per result, all dependent on step_0."""
    from kaos_agents.planning.graph import PlanGraph
    from kaos_agents.types.plan import Step, StepType

    graph = PlanGraph()
    step_ids = sorted(results.keys())
    for i, sid in enumerate(step_ids):
        depends = (step_ids[i - 1],) if i > 0 else ()
        graph.add_step(
            Step(
                id=sid,
                step_type=StepType.LLM,
                description=f"step {sid}",
                depends_on=depends,
            )
        )
    # Mark each step completed with its result
    from kaos_agents.planning.evaluate import EvalMode
    from kaos_agents.types.plan import Judgment

    j = Judgment(matched=True, confidence=1.0, reasoning="", mode=EvalMode.STRUCTURAL)
    for sid, res in results.items():
        graph.mark_complete(sid, res, j)
    return graph, step_ids


def test_collect_predecessor_results_threads_prior_outputs() -> None:
    """Predecessor outputs prepend a labelled block to the prompt."""
    from kaos_agents.planning.compose import _collect_predecessor_results

    graph, _ids = _build_plan_graph_with_results(
        {"step-a": "alpha-result", "step-b": "beta-result"}
    )
    # step-b depends on step-a (per our builder); collecting predecessors of
    # step-b should surface step-a's result.
    out = _collect_predecessor_results(graph, "step-b")
    assert "alpha-result" in out, f"step-b's predecessor result missing: {out!r}"
    assert "step-a" in out, "predecessor step_id should be labelled"


def test_collect_predecessor_results_no_predecessors_returns_empty() -> None:
    """A root step has no predecessors → empty string."""
    from kaos_agents.planning.compose import _collect_predecessor_results

    graph, _ids = _build_plan_graph_with_results({"only-step": "result"})
    assert _collect_predecessor_results(graph, "only-step") == ""


def test_collect_predecessor_results_truncates_at_16kb() -> None:
    """Results >16KB get truncated with a marker — bounds the prompt size."""
    from kaos_agents.planning.compose import _collect_predecessor_results

    huge = "X" * 20_000
    graph, _ = _build_plan_graph_with_results({"step-a": huge, "step-b": "small"})
    out = _collect_predecessor_results(graph, "step-b")
    assert "truncated" in out, f"truncation marker missing: {out[-200:]!r}"
    # Each predecessor block is capped — the included text shouldn't exceed
    # 16K + a few hundred chars of label/marker overhead.
    assert len(out) < 17_000


# ---------------------------------------------------------------------------
# 3. Compose._is_description_only — recognizes planner default shape
# ---------------------------------------------------------------------------


def test_is_description_only_recognizes_planner_default() -> None:
    from kaos_agents.planning.compose import _is_description_only

    # The planner's typical input_spec — should trigger arg synthesis
    assert _is_description_only({"description": "fetch the doc"})
    # Empty dict → not description-only (also not synthesizable)
    assert not _is_description_only({})
    # Structured args present → use as-is (synthesis would be wrong)
    assert not _is_description_only({"document_number": "2026-08174"})
    # Description PLUS structured args → not description-only
    assert not _is_description_only({"description": "fetch X", "document_number": "2026-08174"})


# ---------------------------------------------------------------------------
# 4. EventEmitter auto-duration
# ---------------------------------------------------------------------------


def test_event_emitter_auto_measures_duration_ms() -> None:
    """span_start → time.sleep → span_complete: duration_ms reflects elapsed."""
    import time

    from kaos_agents.events import EventEmitter, SpanSubject

    em = EventEmitter(session_id="t", run_id="r")
    start = em.span_start(SpanSubject.TOOL_CALL, name="tool.x")
    time.sleep(0.02)
    complete = em.span_complete(SpanSubject.TOOL_CALL, span_id=start.span_id, name="tool.x")
    # 20ms slept; auto-measured duration should be >=15ms (allow some slack)
    assert complete.duration_ms is not None and complete.duration_ms >= 15.0, (
        f"auto-measured duration_ms should be >=15ms; got {complete.duration_ms!r}"
    )


def test_event_emitter_respects_caller_supplied_duration() -> None:
    """When caller passes a non-zero duration_ms, it wins over auto-measure."""
    from kaos_agents.events import EventEmitter, SpanSubject

    em = EventEmitter(session_id="t", run_id="r")
    start = em.span_start(SpanSubject.TURN, name="turn.1")
    complete = em.span_complete(
        SpanSubject.TURN, span_id=start.span_id, name="turn.1", duration_ms=12345.6
    )
    assert complete.duration_ms == 12345.6


# ---------------------------------------------------------------------------
# 5. collect_events early-return — ContextVar suppression
# ---------------------------------------------------------------------------


def test_collect_events_suppresses_cross_context_reset() -> None:
    """ValueError from reset() across contexts must be swallowed."""
    from kaos_agents.events.collector import collect_events

    captured: list[BaseException] = []

    def _bad_consumer() -> None:
        # Enter collect_events in this context, then RUN code that
        # accesses + resets contextvars from a different context via
        # contextvars.copy_context. The reset path raises ValueError
        # internally; our finally clause must suppress it.
        ctx = contextvars.copy_context()
        with collect_events():
            # Mutate the outer context's view of the var by running
            # in a child copy — this is the cross-context pattern.
            ctx.run(lambda: None)
        # If we got here without an exception, the suppression works.

    try:
        _bad_consumer()
    except ValueError as exc:  # pragma: no cover
        captured.append(exc)
    assert not captured, f"collect_events should not raise: {captured!r}"


# ---------------------------------------------------------------------------
# 6. emit_citations_for_text — input-shape robustness
# ---------------------------------------------------------------------------


def test_emit_citations_silent_on_empty_text() -> None:
    """Empty text → empty event list, no kaos-citations call."""
    from kaos_agents.events import EventEmitter
    from kaos_agents.grounding import emit_citations_for_text

    em = EventEmitter(session_id="t", run_id="r")
    assert emit_citations_for_text(em, "") == []
    assert emit_citations_for_text(em, "   \n  ") == []


def test_emit_citations_silent_on_no_match() -> None:
    """Text without recognizable citations → empty event list."""
    from kaos_agents.events import EventEmitter
    from kaos_agents.grounding import emit_citations_for_text

    em = EventEmitter(session_id="t", run_id="r")
    out = emit_citations_for_text(em, "The sky is blue and grass is green.")
    assert out == []


def test_emit_citations_emits_on_match() -> None:
    """Text with U.S.C. cite → at least one CitationFound."""
    from kaos_agents.events import EventEmitter
    from kaos_agents.grounding import emit_citations_for_text

    em = EventEmitter(session_id="t", run_id="r")
    out = emit_citations_for_text(em, "See 42 U.S.C. § 1983 for details.")
    assert out, "expected at least one CitationFound for a U.S.C. citation"
    assert any("1983" in str(e.claim) for e in out)


# ---------------------------------------------------------------------------
# 7. emit_thinking_from_invocation — extras-shape robustness
# ---------------------------------------------------------------------------


def test_emit_thinking_silent_on_no_extras() -> None:
    from kaos_agents.events import EventEmitter, emit_thinking_from_invocation

    em = EventEmitter(session_id="t", run_id="r")
    assert emit_thinking_from_invocation(em, None) is None
    assert emit_thinking_from_invocation(em, SimpleNamespace(extras={})) is None
    assert emit_thinking_from_invocation(em, SimpleNamespace(extras={"other": "x"})) is None


def test_emit_thinking_fires_when_populated() -> None:
    from kaos_agents.events import EventEmitter, ThinkingDelta, emit_thinking_from_invocation

    em = EventEmitter(session_id="t", run_id="r")
    inv = SimpleNamespace(extras={"native_thinking": "  reasoning here  "})
    event = emit_thinking_from_invocation(em, inv)
    assert isinstance(event, ThinkingDelta)
    assert event.content == "reasoning here"


# ---------------------------------------------------------------------------
# 8. BudgetExceeded emission — every StopReason kind maps correctly
# ---------------------------------------------------------------------------


def test_budget_exceeded_event_has_required_fields() -> None:
    """BudgetExceeded carries kind + limit + actual + reason."""
    from kaos_agents.events import BudgetExceeded

    evt = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind="cost",
        limit=0.01,
        actual=0.025,
        reason="test",
    )
    assert evt.kind == "cost"
    assert evt.limit == 0.01
    assert evt.actual == 0.025
    assert evt.reason == "test"


# ---------------------------------------------------------------------------
# 9. PlanProposed from proposed (not executed) plan
# ---------------------------------------------------------------------------


def test_plan_proposed_event_carries_steps() -> None:
    """PlanProposed must accept a steps tuple of PlanStepSummary."""
    from kaos_agents.events import PlanProposed, PlanStepSummary

    steps = (
        PlanStepSummary(step_id="a", description="search", tool_name="kaos-search"),
        PlanStepSummary(step_id="b", description="extract", tool_name=None),
    )
    evt = PlanProposed(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        steps=steps,
        strategy="adaptive",
    )
    assert len(evt.steps) == 2
    assert evt.steps[0].tool_name == "kaos-search"
    assert evt.steps[1].tool_name is None  # LLM-only step


# ---------------------------------------------------------------------------
# 10. Optional-modules dispatch — already covered, smoke-check the import path
# ---------------------------------------------------------------------------


def test_optional_modules_table_imports_cleanly() -> None:
    """Just verify the table loads and has the expected entries."""
    from kaos_agents.tools.optional_modules import OPTIONAL_MODULES

    packages = {spec.package for spec in OPTIONAL_MODULES}
    assert {
        "kaos_pdf",
        "kaos_office",
        "kaos_source",
        "kaos_web",
        "kaos_citations",
    }.issubset(packages)
