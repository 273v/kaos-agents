"""Theme A: tool-call failure observability via ``active_emitter`` (2026-05-25).

The audit found three sites in the agent runtime that silently
swallowed tool-execution failures (no log warning, no typed event,
no Span lifecycle):

1. :func:`kaos_agents.capabilities.retrieve._invoke_tool` — ``except
   Exception: logger.debug; return None`` on ``tool.execute()`` crash.
2. :func:`kaos_agents.actions.tool_bridge.kaos_tool_to_llm_tool` —
   ``asyncio.timeout`` ``TimeoutError`` handler logged at warning and
   raised but emitted no typed event.
3. :func:`kaos_agents.planning.act._act_tool` — ``asyncio.timeout``
   ``TimeoutError`` handler returned an error dict but emitted no
   typed event.

The fix: each site now consults
:func:`~kaos_agents.events.active_emitter` and, when an emitter is
in scope, emits a typed event so the failure is visible in the
event stream. Tests below set up a ``collect_events`` + ``use_emitter``
pair and assert the typed events surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult
from kaos_llm_core.errors import ToolReportedError

from kaos_agents.events import (
    EventEmitter,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    collect_events,
    use_emitter,
)

# ---------------------------------------------------------------------------
# Site 1: capabilities.retrieve._invoke_tool — silent skip on tool crash
# ---------------------------------------------------------------------------


class _RaisingRetrievalTool(KaosTool):
    """A SEARCH-shaped tool whose ``execute`` always raises."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-raising-search",
            description="Always raises.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.QUERY,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        raise RuntimeError("synthetic retrieval failure")


class TestRetrieveInvokeToolEmitsRunErrorOnCrash:
    """``_invoke_tool`` emits a :class:`RunError` when the tool raises."""

    @pytest.mark.asyncio
    async def test_invoke_tool_emits_run_error_when_tool_raises(self) -> None:
        """Theme A site 1: silent-skip is now observable.

        Pre-fix: ``_invoke_tool`` returned ``None`` and only logged
        at debug level — the federated retrieve loop ignored the
        crash silently. Now we emit a typed ``RunError`` with
        ``error_type="tool_execution_failure"`` so audit / OTel /
        SPA consumers see the skip.
        """
        from kaos_agents.capabilities.retrieve import _invoke_tool

        emitter = EventEmitter(session_id="sess-r1", run_id="run-r1")
        # Patch _build_tool_args path: this tool has no declared
        # input schema, so the helper's "query" check would skip it
        # before reaching .execute(). Force a non-empty args path by
        # subclassing with a schema that accepts ``query``.
        from kaos_core.types.parameters import ParameterSchema

        class _SchemaTool(_RaisingRetrievalTool):
            @property
            def metadata(self) -> ToolMetadata:
                base = super().metadata
                return ToolMetadata(
                    name=base.name,
                    description=base.description,
                    category=base.category,
                    capability=base.capability,
                    module_name=base.module_name,
                    version=base.version,
                    annotations=base.annotations,
                    input_schema=[
                        ParameterSchema(name="query", type="string", description="q"),
                    ],
                )

        tool = _SchemaTool()

        with collect_events() as collector, use_emitter(emitter):
            result = await _invoke_tool(tool, "anything", max_n=5)

        # Federated contract preserved: caller sees None.
        assert result is None
        # NEW: a RunError surfaced in the event stream.
        run_errors = [e for e in collector.events if isinstance(e, RunError)]
        assert len(run_errors) == 1, (
            "_invoke_tool must emit exactly one RunError when the tool raises; "
            f"got {len(run_errors)} (collector saw {len(collector.events)} events)."
        )
        err = run_errors[0]
        assert err.error_type == "tool_execution_failure"
        assert "test-raising-search" in err.message
        assert "synthetic retrieval failure" in err.message

    @pytest.mark.asyncio
    async def test_invoke_tool_no_emitter_back_compat(self) -> None:
        """No ``use_emitter`` scope — ``_invoke_tool`` must still skip silently.

        Back-compat: the federated-retrieval contract is unchanged
        for callers that don't open a ``use_emitter`` block (most
        non-agent callers — direct unit tests, ad-hoc scripts).
        """
        from kaos_core.types.parameters import ParameterSchema

        from kaos_agents.capabilities.retrieve import _invoke_tool

        class _SchemaTool(_RaisingRetrievalTool):
            @property
            def metadata(self) -> ToolMetadata:
                base = super().metadata
                return ToolMetadata(
                    name=base.name,
                    description=base.description,
                    category=base.category,
                    capability=base.capability,
                    module_name=base.module_name,
                    version=base.version,
                    annotations=base.annotations,
                    input_schema=[
                        ParameterSchema(name="query", type="string", description="q"),
                    ],
                )

        # No use_emitter scope; the helper must NOT raise and MUST
        # return None.
        result = await _invoke_tool(_SchemaTool(), "x", max_n=3)
        assert result is None


# ---------------------------------------------------------------------------
# Site 2: actions.tool_bridge — TimeoutError handler now emits Span(TOOL_CALL, ERROR)
# ---------------------------------------------------------------------------


class _SlowKaosTool(KaosTool):
    """Sleeps past the deadline; forces the tool_bridge timeout path."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test-slow-bridge-tool",
            description="Sleeps then returns.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult.create_text("done")


class TestToolBridgeTimeoutEmitsSpan:
    """The tool_bridge timeout handler emits Span(TOOL_CALL, ERROR)."""

    @pytest.mark.asyncio
    async def test_timeout_emits_span_tool_call_error(self) -> None:
        """Theme A site 2: timeout produces a typed Span(TOOL_CALL, ERROR).

        Pre-fix: the warning log fired and ToolReportedError was raised,
        but no Span lifecycle event ever surfaced for the failed tool
        call — the post-execution span emitter in agent_loop only runs
        on a ToolExecution record, which doesn't exist on timeout.
        """
        from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool

        emitter = EventEmitter(session_id="sess-tb", run_id="run-tb")
        slow = _SlowKaosTool(delay_seconds=2.0)
        bridged = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=0.05)

        with (
            collect_events() as collector,
            use_emitter(emitter),
            pytest.raises(ToolReportedError),
        ):
            await bridged.invoke({})

        # The Span(TOOL_CALL, ERROR) must be in the collector.
        tool_call_errors = [
            e
            for e in collector.events
            if isinstance(e, Span)
            and e.subject == SpanSubject.TOOL_CALL
            and e.phase == SpanPhase.ERROR
        ]
        assert len(tool_call_errors) == 1, (
            "tool_bridge timeout must emit exactly one Span(TOOL_CALL, ERROR); "
            f"got {len(tool_call_errors)} (collector saw {len(collector.events)} events)."
        )
        span = tool_call_errors[0]
        assert span.error_type == "tool_timeout"
        assert "test-slow-bridge-tool" in (span.error_message or "")
        assert span.attributes.get("tool_name") == "test-slow-bridge-tool"
        assert span.attributes.get("is_error") is True
        assert span.attributes.get("timeout_seconds") == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_timeout_no_emitter_back_compat(self) -> None:
        """Bridge timeout works without ``use_emitter`` — raises ToolReportedError.

        Back-compat: tool_bridge has many non-agent callers (direct
        unit tests, ad-hoc invocations). Absent an emitter scope,
        the original raise-on-timeout behaviour is preserved.
        """
        from kaos_agents.actions.tool_bridge import kaos_tool_to_llm_tool

        slow = _SlowKaosTool(delay_seconds=2.0)
        bridged = kaos_tool_to_llm_tool(slow, tool_timeout_seconds=0.05)

        # No use_emitter scope — must still raise ToolReportedError.
        with pytest.raises(ToolReportedError):
            await bridged.invoke({})


# ---------------------------------------------------------------------------
# Site 3: planning.act._act_tool — TimeoutError handler now emits Span(TOOL_CALL, ERROR)
# ---------------------------------------------------------------------------


def _make_slow_llm_tool() -> Any:
    """Build a real kaos-llm-core ``Tool`` whose executor sleeps past the deadline.

    Using the real ``Tool`` (not a duck-typed stub) keeps ``ty`` happy
    when we pass it to ``_act_tool(tool=...)`` and exercises the same
    surface that the plan-execute path uses in production.
    """
    from kaos_llm_client.types import ToolDefinition
    from kaos_llm_core.programs.tool import Tool

    async def _sleep_then_done(**_kwargs: Any) -> str:
        await asyncio.sleep(2.0)
        return "done"

    td = ToolDefinition(
        name="test-act-slow-tool",
        description="Sleeps past the deadline.",
        parameters={"type": "object", "properties": {}},
    )
    return Tool(definition=td, executor=_sleep_then_done)


class TestActToolTimeoutEmitsSpan:
    """``_act_tool``'s timeout handler emits Span(TOOL_CALL, ERROR)."""

    @pytest.mark.asyncio
    async def test_act_tool_timeout_emits_span(self) -> None:
        """Theme A site 3: plan-execute timeout produces a typed Span.

        Pre-fix: the timeout returned an error dict that was recorded
        in the step trace, but no Span ever surfaced — OTel exporters
        and the SPA tool-call stream saw nothing for the timed-out
        invocation.
        """
        from kaos_agents.planning.act import _act_tool

        emitter = EventEmitter(session_id="sess-act", run_id="run-act")

        with collect_events() as collector, use_emitter(emitter):
            result = await _act_tool(
                _make_slow_llm_tool(),
                {"x": 1},
                timeout_seconds=0.05,
            )

        # The dict-return contract is preserved.
        assert result["is_error"] is True
        assert "timed out" in result["output"].lower()

        tool_call_errors = [
            e
            for e in collector.events
            if isinstance(e, Span)
            and e.subject == SpanSubject.TOOL_CALL
            and e.phase == SpanPhase.ERROR
        ]
        assert len(tool_call_errors) == 1, (
            "_act_tool timeout must emit exactly one Span(TOOL_CALL, ERROR); "
            f"got {len(tool_call_errors)} (collector saw {len(collector.events)} events)."
        )
        span = tool_call_errors[0]
        assert span.error_type == "tool_timeout"
        assert "test-act-slow-tool" in (span.error_message or "")
        assert span.attributes.get("tool_name") == "test-act-slow-tool"
        assert span.attributes.get("is_error") is True

    @pytest.mark.asyncio
    async def test_act_tool_timeout_no_emitter_back_compat(self) -> None:
        """``_act_tool`` works without ``use_emitter`` — returns error dict.

        Back-compat: many planning tests instantiate ``_act_tool``
        directly without opening a ``use_emitter`` scope.
        """
        from kaos_agents.planning.act import _act_tool

        result = await _act_tool(
            _make_slow_llm_tool(),
            {"x": 1},
            timeout_seconds=0.05,
        )
        assert result["is_error"] is True
        assert "timed out" in result["output"].lower()
