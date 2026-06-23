"""Bridge KaosTool (kaos-core) to Tool (kaos-llm-core) for ReAct consumption.

This is the single adapter that lets the agent's ReAct loop call any
of the 182+ MCP tools registered on the KaosRuntime.

WS-0.1 — permission gate. The adapter accepts an optional
:class:`PermissionPolicy` that is consulted **before** the underlying
tool's ``execute()`` runs. This closes a safety race where the
Runner's post-hoc permission check (``runner.py:159``) saw
``ToolCallStart`` events only after ReAct had already invoked the tool
and committed side effects. With the gate in place:

- ``PermissionDecision.DENY`` → the underlying tool is NOT called;
  the executor returns a structured error string and records the
  refusal for observability.
- ``PermissionDecision.ASK`` → same — the tool is NOT called; the
  Runner's post-hoc machinery still pauses the run when it sees the
  trajectory-replay ``ToolCallStart``, but now the side effect has been
  prevented rather than just after-the-fact-annotated.
- ``PermissionDecision.ALLOW`` (or no policy) → the executor proxies
  to the underlying ``KaosTool.execute()`` unchanged.

The post-hoc Runner check is retained as defense in depth; the gate
is the primary barrier.
"""

from __future__ import annotations

import asyncio
import json
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.types import PermissionDecision

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.base.tool import KaosTool
    from kaos_core.registry.container import KaosRuntime

    from kaos_agents.runtime.permissions import PermissionPolicy

logger = get_logger(__name__)


def kaos_tool_to_llm_tool(
    kaos_tool: KaosTool,
    context: KaosContext | None = None,
    *,
    permission_policy: PermissionPolicy | None = None,
    tool_timeout_seconds: float | None = None,
) -> Any:
    """Wrap a KaosTool as a kaos-llm-core Tool for use in ReAct.

    The returned Tool has:
    - definition: ToolDefinition built from KaosTool.metadata
    - executor: async function that calls KaosTool.execute()

    Args:
        kaos_tool: The KaosTool instance to wrap.
        context: Optional KaosContext passed to execute() for session/trace tracking.
        permission_policy: Optional pre-execution gate (WS-0.1). When set,
            the policy is consulted **before** the underlying
            ``KaosTool.execute()`` runs; DENY or ASK decisions prevent
            the tool from running and return a structured error string.
            ``None`` installs :meth:`PermissionPolicy.default_safe` (KC17-P0-2)
            so destructiveHint / humanConfirmationRequired tools escalate
            to human approval. Pre-KC17 behaviour ("None = bypass all
            checks") is gone — callers that genuinely need that must
            construct a ``PermissionPolicy`` with explicit allow rules.
        tool_timeout_seconds: Per-invocation timeout (B0.2). The
            ``await kaos_tool.execute(...)`` call is wrapped in
            ``async with asyncio.timeout(timeout_seconds)``; on
            ``TimeoutError`` the executor raises
            :class:`ToolReportedError` so ReAct records the failure
            with ``is_error=True`` and the agent can re-plan. ``None``
            (default) reads ``KaosAgentSettings().tool_timeout_seconds``
            so chat-pattern ReAct picks up the same configurable bound
            as planning's ``_act_tool``. Pre-B0.2 the bridge had no
            timeout and a slow tool could pin a turn indefinitely.

    Returns:
        A kaos-llm-core Tool instance.

    Raises:
        ImportError: If kaos-llm-core or kaos-llm-client are not installed.
            Install with: ``uv add 'kaos-agents[llm]'``
    """
    from kaos_agents._llm_imports import require_llm_client, require_llm_core

    require_llm_client()
    require_llm_core()
    from kaos_llm_client.types import ToolDefinition
    from kaos_llm_core.programs.tool import Tool

    meta = kaos_tool.metadata

    definition = ToolDefinition(
        name=meta.name,
        description=meta.description or "",
        parameters=meta.get_input_json_schema(),
    )

    # KC17-P0-2: when no caller-supplied policy, install the default-safe
    # policy. This makes destructiveHint=True / humanConfirmationRequired
    # tools ESCALATE rather than silently execute. The pre-KC17 contract
    # was "no policy = permissive bypass" which let MCP/API surfaces fire
    # destructive tools without approval — see kaos-agents.md P0-2.
    effective_policy: PermissionPolicy
    if permission_policy is None:
        from kaos_agents.runtime.permissions import PermissionPolicy as _PP

        effective_policy = _PP.default_safe()
    else:
        effective_policy = permission_policy

    # B0.2: per-invocation timeout. None = inherit from settings so the
    # chat-pattern ReAct dispatch picks up the same bound as planning's
    # _act_tool.
    if tool_timeout_seconds is None:
        from kaos_agents.settings import KaosAgentSettings

        effective_timeout = KaosAgentSettings().tool_timeout_seconds
    else:
        effective_timeout = tool_timeout_seconds

    async def executor(**kwargs: Any) -> str:
        """Execute the wrapped KaosTool and return the result as text.

        Consults ``effective_policy`` before invoking the underlying
        tool — this is the WS-0.1 pre-execution gate that prevents
        destructive tools from running before ASK / DENY decisions are
        resolved. The Runner retains a post-hoc check on
        ``ToolCallStart`` events for defense in depth.

        Post KC17-P0-2 ``effective_policy`` is never ``None`` — callers
        that pass ``permission_policy=None`` get the default-safe policy
        installed above so destructive tools escalate to human approval
        instead of running silently.
        """
        decision = effective_policy.evaluate(meta.name, meta.annotations)
        if decision == PermissionDecision.DENY:
            logger.debug(
                "tool_bridge: permission policy DENIED tool=%r before execution",
                meta.name,
            )
            return json.dumps(
                {
                    "error": True,
                    "message": (
                        f"Tool {meta.name!r} denied by permission policy. "
                        "The underlying tool was NOT invoked and no side "
                        "effect was produced."
                    ),
                    "permission_decision": "deny",
                }
            )
        if decision == PermissionDecision.ASK:
            logger.debug(
                "tool_bridge: permission policy requested APPROVAL for tool=%r; "
                "skipping execution until resumed",
                meta.name,
            )
            return json.dumps(
                {
                    "error": True,
                    "message": (
                        f"Tool {meta.name!r} requires approval before running. "
                        "The underlying tool was NOT invoked; the run will "
                        "pause and resume after approval."
                    ),
                    "permission_decision": "ask",
                }
            )

        import time as _time

        from kaos_llm_core.errors import ToolReportedError

        # B0.7: pre-execution PII scrubber. Closes the adversarial
        # F-03 path where an agent forwards client SSN / EIN / credit-
        # card numbers from corpus content into a third-party tool
        # call (kaos-web-search, kaos-source-edgar-*, anything that
        # logs its args on the receiving side). The scrubber returns
        # a new kwargs dict; the original payload is left intact for
        # the caller. We log the matched pattern names so the audit
        # trail records that PII was filtered before dispatch.
        from kaos_agents.runtime.pii_scrubber import scrub_tool_args

        _scrub = scrub_tool_args(kwargs)
        if _scrub.scrubbed:
            logger.debug(
                "tool_bridge.pii_scrubbed: tool=%s patterns=%s",
                meta.name,
                ",".join(_scrub.matches),
            )
            kwargs = _scrub.kwargs

        logger.debug("tool_bridge.execute_start: tool=%s", meta.name)
        t_start = _time.perf_counter()
        try:
            async with asyncio.timeout(effective_timeout):
                result = await kaos_tool.execute(kwargs, context=context)
        except TimeoutError as exc:
            t_elapsed = (_time.perf_counter() - t_start) * 1000
            logger.warning(
                "tool_bridge.execute_timeout: tool=%s timeout_s=%.1f elapsed_ms=%.0f",
                meta.name,
                effective_timeout,
                t_elapsed,
            )
            # Theme A (2026-05-25): make the timeout observable in the
            # event stream. Pre-fix: the warning log fired but no typed
            # event was ever emitted, so audit / OTel / SPA consumers
            # had no record that a TOOL_CALL had timed out — the
            # downstream-visible signal was just a ToolReportedError
            # appearing in the next ReAct observation, with no
            # standalone span describing the failed call. The TOOL_CALL
            # span that ``loop.agent_loop._emit_tool_call_spans`` emits
            # is built from the post-execution ``ToolExecution`` record;
            # on a timeout no execution record is produced for this
            # call, so without an in-place emit the failure leaves no
            # trace in the event stream.
            #
            # ``span_error`` auto-generates a span_id; the collector's
            # current parent (the active TOOL_USE / TURN span) becomes
            # the parent automatically via ``EventEmitter.span`` parent
            # synthesis. Gracefully degrades when no emitter is in
            # scope (most non-agent callers of this bridge).
            from kaos_agents.events import SpanSubject, active_emitter

            _em = active_emitter()
            if _em is not None:
                # span_error requires a span_id; we don't have a prior
                # START to match against from this site (the canonical
                # TOOL_CALL.START is emitted post-execution by
                # ``loop.agent_loop._emit_tool_call_spans`` from the
                # ToolExecution record — but on timeout no execution
                # record exists). Use the lower-level ``span()`` with
                # span_id="" so ``EventEmitter.span`` synthesizes a
                # fresh ID (``resolved_span_id = span_id or _new_span_id()``).
                from kaos_agents.events.spans import SpanPhase

                _em.span(
                    SpanSubject.TOOL_CALL,
                    SpanPhase.ERROR,
                    span_id="",
                    name=f"tool.{meta.name}",
                    duration_ms=t_elapsed,
                    error_type="tool_timeout",
                    error_message=(
                        f"Tool {meta.name!r} timed out after "
                        f"{effective_timeout:.1f}s (elapsed {t_elapsed:.0f}ms)."
                    ),
                    attributes={
                        "tool_name": meta.name,
                        "timeout_seconds": effective_timeout,
                        "elapsed_ms": t_elapsed,
                        "is_error": True,
                    },
                )
            # Same pattern as the result.isError branch below: raise
            # ToolReportedError so ReAct's _invoke_one records the
            # observation with is_error=True. The agent then sees the
            # failure in its memory and can re-plan (e.g. swap tools or
            # ask for a simpler query). Returning a string here would
            # be recorded as is_error=False — the same failure mode that
            # the inventory P0-1 #437 fix closed for explicit tool errors.
            raise ToolReportedError(
                {
                    "error": True,
                    "message": (
                        f"Tool {meta.name!r} timed out after "
                        f"{effective_timeout:.1f}s. The underlying tool was "
                        "still running when the deadline fired. Try a "
                        "simpler query, narrow the scope, or pick a faster "
                        "tool variant."
                    ),
                    "timeout_seconds": effective_timeout,
                }
            ) from exc
        t_elapsed = (_time.perf_counter() - t_start) * 1000
        logger.debug(
            "tool_bridge.execute_result: tool=%s is_error=%s latency_ms=%.0f text_len=%d",
            meta.name,
            result.isError,
            t_elapsed,
            len(result.text or ""),
        )
        if result.isError:
            # Raise ToolReportedError so kaos-llm-core's ReAct dispatcher
            # records is_error=True on the resulting ToolObservation while
            # preserving the tool's own remediation hint as the payload.
            #
            # Pre-fix: this returned a JSON string. ReAct's `_invoke_one`
            # treats any non-exceptional return as success (is_error=False)
            # by design (audit finding #3), so the failure flag was lost
            # before the agent's memory ever saw it. Downstream — UI chips,
            # the critic's `_is_stuck` heuristic, the audit CLI's Red Flag A,
            # the SPA's tool_call_summaries — all saw "done ✓" while the
            # body said "error". Closes inventory P0-1 #437.
            #
            # P0-1 / inventory item, kaos-modules/docs/plans/
            # 2026-05-19-consolidated-issue-inventory.md.
            error_payload = {
                "error": True,
                "message": result.text or "Tool execution failed",
            }
            raise ToolReportedError(error_payload)
        # Return BOTH the human-readable summary AND the structured
        # content when a tool produced both. Returning only `result.text`
        # was the root cause of plan-execute's "needs_replan after 1
        # step" symptom: tools like kaos-source-fr-search return a
        # one-line summary as `text` ("Found 11 documents") and the
        # actual fields (document_number, title, publication_date,
        # citation, abstract) as `structuredContent`. The bridge was
        # dropping `structuredContent`, so the semantic evaluator (and
        # any LLM caller) saw only the summary and judged the step
        # "missing required fields" — wrongly. Concatenating both
        # surfaces the data without losing the summary's signal.
        text_part = result.text or ""
        structured = result.structuredContent
        if structured:
            structured_json = json.dumps(structured, default=str)
            if text_part:
                return f"{text_part}\n\n{structured_json}"
            return structured_json
        if text_part:
            return text_part
        return json.dumps(result.to_mcp_dict(), default=str)

    return Tool(definition=definition, executor=executor)


def _matches_filter(tool_name: str, patterns: list[str]) -> bool:
    """Return True if ``tool_name`` matches any of the given glob patterns.

    Each pattern is Unix-shell-style (``fnmatch`` semantics):

    - Exact names (``"kaos-pdf-extract"``) match by equality.
    - Globs (``"kaos-source-*"``, ``"kaos-*-search"``) wildcard-match.
    - Mixed lists work: ``["kaos-pdf-extract", "kaos-source-*"]`` matches
      one exact tool plus every kaos-source tool.

    WS-0.5: prior to 2026-04 this function did exact-name matching only,
    despite :class:`kaos_agents.config.Agent.tools` being documented as
    "glob patterns matched against runtime tool names". That silent
    mismatch meant user filters like ``kaos-source-*`` yielded zero
    tools — the recipe ergonomics contract was broken.
    """
    return any(fnmatch(tool_name, pattern) for pattern in patterns)


def bridge_runtime_tools(
    runtime: KaosRuntime,
    context: KaosContext | None = None,
    *,
    filter_names: list[str] | None = None,
    max_tools: int = 30,
    permission_policy: PermissionPolicy | None = None,
    relevance_query: str | None = None,
    retrieval_lexicon: Any = None,
    tool_timeout_seconds: float | None = None,
) -> list[Any]:
    """Bridge KaosRuntime tools to ReAct-compatible Tools.

    Args:
        runtime: The KaosRuntime with registered tools.
        context: Optional KaosContext for tool execution.
        filter_names: If provided, only bridge tools whose name matches
            any entry (:func:`fnmatch` glob semantics). Pass exact names,
            ``kaos-source-*`` wildcards, or a mix. ``None`` bridges every
            tool (capped at ``max_tools``).
        max_tools: Maximum number of tools to bridge (default 30).
            ReAct performance degrades with too many tools.
        permission_policy: Optional pre-execution gate (WS-0.1). Threaded
            into each bridged tool's executor so DENY / ASK decisions
            prevent the underlying ``KaosTool.execute()`` from running.
            ``None`` installs :meth:`PermissionPolicy.default_safe`
            (KC17-P0-2) inside each bridged executor so destructive tools
            escalate to human approval. Pre-KC17 "None = bypass" behaviour
            is gone.
        relevance_query: When set, replaces the iteration-order cap
            with a BM25 retrieval over the tool catalog. The top
            ``max_tools`` by relevance to ``relevance_query`` are
            bridged; the rest are excluded. Composes with
            ``filter_names`` (the glob filter applies first, then
            retrieval ranks the survivors). Typically the user's
            current message or step description.
        retrieval_lexicon: Optional ``kaos_nlp_core.lexicon.Lexicon``
            for synonym / hypernym expansion at retrieval time. Pass
            a populated OpenGloss lexicon to turn near-misses ("FR"
            vs "Federal Register") into hits. Only consulted when
            ``relevance_query`` is set. Default ``None`` (plain BM25).
        tool_timeout_seconds: B0.2 per-invocation timeout threaded
            into each bridged executor. ``None`` (default) lets each
            executor inherit from ``KaosAgentSettings().tool_timeout_seconds``;
            pass a float to override for a single bridge call (tests,
            short-deadline gateways).

    Returns:
        List of kaos-llm-core Tool instances.
    """
    candidates = list(runtime.tools.list_tool_objects())

    # Step 1: apply name-glob filter (when set) to narrow the pool.
    if filter_names:
        candidates = [t for t in candidates if _matches_filter(t.metadata.name, filter_names)]

    # Step 2: retrieval-rank when the caller passed a query.
    # Without a query we fall back to iteration order (legacy behavior).
    if relevance_query and len(candidates) > max_tools:
        from kaos_agents.runtime.tool_retrieval import ToolRetrieval

        retrieval = ToolRetrieval(candidates, lexicon=retrieval_lexicon)
        ranked = retrieval.select(relevance_query, top_k=max_tools)
        # ``ranked`` may be shorter than max_tools if many tools tied
        # at score 0 and were filtered. In that case append the
        # iteration-order tail to fill up to max_tools — keeps the
        # ReAct surface roughly stable while still preferring
        # high-relevance hits.
        ranked_ids = {id(t) for t in ranked}
        for t in candidates:
            if len(ranked) >= max_tools:
                break
            if id(t) not in ranked_ids:
                ranked.append(t)
        candidates = ranked
        logger.debug(
            "tool_bridge: retrieval-ranked %d tools by relevance to query (len=%d chars)",
            len(candidates),
            len(relevance_query),
        )

    # Step 3: cap and bridge.
    tools = []
    for kaos_tool in candidates:
        tools.append(
            kaos_tool_to_llm_tool(
                kaos_tool,
                context,
                permission_policy=permission_policy,
                tool_timeout_seconds=tool_timeout_seconds,
            )
        )
        if len(tools) >= max_tools:
            if not relevance_query and len(candidates) > max_tools:
                # Only warn on iteration-order capping — retrieval capping
                # is deliberate and the user opted in.
                logger.warning(
                    "tool_bridge: capped at %d tools (total available: %d). "
                    "Pass relevance_query for query-aware selection, or "
                    "use filter_names to narrow.",
                    max_tools,
                    len(candidates),
                )
            break

    logger.debug("tool_bridge: bridged %d tools", len(tools))
    return tools
