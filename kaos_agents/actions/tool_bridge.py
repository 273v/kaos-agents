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
            ``None`` preserves legacy permissive behavior.

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

    async def executor(**kwargs: Any) -> str:
        """Execute the wrapped KaosTool and return the result as text.

        Consults ``permission_policy`` before invoking the underlying
        tool — this is the WS-0.1 pre-execution gate that prevents
        destructive tools from running before ASK / DENY decisions are
        resolved. The Runner retains a post-hoc check on
        ``ToolCallStart`` events for defense in depth.
        """
        if permission_policy is not None:
            decision = permission_policy.evaluate(meta.name, meta.annotations)
            if decision == PermissionDecision.DENY:
                logger.info(
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
                logger.info(
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

        logger.debug("tool_bridge.execute_start: tool=%s", meta.name)
        t_start = _time.perf_counter()
        result = await kaos_tool.execute(kwargs, context=context)
        t_elapsed = (_time.perf_counter() - t_start) * 1000
        logger.debug(
            "tool_bridge.execute_result: tool=%s is_error=%s latency_ms=%.0f text_len=%d",
            meta.name,
            result.isError,
            t_elapsed,
            len(result.text or ""),
        )
        if result.isError:
            return json.dumps({"error": True, "message": result.text or "Tool execution failed"})
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
            ``None`` preserves legacy permissive behavior.

    Returns:
        List of kaos-llm-core Tool instances.
    """
    tools = []
    for kaos_tool in runtime.tools.list_tool_objects():
        if filter_names and not _matches_filter(kaos_tool.metadata.name, filter_names):
            continue
        tools.append(kaos_tool_to_llm_tool(kaos_tool, context, permission_policy=permission_policy))
        if len(tools) >= max_tools:
            logger.warning(
                "tool_bridge: capped at %d tools (total available: %d). "
                "Use filter_names to select relevant tools.",
                max_tools,
                len(list(runtime.tools.list_tool_objects())),
            )
            break

    logger.debug("tool_bridge: bridged %d tools", len(tools))
    return tools
