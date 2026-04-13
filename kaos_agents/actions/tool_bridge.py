"""Bridge KaosTool (kaos-core) to Tool (kaos-llm-core) for ReAct consumption.

This is the single adapter that lets the agent's ReAct loop call any
of the 182+ MCP tools registered on the KaosRuntime.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.base.tool import KaosTool
    from kaos_core.registry.container import KaosRuntime

logger = get_logger(__name__)


def kaos_tool_to_llm_tool(kaos_tool: KaosTool, context: KaosContext | None = None) -> Any:
    """Wrap a KaosTool as a kaos-llm-core Tool for use in ReAct.

    The returned Tool has:
    - definition: ToolDefinition built from KaosTool.metadata
    - executor: async function that calls KaosTool.execute()

    Args:
        kaos_tool: The KaosTool instance to wrap.
        context: Optional KaosContext passed to execute() for session/trace tracking.

    Returns:
        A kaos-llm-core Tool instance.
    """
    from kaos_llm_client.types import ToolDefinition
    from kaos_llm_core.programs.tool import Tool

    meta = kaos_tool.metadata

    definition = ToolDefinition(
        name=meta.name,
        description=meta.description or "",
        parameters=meta.get_input_json_schema(),
    )

    async def executor(**kwargs: Any) -> str:
        """Execute the wrapped KaosTool and return the result as text."""
        result = await kaos_tool.execute(kwargs, context=context)
        if result.isError:
            return json.dumps({"error": True, "message": result.text or "Tool execution failed"})
        # Return text content or structured content as JSON
        if result.text:
            return result.text
        return json.dumps(result.to_mcp_dict(), default=str)

    return Tool(definition=definition, executor=executor)


def bridge_runtime_tools(
    runtime: KaosRuntime,
    context: KaosContext | None = None,
    *,
    filter_names: list[str] | None = None,
    max_tools: int = 30,
) -> list[Any]:
    """Bridge KaosRuntime tools to ReAct-compatible Tools.

    Args:
        runtime: The KaosRuntime with registered tools.
        context: Optional KaosContext for tool execution.
        filter_names: If provided, only bridge tools with these names.
            If None, bridges all tools (capped at max_tools).
        max_tools: Maximum number of tools to bridge (default 30).
            ReAct performance degrades with too many tools.

    Returns:
        List of kaos-llm-core Tool instances.
    """
    tools = []
    for kaos_tool in runtime.tools.list_tool_objects():
        if filter_names and kaos_tool.metadata.name not in filter_names:
            continue
        tools.append(kaos_tool_to_llm_tool(kaos_tool, context))
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
