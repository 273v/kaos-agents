"""ChatAgent — conversation pattern with tool calling via ReAct.

Extends BaseAgent with:
- Tool-using responses via ReAct (inner loop)
- Runtime tool bridge (KaosTool → ReAct Tool)
- Tool call recording for memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.actions.tool_bridge import bridge_runtime_tools
from kaos_agents.agent import BaseAgent
from kaos_agents.models import ToolCallRecord

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem, MemoryType
    from kaos_agents.settings import KaosAgentSettings

logger = get_logger(__name__)


class ChatAgent(BaseAgent):
    """Conversational agent with tool calling.

    Extends BaseAgent to handle tool_use intent via ReAct. Bridges
    KaosRuntime tools for the ReAct inner loop.

    Usage:
        agent = ChatAgent(
            vfs=runtime.vfs,
            runtime=runtime,
            context=context,
            model="anthropic:claude-sonnet-4-6",
        )
        response = await agent.turn("Extract dates from contract.pdf", session_id="abc")
    """

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        model: str = "anthropic:claude-sonnet-4-6",
        tool_filter: list[str] | None = None,
        max_tools: int = 30,
        max_react_iterations: int = 10,
        settings: KaosAgentSettings | None = None,
    ) -> None:
        super().__init__(vfs, model=model, settings=settings)
        self._runtime = runtime
        self._context = context
        self._tool_filter = tool_filter
        self._max_tools = max_tools
        self._max_react_iterations = max_react_iterations

    async def _handle_tool_use(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle tool-using request via ReAct loop."""
        if self._runtime is None:
            logger.warning("chat_agent: no runtime — falling back to simple response")
            response = await self._simple_respond(message, memory)
            return response, []

        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.react import ReAct

        class ToolTask(Signature):
            """Complete the user's request using the available tools."""

            question: str = InputField(description="The user's request")
            context: str = InputField(description="Conversation context")
            answer: str = OutputField(description="Your final answer to the user")

        # Bridge runtime tools
        tools = bridge_runtime_tools(
            self._runtime,
            self._context,
            filter_names=self._tool_filter,
            max_tools=self._max_tools,
        )

        if not tools:
            logger.warning("chat_agent: no tools available — falling back to simple response")
            response = await self._simple_respond(message, memory)
            return response, []

        # Build conversation context
        from kaos_agents.memory.types import MemoryType

        recent = memory.get_recent(MemoryType.MESSAGES, 10)
        context_text = "\n".join(item.content for item in recent) if recent else ""

        # Run ReAct
        react = ReAct(
            ToolTask,
            tools=tools,
            model=self._model,
            max_iterations=self._max_react_iterations,
            instructions="Complete the user's request using the available tools. "
            "Be thorough and cite your sources.",
        )

        try:
            result = await react(question=message, context=context_text)
        except Exception as exc:
            logger.warning("chat_agent: ReAct failed: %s", exc)
            response = await self._simple_respond(
                message,
                memory,
                extra_instruction=f"A tool-calling attempt failed: {exc}. "
                "Respond helpfully without tools.",
            )
            return response, []

        # Extract tool calls from trajectory
        tool_calls: list[ToolCallRecord] = []
        for iteration in result.trajectory:
            for obs in iteration.tool_results:
                result_text = str(obs.result)[:200] if obs.result else ""
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name=obs.tool_name,
                        arguments=obs.arguments or {},
                        result_summary=result_text,
                        is_error=obs.is_error,
                    )
                )

        response_text = str(result.answer) if result.outputs and result.answer else ""
        if not response_text and result.trajectory:
            # Fallback: use the last iteration's text
            response_text = result.trajectory[-1].text

        logger.debug(
            "chat_agent: ReAct completed — %d iterations, %d tool calls, stop=%s",
            result.iterations_used,
            len(tool_calls),
            result.stop_reason,
        )

        return response_text, tool_calls
