"""BaseAgent — the core agent loop.

Stateless agent that orchestrates memory, tool calling, and LLM dispatch.
The agent is reconstructed per MCP call from session_id. All persistent
state lives in SessionMemory, which hydrates from VFS.

The 8-step turn:
1. Hydrate memory from store (or create fresh)
2. Begin turn (clear ephemeral sections)
3. Add user message to MESSAGES section
4. Assemble context from memory
5. Classify intent
6. Dispatch to handler (respond, tool_use, research, plan, clarify)
7. Update memory (response, actions, findings)
8. End turn, persist, return response
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.context.classify import classify_intent
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.types import MemoryType
from kaos_agents.models import AgentResponse, IntentResult, IntentType, ToolCallRecord
from kaos_agents.settings import KaosAgentSettings

if TYPE_CHECKING:
    from kaos_core.vfs.core import VirtualFileSystem

logger = get_logger(__name__)


class BaseAgent:
    """Core agent with the 8-step turn loop.

    Subclasses (ChatAgent, PlanExecuteAgent) override dispatch handlers
    for each intent type. BaseAgent provides the loop scaffolding.

    The agent is stateless — constructed per call, not per session.
    All state lives in SessionMemory.
    """

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        model: str = "anthropic:claude-haiku-4-5",
        settings: KaosAgentSettings | None = None,
    ) -> None:
        self._settings = KaosAgentSettings.resolve(settings)
        self._store = SessionStore(vfs)
        self._model = model

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Execute a single agent turn.

        This is the main entry point. Implements the 8-step loop.

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Returns:
            AgentResponse with the agent's reply and metadata.
        """
        # Step 1: Hydrate memory
        memory = await self._store.load_or_create(session_id)

        # Step 2: Begin turn
        memory.begin_turn()

        # Step 3: Add user message
        memory.add(MemoryType.MESSAGES, f"user: {message}")

        # Step 4: Assemble context
        context_items = memory.get_sections(
            [MemoryType.MESSAGES, MemoryType.ACTIONS, MemoryType.FINDINGS, MemoryType.DOCUMENTS],
            total_budget_tokens=self._settings.default_context_budget_tokens,
            priority_order=[
                MemoryType.MESSAGES,
                MemoryType.FINDINGS,
                MemoryType.DOCUMENTS,
                MemoryType.ACTIONS,
            ],
        )

        # Step 5: Classify intent (with assembled context)
        intent = await self._classify(message, memory, context_items)

        logger.debug(
            "agent.turn: session=%s intent=%s confidence=%.2f",
            session_id,
            intent.intent.value,
            intent.confidence,
        )

        # Step 6: Dispatch to handler
        response_text, tool_calls = await self._dispatch(intent, message, memory, context_items)

        # Step 7: Update memory
        memory.add(MemoryType.MESSAGES, f"assistant: {response_text}")
        if tool_calls:
            for tc in tool_calls:
                summary = f"Tool: {tc.tool_name}({tc.arguments}) → {tc.result_summary}"
                memory.add(MemoryType.ACTIONS, summary)

        # Step 8: End turn, persist
        memory.end_turn()
        await self._store.save(memory)

        return AgentResponse.create(
            text=response_text,
            intent=intent,
            tool_calls=tuple(tool_calls),
            turn_number=memory.turn_count,
            metadata={"session_id": session_id},
        )

    # -- Overridable dispatch handlers ---------------------------------------

    async def _classify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> IntentResult:
        """Classify user intent. Override for custom classification."""
        # Build context text from assembled items for the classifier
        context_text = ""
        if context_items:
            parts = []
            for _mt, items in context_items.items():
                if items:
                    parts.append("\n".join(item.content for item in items))
            context_text = "\n".join(parts)

        return await classify_intent(message, memory, model=self._model, context_text=context_text)

    async def _dispatch(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Dispatch to the appropriate handler based on intent.

        Returns (response_text, tool_calls).
        Subclasses override to add tool_use, research, plan handlers.
        """
        if intent.intent == IntentType.RESPOND:
            return await self._handle_respond(message, memory, context_items)
        if intent.intent == IntentType.CLARIFY:
            return await self._handle_clarify(message, memory, context_items)
        if intent.intent == IntentType.TOOL_USE:
            return await self._handle_tool_use(message, memory, context_items)
        if intent.intent == IntentType.RESEARCH:
            return await self._handle_research(message, memory, context_items)
        if intent.intent == IntentType.PLAN:
            return await self._handle_plan(message, memory, context_items)

        # Fallback
        return await self._handle_respond(message, memory, context_items)

    async def _handle_respond(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle simple conversational response. Uses a Call."""
        response = await self._simple_respond(message, memory, context_items=context_items)
        return response, []

    async def _handle_clarify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle clarification request."""
        response = await self._simple_respond(
            message,
            memory,
            extra_instruction="The user's request is ambiguous. Ask a clarifying question.",
        )
        return response, []

    async def _handle_tool_use(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle tool-using request. Override in ChatAgent to use ReAct."""
        # BaseAgent falls back to simple response (no tools configured)
        return await self._handle_respond(message, memory, context_items)

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle research/document Q&A. Override to use RAG."""
        return await self._handle_respond(message, memory, context_items)

    async def _handle_plan(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle multi-step plan. Override in PlanExecuteAgent."""
        return await self._handle_respond(message, memory, context_items)

    # -- Internal helpers ----------------------------------------------------

    async def _simple_respond(
        self,
        message: str,
        memory: SessionMemory,
        *,
        extra_instruction: str = "",
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> str:
        """Generate a simple text response via Call."""
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core import Call, InputField, OutputField, Signature

        class Respond(Signature):
            """You are a helpful assistant. Respond to the user's message."""

            message: str = InputField(description="The user's message")
            conversation_history: str = InputField(description="Recent conversation for context")
            response: str = OutputField(description="Your response to the user")

        # Use pre-assembled context if available, otherwise build from memory
        if context_items:
            parts = []
            for mt, items in context_items.items():
                if items:
                    parts.append(
                        f"=== {mt.value.upper()} ===\n" + "\n".join(i.content for i in items)
                    )
            history = "\n\n".join(parts) if parts else "(new conversation)"
        else:
            recent = memory.get_recent(MemoryType.MESSAGES, 10)
            history = "\n".join(item.content for item in recent) if recent else "(new conversation)"

        instructions = "You are a helpful assistant."
        if extra_instruction:
            instructions = f"{instructions} {extra_instruction}"

        call = Call(Respond, model=self._model, instructions=instructions)
        result = await call(message=message, conversation_history=history)
        return str(result.response)
