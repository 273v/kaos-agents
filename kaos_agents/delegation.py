"""Delegation — agent-to-agent composition.

Three delegation modes supported:

1. **Agent-as-tool** (``agent_as_tool``): wrap an Agent as a callable tool
   that the parent agent can invoke via ReAct. The sub-agent runs in an
   isolated session, returns a response string, and the parent continues.

2. **Sub-agents** (``Agent.delegated_agents``): declarative version of
   agent-as-tool. Attach named sub-agents to an Agent config; the Runner
   auto-wraps them as tools before dispatch.

3. **Handoff** (``Agent.handoffs``): transfer full control to another agent.
   The current agent's LLM emits a handoff signal, and the Runner switches
   to the target agent mid-stream. Handoff shares the session_id with the
   parent (same memory).

All three modes emit events for observability:
- ``SubagentStart`` / ``SubagentComplete`` for tool-style delegation
- ``HandoffStart`` for control transfer

Depth tracking via ``contextvars.ContextVar`` prevents infinite recursion.
Each delegation call increments the depth; exceeding ``max_depth`` raises
``DelegationDepthExceeded``.

Design grounded in competitive research:
- OpenAI Agents SDK: handoffs as first-class agent transfers + agent-as-tool
- Anthropic Claude SDK: subagents for context isolation and parallel work
- Pydantic-AI: synchronous delegation within tools (no cross-agent streaming)

We provide all three modes because they have different semantics and
different use cases. Handoff is for routing ("send legal questions to
legal_agent"). Subagent is for isolated sub-tasks ("ask writer to draft
a memo from these findings"). Agent-as-tool is the explicit tool-wrapper
form of subagent.

Usage::

    from kaos_agents import Agent, agent_as_tool, Runner

    writer = Agent(
        instructions="Draft reports from research findings.",
        model="anthropic:claude-sonnet-4-6",
    )
    draft_tool = agent_as_tool(
        writer,
        name="draft_report",
        description="Draft a report from research findings.",
    )

    coordinator = Agent(
        instructions="Research and report drafting coordinator.",
        tools=("kaos-source-*",),
        delegated_agents=(draft_tool,),
    )

    runner = Runner(coordinator, runtime=runtime)
    async for event in runner.run("Draft a report on recent policy changes", "s1"):
        ...
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.errors import KaosAgentError

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.config import Agent

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Depth tracking
# ---------------------------------------------------------------------------

# ContextVar for tracking delegation depth across nested async calls.
# Each sub-agent call increments; each return decrements.
_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "delegation_depth",
    default=0,
)


class DelegationDepthExceeded(KaosAgentError):
    """Raised when delegation exceeds the configured max_depth.

    Prevents infinite recursion in agent-to-agent calls.
    """


def current_delegation_depth() -> int:
    """Get the current delegation depth in this async context.

    Useful for hooks and telemetry. 0 = root agent (no delegation).
    """
    return _delegation_depth.get()


# ---------------------------------------------------------------------------
# DelegatedAgent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DelegatedAgent:
    """Declaration of a sub-agent that can be called as a tool.

    Wraps an Agent config with a tool-style interface. Pass to
    ``Agent(delegated_agents=(...))`` or use directly via
    ``DelegatedAgent.call()``.

    Args:
        agent: The sub-agent configuration to execute.
        name: Tool name (must be a valid Python identifier).
        description: Tool description shown to the parent's LLM.
        max_depth: Maximum delegation depth before raising
            DelegationDepthExceeded. Default 3.
    """

    agent: Agent
    name: str
    description: str
    max_depth: int = 3

    async def call(
        self,
        task: str,
        *,
        parent_session_id: str = "",
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        vfs: VirtualFileSystem | None = None,
    ) -> str:
        """Execute the sub-agent and return its response text.

        Creates a new Runner with the sub-agent config and runs a single turn.
        The sub-session is derived from ``parent_session_id`` to avoid collisions.

        Args:
            task: The task to send to the sub-agent.
            parent_session_id: Parent session (for deriving sub-session id).
            runtime: KaosRuntime for the sub-agent's tool execution.
            context: KaosContext for the sub-agent's request metadata.
            vfs: VirtualFileSystem for the sub-agent's memory.

        Returns:
            The sub-agent's response text.

        Raises:
            DelegationDepthExceeded: If current depth >= max_depth.
        """
        current = _delegation_depth.get()
        if current >= self.max_depth:
            raise DelegationDepthExceeded(
                f"Delegation depth {current} reached max_depth {self.max_depth} "
                f"for sub-agent '{self.name}'. "
                f"Increase max_depth on the DelegatedAgent, or restructure the "
                f"agent graph to reduce recursion. "
                f"Alternative: use Agent.handoffs for flat routing instead of nested calls."
            )

        from kaos_agents.runner import Runner

        sub_session_id = (
            f"{parent_session_id}/subagent-{self.name}"
            if parent_session_id
            else f"subagent-{self.name}"
        )
        sub_runner = Runner(
            self.agent,
            runtime=runtime,
            context=context,
            vfs=vfs,
        )

        token = _delegation_depth.set(current + 1)
        try:
            logger.debug(
                "delegation: invoking sub-agent '%s' at depth %d (session=%s)",
                self.name,
                current + 1,
                sub_session_id,
            )
            response = await sub_runner.turn(task, sub_session_id)
            logger.debug(
                "delegation: sub-agent '%s' returned %d chars",
                self.name,
                len(response.text),
            )
            return response.text
        finally:
            _delegation_depth.reset(token)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agent_as_tool(
    agent: Agent,
    *,
    name: str,
    description: str,
    max_depth: int = 3,
) -> DelegatedAgent:
    """Wrap an Agent as a sub-agent that can be called as a tool.

    The returned ``DelegatedAgent`` can be:
    - Passed to ``Agent(delegated_agents=(...))`` for declarative use.
    - Called directly via ``DelegatedAgent.call(task)`` for ad-hoc use.

    Args:
        agent: The sub-agent configuration.
        name: Tool name (valid Python identifier, lowercase with underscores).
        description: What the sub-agent does; shown to the parent's LLM.
        max_depth: Maximum nested delegation depth. Default 3.

    Returns:
        A frozen DelegatedAgent declaration.

    Example::

        writer = Agent(instructions="Draft reports from findings.")
        draft_tool = agent_as_tool(
            writer,
            name="draft_report",
            description="Draft a report from research findings.",
        )
    """
    return DelegatedAgent(
        agent=agent,
        name=name,
        description=description,
        max_depth=max_depth,
    )
