"""AgentToolSpec — declarative dependency contract for tools used by KaosAgent.

A tool author declares: "this tool needs these memory sections, this
LLM role, this addition to the system prompt." The agent's bridge
reads the spec at wiring time and threads the right slice of memory,
the right model, and the right prompt material — without the tool
having to interrogate ``KaosContext._config`` itself.

This is the kelvin-agent ``MemoryInstructAction.memory_section_names`` /
``engine_role`` / ``base_instructions`` pattern (see
``kelvin/agent/actions/patterns/memory_instruct.py:28-60``) lifted into
a flat value-type contract. Chosen over a multi-inheritance
``LLMTool``/``MemoryTool``/``CorpusTool`` mixin tree because:

- KaosTool lives in kaos-core; we can't add a kaos-agents-aware mixin
  to it without a circular layer dependency.
- Multi-inheritance produces method-resolution surprises that pydantic-
  flavored class hierarchies handle worse than flat data.
- A registered spec keyed by tool name is debuggable from a single
  location (``default_tool_spec_registry``) instead of scattered across
  classes.

Tools are unaware of the spec. The spec is *registered for* a tool by
its name. This decouples the contract from kaos-core's KaosTool ABC
and lets out-of-tree tool authors register specs without changing
their tool implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_agents.types.memory import MemoryType
    from kaos_agents.types.providers import ModelRole


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    """Declarative dependencies a tool wants the bridge to satisfy.

    Frozen so a registered spec is safe to share across runs without
    accidental mutation. All fields default to "no opinion" — the
    ``AgentToolSpec()`` empty instance is the implicit default for
    every tool that hasn't been explicitly registered.

    Attributes:
        memory_sections: Sections of :class:`SessionMemory` the tool's
            execution should be parameterized by. The bridge fetches
            these and threads them into the tool's input or surrounding
            prompt material. Empty tuple = no memory dependency.

            Example: ``(MemoryType.FINDINGS, MemoryType.DOCUMENTS)`` —
            the tool sees verified findings + the corpus when it runs.

        engine_role: When the tool itself drives an LLM call, which
            :class:`ModelRole` the bridge resolves. Threads through
            :class:`ProviderConfig` so cost-aware model selection
            applies. ``None`` = the tool doesn't drive an LLM (or
            uses the agent's default model).

            Example: ``ModelRole.EXTRACT`` — the tool is an
            extraction primitive; resolve to the EXTRACT-role model.

        extra_instructions: Text the bridge appends to the agent's
            ReAct system prompt when this tool is in scope. Useful
            for declaring guardrails ("never delete a file without
            confirming with the user") or hints ("prefer the JSON
            output format for this tool's results"). Empty string =
            no addition.

        field_set: Optional :class:`FieldSet`-like projection (TODO:
            v2 — depends on the FieldSet port from Track 6). Reserved
            for now; ignored by the chunk-A1 bridge.
    """

    memory_sections: tuple[MemoryType, ...] = ()
    engine_role: ModelRole | None = None
    extra_instructions: str = ""
    field_set: str | None = None  # placeholder; Track 6 introduces typed FieldSet

    @property
    def is_empty(self) -> bool:
        """``True`` when the spec carries no opinion for any field.

        The bridge can short-circuit memory/prompt-assembly when this
        is true — most built-in tools (search / fetch / extract single
        document) need no agent-side context threading.
        """
        return (
            not self.memory_sections
            and self.engine_role is None
            and not self.extra_instructions
            and self.field_set is None
        )

    @classmethod
    def empty(cls) -> AgentToolSpec:
        """The sentinel "no opinion" spec — useful as a default."""
        return _EMPTY_SPEC


# Module-level singleton so ``AgentToolSpec.empty()`` is identity-stable.
_EMPTY_SPEC = AgentToolSpec()


@dataclass(frozen=True, slots=True)
class AgentToolRegistration:
    """A registered (tool_name, spec) pair.

    Used by :class:`AgentToolSpecRegistry` for typed lookups; tool
    authors don't construct these directly — they call
    :meth:`AgentToolSpecRegistry.register`.
    """

    tool_name: str
    spec: AgentToolSpec
    tags: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "AgentToolRegistration",
    "AgentToolSpec",
]
