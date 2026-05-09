"""AgentEnvelope — content-addressed declarative form of an Agent.

Mirrors kaos_llm_core.programs.envelope.ProgramEnvelope at the agent
layer. Phase 0 captures the fields the existing Agent dataclass already
has — later phases (1-3) add per-subsystem envelopes (perceiver,
actor, planner, termination, escalation) as they ship.

agent_hash() mirrors program_hash() — same canonical-JSON discipline,
same sha256 prefix.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kaos_agents.config import AgentPattern
from kaos_agents.settings import DEFAULT_MODEL


class AgentEnvelope(BaseModel):
    """Content-addressed Agent declaration.

    Serialize to JSON for cross-process A2A delegation, recipe storage,
    or replay. Round-trips via Agent.from_envelope() / Agent.to_envelope()
    (added in Phase 0.D).

    Recursive: delegated_agents and handoffs are themselves
    AgentEnvelopes, so a HierarchicalPlanner's full agent tree
    serializes to one nested document.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        # Pydantic v2 — emit deterministic ordering for hashing
        populate_by_name=True,
    )

    pattern: AgentPattern = AgentPattern.CHAT
    instructions: str = "You are a helpful assistant."
    model: str = DEFAULT_MODEL
    tools: tuple[str, ...] = ()
    provider: dict[str, Any] | None = None  # ProviderConfig.model_dump() output
    settings_overrides: dict[str, Any] = Field(default_factory=dict)

    # Pattern-specific overrides (mirror Agent's top-level fields)
    max_tools: int | None = None
    max_react_iterations: int | None = None
    max_plan_steps: int | None = None
    rag_top_k: int | None = None
    rag_max_retries: int | None = None

    # Delegation
    delegated_agents: tuple[AgentEnvelope, ...] = ()
    handoffs: tuple[AgentEnvelope, ...] = ()
    max_delegation_depth: int = 3

    # Identity / metadata
    name: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    recipe_id: str | None = None

    def agent_hash(self) -> str:
        """Stable sha256 hash of the canonical JSON envelope.

        Mirrors kaos_llm_core.programs.envelope.hashing.program_hash:
        defaults filled, fields sorted, sha256 prefix.
        """
        return agent_hash(self)


# Late-import-friendly resolution for the recursive forward refs.
AgentEnvelope.model_rebuild()


def agent_hash(envelope: AgentEnvelope | dict[str, Any]) -> str:
    """Return a stable sha256 hash of an envelope.

    Both raw dicts and parsed AgentEnvelope instances are accepted.
    Both are normalized through AgentEnvelope first so the hash
    always covers the same canonical form.
    """
    parsed = (
        envelope if isinstance(envelope, AgentEnvelope) else AgentEnvelope.model_validate(envelope)
    )
    env_dict = parsed.model_dump(by_alias=True, exclude_none=False)
    canonical = json.dumps(env_dict, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = ["AgentEnvelope", "agent_hash"]
