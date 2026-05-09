"""Phase 0.D — unit tests for Agent.clone_with / to_envelope / from_envelope.

These tests cover the round-trip between the runtime ``Agent`` dataclass
and the content-addressed ``AgentEnvelope``, including:

- ``clone_with`` semantics (identity, override, error on unknown field)
- ``to_envelope`` projection of scalar fields
- Round-trip identity for the fields the envelope carries
- Recursive round-trip through ``handoffs`` (nested Agents)
- Dict input acceptance for ``from_envelope``
- ``agent_hash`` determinism across two equal Agents
- ``agent_hash`` change after ``clone_with`` of an envelope-projected field

Phase 0.D limitations (documented in the methods' docstrings):
- ``settings`` (KaosAgentSettings) is dropped on round-trip
- ``refusal_policy`` is dropped on round-trip
"""

from __future__ import annotations

import pytest

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.core.envelope import AgentEnvelope

# ---------------------------------------------------------------------------
# clone_with
# ---------------------------------------------------------------------------


def test_clone_with_returns_new_instance_with_same_fields() -> None:
    """clone_with() with no overrides returns a structurally equal but
    distinct Agent (frozen dataclass equality is by value, but identity
    must differ)."""
    agent = Agent(instructions="hello", model="anthropic:claude-haiku-4-5")
    clone = agent.clone_with()
    assert clone == agent
    assert clone is not agent


def test_clone_with_swaps_model_preserves_other_fields() -> None:
    """clone_with(model=...) swaps the model field and preserves everything
    else."""
    agent = Agent(
        instructions="research assistant",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*",),
        pattern=AgentPattern.PLAN,
        max_plan_steps=12,
        name="planner",
    )
    clone = agent.clone_with(model="anthropic:claude-sonnet-4-6")
    assert clone.model == "anthropic:claude-sonnet-4-6"
    assert clone.instructions == agent.instructions
    assert clone.tools == agent.tools
    assert clone.pattern == agent.pattern
    assert clone.max_plan_steps == agent.max_plan_steps
    assert clone.name == agent.name
    # And the original is unchanged.
    assert agent.model == "anthropic:claude-haiku-4-5"


def test_clone_with_unknown_field_raises_typeerror() -> None:
    """dataclasses.replace raises TypeError on unknown field names — we
    rely on that and don't add extra validation."""
    agent = Agent()
    with pytest.raises(TypeError):
        agent.clone_with(unknown_field="x")


# ---------------------------------------------------------------------------
# to_envelope
# ---------------------------------------------------------------------------


def test_to_envelope_returns_envelope_with_matching_fields() -> None:
    """A bare Agent's envelope reflects its scalar fields."""
    agent = Agent(
        instructions="hi there",
        model="anthropic:claude-haiku-4-5",
        pattern=AgentPattern.PLAN,
    )
    env = agent.to_envelope()
    assert isinstance(env, AgentEnvelope)
    assert env.pattern == AgentPattern.PLAN
    assert env.instructions == "hi there"
    assert env.model == "anthropic:claude-haiku-4-5"


def test_to_envelope_projects_full_field_surface() -> None:
    """Every envelope-tracked field projects through correctly, including
    overrides and metadata."""
    agent = Agent(
        instructions="multi-field",
        model="anthropic:claude-haiku-4-5",
        pattern=AgentPattern.RESEARCH,
        tools=("kaos-source-*", "kaos-web-*"),
        max_tools=11,
        max_react_iterations=5,
        max_plan_steps=8,
        rag_top_k=4,
        rag_max_retries=2,
        max_delegation_depth=4,
        name="researcher",
        metadata=(("k", "v"), ("owner", "tests")),
    )
    env = agent.to_envelope()
    assert env.tools == ("kaos-source-*", "kaos-web-*")
    assert env.max_tools == 11
    assert env.max_react_iterations == 5
    assert env.max_plan_steps == 8
    assert env.rag_top_k == 4
    assert env.rag_max_retries == 2
    assert env.max_delegation_depth == 4
    assert env.name == "researcher"
    assert env.metadata == (("k", "v"), ("owner", "tests"))


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


def test_round_trip_preserves_envelope_projected_fields() -> None:
    """to_envelope → from_envelope round-trips the envelope-projected
    fields. settings and refusal_policy are dropped (Phase 0.D limitation)."""
    agent = Agent(
        instructions="Hi",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*",),
        pattern=AgentPattern.PLAN,
        max_tools=10,
        name="researcher",
    )
    env = agent.to_envelope()
    rebuilt = Agent.from_envelope(env)

    assert rebuilt.instructions == agent.instructions
    assert rebuilt.model == agent.model
    assert rebuilt.tools == agent.tools
    assert rebuilt.pattern == agent.pattern
    assert rebuilt.max_tools == agent.max_tools
    assert rebuilt.name == agent.name
    # Phase 0.D limitations explicitly verified:
    assert rebuilt.settings is None
    assert rebuilt.refusal_policy is None


def test_round_trip_with_provider_config() -> None:
    """ProviderConfig (frozen dataclass) survives projection through the
    envelope's dict[str, Any] provider field."""
    from kaos_agents.types.providers import ModelRole, ProviderConfig

    provider = ProviderConfig(
        default="anthropic:claude-haiku-4-5",
        role_models={ModelRole.PLAN: "anthropic:claude-sonnet-4-6"},
    )
    agent = Agent(instructions="provider-aware", provider=provider)
    rebuilt = Agent.from_envelope(agent.to_envelope())
    assert rebuilt.provider is not None
    assert rebuilt.provider.default == "anthropic:claude-haiku-4-5"
    assert rebuilt.provider.role_models == {ModelRole.PLAN: "anthropic:claude-sonnet-4-6"}


# ---------------------------------------------------------------------------
# Recursive handoffs round-trip
# ---------------------------------------------------------------------------


def test_recursive_handoffs_round_trip() -> None:
    """Nested handoffs (A → B → C) survive the envelope round-trip."""
    c = Agent(instructions="C is the leaf", name="c")
    b = Agent(instructions="B routes to C", name="b", handoffs=(c,))
    a = Agent(instructions="A routes to B", name="a", handoffs=(b,))

    env = a.to_envelope()
    rebuilt = Agent.from_envelope(env)

    assert rebuilt.name == "a"
    assert len(rebuilt.handoffs) == 1
    assert rebuilt.handoffs[0].name == "b"
    assert rebuilt.handoffs[0].instructions == b.instructions
    assert len(rebuilt.handoffs[0].handoffs) == 1
    assert rebuilt.handoffs[0].handoffs[0].instructions == c.instructions
    assert rebuilt.handoffs[0].handoffs[0].name == "c"


def test_delegated_agents_round_trip_as_plain_agents() -> None:
    """delegated_agents wrapped in DelegatedAgent flatten to plain Agents
    on the rebuilt side (lossy — caller responsible for re-wrapping)."""
    from kaos_agents.runtime.delegation import DelegatedAgent

    sub = Agent(instructions="sub-agent body", name="sub")
    parent = Agent(
        instructions="parent",
        name="parent",
        delegated_agents=(DelegatedAgent(agent=sub, name="sub_tool", description="d"),),
    )
    env = parent.to_envelope()
    rebuilt = Agent.from_envelope(env)

    assert len(rebuilt.delegated_agents) == 1
    rebuilt_sub = rebuilt.delegated_agents[0]
    # Lossy direction: it's an Agent now, not a DelegatedAgent.
    assert isinstance(rebuilt_sub, Agent)
    assert rebuilt_sub.instructions == "sub-agent body"
    assert rebuilt_sub.name == "sub"


def test_to_envelope_skips_delegated_entry_without_agent_attr() -> None:
    """Defensive: an entry that doesn't expose ``.agent`` (or itself
    isn't an Agent) is silently dropped — debug-logged in the
    implementation, not test-asserted."""

    class Bogus:
        """Looks vaguely like a wrapper but has no .agent and is not an Agent."""

    parent = Agent(
        instructions="parent",
        delegated_agents=(Bogus(),),
    )
    env = parent.to_envelope()
    assert env.delegated_agents == ()


# ---------------------------------------------------------------------------
# from_envelope dict input
# ---------------------------------------------------------------------------


def test_from_envelope_accepts_dict_input() -> None:
    """from_envelope normalises a raw dict through AgentEnvelope first
    so partial dicts pick up envelope defaults."""
    rebuilt = Agent.from_envelope({"pattern": "chat", "instructions": "x"})
    assert isinstance(rebuilt, Agent)
    assert rebuilt.pattern == AgentPattern.CHAT
    assert rebuilt.instructions == "x"
    # Defaults from the envelope (not from Agent) propagate cleanly.
    assert rebuilt.tools == ()
    assert rebuilt.max_delegation_depth == 3


# ---------------------------------------------------------------------------
# agent_hash interactions
# ---------------------------------------------------------------------------


def test_to_envelope_then_agent_hash_is_deterministic_across_equal_agents() -> None:
    """Two structurally equal Agents produce the same envelope hash."""
    a1 = Agent(
        instructions="canon",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*",),
        pattern=AgentPattern.PLAN,
        max_tools=10,
        name="researcher",
    )
    a2 = Agent(
        instructions="canon",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*",),
        pattern=AgentPattern.PLAN,
        max_tools=10,
        name="researcher",
    )
    h1 = a1.to_envelope().agent_hash()
    h2 = a2.to_envelope().agent_hash()
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_clone_with_changes_envelope_hash_when_field_changes() -> None:
    """clone_with() that mutates an envelope-projected field produces a
    different agent_hash than the original."""
    base = Agent(
        instructions="canon",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*",),
    )
    base_hash = base.to_envelope().agent_hash()

    swapped_model = base.clone_with(model="anthropic:claude-sonnet-4-6")
    assert swapped_model.to_envelope().agent_hash() != base_hash

    swapped_tools = base.clone_with(tools=("kaos-source-*", "kaos-web-*"))
    assert swapped_tools.to_envelope().agent_hash() != base_hash

    swapped_name = base.clone_with(name="alpha")
    assert swapped_name.to_envelope().agent_hash() != base_hash
