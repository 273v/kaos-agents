"""Phase 0.A — unit tests for kaos_agents.core.envelope."""

from __future__ import annotations

from kaos_agents.config import AgentPattern
from kaos_agents.core.envelope import AgentEnvelope, agent_hash


def test_default_envelope_construction() -> None:
    env = AgentEnvelope()
    assert env.pattern == AgentPattern.CHAT
    assert env.instructions == "You are a helpful assistant."
    assert env.tools == ()
    assert env.delegated_agents == ()
    assert env.handoffs == ()
    assert env.max_delegation_depth == 3
    assert env.metadata == ()


def test_round_trip_via_model_dump_json() -> None:
    env = AgentEnvelope(
        pattern=AgentPattern.PLAN,
        instructions="You are a planner.",
        model="anthropic:claude-haiku-4-5",
        tools=("kaos-source-*", "kaos-web-*"),
        max_plan_steps=50,
        name="planner-1",
        metadata=(("owner", "tests"), ("version", 2)),
    )
    payload = env.model_dump_json()
    parsed = AgentEnvelope.model_validate_json(payload)
    assert parsed == env
    # Hash is stable across round-trip.
    assert env.agent_hash() == parsed.agent_hash()


def test_agent_hash_deterministic_for_equal_envelopes() -> None:
    env1 = AgentEnvelope(instructions="hello", tools=("a", "b"))
    env2 = AgentEnvelope(instructions="hello", tools=("a", "b"))
    assert env1 == env2
    assert env1.agent_hash() == env2.agent_hash()
    assert env1.agent_hash().startswith("sha256:")


def test_agent_hash_differs_when_any_field_differs() -> None:
    base = AgentEnvelope(instructions="base", tools=("a",))
    h_base = base.agent_hash()

    # Change instructions
    h_instr = AgentEnvelope(instructions="other", tools=("a",)).agent_hash()
    assert h_instr != h_base

    # Change tools
    h_tools = AgentEnvelope(instructions="base", tools=("a", "b")).agent_hash()
    assert h_tools != h_base

    # Change pattern
    h_pat = AgentEnvelope(
        instructions="base", tools=("a",), pattern=AgentPattern.RESEARCH
    ).agent_hash()
    assert h_pat != h_base

    # Change name (an Optional field — None vs str)
    h_name = AgentEnvelope(instructions="base", tools=("a",), name="foo").agent_hash()
    assert h_name != h_base

    # Change recipe_id
    h_recipe = AgentEnvelope(instructions="base", tools=("a",), recipe_id="my-recipe").agent_hash()
    assert h_recipe != h_base


def test_recursive_delegated_agents_round_trip() -> None:
    inner = AgentEnvelope(instructions="inner", name="child", tools=("kaos-tool-x",))
    outer = AgentEnvelope(
        instructions="outer",
        name="parent",
        delegated_agents=(inner,),
        handoffs=(inner,),
    )
    payload = outer.model_dump_json()
    parsed = AgentEnvelope.model_validate_json(payload)
    assert parsed == outer
    assert parsed.delegated_agents[0] == inner
    assert parsed.handoffs[0] == inner
    assert parsed.agent_hash() == outer.agent_hash()


def test_agent_hash_accepts_dict_form() -> None:
    """agent_hash() accepts both AgentEnvelope and dict; dict path
    must produce the same hash as the parsed AgentEnvelope path."""
    env = AgentEnvelope(
        instructions="hashable",
        tools=("a", "b"),
        max_plan_steps=10,
        metadata=(("k", "v"),),
    )
    raw = env.model_dump(by_alias=True, exclude_none=False)
    h_from_obj = agent_hash(env)
    h_from_dict = agent_hash(raw)
    assert h_from_obj == h_from_dict
    assert h_from_obj.startswith("sha256:")
    # 6 chars for prefix "sha256:" + 64 hex chars
    assert len(h_from_obj) == len("sha256:") + 64


def test_agent_hash_partial_dict_normalizes() -> None:
    """A partial dict (only a couple of fields specified) is normalized
    through AgentEnvelope first, so its hash equals the hash of the
    fully-defaulted envelope built from the same explicit fields."""
    h_partial = agent_hash({"instructions": "x"})
    h_parsed = AgentEnvelope(instructions="x").agent_hash()
    assert h_partial == h_parsed
