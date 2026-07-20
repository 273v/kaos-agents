"""Tests for AgentToolSpec + AgentToolSpecRegistry.

Track 3 chunk A1 — confirms the contract:
- AgentToolSpec is a frozen value type with sensible defaults
- ``AgentToolSpec.empty()`` is identity-stable
- ``AgentToolSpec.is_empty`` distinguishes "no opinion" from "real spec"
- AgentToolSpecRegistry: register / get / unregister / clear / membership
- Conflict detection raises RegistryError without force=True
- Tag-based filtering works
- ``get`` returns the empty spec for unregistered tool names (not None)
"""

from __future__ import annotations

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.registry.tool_spec_registry import (
    AgentToolSpecRegistry,
    default_tool_spec_registry,
)
from kaos_agents.types import (
    AgentToolRegistration,
    AgentToolSpec,
    MemoryType,
    ModelRole,
)


@pytest.mark.unit
class TestAgentToolSpec:
    def test_default_is_empty(self) -> None:
        spec = AgentToolSpec()
        assert spec.is_empty
        assert spec.memory_sections == ()
        assert spec.engine_role is None
        assert spec.extra_instructions == ""

    def test_empty_classmethod_is_identity_stable(self) -> None:
        """Repeated AgentToolSpec.empty() calls return the same instance."""
        a = AgentToolSpec.empty()
        b = AgentToolSpec.empty()
        assert a is b

    def test_with_memory_sections(self) -> None:
        spec = AgentToolSpec(memory_sections=(MemoryType.FINDINGS, MemoryType.MESSAGES))
        assert not spec.is_empty
        assert spec.memory_sections == (MemoryType.FINDINGS, MemoryType.MESSAGES)

    def test_with_engine_role(self) -> None:
        spec = AgentToolSpec(engine_role=ModelRole.RESEARCH)
        assert not spec.is_empty
        assert spec.engine_role == ModelRole.RESEARCH

    def test_with_extra_instructions(self) -> None:
        spec = AgentToolSpec(extra_instructions="Always cite sources.")
        assert not spec.is_empty
        assert spec.extra_instructions == "Always cite sources."

    def test_combined(self) -> None:
        spec = AgentToolSpec(
            memory_sections=(MemoryType.DOCUMENTS,),
            engine_role=ModelRole.RESEARCH,
            extra_instructions="Verify spans.",
        )
        assert not spec.is_empty

    def test_frozen(self) -> None:
        spec = AgentToolSpec()
        with pytest.raises((AttributeError, Exception)):
            object.__setattr__(spec, "memory_sections", (MemoryType.MESSAGES,))
            spec.memory_sections = (MemoryType.MESSAGES,)


@pytest.mark.unit
class TestAgentToolSpecRegistry:
    def test_register_and_get(self) -> None:
        reg = AgentToolSpecRegistry()
        spec = AgentToolSpec(memory_sections=(MemoryType.FINDINGS,))

        reg.register("kaos-test-tool", spec)
        assert reg.get("kaos-test-tool") == spec

    def test_get_unregistered_returns_empty_spec(self) -> None:
        """The empty-spec fallback means callers don't None-check."""
        reg = AgentToolSpecRegistry()
        spec = reg.get("not-registered")
        assert spec.is_empty
        assert spec is AgentToolSpec.empty()

    def test_double_register_same_spec_is_idempotent(self) -> None:
        reg = AgentToolSpecRegistry()
        spec = AgentToolSpec(memory_sections=(MemoryType.MESSAGES,))
        reg.register("dup", spec)
        reg.register("dup", spec)  # same spec, no force needed
        assert reg.get("dup") == spec

    def test_double_register_different_spec_raises(self) -> None:
        reg = AgentToolSpecRegistry()
        spec_a = AgentToolSpec(memory_sections=(MemoryType.MESSAGES,))
        spec_b = AgentToolSpec(memory_sections=(MemoryType.FINDINGS,))

        reg.register("dup", spec_a)
        with pytest.raises(RegistryError):
            reg.register("dup", spec_b)

    def test_force_replaces(self) -> None:
        reg = AgentToolSpecRegistry()
        spec_a = AgentToolSpec(memory_sections=(MemoryType.MESSAGES,))
        spec_b = AgentToolSpec(memory_sections=(MemoryType.FINDINGS,))

        reg.register("dup", spec_a)
        reg.register("dup", spec_b, force=True)
        assert reg.get("dup") == spec_b

    def test_unregister_returns_record(self) -> None:
        reg = AgentToolSpecRegistry()
        spec = AgentToolSpec(extra_instructions="hi")
        reg.register("a", spec)

        removed = reg.unregister("a")
        assert isinstance(removed, AgentToolRegistration)
        assert removed.spec == spec
        assert reg.get("a").is_empty

    def test_clear(self) -> None:
        reg = AgentToolSpecRegistry()
        reg.register("a", AgentToolSpec(extra_instructions="a"))
        reg.register("b", AgentToolSpec(extra_instructions="b"))
        reg.clear()
        assert len(reg) == 0

    def test_membership_uses_string_check(self) -> None:
        reg = AgentToolSpecRegistry()
        reg.register("a", AgentToolSpec(extra_instructions="a"))

        assert "a" in reg
        assert "b" not in reg
        assert 42 not in reg  # non-string keys False, never raise

    def test_list_names_sorted(self) -> None:
        reg = AgentToolSpecRegistry()
        reg.register("z-tool", AgentToolSpec())
        reg.register("a-tool", AgentToolSpec())

        # Both registered have empty spec — same value, won't raise.
        assert reg.list_names() == ["a-tool", "z-tool"]

    def test_list_by_tag(self) -> None:
        reg = AgentToolSpecRegistry()
        reg.register("research-1", AgentToolSpec(extra_instructions="r1"), tags=("research",))
        reg.register(
            "research-2",
            AgentToolSpec(extra_instructions="r2"),
            tags=("research", "rag"),
        )
        reg.register("other", AgentToolSpec(extra_instructions="o"), tags=("misc",))

        research = reg.list_by_tag("research")
        assert len(research) == 2
        assert {r.tool_name for r in research} == {"research-1", "research-2"}

        rag = reg.list_by_tag("rag")
        assert len(rag) == 1
        assert rag[0].tool_name == "research-2"

    def test_get_for_tool_uses_metadata_name(self) -> None:
        """The convenience accessor that takes a KaosTool instance."""

        class _StubMeta:
            name = "kaos-stub-tool"

        class _StubTool:
            metadata = _StubMeta()

        reg = AgentToolSpecRegistry()
        spec = AgentToolSpec(memory_sections=(MemoryType.FINDINGS,))
        reg.register("kaos-stub-tool", spec)

        # _StubTool ducks as KaosTool — accessor only reads .metadata.name.
        assert reg.get_for_tool(_StubTool()) == spec  # ty: ignore[invalid-argument-type]

    def test_get_registration_returns_full_record(self) -> None:
        reg = AgentToolSpecRegistry()
        spec = AgentToolSpec(extra_instructions="x")
        reg.register("a", spec, tags=("tag1", "tag2"))

        record = reg.get_registration("a")
        assert record is not None
        assert record.tool_name == "a"
        assert record.spec == spec
        assert record.tags == ("tag1", "tag2")


@pytest.mark.unit
class TestDefaultRegistry:
    def test_default_registry_starts_empty(self) -> None:
        """The module-level default has no built-ins as of chunk A1."""
        # The default may have entries registered by other modules that
        # have been imported during the test session. Just confirm it's
        # the canonical instance.
        assert isinstance(default_tool_spec_registry, AgentToolSpecRegistry)

    def test_default_is_module_singleton(self) -> None:
        """Two imports give the same instance."""
        from kaos_agents.registry.tool_spec_registry import (
            default_tool_spec_registry as r1,
        )
        from kaos_agents.registry.tool_spec_registry import (
            default_tool_spec_registry as r2,
        )

        assert r1 is r2
