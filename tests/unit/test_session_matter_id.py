"""Unit tests for SessionMemory.matter_id (plan §Issue 2).

Per-matter tenancy: every kaos-agents session carries an optional
``matter_id`` (firm-side ethical-wall identifier) that downstream
consumers can read to enforce cross-matter isolation. The field is
optional with ``None`` as backward-compat default; round-trips
through ``to_dict`` / ``from_dict`` so VFS-persisted sessions retain
their matter scope across process restarts.

Tests:

* default = None when constructor is called without ``matter_id``
* explicit string is preserved
* persistence round-trip restores the matter_id
* pre-0.1.8 snapshots (no ``matter_id`` key) hydrate as ``None``
"""

from __future__ import annotations

import pytest

from kaos_agents.memory.session import SessionMemory


@pytest.mark.unit
def test_matter_id_defaults_to_none() -> None:
    mem = SessionMemory("session-unscoped")
    assert mem.matter_id is None


@pytest.mark.unit
def test_matter_id_preserves_explicit_value() -> None:
    mem = SessionMemory("session-scoped", matter_id="ABC-2026-0042")
    assert mem.matter_id == "ABC-2026-0042"


@pytest.mark.unit
def test_matter_id_round_trips_through_persistence() -> None:
    """A session created with a matter_id must survive a to_dict /
    from_dict cycle without losing scope. The ethical-wall identifier
    is load-bearing: dropping it on persistence would silently
    declassify a matter-scoped session into the unscoped pool.
    """
    mem = SessionMemory("session-roundtrip", matter_id="XYZ-2026-0099")
    data = mem.to_dict()
    assert data["matter_id"] == "XYZ-2026-0099"

    restored = SessionMemory.from_dict(data)
    assert restored.matter_id == "XYZ-2026-0099"
    assert restored.session_id == "session-roundtrip"


@pytest.mark.unit
def test_legacy_snapshot_without_matter_id_hydrates_as_none() -> None:
    """Pre-0.1.8 persisted sessions don't carry the ``matter_id`` key.
    They must rehydrate as ``None`` (unscoped) — defaulting them to
    any other value would silently retroactively scope them into a
    matter the user never opted into.
    """
    legacy_snapshot = {
        "session_id": "session-legacy",
        "turn_count": 3,
        "chars_per_token": 4.0,
        "corpus_ever_attached": False,
        # NOTE: deliberately no ``matter_id`` key — simulates an older
        # snapshot from before this field shipped.
        "sections": {},
    }
    restored = SessionMemory.from_dict(legacy_snapshot)
    assert restored.matter_id is None
    assert restored.session_id == "session-legacy"
    assert restored.turn_count == 3


@pytest.mark.unit
def test_matter_id_is_independent_of_session_id() -> None:
    """Many sessions can belong to the same matter. The two ids are
    semantically distinct and must not be conflated.
    """
    mem_a = SessionMemory("01ABC", matter_id="MATTER-1")
    mem_b = SessionMemory("01DEF", matter_id="MATTER-1")
    mem_c = SessionMemory("01XYZ", matter_id="MATTER-2")
    assert mem_a.matter_id == mem_b.matter_id
    assert mem_a.session_id != mem_b.session_id
    assert mem_c.matter_id != mem_a.matter_id


# ── SessionStore.load_or_create matter_id propagation ─────────────


@pytest.mark.asyncio
async def test_load_or_create_propagates_matter_id_to_new_session() -> None:
    """``SessionStore.load_or_create(sid, matter_id=...)`` must construct
    the fresh ``SessionMemory`` with that matter scope. This is the
    integration point the SPA backend / chat router will call when a
    matter-scoped session is opened.
    """
    from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

    from kaos_agents.memory.store import SessionStore

    vfs = VirtualFileSystem(
        config=VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    )
    store = SessionStore(vfs)
    mem = await store.load_or_create(
        "01STOREMATTER",
        load_recipes=False,
        matter_id="ABC-2026-0042",
    )
    assert mem.matter_id == "ABC-2026-0042"


@pytest.mark.asyncio
async def test_load_or_create_omits_matter_id_when_unscoped() -> None:
    """No ``matter_id`` arg → unscoped (None). Backward-compat for
    every existing caller that doesn't know about the field yet."""
    from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

    from kaos_agents.memory.store import SessionStore

    vfs = VirtualFileSystem(
        config=VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    )
    store = SessionStore(vfs)
    mem = await store.load_or_create("01STORENOSCOPE", load_recipes=False)
    assert mem.matter_id is None


@pytest.mark.asyncio
async def test_load_or_create_preserves_existing_matter_id_on_reload() -> None:
    """An existing session's persisted matter_id is the source of truth.
    Calling ``load_or_create(sid, matter_id="OVERRIDE")`` on an already-
    existing session loads the persisted matter, NOT the new one. This
    prevents a stale arg from silently re-scoping a live session.
    """
    from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

    from kaos_agents.memory.store import SessionStore

    vfs = VirtualFileSystem(
        config=VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    )
    store = SessionStore(vfs)
    # Create + save a session scoped to MATTER-1.
    sid = "01STOREPRESERVE"
    mem = await store.load_or_create(sid, load_recipes=False, matter_id="MATTER-1")
    assert mem.matter_id == "MATTER-1"
    await store.save(mem)

    # Reload with a different matter_id arg — the persisted value wins.
    reloaded = await store.load_or_create(sid, load_recipes=False, matter_id="MATTER-OVERRIDE")
    assert reloaded.matter_id == "MATTER-1"
