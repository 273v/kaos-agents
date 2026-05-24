"""Tests for SessionStore — VFS-backed persistence."""

from __future__ import annotations

import pytest
from kaos_core.vfs.core import VirtualFileSystem

from kaos_agents.errors import SessionCorruptedError, SessionNotFoundError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import (
    SessionStore,
    _safe_component,
    _session_graph_path,
    _session_path,
    _unsafe_component,
)
from kaos_agents.types.memory import MemoryType


@pytest.fixture
def vfs() -> VirtualFileSystem:
    """In-memory VFS for testing (no disk I/O)."""
    from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig

    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


@pytest.fixture
def store(vfs: VirtualFileSystem) -> SessionStore:
    return SessionStore(vfs)


class TestSessionStoreSaveLoad:
    async def test_save_and_load(self, store: SessionStore):
        mem = SessionMemory("test-1")
        mem.add(MemoryType.MESSAGES, "Hello!")
        mem.add(MemoryType.ACTIONS, "Called a tool")
        mem.end_turn()

        path = await store.save(mem)
        assert "test-1" in path

        loaded = await store.load("test-1")
        assert loaded.session_id == "test-1"
        assert loaded.turn_count == 1
        assert loaded.section_item_count(MemoryType.MESSAGES) == 1
        assert loaded.section_item_count(MemoryType.ACTIONS) == 1
        assert loaded.total_tokens == mem.total_tokens

    async def test_load_nonexistent_raises(self, store: SessionStore):
        with pytest.raises(SessionNotFoundError, match="No saved session"):
            await store.load("nonexistent")

    async def test_load_corrupted_json_raises(self, store: SessionStore, vfs: VirtualFileSystem):
        # Write invalid JSON bytes
        await vfs.write("kaos-agents/sessions/bad/memory.json", b"not json{{{")
        with pytest.raises(SessionCorruptedError, match="cannot be deserialized"):
            await store.load("bad")

    async def test_load_structurally_invalid_raises(
        self, store: SessionStore, vfs: VirtualFileSystem
    ):
        # Write valid JSON but missing required fields (session_id)
        await vfs.write(
            "kaos-agents/sessions/bad-struct/memory.json",
            b'{"not_session_id": "oops", "sections": {}}',
        )
        with pytest.raises(SessionCorruptedError, match="invalid structure"):
            await store.load("bad-struct")

    async def test_overwrite_on_save(self, store: SessionStore):
        mem = SessionMemory("test-2")
        mem.add(MemoryType.MESSAGES, "first")
        await store.save(mem)

        mem.add(MemoryType.MESSAGES, "second")
        await store.save(mem)

        loaded = await store.load("test-2")
        assert loaded.section_item_count(MemoryType.MESSAGES) == 2


class TestSessionStoreExists:
    async def test_exists_false_for_new(self, store: SessionStore):
        assert not await store.exists("new-session")

    async def test_exists_true_after_save(self, store: SessionStore):
        mem = SessionMemory("saved")
        await store.save(mem)
        assert await store.exists("saved")


class TestSessionStoreDelete:
    async def test_delete_existing(self, store: SessionStore):
        mem = SessionMemory("to-delete")
        await store.save(mem)
        assert await store.delete("to-delete")
        assert not await store.exists("to-delete")

    async def test_delete_nonexistent(self, store: SessionStore):
        assert not await store.delete("nonexistent")

    async def test_delete_sweeps_graph_ttl(self, store: SessionStore, vfs: VirtualFileSystem):
        """KC17-P1-1: delete must remove graph.ttl alongside memory.json.

        Pre-KC17, delete() called vfs.delete(memory.json) ONLY, leaving
        the per-session graph.ttl on disk. A subsequent ``exists()``
        would return True; a privacy/right-to-delete defect.
        """
        # Write both files directly to simulate a real session that
        # built a knowledge graph.
        await vfs.write(
            "kaos-agents/sessions/with-graph/memory.json",
            b'{"session_id":"with-graph","turn_count":0,"sections":{},"chars_per_token":4.0}',
        )
        await vfs.write(
            "kaos-agents/sessions/with-graph/graph.ttl",
            b"@prefix ex: <http://example/> . ex:s ex:p ex:o .",
        )

        deleted = await store.delete("with-graph")
        assert deleted is True

        # Both files should be gone.
        assert not await vfs.exists("kaos-agents/sessions/with-graph/memory.json")
        assert not await vfs.exists("kaos-agents/sessions/with-graph/graph.ttl")
        assert not await store.exists("with-graph")

    async def test_delete_idempotent_when_empty(self, store: SessionStore):
        """Calling delete twice returns False the second time."""
        mem = SessionMemory("idempotent-delete")
        await store.save(mem)
        assert await store.delete("idempotent-delete") is True
        assert await store.delete("idempotent-delete") is False


class TestAtomicSave:
    """KC17-P1-3 — SessionStore.save survives SIGTERM between writes.

    Pre-KC17 ``save()`` called ``vfs.write(memory.json)`` followed by
    ``vfs.write(graph.ttl)`` as two non-atomic ops. A SIGTERM between
    them left a torn on-disk state that the next ``load()`` consumed.
    The fix routes both writes through ``_atomic_write`` (temp+fsync+
    os.replace on disk-backed VFS).
    """

    async def test_atomic_write_temp_path_left_clean_on_disk(self, tmp_path) -> None:
        """The .tmp file must be cleaned up after a successful rename.

        Uses the real disk VFS to exercise the temp+replace path.
        """
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        config = VFSConfig(default_backend=StorageBackend.DISK, disk_base_path=tmp_path)
        vfs = VirtualFileSystem(config=config)
        store = SessionStore(vfs)

        mem = SessionMemory("atomic-clean")
        mem.add(MemoryType.MESSAGES, "alpha")
        await store.save(mem)

        # The .tmp file must NOT survive a successful save. Since 0.1.17
        # ``SessionStore.save`` writes under ``context_id=session_id`` so
        # the disk path lands in the per-session VFS namespace
        # (``{session_id}/...``) rather than the shared ``default/`` scope
        # — see tests/unit/test_session_isolation.py for the rationale.
        memory_path = (
            tmp_path / "atomic-clean" / "kaos-agents" / "sessions" / "atomic-clean" / "memory.json"
        )
        tmp_file = memory_path.with_suffix(".json.tmp")
        assert memory_path.exists()
        assert not tmp_file.exists(), (
            f"KC17-P1-3 regression: .tmp file {tmp_file} survived a "
            "successful save — temp/rename did not run, or the rename "
            "did not consume the temp file"
        )

    async def test_sigterm_between_writes_leaves_old_state(self, tmp_path, monkeypatch) -> None:
        """KC17-P1-3 regression: simulate SIGTERM mid-save by raising on
        the SECOND atomic write. The disk must hold the OLD state for
        BOTH files — neither memory.json nor graph.ttl is partially-new.
        """
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.memory import store as store_module

        config = VFSConfig(default_backend=StorageBackend.DISK, disk_base_path=tmp_path)
        vfs = VirtualFileSystem(config=config)
        store = SessionStore(vfs)

        # First save — establish the baseline old state. Force a graph
        # write by giving the session a non-empty knowledge graph.
        try:
            from kaos_graph import Graph
        except ImportError:
            pytest.skip("kaos-graph not available")

        original = SessionMemory("atomic-sigterm")
        original.add(MemoryType.MESSAGES, "OLD message")
        original.graph = Graph(directed=True, multi=True, name="atomic-sigterm")
        original.graph.add_node("urn:old")
        original.graph.add_node("urn:other")
        original.graph.add_edge("urn:old", "urn:other", attributes={"k": "v"})
        await store.save(original)

        # Snapshot the OLD bytes on disk so we can prove they survived
        # the simulated crash. Per-session scope since 0.1.17.
        memory_path = (
            tmp_path
            / "atomic-sigterm"
            / "kaos-agents"
            / "sessions"
            / "atomic-sigterm"
            / "memory.json"
        )
        graph_path = memory_path.parent / "graph.ttl"
        old_memory_bytes = memory_path.read_bytes()
        old_graph_bytes = graph_path.read_bytes()
        assert b"OLD message" in old_memory_bytes

        # Now mutate the in-memory state to "NEW" and simulate the
        # SIGTERM: the first _atomic_write (memory.json) succeeds, the
        # SECOND (graph.ttl) raises.
        updated = await store.load("atomic-sigterm")
        updated.add(MemoryType.MESSAGES, "NEW message")
        updated.graph.add_node("urn:newer")
        updated.graph.add_edge("urn:old", "urn:newer", attributes={"k": "v2"})

        call_count = {"n": 0}
        original_atomic = store_module._atomic_write

        async def crash_on_second(vfs_arg, path, data, **kwargs):  # type: ignore[no-untyped-def]
            # ``**kwargs`` absorbs ``context_id`` (added in 0.1.17 for
            # per-session VFS scoping — see _atomic_write).
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First write: memory.json — let it succeed.
                await original_atomic(vfs_arg, path, data, **kwargs)
                return
            # Second write: graph.ttl — simulate SIGTERM.
            raise InterruptedError("simulated SIGTERM mid-save")

        monkeypatch.setattr(store_module, "_atomic_write", crash_on_second)

        with pytest.raises(InterruptedError):
            await store.save(updated)

        # On disk: memory.json is NEW, graph.ttl is OLD. But more
        # importantly, the load path must see a consistent pair — the
        # session must load WITHOUT a corrupted-deserialization error.
        # The MOST important invariant: graph.ttl is still the OLD
        # bytes because the second write never completed. The new
        # graph.tll.tmp must also NOT exist (we never started writing it).
        assert graph_path.read_bytes() == old_graph_bytes, (
            "KC17-P1-3 regression: graph.ttl was modified despite the "
            "second atomic_write raising — atomicity broken"
        )
        tmp_graph = graph_path.with_suffix(".ttl.tmp")
        assert not tmp_graph.exists()

        # The session must still load (no torn deserialization). Use
        # a fresh store on the same VFS.
        store2 = SessionStore(vfs)
        loaded = await store2.load("atomic-sigterm")
        assert loaded.session_id == "atomic-sigterm"

    async def test_save_uses_temp_path_on_disk(self, tmp_path, monkeypatch) -> None:
        """The actual disk write goes through a .tmp file before rename."""
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        config = VFSConfig(default_backend=StorageBackend.DISK, disk_base_path=tmp_path)
        vfs = VirtualFileSystem(config=config)
        store = SessionStore(vfs)

        observed: list[str] = []

        # Patch os.replace inside the store module to record the rename.
        from kaos_agents.memory import store as store_module

        original_replace = store_module.os.replace

        def spy_replace(src, dst):  # type: ignore[no-untyped-def]
            observed.append(f"replace {src} -> {dst}")
            return original_replace(src, dst)

        monkeypatch.setattr(store_module.os, "replace", spy_replace)

        mem = SessionMemory("temp-path-spy")
        await store.save(mem)

        assert any(".tmp" in entry for entry in observed), (
            f"KC17-P1-3: expected an os.replace from a .tmp path, got {observed}"
        )


class TestSessionStoreList:
    async def test_list_empty(self):
        """Use a fresh VFS to test empty listing."""
        from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig

        config = VFSConfig(
            default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL
        )
        fresh_store = SessionStore(VirtualFileSystem(config=config))
        sessions = await fresh_store.list_sessions()
        assert sessions == []

    async def test_list_multiple(self):
        """Use a fresh VFS to test multi-session listing."""
        from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig

        config = VFSConfig(
            default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL
        )
        fresh_store = SessionStore(VirtualFileSystem(config=config))
        for name in ["alpha", "beta", "gamma"]:
            mem = SessionMemory(name)
            await fresh_store.save(mem)

        sessions = await fresh_store.list_sessions()
        assert set(sessions) == {"alpha", "beta", "gamma"}


class TestSessionStoreLoadOrCreate:
    async def test_creates_new(self, store: SessionStore):
        mem = await store.load_or_create("new-session")
        assert mem.session_id == "new-session"
        # New sessions have recipes loaded into PLAN_EXAMPLES
        assert mem.section_item_count(MemoryType.PLAN_EXAMPLES) >= 5

    async def test_creates_new_without_recipes(self, store: SessionStore):
        mem = await store.load_or_create("no-recipes", load_recipes=False)
        assert mem.session_id == "no-recipes"
        assert mem.total_tokens == 0

    async def test_loads_existing(self, store: SessionStore):
        original = SessionMemory("existing")
        original.add(MemoryType.MESSAGES, "persisted message")
        await store.save(original)

        loaded = await store.load_or_create("existing")
        assert loaded.section_item_count(MemoryType.MESSAGES) == 1


class TestMultiTurnPersistence:
    async def test_five_turn_persist_restore(self, store: SessionStore):
        """Simulate 5 turns with persist/restore between each."""
        session_id = "multi-turn"

        for turn in range(5):
            mem = await store.load_or_create(session_id)
            mem.begin_turn()
            mem.add(MemoryType.MESSAGES, f"User: turn {turn}")
            mem.add(MemoryType.MESSAGES, f"Assistant: response {turn}")
            mem.add(MemoryType.FINDINGS, f"Finding from turn {turn}")
            mem.end_turn()
            await store.save(mem)

        # Final load
        final = await store.load(session_id)
        assert final.turn_count == 5
        assert final.section_item_count(MemoryType.FINDINGS) == 5
        # Messages may have been evicted if over budget, but some should remain
        assert final.section_item_count(MemoryType.MESSAGES) > 0


class TestSessionPathEncoding:
    """Cross-platform safety for the on-disk path component.

    Tenant-scoped session ids use ``<tenant>:<session>`` and ``:`` is a
    reserved character on the Windows filesystem (NTFS alternate data
    streams). The disk backend would write ``./.kaos-vfs/.../<tenant>:
    <session>/memory.json`` happily on POSIX and fail with
    ``NotADirectoryError`` on Windows. ``_safe_component`` percent-
    encodes the reserved set so the on-disk path is valid on every
    OS we target; ``_unsafe_component`` is its inverse so
    ``list_sessions`` returns the caller-supplied id verbatim.
    """

    def test_plain_id_is_unchanged(self) -> None:
        assert _safe_component("plain-id_1.2~3") == "plain-id_1.2~3"

    def test_colon_is_encoded(self) -> None:
        # The tenant-scoping separator must not survive into the path
        # because Windows rejects ``:`` in filename components.
        assert _safe_component("tenant:session") == "tenant%3Asession"
        assert ":" not in _safe_component("tenant:session")

    def test_windows_reserved_chars_are_encoded(self) -> None:
        # Per https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file
        for ch in '<>:"|?*\\':
            encoded = _safe_component(f"a{ch}b")
            assert ch not in encoded, f"{ch!r} must be percent-encoded in {encoded!r}"

    def test_roundtrip(self) -> None:
        for original in (
            "plain",
            "tenant:session",
            "with spaces",
            "with/slash",
            "weird<>chars",
            "1234567890abcdef:my-session",
        ):
            assert _unsafe_component(_safe_component(original)) == original

    def test_session_path_uses_safe_component(self) -> None:
        path = _session_path("575525f04c94:my-session")
        assert ":" not in path
        assert path == "kaos-agents/sessions/575525f04c94%3Amy-session/memory.json"

    def test_session_graph_path_uses_safe_component(self) -> None:
        path = _session_graph_path("575525f04c94:my-session")
        assert ":" not in path
        assert path == "kaos-agents/sessions/575525f04c94%3Amy-session/graph.ttl"

    async def test_save_load_with_tenant_scoped_id(self, store: SessionStore) -> None:
        # The exact ``<hex>:<name>`` shape the API tier produces.
        # On Windows this used to fail with NotADirectoryError; the
        # safe-component encoding fixes it without changing the
        # caller-supplied id.
        scoped_id = "575525f04c94:my-session"
        mem = SessionMemory(scoped_id)
        mem.add(MemoryType.MESSAGES, "tenant-scoped hello")
        mem.end_turn()
        await store.save(mem)

        loaded = await store.load(scoped_id)
        assert loaded.session_id == scoped_id
        assert loaded.section_item_count(MemoryType.MESSAGES) == 1

    async def test_list_sessions_returns_decoded_ids(self, store: SessionStore) -> None:
        # The directory on disk is encoded but list_sessions must
        # return the original id, not the percent-escaped form.
        scoped_id = "575525f04c94:list-me"
        mem = SessionMemory(scoped_id)
        mem.add(MemoryType.MESSAGES, "hello")
        await store.save(mem)

        ids = await store.list_sessions()
        assert scoped_id in ids
        assert "575525f04c94%3Alist-me" not in ids
