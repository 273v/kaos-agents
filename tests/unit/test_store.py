"""Tests for SessionStore — VFS-backed persistence."""

from __future__ import annotations

import pytest
from kaos_core.vfs.core import VirtualFileSystem

from kaos_agents.errors import SessionCorruptedError, SessionNotFoundError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.types import MemoryType


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

    async def test_load_corrupted_raises(self, store: SessionStore, vfs: VirtualFileSystem):
        # Write garbage data
        await vfs.write("kaos-agents/sessions/bad/memory.json", b"not json{{{")
        with pytest.raises(SessionCorruptedError, match="cannot be deserialized"):
            await store.load("bad")

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
