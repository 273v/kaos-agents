"""Integration tests for SessionStore with real disk VFS.

These tests write to the filesystem via the DiskBackend, verifying
that persistence survives across separate VFS instances (simulating
process restarts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.errors import SessionCorruptedError, SessionNotFoundError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.types import MemoryType


def _disk_vfs(tmp_path: Path) -> VirtualFileSystem:
    """Create a disk-backed VFS rooted in a pytest tmp directory."""
    config = VFSConfig(
        default_backend=StorageBackend.DISK,
        disk_base_path=tmp_path,
        isolation_mode=IsolationMode.GLOBAL,
    )
    return VirtualFileSystem(config=config)


@pytest.mark.integration
class TestDiskPersistence:
    async def test_save_and_load_survives_new_vfs(self, tmp_path: Path):
        """Data written by one VFS instance is readable by another."""
        vfs1 = _disk_vfs(tmp_path)
        store1 = SessionStore(vfs1)

        mem = SessionMemory("disk-test")
        mem.add(MemoryType.MESSAGES, "Hello from disk")
        mem.add(MemoryType.FINDINGS, "Finding persisted to disk")
        mem.end_turn()
        await store1.save(mem)

        # Create a completely new VFS instance (simulates process restart)
        vfs2 = _disk_vfs(tmp_path)
        store2 = SessionStore(vfs2)

        loaded = await store2.load("disk-test")
        assert loaded.session_id == "disk-test"
        assert loaded.turn_count == 1
        assert loaded.section_item_count(MemoryType.MESSAGES) == 1
        assert loaded.section_item_count(MemoryType.FINDINGS) == 1

    async def test_multi_turn_disk_persistence(self, tmp_path: Path):
        """Five turns persisted and restored from disk."""
        session_id = "multi-turn-disk"

        for turn in range(5):
            vfs = _disk_vfs(tmp_path)
            store = SessionStore(vfs)
            mem = await store.load_or_create(session_id)
            mem.begin_turn()
            mem.add(MemoryType.MESSAGES, f"User: question {turn}")
            mem.add(MemoryType.MESSAGES, f"Agent: answer {turn}")
            mem.end_turn()
            await store.save(mem)

        # Final load from fresh VFS
        final_vfs = _disk_vfs(tmp_path)
        final_store = SessionStore(final_vfs)
        final = await final_store.load(session_id)
        assert final.turn_count == 5
        assert final.section_item_count(MemoryType.MESSAGES) > 0

    async def test_list_sessions_on_disk(self, tmp_path: Path):
        """Sessions saved to disk are discoverable via list_sessions."""
        vfs = _disk_vfs(tmp_path)
        store = SessionStore(vfs)

        for name in ["alpha", "beta", "gamma"]:
            mem = SessionMemory(name)
            await store.save(mem)

        sessions = await store.list_sessions()
        assert set(sessions) == {"alpha", "beta", "gamma"}

    async def test_delete_on_disk(self, tmp_path: Path):
        """Deleted sessions are removed from disk."""
        vfs = _disk_vfs(tmp_path)
        store = SessionStore(vfs)

        mem = SessionMemory("to-delete")
        await store.save(mem)
        assert await store.exists("to-delete")

        await store.delete("to-delete")
        assert not await store.exists("to-delete")

        # A new VFS also shouldn't find it
        vfs2 = _disk_vfs(tmp_path)
        store2 = SessionStore(vfs2)
        assert not await store2.exists("to-delete")

    async def test_corrupted_file_on_disk(self, tmp_path: Path):
        """Corrupted JSON on disk raises SessionCorruptedError."""
        vfs = _disk_vfs(tmp_path)
        store = SessionStore(vfs)

        # Write garbage
        await vfs.write("kaos-agents/sessions/bad/memory.json", b"not-json{{{")

        with pytest.raises(SessionCorruptedError, match="cannot be deserialized"):
            await store.load("bad")

    async def test_structurally_invalid_json_on_disk(self, tmp_path: Path):
        """Valid JSON but missing required fields raises SessionCorruptedError."""
        vfs = _disk_vfs(tmp_path)
        store = SessionStore(vfs)

        # Write valid JSON but missing session_id
        await vfs.write(
            "kaos-agents/sessions/bad-structure/memory.json",
            b'{"not_session_id": "oops"}',
        )

        with pytest.raises(SessionCorruptedError, match="invalid structure"):
            await store.load("bad-structure")

    async def test_nonexistent_session_raises(self, tmp_path: Path):
        """Loading a non-existent session raises SessionNotFoundError."""
        vfs = _disk_vfs(tmp_path)
        store = SessionStore(vfs)

        with pytest.raises(SessionNotFoundError):
            await store.load("nonexistent")
