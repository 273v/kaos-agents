"""Unit tests for PA5 — AgentChatTool VFS artifact auto-hydration.

Verifies the contract documented in
``kaos_agents/runtime/artifact_hydration.py``:

- A ``kaos://artifacts/<id>`` URI in a user message triggers a VFS read
  and injects the body into ``SessionMemory.DOCUMENTS``.
- A JSON manifest blob in a user message triggers the same.
- An already-hydrated artifact does NOT re-fetch on the next turn.
- An artifact with ``size >= SUMMARY_THRESHOLD`` injects a handle-only
  reference (no body) into memory.

The tests exercise the hydration helper directly with a real
``KaosRuntime`` + in-memory VFS + ``ArtifactStore`` so the behaviour is
verified end-to-end without LLM calls.
"""

from __future__ import annotations

import json

import pytest
from kaos_core.artifacts.models import INLINE_THRESHOLD, SUMMARY_THRESHOLD
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.runtime.artifact_hydration import (
    HydratedArtifact,
    already_hydrated_artifact_ids,
    find_artifact_references,
    hydrate_artifacts_from_message,
)
from kaos_agents.types.memory import MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> KaosRuntime:
    """Build an isolated runtime with an in-memory VFS for each test."""
    vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
    return KaosRuntime(vfs=vfs)


async def _write_artifact(
    runtime: KaosRuntime,
    *,
    name: str,
    body: bytes,
    session_id: str = "test-session",
    mime_type: str = "text/plain",
) -> str:
    """Write ``body`` to the VFS and register an ArtifactManifest.

    Returns the artifact_id.
    """
    vfs = runtime.vfs
    path = f"/artifacts/{name}"
    await vfs.write(path, body, context_id=session_id)
    manifest = await runtime.artifacts.create_from_path(
        path,
        context_id=session_id,
        session_id=session_id,
        name=name,
        mime_type=mime_type,
    )
    return manifest.artifact_id


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindArtifactReferences:
    """Verify the URI / shorthand / manifest-blob scanner."""

    def test_finds_kaos_artifact_uri(self):
        msg = "Please summarize kaos://artifacts/abc12345-0000-0000-0000-000000000001"
        assert find_artifact_references(msg) == ("abc12345-0000-0000-0000-000000000001",)

    def test_finds_kaos_artifact_body_uri(self):
        msg = "Body is at kaos://artifacts/abc12345-0000-0000-0000-000000000001/body"
        assert find_artifact_references(msg) == ("abc12345-0000-0000-0000-000000000001",)

    def test_finds_artifact_shorthand(self):
        msg = "See artifact://abc12345-0000-0000-0000-000000000001 for the report."
        assert find_artifact_references(msg) == ("abc12345-0000-0000-0000-000000000001",)

    def test_finds_manifest_json_blob(self):
        manifest_blob = json.dumps(
            {
                "artifact_id": "abc12345-0000-0000-0000-000000000001",
                "uri": "kaos://artifacts/abc12345-0000-0000-0000-000000000001",
                "name": "report.pdf",
                "size": 1234,
            }
        )
        msg = f"Got the manifest: {manifest_blob}"
        assert find_artifact_references(msg) == ("abc12345-0000-0000-0000-000000000001",)

    def test_deduplicates_multi_shape_references(self):
        """The same artifact ID referenced via 2+ shapes appears once."""
        msg = (
            "First: kaos://artifacts/abc12345-0000-0000-0000-000000000001/body. "
            "Also: artifact://abc12345-0000-0000-0000-000000000001."
        )
        assert find_artifact_references(msg) == ("abc12345-0000-0000-0000-000000000001",)

    def test_preserves_first_seen_order_across_multiple_artifacts(self):
        msg = (
            "kaos://artifacts/aaaa1111-0000-0000-0000-000000000001 "
            "kaos://artifacts/bbbb2222-0000-0000-0000-000000000002"
        )
        refs = find_artifact_references(msg)
        assert refs == (
            "aaaa1111-0000-0000-0000-000000000001",
            "bbbb2222-0000-0000-0000-000000000002",
        )

    def test_empty_message_returns_empty(self):
        assert find_artifact_references("") == ()
        assert find_artifact_references("just a normal question, no URIs") == ()

    def test_manifest_blob_without_kaos_uri_is_skipped(self):
        """A JSON object with artifact_id but no kaos:// URI doesn't fire.

        Prevents false positives when downstream tools emit JSON with an
        ``artifact_id`` key that points at a non-VFS resource.
        """
        blob = json.dumps(
            {
                "artifact_id": "abc12345-0000-0000-0000-000000000001",
                "uri": "https://example.com/some-other-thing",
            }
        )
        # Should NOT pick up via the manifest blob path. But the bare
        # hex ID isn't in any URL we recognise either, so result is
        # empty.
        assert find_artifact_references(blob) == ()


# ---------------------------------------------------------------------------
# already_hydrated_artifact_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAlreadyHydratedTracking:
    def test_empty_memory_returns_empty_set(self):
        memory = SessionMemory("test-session")
        assert already_hydrated_artifact_ids(memory) == set()

    def test_finds_hydrated_marker_in_documents(self):
        memory = SessionMemory("test-session")
        memory.add(
            MemoryType.DOCUMENTS,
            "fake content",
            metadata={"hydrated_artifact_id": "aaa-111"},
        )
        assert already_hydrated_artifact_ids(memory) == {"aaa-111"}

    def test_ignores_non_hydrated_documents(self):
        """Documents added by other code (not auto-hydrate) are ignored."""
        memory = SessionMemory("test-session")
        memory.add(MemoryType.DOCUMENTS, "manually injected content")
        memory.add(
            MemoryType.DOCUMENTS,
            "hydrated content",
            metadata={"hydrated_artifact_id": "aaa-111"},
        )
        assert already_hydrated_artifact_ids(memory) == {"aaa-111"}


# ---------------------------------------------------------------------------
# hydrate_artifacts_from_message — end-to-end with real VFS + ArtifactStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHydrateArtifactsFromMessage:
    async def test_no_runtime_is_noop(self):
        memory = SessionMemory("test-session")
        msg = "Please look at kaos://artifacts/abc12345-0000-0000-0000-000000000001"
        result = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=None,
        )
        assert result == ()
        assert memory.section_item_count(MemoryType.DOCUMENTS) == 0

    async def test_no_artifact_references_is_noop(self):
        runtime = _make_runtime()
        memory = SessionMemory("test-session")
        result = await hydrate_artifacts_from_message(
            "just a normal question",
            memory=memory,
            runtime=runtime,
        )
        assert result == ()

    async def test_kaos_uri_in_message_triggers_hydration(self):
        """End-to-end: write a real artifact, reference it via URI, observe
        the hydrator inject the body into DOCUMENTS."""
        runtime = _make_runtime()
        body = b"This is the contract under review. It mentions Section 3."
        artifact_id = await _write_artifact(
            runtime,
            name="contract.txt",
            body=body,
        )

        memory = SessionMemory("test-session")
        msg = f"What does kaos://artifacts/{artifact_id} say about Section 3?"
        hydrated = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )

        assert len(hydrated) == 1
        h = hydrated[0]
        assert h.artifact_id == artifact_id
        assert h.tier == "inline"
        assert h.bytes_loaded == len(body)
        # And the body should be in DOCUMENTS now.
        items = memory.get(MemoryType.DOCUMENTS)
        assert len(items) == 1
        assert "Section 3" in items[0].content
        assert items[0].metadata.get("hydrated_artifact_id") == artifact_id
        assert items[0].metadata.get("source") == "auto_hydrate"

    async def test_manifest_json_blob_in_message_triggers_hydration(self):
        """Pasting an ``ArtifactManifest.model_dump()`` payload should
        trigger hydration just like a bare URI."""
        runtime = _make_runtime()
        body = b"Manifest-shaped reference test body."
        artifact_id = await _write_artifact(
            runtime,
            name="memo.txt",
            body=body,
        )

        memory = SessionMemory("test-session")
        manifest_blob = json.dumps(
            {
                "artifact_id": artifact_id,
                "uri": f"kaos://artifacts/{artifact_id}",
                "name": "memo.txt",
                "size": len(body),
            }
        )
        msg = f"Here's what the upstream tool returned: {manifest_blob}"
        hydrated = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )

        assert len(hydrated) == 1
        assert hydrated[0].artifact_id == artifact_id
        assert memory.section_item_count(MemoryType.DOCUMENTS) == 1

    async def test_already_hydrated_artifact_is_skipped(self):
        """Second turn referencing the same artifact does NOT re-fetch."""
        runtime = _make_runtime()
        body = b"Body that should only get hydrated once."
        artifact_id = await _write_artifact(
            runtime,
            name="once.txt",
            body=body,
        )

        memory = SessionMemory("test-session")
        msg = f"First turn: kaos://artifacts/{artifact_id}"
        first = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )
        assert len(first) == 1
        assert memory.section_item_count(MemoryType.DOCUMENTS) == 1

        # Same memory, same URI in the next message — should be a no-op.
        msg_again = f"Follow-up: tell me more about kaos://artifacts/{artifact_id}"
        second = await hydrate_artifacts_from_message(
            msg_again,
            memory=memory,
            runtime=runtime,
        )
        assert second == ()
        # DOCUMENTS section unchanged: still exactly one item.
        assert memory.section_item_count(MemoryType.DOCUMENTS) == 1

    async def test_oversized_artifact_returns_handle_only(self):
        """An artifact larger than SUMMARY_THRESHOLD must NOT have its
        full body inlined — only a handle reference."""
        runtime = _make_runtime()
        # Construct a body just over the SUMMARY_THRESHOLD (256 KB).
        oversize_body = b"X" * (SUMMARY_THRESHOLD + 1024)
        artifact_id = await _write_artifact(
            runtime,
            name="huge.bin",
            body=oversize_body,
            mime_type="application/octet-stream",
        )

        memory = SessionMemory("test-session")
        msg = f"Process this big blob: kaos://artifacts/{artifact_id}"
        hydrated = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )

        assert len(hydrated) == 1
        h = hydrated[0]
        assert h.tier == "handle"
        # No body bytes were transferred — handle-only.
        assert h.bytes_loaded == 0

        items = memory.get(MemoryType.DOCUMENTS)
        assert len(items) == 1
        # The injected content must NOT contain the raw body — just a
        # handle pointer.
        assert "XXXX" not in items[0].content
        assert f"kaos://artifacts/{artifact_id}" in items[0].content
        # And the metadata records the tier so observers can see it.
        assert items[0].metadata.get("tier") == "handle"

    async def test_mid_size_artifact_returns_summary_tier(self):
        """A body between INLINE and SUMMARY thresholds inlines the
        first INLINE_THRESHOLD bytes with a "fetch full body" pointer."""
        runtime = _make_runtime()
        midsize_body = b"M" * (INLINE_THRESHOLD * 2)  # ~32 KB
        artifact_id = await _write_artifact(
            runtime,
            name="midsize.txt",
            body=midsize_body,
        )

        memory = SessionMemory("test-session")
        msg = f"kaos://artifacts/{artifact_id}"
        hydrated = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )

        assert len(hydrated) == 1
        h = hydrated[0]
        assert h.tier == "summary"
        # Loaded exactly INLINE_THRESHOLD bytes (capped by max_bytes).
        assert h.bytes_loaded == INLINE_THRESHOLD

    async def test_unresolvable_artifact_is_skipped_not_raised(self):
        """A bogus artifact ID in the message logs a warning and is
        skipped — it does NOT crash the hydration pipeline."""
        runtime = _make_runtime()
        memory = SessionMemory("test-session")
        msg = "kaos://artifacts/aaaa1111-dead-beef-cafe-000000000001"

        hydrated = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )

        assert hydrated == ()
        assert memory.section_item_count(MemoryType.DOCUMENTS) == 0

    async def test_returns_typed_hydrated_artifact(self):
        """The returned tuple contains ``HydratedArtifact`` instances
        with all expected fields populated."""
        runtime = _make_runtime()
        body = b"typed return value test"
        artifact_id = await _write_artifact(
            runtime,
            name="typed.txt",
            body=body,
        )
        memory = SessionMemory("test-session")
        msg = f"kaos://artifacts/{artifact_id}"
        result = await hydrate_artifacts_from_message(
            msg,
            memory=memory,
            runtime=runtime,
        )
        assert isinstance(result[0], HydratedArtifact)
        assert result[0].name == "typed.txt"
        assert result[0].mime_type == "text/plain"
        assert result[0].size == len(body)


# ---------------------------------------------------------------------------
# AgentChatTool integration — verifies the wiring is plumbed end-to-end
# without invoking the LLM (we use a missing-API-key error to short-circuit
# AFTER hydration has run).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatToolAutoHydrate:
    """Verify ``AgentChatTool.execute`` runs hydration before the LLM call.

    We intercept the runner with a mock so no LLM call happens but we
    still observe the hydration side-effect on session memory.
    """

    async def test_chat_tool_hydrates_before_running_turn(self, monkeypatch):
        from kaos_core.base.context import KaosContext

        from kaos_agents.memory.store import SessionStore
        from kaos_agents.tools.registry import AgentChatTool

        runtime = _make_runtime()
        body = b"Auto-hydrated body content for chat tool integration test."
        artifact_id = await _write_artifact(
            runtime,
            name="chat-test.txt",
            body=body,
            session_id="chat-tool-session",
        )

        # Stub the runner's actual turn so we never call the LLM.
        async def _fake_run_turn(runner, message, session_id):
            from kaos_agents.types.intents import IntentResult, IntentType
            from kaos_agents.types.response import AgentResponse

            status = {
                "paused_for_approval": False,
                "pending_tool_name": None,
                "run_state_ref": None,
                "budget_exceeded": False,
                "budget_kind": None,
                "run_error_event": None,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
            intent_result = IntentResult(
                intent=IntentType.RESPOND,
                confidence=1.0,
                reasoning="stubbed",
            )
            response = AgentResponse.create(
                text="(stubbed response)",
                intent=intent_result,
                tool_calls=(),
                turn_number=1,
                tokens_used=0,
                cost_usd=0.0,
                total_tokens=0,
                metadata={"session_id": session_id},
            )
            return response, status

        monkeypatch.setattr(
            "kaos_agents.tools.registry._run_turn_with_status",
            _fake_run_turn,
        )

        # Context session MUST match the inputs.session_id since 0.1.17 —
        # the memory/chat tools now refuse cross-session retargeting
        # (see ``tests/unit/test_session_isolation.py``). For this hydration
        # smoke test we use the same id for both so we exercise the
        # hydration side effect, not the cross-session refusal path.
        context = KaosContext(session_id="chat-tool-session", runtime=runtime)
        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": (
                    f"Please review kaos://artifacts/{artifact_id} and tell me what it says."
                ),
                "session_id": "chat-tool-session",
            },
            context=context,
        )

        assert result.isError is False
        structured = result.require_structured()
        assert "hydrated_artifacts" in structured
        assert len(structured["hydrated_artifacts"]) == 1
        assert structured["hydrated_artifacts"][0]["artifact_id"] == artifact_id

        # And the body should have been written through SessionStore so a
        # follow-up turn sees it without re-hydration.
        store = SessionStore(runtime.vfs)
        memory = await store.load_or_create("chat-tool-session")
        items = memory.get(MemoryType.DOCUMENTS)
        assert len(items) == 1
        assert artifact_id in items[0].metadata.get("hydrated_artifact_id", "")
