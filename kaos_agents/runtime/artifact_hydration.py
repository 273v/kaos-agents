"""Auto-hydration of VFS artifact references from agent messages.

PA5 — ``AgentChatTool`` previously required the caller to manually attach
VFS artifacts to the request. The natural pattern is: an upstream tool
produces an artifact (e.g. ``kaos-pdf-parse`` returns a manifest URI), the
agent chat tool then gets a follow-up "what's in that doc?" — the
artifact should auto-hydrate from the message context, not require the
caller to re-pass ``artifact_ids=[...]``.

This module scans message text for artifact-shaped references and pulls
the body (or summary, depending on the artifact's size relative to
``INLINE_THRESHOLD`` / ``SUMMARY_THRESHOLD``) into ``SessionMemory`` under
the :attr:`MemoryType.DOCUMENTS` section. De-duplication is keyed on
the artifact ID via ``MemoryItem.metadata["hydrated_artifact_id"]`` —
the second turn referencing the same URI is a no-op.

Surface area is intentionally narrow: scan, resolve, read (respecting
size tiers), inject. The runner / pattern do not need to know hydration
happened — the document content is simply present in the DOCUMENTS
section when the chat pattern assembles context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_core.artifacts.models import INLINE_THRESHOLD, SUMMARY_THRESHOLD
from kaos_core.logging import get_logger

from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from kaos_core.artifacts.store import ArtifactStore
    from kaos_core.registry.container import KaosRuntime

    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# URI patterns
# ---------------------------------------------------------------------------
# Match ``kaos://artifacts/<id>`` with optional ``/body``, ``/manifest``,
# ``/chunk/<n>``, ``/range/<start>/<length>``. We treat all of these as
# pointing at the same artifact for hydration purposes — the agent wants
# the document content, not a specific byte range.
_KAOS_URI_RE = re.compile(
    r"kaos://artifacts/([0-9a-fA-F][0-9a-fA-F-]{7,})(?:/[\w/]+)?",
)

# Match the shorthand ``artifact://<id>`` URI scheme used by some upstream
# tool surfaces and by humans pasting ID-bearing strings.
_ARTIFACT_SHORTHAND_RE = re.compile(
    r"artifact://([0-9a-fA-F][0-9a-fA-F-]{7,})",
)

# Manifest-shaped JSON blob detection. We look for a JSON object that
# contains both ``"artifact_id"`` AND a ``"uri"`` field naming a kaos://
# artifact. This is a conservative match — random JSON with one of the
# keys won't trigger; only an actual ``ArtifactManifest.model_dump()``
# payload will.
_MANIFEST_BLOB_RE = re.compile(
    r'\{[^{}]*"artifact_id"\s*:\s*"([0-9a-fA-F][0-9a-fA-F-]{7,})"[^{}]*\}',
    re.DOTALL,
)

# Hard cap on how much body content we pull into memory per turn. Even
# if multiple artifacts are referenced, total injected text is bounded by
# this to avoid blowing the context window on the very first turn.
# Aligned with SUMMARY_THRESHOLD so a single mid-size artifact still
# fits inline.
_PER_TURN_HYDRATION_BUDGET = SUMMARY_THRESHOLD  # 256 KB total per turn


@dataclass(frozen=True, slots=True)
class HydratedArtifact:
    """A single auto-hydrated artifact reference.

    Returned by :func:`hydrate_artifacts_from_message` so callers (the
    chat tool, tests) can observe what was injected without re-reading
    session memory.
    """

    artifact_id: str
    uri: str
    size: int
    mime_type: str | None
    name: str
    tier: str  # "inline" | "summary" | "handle"
    bytes_loaded: int


def find_artifact_references(message: str) -> tuple[str, ...]:
    """Scan a message string for artifact-shaped references.

    Returns a tuple of unique artifact IDs in first-seen order. The
    order matters because the chat tool injects in encounter order
    (older first → newer last in DOCUMENTS).

    Scans for:
    - ``kaos://artifacts/<id>`` (and any sub-path)
    - ``artifact://<id>``
    - JSON manifest blobs with ``"artifact_id": "<id>"`` and a ``"uri"``
      field naming a kaos:// artifact

    The artifact ID format is the loose ``<hex>[-hex...]`` pattern used
    by ``uuid4()`` (32 hex chars + 4 hyphens) — we accept anything
    starting with a hex char and at least 8 chars long so test fixtures
    that use shorter IDs also work.
    """
    if not message:
        return ()

    seen: dict[str, None] = {}  # dict preserves insertion order

    # 1) JSON manifest blobs first (most specific). When a blob matches,
    # the embedded kaos:// URI inside it will ALSO match the kaos://
    # regex below — but the dict-de-dup keeps each artifact ID exactly
    # once regardless of how many shapes pointed at it.
    for match in _MANIFEST_BLOB_RE.finditer(message):
        # Validate it's actually a parseable manifest with a kaos:// URI.
        # If parsing fails, fall through to the URI regex below which
        # will catch the URI directly.
        blob_text = match.group(0)
        try:
            blob = json.loads(blob_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(blob, dict):
            continue
        artifact_id = blob.get("artifact_id")
        uri = blob.get("uri", "")
        if (
            isinstance(artifact_id, str)
            and isinstance(uri, str)
            and uri.startswith("kaos://artifacts/")
        ):
            seen.setdefault(artifact_id, None)

    # 2) kaos:// URIs
    for match in _KAOS_URI_RE.finditer(message):
        seen.setdefault(match.group(1), None)

    # 3) artifact:// shorthand
    for match in _ARTIFACT_SHORTHAND_RE.finditer(message):
        seen.setdefault(match.group(1), None)

    return tuple(seen.keys())


def already_hydrated_artifact_ids(memory: SessionMemory) -> set[str]:
    """Return the set of artifact IDs already auto-hydrated into this
    session's :attr:`MemoryType.DOCUMENTS` section.

    Used by :func:`hydrate_artifacts_from_message` to skip re-fetching
    artifacts that were injected on a prior turn. The marker is the
    ``hydrated_artifact_id`` key in each memory item's metadata — we
    don't trust content-substring matching because the artifact body
    may have been summarised at injection time.
    """
    if not memory.has_section(MemoryType.DOCUMENTS):
        return set()
    items = memory.get(MemoryType.DOCUMENTS)
    seen: set[str] = set()
    for item in items:
        artifact_id = item.metadata.get("hydrated_artifact_id")
        if isinstance(artifact_id, str):
            seen.add(artifact_id)
    return seen


def _format_inline(manifest: Any, text: str) -> str:
    """Render an inline-tier artifact as the memory item content."""
    header = (
        f"[Artifact: {manifest.name} ({manifest.artifact_id})] "
        f"mime={manifest.mime_type or 'unknown'} size={manifest.size}B"
    )
    return f"{header}\n\n{text}"


def _format_summary(manifest: Any, text: str) -> str:
    """Render a summary-tier artifact as the memory item content.

    The artifact body is between INLINE and SUMMARY thresholds — we
    inline the first ``INLINE_THRESHOLD`` bytes (decoded as text) and
    note the truncation so the agent knows to fetch more via
    ``kaos-mcp`` resource APIs if needed.
    """
    header = (
        f"[Artifact: {manifest.name} ({manifest.artifact_id})] "
        f"mime={manifest.mime_type or 'unknown'} size={manifest.size}B "
        f"(showing first {len(text)}B; fetch {manifest.body_uri} for full body)"
    )
    return f"{header}\n\n{text}"


def _format_handle(manifest: Any) -> str:
    """Render a handle-only-tier artifact (>SUMMARY_THRESHOLD).

    Body is NOT loaded — we only inject the manifest reference so the
    agent knows the artifact exists and how to fetch ranges of it.
    """
    return (
        f"[Artifact: {manifest.name} ({manifest.artifact_id})] "
        f"mime={manifest.mime_type or 'unknown'} size={manifest.size}B "
        f"(handle-only; exceeds {SUMMARY_THRESHOLD}B inline budget — "
        f"fetch ranges via {manifest.body_uri})"
    )


async def _read_text_or_handle(
    store: ArtifactStore,
    manifest: Any,
    *,
    max_bytes: int,
    caller_session_id: str | None = None,
) -> tuple[str, str, int]:
    """Read up to ``max_bytes`` of an artifact's body as text.

    ``caller_session_id`` is forwarded to ``ArtifactStore.read_body`` so
    the kaos-core ownership guard
    (``kaos_core.artifacts.store.ArtifactStore._resolve``) fires when an
    artifact belongs to a different session. Passing ``None`` opts out
    of the check (trusted in-process callers); the auto-hydration call
    site at ``hydrate_artifacts_from_message`` forwards the requesting
    session id and MUST NOT default to ``manifest.session_id`` (that
    would compare the artifact's owner to itself and always pass).

    Returns ``(tier, content, bytes_loaded)`` where tier is one of
    ``"inline"`` / ``"summary"`` / ``"handle"``.
    """
    if manifest.size >= SUMMARY_THRESHOLD:
        return "handle", _format_handle(manifest), 0

    # For inline + summary tiers we read at most ``max_bytes`` raw bytes
    # and decode with replacement so a binary or partially-binary body
    # still produces a usable string (no UnicodeDecodeError on the
    # hydration path).
    read_length = min(manifest.size, max_bytes)
    try:
        payload = await store.read_body(
            manifest.artifact_id,
            start=0,
            length=read_length if read_length > 0 else None,
            caller_session_id=caller_session_id,
        )
    except Exception as exc:
        logger.warning(
            "artifact_hydration: failed to read body for %s: %s",
            manifest.artifact_id,
            exc,
        )
        return "handle", _format_handle(manifest), 0

    text = payload.decode("utf-8", errors="replace")

    if manifest.size < INLINE_THRESHOLD:
        return "inline", _format_inline(manifest, text), len(payload)
    return "summary", _format_summary(manifest, text), len(payload)


async def hydrate_artifacts_from_message(
    message: str,
    *,
    memory: SessionMemory,
    runtime: KaosRuntime | None,
    per_turn_budget_bytes: int = _PER_TURN_HYDRATION_BUDGET,
    caller_session_id: str | None = None,
) -> tuple[HydratedArtifact, ...]:
    """Scan ``message`` for artifact references and inject bodies into
    ``memory.DOCUMENTS``.

    ``caller_session_id`` identifies the session on whose behalf the
    hydration is running. It is forwarded to
    ``ArtifactStore._resolve_async`` and ``ArtifactStore.read_body`` so
    kaos-core's ownership guard
    (``kaos_core.artifacts.store.ArtifactStore._resolve``) refuses to
    surface an artifact minted by a different session.

    Callers MUST pass the requesting session's id; the default
    ``None`` exists for trusted in-process callers (cleanup, migration,
    tests that intentionally cross sessions). When ``None`` is passed,
    the kaos-core guard is bypassed — that is a deliberate escape
    hatch, not the production path.

    No-op when:
    - ``runtime`` is None (no ArtifactStore available)
    - ``memory`` has no DOCUMENTS section configured
    - no artifact-shaped references are found in the message
    - all referenced artifacts are already hydrated for this session

    Body content is injected as a memory item with metadata
    ``{"hydrated_artifact_id": <id>, "source": "auto_hydrate"}`` so the
    next turn can detect "already done" without re-reading the body.

    Returns the tuple of newly-hydrated artifacts (empty when nothing
    fired). Order matches first-encounter order in ``message``.

    Errors during a single artifact's resolve/read are logged at WARNING
    and skipped — one bad URI does not break the whole turn. A
    cross-session denial from the kaos-core guard surfaces as a single
    skipped artifact (not a turn-level failure), so a malicious or
    confused caller asking for a sibling-session artifact gets nothing
    back rather than an error trail that confirms the artifact exists.
    """
    if not message or runtime is None:
        return ()
    if not memory.has_section(MemoryType.DOCUMENTS):
        return ()

    candidate_ids = find_artifact_references(message)
    if not candidate_ids:
        return ()

    already = already_hydrated_artifact_ids(memory)
    fresh_ids = tuple(aid for aid in candidate_ids if aid not in already)
    if not fresh_ids:
        return ()

    try:
        store = runtime.artifacts
    except Exception as exc:
        logger.warning("artifact_hydration: runtime has no ArtifactStore: %s", exc)
        return ()

    hydrated: list[HydratedArtifact] = []
    remaining_budget = per_turn_budget_bytes

    for artifact_id in fresh_ids:
        if remaining_budget <= 0:
            logger.info(
                "artifact_hydration: per-turn budget exhausted; skipping remaining %d artifact(s)",
                len(fresh_ids) - len(hydrated),
            )
            break

        # Resolve manifest under the caller's session. ``caller_session_id``
        # is forwarded to kaos-core's ownership guard (see
        # ``kaos_core.artifacts.store.ArtifactStore._resolve``): when the
        # artifact's owning session != the caller's session the guard
        # raises ``ResourceError("Unknown artifact")``. We treat that
        # identically to "not found" so a probe for a sibling-session
        # artifact ID yields exactly the same observable behavior as a
        # probe for a non-existent artifact ID — no oracle.
        try:
            manifest = await store._resolve_async(
                artifact_id,
                caller_session_id=caller_session_id,
            )
        except Exception as exc:
            logger.warning(
                "artifact_hydration: unable to resolve artifact %s: %s",
                artifact_id,
                exc,
            )
            continue

        max_bytes_for_this = min(remaining_budget, INLINE_THRESHOLD)
        tier, content, bytes_loaded = await _read_text_or_handle(
            store,
            manifest,
            max_bytes=max_bytes_for_this,
            caller_session_id=caller_session_id,
        )

        try:
            memory.add(
                MemoryType.DOCUMENTS,
                content,
                tags=("auto_hydrate", f"artifact:{artifact_id}"),
                metadata={
                    "hydrated_artifact_id": artifact_id,
                    "source": "auto_hydrate",
                    "tier": tier,
                    "manifest_uri": manifest.manifest_uri,
                    "body_uri": manifest.body_uri,
                    "mime_type": manifest.mime_type,
                    "size": manifest.size,
                },
            )
        except Exception as exc:
            # Memory may be REFUSE / over-budget. Skip this artifact
            # but keep going — partial hydration is still useful.
            logger.warning(
                "artifact_hydration: unable to add artifact %s to DOCUMENTS: %s",
                artifact_id,
                exc,
            )
            continue

        hydrated.append(
            HydratedArtifact(
                artifact_id=artifact_id,
                uri=manifest.body_uri,
                size=manifest.size,
                mime_type=manifest.mime_type,
                name=manifest.name,
                tier=tier,
                bytes_loaded=bytes_loaded,
            )
        )
        remaining_budget -= bytes_loaded
        logger.info(
            "artifact_hydration: injected artifact=%s tier=%s bytes=%d",
            artifact_id,
            tier,
            bytes_loaded,
        )

    return tuple(hydrated)


__all__ = [
    "HydratedArtifact",
    "already_hydrated_artifact_ids",
    "find_artifact_references",
    "hydrate_artifacts_from_message",
]
