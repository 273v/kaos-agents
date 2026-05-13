"""StateSnapshot — capture + restore full agent state for replay.

Distinct from :class:`kaos_agents.runtime.interrupts.RunState` (tool-
approval pause/resume): a StateSnapshot captures the full
:class:`TurnInvocation` (events, usage, intent, plan, escalations) at
a moment in time. Used by:

  - Admin override flows (rollback to a previous turn state).
  - Reproducible-run testing (capture once, replay many).
  - Post-mortem debugging (load the state of a failed run).

Persistence shape: a single JSON document. Phase 5.F ships the typed
value type + serialization helpers; the broader replay machinery
(rebuilding an AgentLoop from a snapshot) is Phase 6+.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kaos_agents.core.invocation import TurnInvocation
from kaos_agents.events import deserialize_event_json, serialize_event_json


def _intent_to_dict(intent: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an intent value into a JSON-safe dict.

    ``IntentResult`` is a frozen dataclass in the agents package
    (:mod:`kaos_agents.types.intents`); pydantic ``model_dump`` is
    used when an intent happens to be a pydantic model in some
    downstream variant. ``None`` round-trips as-is.
    """
    if intent is None:
        return None
    # Pydantic model variant.
    model_dump = getattr(intent, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except Exception:
            # Defensive: fall through to dataclass path if model_dump
            # raises (e.g. extra-strict pydantic config).
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    # Dataclass variant — IntentResult is the canonical case.
    if dataclasses.is_dataclass(intent) and not isinstance(intent, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(intent):
            value = getattr(intent, f.name)
            # Convert nested usage dataclass to dict.
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                out[f.name] = dataclasses.asdict(value)
            else:
                out[f.name] = value
        # Stringify enums so JSON survives them.
        for key, value in list(out.items()):
            if hasattr(value, "value") and not isinstance(value, (int, float, bool, str)):
                out[key] = value.value
        return out
    # Last-resort fallback: capture whatever raw_input the object exposes.
    return {"raw_input": getattr(intent, "raw_input", "")}


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A captured agent-state point. Read-only by construction.

    Fields mirror the canonical TurnInvocation contract plus a
    ``snapshot_id`` and ``captured_at`` for archival.
    """

    snapshot_id: str
    captured_at: datetime
    session_id: str
    run_id: str
    turn_number: int
    agent_envelope_hash: str
    output: str
    intent: dict[str, Any] | None
    events_json: tuple[str, ...]  # serialized events; deserialize on demand
    usage: dict[str, Any]
    cost_usd: float
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_invocation(
        cls,
        invocation: TurnInvocation,
        *,
        snapshot_id: str | None = None,
    ) -> StateSnapshot:
        """Build a StateSnapshot from a live :class:`TurnInvocation`."""
        from uuid import uuid4

        # Serialize events one by one so the snapshot survives
        # round-tripping even when an event contains non-trivial fields
        # (e.g. EscalationRequired with dict details).
        events_json = tuple(serialize_event_json(e) for e in invocation.events)
        intent_dict = _intent_to_dict(invocation.intent)
        usage_dict = {
            "input_tokens": invocation.usage.input_tokens,
            "output_tokens": invocation.usage.output_tokens,
            "total_tokens": invocation.usage.total_tokens,
            "cost_usd": invocation.usage.cost_usd,
        }
        return cls(
            snapshot_id=snapshot_id or f"snap_{uuid4().hex[:12]}",
            captured_at=datetime.now(UTC),
            session_id=invocation.session_id,
            run_id=invocation.run_id,
            turn_number=invocation.turn_number,
            agent_envelope_hash=invocation.agent_envelope_hash,
            output=invocation.output,
            intent=intent_dict,
            events_json=events_json,
            usage=usage_dict,
            cost_usd=invocation.cost_usd,
            extras=dict(invocation.extras),
        )

    def deserialized_events(self) -> tuple[Any, ...]:
        """Return events as concrete :class:`KaosEvent` instances.

        Lazy: snapshots store events as JSON strings to keep the value
        type cheap to serialize; callers that want typed events call
        this method.
        """
        return tuple(deserialize_event_json(s) for s in self.events_json)

    def to_json(self) -> str:
        """Serialize the snapshot to a single JSON string."""
        return json.dumps(
            {
                "snapshot_id": self.snapshot_id,
                "captured_at": self.captured_at.isoformat(),
                "session_id": self.session_id,
                "run_id": self.run_id,
                "turn_number": self.turn_number,
                "agent_envelope_hash": self.agent_envelope_hash,
                "output": self.output,
                "intent": self.intent,
                "events_json": list(self.events_json),
                "usage": self.usage,
                "cost_usd": self.cost_usd,
                "extras": self.extras,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> StateSnapshot:
        """Deserialize from the :meth:`to_json` form."""
        data = json.loads(payload)
        return cls(
            snapshot_id=data["snapshot_id"],
            captured_at=datetime.fromisoformat(data["captured_at"]),
            session_id=data["session_id"],
            run_id=data["run_id"],
            turn_number=int(data["turn_number"]),
            agent_envelope_hash=data["agent_envelope_hash"],
            output=data["output"],
            intent=data.get("intent"),
            events_json=tuple(data.get("events_json", [])),
            usage=dict(data["usage"]),
            cost_usd=float(data["cost_usd"]),
            extras=dict(data.get("extras", {})),
        )


async def save_snapshot(
    snapshot: StateSnapshot,
    vfs: Any,
    *,
    path: str | None = None,
    context_id: str | None = None,
) -> str:
    """Save a snapshot to the VFS. Returns the path written.

    Default path: ``snapshots/<snapshot_id>.json``.
    """
    target = path or f"snapshots/{snapshot.snapshot_id}.json"
    payload = snapshot.to_json().encode("utf-8")
    await vfs.write(target, payload, context_id=context_id)
    return target


async def load_snapshot(
    vfs: Any,
    path: str,
    *,
    context_id: str | None = None,
) -> StateSnapshot:
    """Load a snapshot from a VFS path."""
    payload = await vfs.read(path, context_id=context_id)
    return StateSnapshot.from_json(payload.decode("utf-8"))


__all__ = ["StateSnapshot", "load_snapshot", "save_snapshot"]
