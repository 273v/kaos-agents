"""Wire-format serialization for KaosEvent streams.

These helpers are thin facades over pydantic's ``model_dump`` /
``model_validate`` plus the
:data:`kaos_agents.registry.event_registry.default_event_registry`
type-string lookup. The dispatch table is the registry — see
:mod:`kaos_agents.registry.event_registry` and
:meth:`kaos_agents.base.event.KaosEvent.__init_subclass__` for the
auto-registration side.

Public surface: :func:`serialize_event`, :func:`deserialize_event`,
plus the JSON-string variants for stream wires.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from kaos_agents.base.event import KaosEvent
from kaos_agents.errors import EventDeserializationError, EventSerializationError
from kaos_agents.registry.event_registry import default_event_registry as _event_registry


def event_type_name(event: KaosEvent) -> str:
    """Get the wire-format type string for an event instance."""
    return type(event).event_type()


def serialize_event(event: KaosEvent) -> dict[str, Any]:
    """Serialize a KaosEvent to a JSON-safe dict with a ``type`` discriminator.

    The ``type`` field is the snake_case version of the class name.
    Nested ``KaosModel`` payloads (ToolCallSummary, PlanStepSummary)
    are serialized as dicts. Tuples become lists.

    Example::

        >>> e = TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=1)
        >>> serialize_event(e)
        {'type': 'turn_start', 'timestamp': 1.0, 'sequence': 0, ...}
    """
    try:
        data = event.model_dump(mode="json")
    except ValidationError as exc:
        raise EventSerializationError(
            f"Failed to serialize {type(event).__name__}: {exc}. "
            "Ensure all event fields contain JSON-serializable values."
        ) from exc
    data["type"] = event_type_name(event)
    return data


def deserialize_event(data: dict[str, Any]) -> KaosEvent:
    """Reconstruct a KaosEvent from a serialized dict.

    Raises:
        EventDeserializationError: If the ``type`` field is missing or unknown,
            or if required fields are missing or have incorrect types.

    Example::

        >>> d = {'type': 'turn_start', 'timestamp': 1.0, 'sequence': 0,
        ...      'session_id': 's', 'run_id': 'r', 'turn_number': 1}
        >>> deserialize_event(d)
        TurnStart(timestamp=1.0, sequence=0, session_id='s', run_id='r',
                  agent_id=None, turn_number=1)
    """
    type_name = data.get("type")
    if not type_name:
        raise EventDeserializationError(
            "Event dict missing 'type' field. "
            "Every serialized event must include a 'type' field with the snake_case event name. "
            "Use serialize_event() to produce correctly formatted dicts."
        )

    cls = _event_registry.get(type_name)
    if cls is None:
        valid = ", ".join(_event_registry.list_types())
        raise EventDeserializationError(
            f"Unknown event type '{type_name}'. "
            f"Valid types: {valid}. "
            "Check that the event type name is snake_case (e.g., 'turn_start', not 'TurnStart')."
        )

    payload = {k: v for k, v in data.items() if k != "type"}
    try:
        return cls.model_validate(payload)
    except ValidationError as exc:
        required = ", ".join(name for name, info in cls.model_fields.items() if info.is_required())
        raise EventDeserializationError(
            f"Failed to construct {cls.__name__}: {exc}. "
            f"Required fields: {required or '(none)'}. "
            f"Provided fields: {', '.join(payload) or '(none)'}."
        ) from exc


def serialize_event_json(event: KaosEvent) -> str:
    """Serialize a KaosEvent to a compact JSON string.

    Raises:
        EventSerializationError: If the event contains values that
            cannot be serialized to JSON (e.g., custom objects).
    """
    try:
        return json.dumps(serialize_event(event), separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EventSerializationError(
            f"Failed to serialize {type(event).__name__} to JSON: {exc}. "
            "Ensure all event fields contain JSON-serializable values "
            "(str, int, float, bool, None, tuple, dict)."
        ) from exc


def deserialize_event_json(data: str) -> KaosEvent:
    """Deserialize a KaosEvent from a JSON string.

    Raises:
        EventDeserializationError: If the JSON is invalid or the event
            cannot be reconstructed.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise EventDeserializationError(
            f"Invalid JSON at position {exc.pos}: {exc.msg}. "
            "Ensure the input is valid JSON. "
            "Use serialize_event_json() to produce correctly formatted strings."
        ) from exc
    if not isinstance(parsed, dict):
        raise EventDeserializationError(
            f"Expected a JSON object, got {type(parsed).__name__}. "
            "Each event must be a JSON object with a 'type' field. "
            "Use serialize_event_json() to produce correctly formatted strings."
        )
    return deserialize_event(parsed)


__all__ = [
    "deserialize_event",
    "deserialize_event_json",
    "event_type_name",
    "serialize_event",
    "serialize_event_json",
]
