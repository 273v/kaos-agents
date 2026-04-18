"""Wire format serializers for AgentEvent streams.

Converts ``AsyncIterator[AgentEvent]`` to wire-format strings for
transport over SSE, JSONL, or WebSocket. These are pure async generators
with no framework dependency — they can be used with FastAPI, Starlette,
aiohttp, or any ASGI framework.

Two formats:

- **SSE** (Server-Sent Events): ``event: {type}\\ndata: {json}\\nid: {seq}\\n\\n``
  For browser ``EventSource`` and HTTP streaming clients.
- **JSONL** (JSON Lines): ``{json}\\n`` per event.
  For SDK clients, log replay, and pipe-based consumers.

Usage with FastAPI::

    from starlette.responses import StreamingResponse
    from kaos_agents.wire import events_to_sse

    @app.post("/v1/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: MessageRequest):
        runner = Runner(agent, runtime=runtime)
        event_stream = runner.run(body.message, session_id)
        return StreamingResponse(
            events_to_sse(event_stream),
            media_type="text/event-stream",
        )
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from kaos_agents.events import AgentEvent, event_type_name, serialize_event


async def events_to_sse(
    events: AsyncIterator[AgentEvent],
) -> AsyncIterator[str]:
    """Convert an event stream to SSE (Server-Sent Events) format.

    Each event becomes an SSE message with:
    - ``event:`` — the snake_case event type name
    - ``data:`` — compact JSON of the full event
    - ``id:`` — the event sequence number (for ``Last-Event-ID`` reconnection)

    Terminated by ``\\n\\n`` per the SSE specification.

    Args:
        events: Async iterator of AgentEvent objects.

    Yields:
        SSE-formatted strings, one per event.
    """
    async for event in events:
        type_name = event_type_name(event)
        data = json.dumps(serialize_event(event), separators=(",", ":"))
        yield f"event: {type_name}\ndata: {data}\nid: {event.sequence}\n\n"


async def events_to_jsonl(
    events: AsyncIterator[AgentEvent],
) -> AsyncIterator[str]:
    """Convert an event stream to JSONL (JSON Lines) format.

    Each event becomes a single line of compact JSON followed by a newline.
    The ``type`` field is included in the JSON for self-describing messages.

    Args:
        events: Async iterator of AgentEvent objects.

    Yields:
        JSONL-formatted strings, one line per event.
    """
    async for event in events:
        yield json.dumps(serialize_event(event), separators=(",", ":")) + "\n"


async def events_to_ws(
    events: AsyncIterator[AgentEvent],
) -> AsyncIterator[dict[str, Any]]:
    """Convert an event stream to WebSocket-ready dicts.

    Each event becomes a dict suitable for ``websocket.send_json()``.
    No string encoding — the WebSocket layer handles JSON serialization.

    Args:
        events: Async iterator of AgentEvent objects.

    Yields:
        Serialized event dicts with ``type`` discriminator.
    """
    async for event in events:
        yield serialize_event(event)
