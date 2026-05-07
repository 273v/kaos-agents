"""Stream deltas — low-level, real-time, token-by-token events.

Consumers that care about token-by-token display handle these
(:class:`TextDelta`, :class:`ThinkingDelta`, :class:`ToolCallArgsDelta`).
``isinstance(event, StreamDelta)`` is the canonical dispatch.

For high-level lifecycle events (turn boundaries, tool boundaries,
plan steps), see :mod:`kaos_agents.events.lifecycle` and friends.
"""

from __future__ import annotations

from kaos_agents.base.event import KaosEvent


class StreamDelta(KaosEvent):
    """Base class for low-level, real-time streaming events.

    Consumers that care about token-by-token display handle these
    (TextDelta, ThinkingDelta, ToolCallArgsDelta).

    ``isinstance(event, StreamDelta)`` is the canonical way to route
    real-time display events.
    """

    @classmethod
    def event_category(cls) -> str:
        return "stream"


class TextDelta(StreamDelta):
    """Streaming text token(s) from the LLM.

    Emitted as the model generates text. Consumers concatenate
    sequential TextDelta.content to build the full response.
    """

    content: str = ""


class ThinkingDelta(StreamDelta):
    """Streaming thinking/reasoning tokens (extended thinking).

    Separate from TextDelta so UIs can render thinking in a
    collapsible section or different style.
    """

    content: str = ""


class ToolCallArgsDelta(StreamDelta):
    """Streaming tool call argument tokens.

    Emitted as the model generates tool call arguments.
    Allows UIs to show partial argument JSON as it builds.
    """

    call_id: str = ""
    content: str = ""
