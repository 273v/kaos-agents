"""Re-emit a captured event stream as an async iterator.

The minimal-viable replay path. ``replay_run`` yields the captured
events in order — useful for testing consumers (wire serializers,
event-stream parsers, hooks) without re-paying LLM cost.

Strict replay — re-executing the underlying actions against a live
runtime with stubbed LLM responses — needs cassette-style HTTP
recording on ``kaos-llm-client`` and is deferred. The event-replay
path here covers the common "did my consumer change break this trace
shape?" question without that infrastructure.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent
    from kaos_agents.replay.recorder import RecordedRun


async def replay_run(
    run: RecordedRun,
    *,
    delay_seconds: float = 0.0,
) -> AsyncIterator[KaosEvent]:
    """Yield the captured events in order.

    Use as a drop-in replacement for ``Runner.run`` when you want to
    feed a downstream consumer (wire serializer, hook chain, replay
    test) a known event sequence without invoking the agent.

    Args:
        run: The :class:`RecordedRun` to replay.
        delay_seconds: Optional artificial delay between events.
            Useful when testing streaming consumers that expect
            inter-event timing (e.g., SSE clients that buffer).
            Default 0 (back-to-back yield, fastest).

    Yields:
        Every event from ``run.events`` in original order.
    """
    for event in run.events:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield event


__all__ = ["replay_run"]
