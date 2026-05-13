"""Run replay + diff — regression testing without re-paying LLM cost.

Captures an agent run's event stream as a JSONL artifact; loads it
back; diffs two captured runs to surface deltas.

Today there's no way to ask "did my change to the ReAct prompt break
anything?" without running a $5 surface-parity sweep. Replay lets the
question be deterministic and free. Diff lets bisection target the
actual regression.

Public surface:

- :class:`RecordedRun` — value type carrying a captured event stream
  + summary metadata (turn count, total cost, final answer text).
- :func:`record_events` — drain an ``AsyncIterator[KaosEvent]`` into
  a ``RecordedRun``. Pairs with ``Runner.run`` — wrap the iterator
  to capture every emitted event.
- :func:`save_run` / :func:`load_run` — JSONL persistence via
  ``kaos_agents.events.serde``.
- :class:`RunDiff` + :func:`diff_runs` — structured comparison of
  two ``RecordedRun`` instances. Reports event-count delta, type
  histograms, the first divergence point, and final-answer text
  diff.

The replay-mode-strict path (re-execute against a live runtime with
stubbed LLM responses) is a separate concern — see :func:`replay_run`
for the v1 implementation, which currently re-emits the captured
stream without re-running the underlying actions. Strict replay
needs Cassette-style HTTP recording on kaos-llm-client; deferred.

Design rationale:

- Events are the authoritative interface. Capturing them is sufficient
  for "what did the run produce?" Re-executing against the runtime
  is "did the runtime still produce that?" — a different question.
- JSONL persistence reuses existing ``serialize_event_json`` /
  ``deserialize_event_json``. No new serialization formats.
- Diffing is structural, not semantic. Two runs with identical event
  sequences are "equal" even if minor text wording differs in
  TextDelta payloads — exposed via an explicit
  ``RunDiff.text_diff`` channel callers can opt into.
"""

from __future__ import annotations

from kaos_agents.replay.diff import (
    EventDelta,
    RunDiff,
    diff_runs,
)
from kaos_agents.replay.recorder import (
    RecordedRun,
    load_run,
    record_events,
    save_run,
)
from kaos_agents.replay.replayer import replay_run

__all__ = [
    "EventDelta",
    "RecordedRun",
    "RunDiff",
    "diff_runs",
    "load_run",
    "record_events",
    "replay_run",
    "save_run",
]
