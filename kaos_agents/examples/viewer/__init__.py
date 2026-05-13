"""Single-page HTML viewer for kaos-agents JSONL telemetry recordings.

The viewer is a static ``index.html`` shipped alongside this module.
It needs no build step and no Python dependencies at runtime; the
:mod:`kaos_agents.examples.viewer.__main__` entry point just spins up
:mod:`http.server` long enough to serve the file (and an optional
JSONL fixture) to the local browser.

Run::

    python -m kaos_agents.examples.viewer
    python -m kaos_agents.examples.viewer /path/to/recording.jsonl

See ``README.md`` (section "Live audit trail") for the recorder side.
"""

from __future__ import annotations
