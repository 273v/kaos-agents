"""kaos-agents CLI entry points — Track 5 chunk T5-2 consolidation.

Subpackage layout:

- :mod:`kaos_agents.cli.chat` — ``kaos-agent chat`` REPL +
  one-shot turn entry point (was ``kaos_agents.cli_chat``)
- :mod:`kaos_agents.cli.extract` — ``kaos-extract`` schema-driven
  extraction CLI (was ``kaos_agents.cli_extract``)

Top-level entry points (registered via ``[project.scripts]`` in
``pyproject.toml``):

- ``kaos-agent     = kaos_agents.cli.chat:main``
- ``kaos-extract   = kaos_agents.cli.extract:main``

Benchmark / test consumers that previously did
``from kaos_agents.cli_chat import _load_files_into_memory`` should
update to ``from kaos_agents.cli.chat import _load_files_into_memory``.
The pre-T5 leaf modules ``kaos_agents.cli_chat`` /
``kaos_agents.cli_extract`` are gone.
"""

from __future__ import annotations

from kaos_agents.cli.chat import main as chat_main
from kaos_agents.cli.extract import main as extract_main

__all__ = [
    "chat_main",
    "extract_main",
]
