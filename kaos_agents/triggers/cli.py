"""CLI-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.cli` under the stable
``CLIPromptTrigger`` name so consumers can write::

    from kaos_agents.triggers.cli import CLIPromptTrigger
    trigger = CLIPromptTrigger("hello", session_id="local", tty=True)

Phase 4+ may replace this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass that consumes
stdin / interactive REPL input as an async iterator. The import path
stays stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

CLIPromptTrigger = Trigger.cli

__all__ = ["CLIPromptTrigger"]
