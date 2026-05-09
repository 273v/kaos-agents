"""Filesystem-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.filesystem` under the stable
``FileSystemTrigger`` name so consumers can write::

    from kaos_agents.triggers.fs import FileSystemTrigger
    trigger = FileSystemTrigger(path="/var/log/app.log", event="modified")

Phase 4+ replaces this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass driven by a
filesystem watcher (e.g. ``watchfiles`` / ``watchdog``). The import
path stays stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

FileSystemTrigger = Trigger.filesystem

__all__ = ["FileSystemTrigger"]
