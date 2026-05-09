"""Re-exports of the canonical event collector for the perception package.

Phase 0.B landed ``kaos_agents.events.collector`` as the canonical
push-based collector. Phase 1.B was built against a stale base where
that module did not yet exist, so it shipped a private duplicate. This
module is now a thin re-export so existing imports inside the perception
package keep working without dragging the duplicate into the type tree.

New code should import from :mod:`kaos_agents.events.collector` directly.
"""

from __future__ import annotations

from kaos_agents.events.collector import collect_events, push_event

__all__ = ["collect_events", "push_event"]
