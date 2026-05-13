"""Coverage contract for KaosEvent subclasses.

Every concrete ``KaosEvent`` subclass must either:

1. **Be emitted** by at least one production code site (a non-test,
   non-events-module file inside ``kaos_agents/``), OR
2. **Be on the explicit "not-yet-wired" allowlist** below — with a
   documented reason and a tracking note. The allowlist is a
   loud-failure mode: classes that exist but have no emit site rot
   silently otherwise (the audit of bug #4 found 5 such classes).

When you add a new event subclass, the audit below will fail until you
either wire an emit site or move the class onto ``KNOWN_UNEMITTED``.
When you wire a previously-allowlisted class, drop it from the list —
the test will fail if a class is on the allowlist but has at least
one production emit site (no stale entries).
"""

from __future__ import annotations

import pkgutil
import re
from pathlib import Path

import pytest

from kaos_agents import events as events_pkg
from kaos_agents.base.event import KaosEvent

# Classes that exist but are not (yet) emitted from production code.
# Adding to this set is fine — but it must come with a tracking note.
KNOWN_UNEMITTED: dict[str, str] = {
    "ToolCallArgsDelta": (
        "Streaming token-by-token tool-argument construction. "
        "kaos-llm-client emits ``StreamChunk(type='tool_call_delta', "
        "tool_call_delta=...)`` for OpenAI and Google providers — the "
        "primitive exists at the provider layer. Bridging it through "
        "to a kaos-agents ToolCallArgsDelta requires switching the "
        "pattern dispatchers (chat.py, research/agent.py, "
        "plan_execute.py) from ``Call.invoke()`` (non-streaming) to "
        "``Call.stream()`` and routing the tool_call_delta chunks. "
        "That's a real architectural change — pattern dispatch "
        "currently materializes the full Invocation before any "
        "trajectory walk happens. Defer to the same streaming-first "
        "rewrite that would also surface incremental TextDelta "
        "tokens (today TextDelta also fires once with the complete "
        "string, not per-token)."
    ),
}


def _all_concrete_event_classes() -> list[type[KaosEvent]]:
    """Enumerate every concrete KaosEvent subclass under kaos_agents.events.

    Excludes:
    - Abstract intermediates (LifecycleEvent, StreamDelta).
    - Classes outside ``kaos_agents.events`` — Triggers live in
      ``kaos_agents.triggers`` and represent agent INPUTS (not events
      emitted by the run loop), so they have a different audit shape.
    """
    # Force-import every events submodule so subclasses register.
    for mod_info in pkgutil.iter_modules(events_pkg.__path__):
        __import__(f"{events_pkg.__name__}.{mod_info.name}")

    seen: list[type[KaosEvent]] = []

    def _walk(cls: type[KaosEvent]) -> None:
        for sub in cls.__subclasses__():
            _walk(sub)
            # Skip abstract intermediates (LifecycleEvent, StreamDelta).
            if sub.__module__.endswith("_intermediates") or sub.__name__ in {
                "LifecycleEvent",
                "StreamDelta",
            }:
                continue
            # Skip Trigger and its subclasses — they're inputs to the
            # agent (the trigger that opens a turn), not events emitted
            # downstream of the run loop. Different audit shape applies.
            if not sub.__module__.startswith("kaos_agents.events"):
                continue
            if sub not in seen:
                seen.append(sub)

    _walk(KaosEvent)
    return seen


def _emit_sites_for(class_name: str) -> list[Path]:
    """Locate production emit sites of a given class name.

    Recognises two flavors of "the production code emitted this event":

    1. ``emitter.emit(ClassName, ...)`` — the standard channel.
    2. Direct construction in a control-flow context that yields/returns
       the event into the run loop's stream — ``return ClassName(...)``,
       ``yield ClassName(...)``, ``Var(... = ClassName(...)``. Used by
       Runner._pause_for_approval (constructs ToolCallApprovalRequired
       outside any emitter scope and returns it for the caller to yield)
       and a handful of other paths where direct construction is the
       cleanest fit.

    ``Span`` is special-cased — its emissions go through the
    ``span_start`` / ``span_complete`` / ``span_error`` helpers, not a
    raw ``.emit(Span, ...)``.
    """
    if class_name == "Span":
        pattern: re.Pattern[str] = re.compile(r"\.span_(start|complete|error)\(")
    else:
        emit_pat = rf"\.emit\(\s*{re.escape(class_name)}\b"
        # Direct construction in a yield/return/assignment context. The
        # regex avoids matching class definitions or imports by anchoring
        # on common control-flow keywords / assignment.
        direct_pat = rf"(?:yield\s+|return\s+|=\s*){re.escape(class_name)}\("
        pattern = re.compile(rf"(?:{emit_pat})|(?:{direct_pat})")
    pkg_root = Path(__file__).parents[3] / "kaos_agents"
    hits: list[Path] = []
    # Files inside events/ that are pure type definitions — emissions
    # in those files are the typed-class declaration, not an event
    # being fired. emitter.py + serde.py are the helper modules that
    # genuinely emit (emit_thinking_from_invocation, etc.) and SHOULD
    # be searched.
    EVENTS_DEFINITION_FILES = {
        "_intermediates.py",
        "budget.py",
        "collector.py",
        "escalation.py",
        "lifecycle.py",
        "memory.py",
        "plan.py",
        "research.py",
        "spans.py",
        "stream.py",
        "tools.py",
        "__init__.py",
    }
    for path in pkg_root.rglob("*.py"):
        rel = path.relative_to(pkg_root.parent)
        if "test" in rel.parts:
            continue
        if rel.parts[1] == "events" and path.name in EVENTS_DEFINITION_FILES:
            continue
        if pattern.search(path.read_text()):
            hits.append(path)
    return hits


@pytest.mark.parametrize("event_cls", _all_concrete_event_classes(), ids=lambda c: c.__name__)
def test_event_class_is_emitted_or_explicitly_unemitted(event_cls: type[KaosEvent]) -> None:
    """Every concrete KaosEvent must be emitted OR on the allowlist."""
    name = event_cls.__name__
    sites = _emit_sites_for(name)

    if sites:
        # Class is emitted — must NOT be on the allowlist (no stale entries).
        assert name not in KNOWN_UNEMITTED, (
            f"{name} is emitted from {[str(p) for p in sites]} but is "
            f"still listed in KNOWN_UNEMITTED — drop the entry."
        )
        return

    # No emit sites — must be on the explicit allowlist with a reason.
    assert name in KNOWN_UNEMITTED, (
        f"{name} has zero `.emit({name}, ...)` call sites in production "
        f"code. Either wire an emit site, or add it to "
        f"KNOWN_UNEMITTED with a tracking note explaining why it's "
        f"declared but unemitted. Silent telemetry holes are how bug "
        f"#4 (PlanProposed never appearing in plan-execute logs) "
        f"shipped — make the gap visible."
    )


def test_known_unemitted_table_has_no_stale_entries() -> None:
    """Sanity: every name in KNOWN_UNEMITTED must be a real KaosEvent class."""
    real = {c.__name__ for c in _all_concrete_event_classes()}
    for name in KNOWN_UNEMITTED:
        assert name in real, (
            f"KNOWN_UNEMITTED references {name!r} which is not a real "
            f"KaosEvent subclass. Did the class get renamed or removed? "
            f"Drop the stale entry."
        )
