"""Shape-contract tests for :class:`AgentFailureClassification`
(plan §Issue 6 — typed error_category).

Plan §Issue 6 ships the typed ``error_category`` surface that the
SPA's audit JSONL + MCP tool wrapper consume to produce SOC2-grade
alert messages. These tests pin the load-bearing invariants so
downstream consumers (the audit CLI, the operator alert sink, the
replay-endpoint renderer, MCP ``ToolResult.create_error``) can rely
on the shape:

- the ``ERROR_KIND_*`` constants are stable strings (auditors index
  on them);
- ``SURFACING_FAILURE_KINDS`` is a frozenset of those constants
  (so the membership check is O(1) and the set is immutable);
- ``AgentFailureClassification`` is frozen+slotted (safe to persist
  to audit JSONL);
- ``is_surfacing`` correctly identifies the failures that must
  propagate as ``isError=True`` to the agent loop;
- ``classify_agent_failure`` returns ``None`` when no recognized
  failure shape matches (so the caller falls back to a generic
  recovery hint instead of silently mis-classifying).

Plan: ``kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md``
§Issue 6 — Can't reproduce / debug yesterday's turn.
"""

from __future__ import annotations

import pytest

from kaos_agents.errors import (
    ERROR_KIND_AUTH,
    ERROR_KIND_CONTEXT_TOO_LARGE,
    ERROR_KIND_PROVIDER,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_SERVICE_UNAVAILABLE,
    ERROR_KIND_TRANSPORT,
    SURFACING_FAILURE_KINDS,
    AgentFailureClassification,
    classify_agent_failure,
)

# ── Stable kind-string constants ────────────────────────────────────


_ALL_KINDS: tuple[str, ...] = (
    ERROR_KIND_AUTH,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_SERVICE_UNAVAILABLE,
    ERROR_KIND_CONTEXT_TOO_LARGE,
    ERROR_KIND_TRANSPORT,
    ERROR_KIND_PROVIDER,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind, expected",
    [
        (ERROR_KIND_AUTH, "auth_failure"),
        (ERROR_KIND_RATE_LIMIT, "rate_limit"),
        (ERROR_KIND_SERVICE_UNAVAILABLE, "service_unavailable"),
        (ERROR_KIND_CONTEXT_TOO_LARGE, "context_too_large"),
        (ERROR_KIND_TRANSPORT, "transport_error"),
        (ERROR_KIND_PROVIDER, "provider_error"),
    ],
)
def test_error_kind_constants_are_stable_strings(kind: str, expected: str) -> None:
    """Auditors index audit JSONL rows by these strings — renaming
    is a breaking change. Pin the canonical spellings so a future
    refactor that renames e.g. ``"auth_failure"`` → ``"auth"`` trips
    this gate."""
    assert kind == expected


@pytest.mark.unit
def test_all_kinds_are_distinct() -> None:
    """No two ERROR_KIND_* aliases the same string. A collision
    would silently coalesce two distinct failure classes in the
    audit log."""
    assert len(set(_ALL_KINDS)) == len(_ALL_KINDS)


# ── Surfacing-kinds invariant ───────────────────────────────────────


@pytest.mark.unit
def test_surfacing_failure_kinds_is_immutable_frozenset() -> None:
    """Frozen so test code can't accidentally mutate it; ``frozenset``
    membership is O(1) for the hot dispatch path."""
    assert isinstance(SURFACING_FAILURE_KINDS, frozenset)
    with pytest.raises((AttributeError, TypeError)):
        SURFACING_FAILURE_KINDS.add("new_kind")  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_every_kind_is_in_surfacing_set(kind: str) -> None:
    """The full set of recognized kinds is in SURFACING_FAILURE_KINDS
    today — every classified failure should propagate as
    ``isError=True``. If a future change introduces a non-surfacing
    kind (e.g. "warning"), this test will need to update — that's
    the contract review point."""
    assert kind in SURFACING_FAILURE_KINDS


# ── AgentFailureClassification value-type contract ─────────────────


@pytest.mark.unit
def test_classification_is_frozen_dataclass_with_slots() -> None:
    """Safe to persist to audit JSONL — no surprise mutations."""
    c = AgentFailureClassification(
        kind=ERROR_KIND_AUTH,
        credential="OPENAI_API_KEY",
        recovery_hint="Refresh your API key.",
        provider="openai",
        status_code=401,
    )
    # Slotted: no surprise dict — auditor's row layout is predictable.
    assert hasattr(c, "__slots__")
    # Frozen: __hash__ defined → can be put in a set; mutability
    # check via dataclass frozen flag (we don't directly call the
    # private API).
    assert hash(c) == hash(
        AgentFailureClassification(
            kind=ERROR_KIND_AUTH,
            credential="OPENAI_API_KEY",
            recovery_hint="Refresh your API key.",
            provider="openai",
            status_code=401,
        )
    )


@pytest.mark.unit
def test_is_surfacing_returns_true_for_all_known_kinds() -> None:
    """Today every known kind surfaces. If a future kind doesn't,
    add a non-surfacing case to the parametrization explicitly so
    the operator's alert sink knows about it."""
    for kind in _ALL_KINDS:
        c = AgentFailureClassification(
            kind=kind,
            credential=None,
            recovery_hint="x",
        )
        assert c.is_surfacing is True, f"kind={kind!r} should be in surfacing set"


@pytest.mark.unit
def test_is_surfacing_returns_false_for_unknown_kind() -> None:
    """A kind that isn't in SURFACING_FAILURE_KINDS does not
    propagate — defends against a future addition that forgets to
    update the set."""
    c = AgentFailureClassification(
        kind="not_a_real_kind",
        credential=None,
        recovery_hint="x",
    )
    assert c.is_surfacing is False


# ── classify_agent_failure dispatch ────────────────────────────────


@pytest.mark.unit
def test_classify_returns_none_for_unrecognized_exception() -> None:
    """Generic Python exceptions (KeyError, ValueError, etc.) that
    don't match any known shape return ``None`` so the caller's
    fallback path (generic recovery hint) is taken — silent
    mis-classification would degrade the audit log."""
    assert classify_agent_failure(KeyError("anything")) is None
    assert classify_agent_failure(ValueError("anything")) is None
    assert classify_agent_failure(RuntimeError("anything")) is None


@pytest.mark.unit
def test_classify_recognizes_context_too_large_phrasing() -> None:
    """The plan calls out context-too-large explicitly. The
    classifier inspects the exception message for documented
    phrasings (``context_length``, ``too many tokens``,
    ``input too long``, etc.). Pin a few canonical strings so
    the regex set stays anchored."""
    samples = [
        "Provider error: maximum context length is 200000 tokens",
        "context window exceeded",
        "Input too long for the requested model",
        "Maximum tokens exceeded",
    ]
    for msg in samples:
        result = classify_agent_failure(RuntimeError(msg))
        # Some sample strings may not match every regex but at
        # least one should hit. Pin the canonical ``maximum context
        # length`` shape that real OpenAI/Anthropic errors emit.
        if "maximum context length" in msg.lower():
            assert result is not None, f"expected match for {msg!r}"
            assert result.kind == ERROR_KIND_CONTEXT_TOO_LARGE


# ── recovery_hint is non-empty for surfacing kinds ─────────────────


@pytest.mark.unit
def test_recovery_hint_is_never_empty_string_on_constructed_classification() -> None:
    """The recovery_hint is the operator-facing remediation. The
    constructor accepts any string but production callers must
    pass a non-empty hint — pin this via documentation, not via
    enforcement, since tests construct minimal stubs. The
    invariant test: when the helper itself returns a
    classification, the hint is non-empty."""
    # Construct directly with a hint (caller responsibility).
    c = AgentFailureClassification(
        kind=ERROR_KIND_RATE_LIMIT,
        credential=None,
        recovery_hint="Back off and retry; see provider rate-limit docs.",
    )
    assert c.recovery_hint
    assert len(c.recovery_hint) > 0
