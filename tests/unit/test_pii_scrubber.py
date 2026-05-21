"""Tests for B0.7 — tool-call argument PII scrubber.

Pre-fix (broad-reliability roadmap §B0.7), an agent that lifted PII
from a corpus document into a tool call (e.g. embedding the user's
client SSN inside ``kaos-web-search(query=...)``) leaked the PII into
third-party logs on the receiving side. The pre-execution gate in
``tool_bridge.executor`` did permission checks but not content
inspection.

Post-fix, every kwargs payload runs through :func:`scrub_tool_args`
before the underlying ``KaosTool.execute()`` invocation. Matched
patterns (SSN / EIN / Luhn-valid credit card) get replaced with a
``SCRUB_MASK`` token; whole-field matches use a labeled
``SCRUB_FIELD_PREFIX`` placeholder so audit trails can distinguish
"PII inside a larger query" from "the entire arg IS PII."

The scrubber is intentionally string-only and conservative: ints /
booleans / unknown types pass through; bare 9-digit runs only match
when surrounded by non-digit context; credit-card candidates must
pass the Luhn check.
"""

from __future__ import annotations

import pytest

from kaos_agents.runtime.pii_scrubber import (
    SCRUB_FIELD_PREFIX,
    SCRUB_MASK,
    ScrubResult,
    scrub_tool_args,
)

# ── Pattern coverage ────────────────────────────────────────────────


class TestSSNDetection:
    """SSN patterns — hyphenated, whole-field, and bare 9-digit."""

    def test_hyphenated_ssn_inside_query_is_masked(self) -> None:
        result = scrub_tool_args({"query": "John Doe 123-45-6789 fraud case"})
        assert result.kwargs["query"] == f"John Doe {SCRUB_MASK} fraud case"
        assert "ssn" in result.matches

    def test_whole_field_ssn_uses_labeled_placeholder(self) -> None:
        """Whole-string SSN → labeled placeholder so the audit trail
        can tell ``ssn=*** `` from ``ssn=SCRUBBED-inside-larger-query``."""
        result = scrub_tool_args({"ssn": "123-45-6789"})
        assert result.kwargs["ssn"].startswith(SCRUB_FIELD_PREFIX)
        assert "ssn" in result.matches

    def test_bare_9_digit_ssn_with_non_digit_context_is_masked(self) -> None:
        result = scrub_tool_args({"q": "applicant 123456789 status"})
        assert result.kwargs["q"] == f"applicant {SCRUB_MASK} status"
        assert "ssn" in result.matches

    def test_bare_9_digit_surrounded_by_digits_does_not_match(self) -> None:
        """``foo:1234567890`` is a 10-digit run; we must not match a
        nested 9-digit substring (false-positive bias)."""
        result = scrub_tool_args({"q": "id 1234567890"})
        assert "ssn" not in result.matches
        assert result.kwargs["q"] == "id 1234567890"

    def test_phone_number_does_not_match_ssn(self) -> None:
        """``(415) 555-1234`` is NOT a SSN pattern; must not fire."""
        result = scrub_tool_args({"phone": "(415) 555-1234"})
        assert "ssn" not in result.matches


class TestEINDetection:
    """EIN: NN-NNNNNNN."""

    def test_ein_is_masked(self) -> None:
        result = scrub_tool_args({"q": "filed by 12-3456789 LLC"})
        assert result.kwargs["q"] == f"filed by {SCRUB_MASK} LLC"
        assert "ein" in result.matches

    def test_ein_no_dash_does_not_match(self) -> None:
        """A 9-digit run without the dash matches SSN, not EIN — but
        the SSN pattern still fires, so PII gets scrubbed either
        way. This is a coverage assertion."""
        result = scrub_tool_args({"q": "filed by 123456789 LLC"})
        # Fires as SSN (bare-9 with non-digit context).
        assert "ssn" in result.matches
        assert "ein" not in result.matches


class TestCreditCardDetection:
    """Credit-card matches require Luhn validity."""

    def test_luhn_valid_credit_card_is_masked(self) -> None:
        """4242 4242 4242 4242 is the Stripe test card — Luhn-valid."""
        result = scrub_tool_args({"q": "charge 4242 4242 4242 4242 declined"})
        assert SCRUB_MASK in result.kwargs["q"]
        assert "credit_card" in result.matches

    def test_luhn_valid_dashed_credit_card_is_masked(self) -> None:
        result = scrub_tool_args({"q": "card 5555-5555-5555-4444 expired"})
        assert SCRUB_MASK in result.kwargs["q"]
        assert "credit_card" in result.matches

    def test_luhn_invalid_run_is_NOT_masked(self) -> None:
        """A 16-digit run that fails Luhn (e.g. invoice / order ID)
        must pass through unchanged. Stops the scrubber false-
        positiving on every long numeric."""
        # 1234 5678 9012 3456 has Luhn checksum 4, not 0.
        result = scrub_tool_args({"q": "invoice 1234 5678 9012 3456 paid"})
        assert "credit_card" not in result.matches
        assert "1234 5678 9012 3456" in result.kwargs["q"]


# ── Nested-structure handling ────────────────────────────────────────


class TestNestedStructureScrubbing:
    """The scrubber recurses through dict / list / tuple containers
    while leaving the shape unchanged. Non-string scalars pass
    through."""

    def test_dict_recursion(self) -> None:
        result = scrub_tool_args(
            {
                "outer": {
                    "inner": "patient 999-12-3456 record",
                },
            }
        )
        assert result.kwargs["outer"]["inner"] == f"patient {SCRUB_MASK} record"
        assert "ssn" in result.matches

    def test_list_recursion(self) -> None:
        result = scrub_tool_args(
            {
                "tags": ["green", "ein 12-3456789", "ready"],
            }
        )
        assert result.kwargs["tags"][1] == f"ein {SCRUB_MASK}"
        assert "ein" in result.matches

    def test_tuple_recursion_preserves_tuple(self) -> None:
        result = scrub_tool_args(
            {
                "pair": ("query", "ssn 123-45-6789"),
            }
        )
        assert isinstance(result.kwargs["pair"], tuple)
        assert result.kwargs["pair"][1] == f"ssn {SCRUB_MASK}"

    def test_non_string_scalars_pass_through(self) -> None:
        result = scrub_tool_args(
            {
                "count": 123_456_789,  # numeric SSN-shape, but int
                "ratio": 0.42,
                "active": True,
                "missing": None,
            }
        )
        assert result.kwargs == {
            "count": 123_456_789,
            "ratio": 0.42,
            "active": True,
            "missing": None,
        }
        assert result.matches == ()


# ── No-op + immutability ────────────────────────────────────────────


class TestNoOpAndImmutability:
    """Clean inputs return matches=() and a fresh dict (no caller
    mutation)."""

    def test_clean_query_passes_through_unchanged(self) -> None:
        original = {"query": "what is the Federal Register filing fee", "limit": 10}
        result = scrub_tool_args(original)
        assert result.kwargs == original
        assert result.matches == ()
        assert result.scrubbed is False

    def test_scrubber_does_not_mutate_caller_kwargs(self) -> None:
        """Returning a fresh kwargs dict is part of the contract —
        the underlying tool's caller must not see a mutated input."""
        original = {"query": "SSN 123-45-6789"}
        original_snapshot = dict(original)
        scrub_tool_args(original)
        assert original == original_snapshot

    def test_multiple_pattern_kinds_in_one_payload(self) -> None:
        """Mixed PII payload — all matched patterns appear in
        ``result.matches``."""
        result = scrub_tool_args(
            {
                "q1": "SSN 123-45-6789",
                "q2": "EIN 12-3456789",
                "q3": "card 4242 4242 4242 4242",
            }
        )
        assert set(result.matches) == {"ssn", "ein", "credit_card"}


# ── ScrubResult shape ────────────────────────────────────────────────


def test_scrub_result_scrubbed_property() -> None:
    assert scrub_tool_args({"x": "no pii here"}).scrubbed is False
    assert scrub_tool_args({"x": "ssn 111-22-3333"}).scrubbed is True


def test_scrub_result_is_frozen() -> None:
    """``ScrubResult`` is a frozen value type — protects audit trails
    that might keep a reference past the executor's return."""
    r = ScrubResult(kwargs={"x": 1}, matches=("ssn",))
    # ty correctly flags ``r.matches = ()`` at static-analysis time;
    # going through setattr keeps the test ty-clean while still
    # exercising the frozen-dataclass guarantee at runtime.
    with pytest.raises((AttributeError, TypeError)):
        setattr(r, "matches", ())  # noqa: B010 — testing frozen guard
