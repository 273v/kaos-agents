"""Unit tests for planning result_check utilities."""

from __future__ import annotations

import re

import pytest

from kaos_agents.planning.result_check import (
    is_empty_result,
    is_error_result,
    is_uninformative_result,
)


@pytest.mark.unit
class TestIsErrorResult:
    def test_error_prefix(self):
        assert is_error_result("ERROR: tool failed") is True

    def test_json_error(self):
        assert is_error_result('{"error": "something went wrong"}') is True

    def test_normal_text(self):
        assert is_error_result("40 CFR Part 60: Standards of Performance") is False

    def test_empty_string(self):
        assert is_error_result("") is False

    def test_error_not_at_start(self):
        assert is_error_result("Some text ERROR: not at start") is False

    def test_json_error_not_at_start(self):
        assert is_error_result('result: {"error": "nested"}') is False


@pytest.mark.unit
class TestIsEmptyResult:
    def test_empty(self):
        assert is_empty_result("") is True

    def test_whitespace(self):
        assert is_empty_result("   \t\n  ") is True

    def test_non_empty(self):
        assert is_empty_result("some result") is False

    def test_single_char(self):
        assert is_empty_result("x") is False


@pytest.mark.unit
class TestIsUninformativeResult:
    """The generic 'no usable signal' predicate.

    Empirical anchor: session ``01KS2DEBYT341F1F16B3BRQRV0`` had 12
    consecutive ``kaos-web-search`` calls with body
    ``"No results found for: <query>"``, each ``is_error=False``. Any
    pattern set we ship has to catch THAT exact phrasing, plus the
    JSON empty-list / zero-count shapes the kaos-content surface
    emits, without false-positiving on legitimate informative results.
    """

    # -- empty / whitespace --

    def test_empty_string_is_uninformative(self):
        assert is_uninformative_result("") is True

    def test_whitespace_only_is_uninformative(self):
        assert is_uninformative_result("   \n  ") is True

    # -- prose phrases --

    def test_kaos_web_no_results_found(self):
        """The exact phrasing from session DEB."""
        assert (
            is_uninformative_result("No results found for: Federal Reserve federal funds rate")
            is True
        )

    def test_no_result_singular(self):
        assert is_uninformative_result("Search returned no result.") is True

    def test_no_matches(self):
        assert is_uninformative_result("No matches in the corpus.") is True

    def test_no_match_singular(self):
        assert is_uninformative_result("No match found for query 'X'.") is True

    def test_no_hits(self):
        assert is_uninformative_result("No hits returned.") is True

    def test_zero_results_with_digit(self):
        assert is_uninformative_result("Found 0 results matching the query.") is True

    def test_zero_hits_with_digit(self):
        assert is_uninformative_result("Returned 0 hits for the query.") is True

    # -- JSON shapes --

    def test_json_empty_results_list(self):
        assert (
            is_uninformative_result('{"results": [], "total_matches": 0, "has_more": false}')
            is True
        )

    def test_json_empty_hits(self):
        assert is_uninformative_result('{"hits": []}') is True

    def test_json_empty_matches(self):
        assert is_uninformative_result('{"matches": []}') is True

    def test_json_empty_items(self):
        assert is_uninformative_result('{"items": []}') is True

    def test_json_zero_count(self):
        assert is_uninformative_result('{"count": 0, "data": null}') is True

    def test_json_zero_total(self):
        assert is_uninformative_result('{"total": 0}') is True

    def test_json_zero_total_matches(self):
        assert is_uninformative_result('{"total_matches": 0, "results": []}') is True

    def test_bare_empty_array(self):
        assert is_uninformative_result("[]") is True

    def test_bare_empty_array_with_whitespace(self):
        assert is_uninformative_result("  [ ]  \n") is True

    # -- negative cases (must NOT fire) --

    def test_informative_search_result(self):
        assert (
            is_uninformative_result(
                'Found 18 matches for "FOMC 2026 calendar" on federalreserve.gov'
            )
            is False
        )

    def test_count_with_leading_digits_does_not_match_zero_pattern(self):
        # "1230 results" has 30 right before the space; the regex
        # anchors on `\b 0 \b` which fails because 0 is preceded by 3.
        assert is_uninformative_result("Returned 1230 results.") is False

    def test_total_matches_nonzero(self):
        assert is_uninformative_result('{"total_matches": 18, "results": [...]}') is False

    def test_results_with_one_item(self):
        assert is_uninformative_result('{"results": [{"id": "x"}], "total_matches": 1}') is False

    def test_word_no_not_at_phrase_boundary(self):
        # "no" inside a longer word like "noteworthy" must not trip
        # the predicate.
        assert is_uninformative_result("Noteworthy findings included...") is False

    def test_normal_prose(self):
        assert (
            is_uninformative_result(
                "The Federal Reserve raised the target range to 5.25-5.50% in 2024."
            )
            is False
        )

    # -- error path is owned by is_error_result --

    def test_error_prefix_is_not_uninformative(self):
        """``is_uninformative_result`` defers errors to ``is_error_result``."""
        assert is_uninformative_result("ERROR: tool failed") is False

    def test_json_error_is_not_uninformative(self):
        assert is_uninformative_result('{"error": "something went wrong"}') is False

    # -- caller-supplied extra patterns --

    def test_extra_pattern_extends_defaults(self):
        custom = (re.compile(r"\bunavailable\b", re.IGNORECASE),)
        assert (
            is_uninformative_result("Service temporarily UNAVAILABLE.", extra_patterns=custom)
            is True
        )
        # Default still fires too.
        assert is_uninformative_result("No results found.", extra_patterns=custom) is True

    def test_extra_pattern_does_not_falsify_negative(self):
        custom = (re.compile(r"\bunavailable\b", re.IGNORECASE),)
        # Neither default nor extra pattern fires.
        assert (
            is_uninformative_result(
                "All systems nominal — returning live data.", extra_patterns=custom
            )
            is False
        )
