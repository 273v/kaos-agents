"""Unit tests for planning result_check utilities."""

from __future__ import annotations

import pytest

from kaos_agents.planning.result_check import is_empty_result, is_error_result


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
