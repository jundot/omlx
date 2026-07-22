# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ErrorBudget dataclass."""
import pytest
from omlx.api.guardrails.budget import ErrorBudget


class TestErrorBudgetDefaults:
    def test_default_max_retries(self):
        b = ErrorBudget()
        assert b.max_retries == 3

    def test_default_max_tool_errors(self):
        b = ErrorBudget()
        assert b.max_tool_errors == 2

    def test_custom_values(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        assert b.max_retries == 5
        assert b.max_tool_errors == 1

    def test_is_frozen(self):
        b = ErrorBudget()
        with pytest.raises(AttributeError):
            b.max_retries = 10  # type: ignore


class TestShouldRetry:
    def test_retry_when_under_both_limits(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=1, tool_error_count=0) is True

    def test_retry_at_exact_limit(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=3, tool_error_count=2) is True

    def test_no_retry_when_retries_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=4, tool_error_count=0) is False

    def test_no_retry_when_tool_errors_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=0, tool_error_count=3) is False


class TestRecommendedAction:
    def test_retry_when_under_limits(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=1, tool_error_count=0) == "retry"

    def test_give_up_when_retries_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=4, tool_error_count=0) == "give_up"

    def test_give_up_when_tool_errors_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=0, tool_error_count=3) == "give_up"


class TestSerialization:
    def test_to_dict(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        d = b.to_dict()
        assert d == {
            "max_retries": 5,
            "max_tool_errors": 1,
            "recommended_action": "retry",
        }

    def test_to_dict_defaults(self):
        b = ErrorBudget()
        d = b.to_dict()
        assert d["max_retries"] == 3
        assert d["max_tool_errors"] == 2
        assert d["recommended_action"] == "retry"

    def test_from_dict(self):
        d = {"max_retries": 7, "max_tool_errors": 3}
        b = ErrorBudget.from_dict(d)
        assert b.max_retries == 7
        assert b.max_tool_errors == 3

    def test_from_dict_missing_keys_uses_defaults(self):
        b = ErrorBudget.from_dict({})
        assert b.max_retries == 3
        assert b.max_tool_errors == 2

    def test_round_trip(self):
        original = ErrorBudget(max_retries=10, max_tool_errors=5)
        restored = ErrorBudget.from_dict(original.to_dict())
        assert restored.max_retries == original.max_retries
        assert restored.max_tool_errors == original.max_tool_errors
