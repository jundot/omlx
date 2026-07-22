# SPDX-License-Identifier: Apache-2.0
"""Unit tests for nudge generator functions."""
from omlx.api.guardrails.nudge import (
    missing_params_nudge,
    retry_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
from omlx.api.guardrails.types import (
    KIND_RETRY,
    KIND_TOOL_ARG_VALIDATION,
    KIND_UNKNOWN_TOOL,
)


class TestRetryNudge:
    def test_returns_user_role(self):
        n = retry_nudge()
        assert n.role == "user"

    def test_kind_is_retry(self):
        n = retry_nudge()
        assert n.kind == KIND_RETRY

    def test_content_instructs_tool_call(self):
        n = retry_nudge()
        assert "tool" in n.content.lower()


class TestUnknownToolNudge:
    def test_returns_tool_role(self):
        n = unknown_tool_nudge("bad_tool", ["search", "read"])
        assert n.role == "tool"

    def test_kind_is_unknown_tool(self):
        n = unknown_tool_nudge("bad_tool", ["search"])
        assert n.kind == KIND_UNKNOWN_TOOL

    def test_content_mentions_bad_tool(self):
        n = unknown_tool_nudge("bad_tool", ["search", "read"])
        assert "bad_tool" in n.content

    def test_content_lists_available_tools(self):
        n = unknown_tool_nudge("bad_tool", ["search", "read", "write"])
        assert "search" in n.content
        assert "read" in n.content
        assert "write" in n.content


class TestToolArgValidationNudge:
    def test_returns_tool_role(self):
        n = tool_arg_validation_nudge("search", "some_string", "str")
        assert n.role == "tool"

    def test_kind_is_tool_arg_validation(self):
        n = tool_arg_validation_nudge("search", "xyz", "str")
        assert n.kind == KIND_TOOL_ARG_VALIDATION

    def test_content_mentions_tool_and_type(self):
        n = tool_arg_validation_nudge("search", "xyz", "str")
        assert "search" in n.content
        assert "str" in n.content


class TestMissingParamsNudge:
    def test_returns_tool_role(self):
        n = missing_params_nudge("search", ["query"])
        assert n.role == "tool"

    def test_kind_is_tool_arg_validation(self):
        n = missing_params_nudge("search", ["query"])
        assert n.kind == KIND_TOOL_ARG_VALIDATION

    def test_content_mentions_missing_params(self):
        n = missing_params_nudge("search", ["query", "limit"])
        assert "query" in n.content
        assert "limit" in n.content

    def test_content_mentions_tool_name(self):
        n = missing_params_nudge("search", ["query"])
        assert "search" in n.content
