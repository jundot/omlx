# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MCP prerequisite enforcement."""
from __future__ import annotations

import dataclasses

import pytest

from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.mcp.prerequisites import PrerequisiteCheck, PrerequisiteChecker


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


class TestPrerequisiteCheckDataclass:
    def test_satisfied_true(self):
        pc = PrerequisiteCheck(satisfied=True, missing=[])
        assert pc.satisfied is True
        assert pc.missing == []

    def test_not_satisfied_with_missing(self):
        pc = PrerequisiteCheck(satisfied=False, missing=["read_file"])
        assert pc.satisfied is False
        assert "read_file" in pc.missing

    def test_is_frozen(self):
        pc = PrerequisiteCheck(satisfied=True, missing=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.satisfied = False  # type: ignore[misc]


class TestPrerequisiteCheckerConstruction:
    def test_empty_prerequisites(self):
        checker = PrerequisiteChecker({})
        results = checker.check([], [])
        assert results == []

    def test_no_prereq_for_tool(self):
        checker = PrerequisiteChecker({"edit_file": {"requires": ["read_file"]}})
        # search has no declared prereqs — no result emitted for it
        results = checker.check([_tc("search")], [])
        assert results == []


class TestNameOnlyPrerequisite:
    def _checker(self):
        return PrerequisiteChecker({"edit_file": {"requires": ["read_file"]}})

    def test_missing_prereq_flagged(self):
        checker = self._checker()
        results = checker.check([_tc("edit_file")], [])
        assert len(results) == 1
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")

    def test_satisfied_when_prior_call_exists(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check([_tc("edit_file")], prior)
        assert len(results) == 1
        assert results[0].passed is True

    def test_different_prior_tool_does_not_satisfy(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check([_tc("edit_file")], prior)
        assert results[0].passed is False

    def test_multiple_name_only_prereqs(self):
        checker = PrerequisiteChecker(
            {"commit": {"requires": ["read_file", "edit_file"]}}
        )
        results = checker.check([_tc("commit")], [])
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")
        assert "edit_file" in (results[0].detail or "")


class TestArgMatchedPrerequisite:
    def _checker(self):
        return PrerequisiteChecker(
            {
                "edit_file": {
                    "requires": [{"tool": "read_file", "match_arg": "path"}]
                }
            }
        )

    def test_missing_entirely(self):
        checker = self._checker()
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], []
        )
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")

    def test_matching_arg_satisfies(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/foo"}',
                        },
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], prior
        )
        assert results[0].passed is True

    def test_non_matching_arg_does_not_satisfy(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/other"}',
                        },
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], prior
        )
        assert results[0].passed is False

    def test_mixed_name_and_arg_prereqs(self):
        checker = PrerequisiteChecker(
            {
                "deploy": {
                    "requires": [
                        "test",
                        {"tool": "build", "match_arg": "target"},
                    ]
                }
            }
        )
        # Only test was called, build was not
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("deploy", '{"target": "prod"}')], prior
        )
        assert results[0].passed is False
        assert "build" in (results[0].detail or "")
        assert "test" not in (results[0].detail or "")


class TestExecutedToolsFromHistory:
    def test_ignores_user_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        prior = [
            {"role": "user", "content": "edit the file"},
            {
                "role": "user",
                "tool_calls": [{"function": {"name": "read"}}],
            },
        ]
        results = checker.check([_tc("edit")], prior)
        assert results[0].passed is False  # user tool_calls ignored

    def test_ignores_tool_role_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        prior = [
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
            {
                "role": "tool",
                "tool_calls": [{"function": {"name": "read"}}],
            },
        ]
        results = checker.check([_tc("edit")], prior)
        assert results[0].passed is False

    def test_multiple_prior_assistant_messages_accumulate(self):
        checker = PrerequisiteChecker(
            {"commit": {"requires": ["read", "write"]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "write", "arguments": "{}"}}
                ],
            },
        ]
        results = checker.check([_tc("commit")], prior)
        assert results[0].passed is True

    def test_empty_prior_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        results = checker.check([_tc("edit")], [])
        assert results[0].passed is False

    def test_malformed_args_in_history_treated_as_empty(self):
        checker = PrerequisiteChecker(
            {"edit": {"requires": [{"tool": "read", "match_arg": "path"}]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "not-json"}}
                ],
            }
        ]
        results = checker.check(
            [_tc("edit", '{"path": "/x"}')], prior
        )
        assert results[0].passed is False  # malformed args can't match
