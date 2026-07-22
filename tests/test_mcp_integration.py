# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MCP forge features wiring.

Exercises the full pipeline without a running model engine:
  - PrerequisiteChecker -> apply_guardrails -> ValidationResult
  - inject_respond_tool / strip_respond_calls round-trip
  - Server source inspection (wiring of prereq checker + respond tool)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from omlx.api.guardrail_wiring import apply_guardrails
from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.respond import (
    RESPOND_TOOL_NAME,
    inject_respond_tool,
    strip_respond_calls,
)
from omlx.api.tool_calling import ToolCallExtraction
from omlx.mcp.prerequisites import PrerequisiteChecker

SERVER_PATH = Path(__file__).resolve().parent.parent / "omlx" / "server.py"


def _server_source() -> str:
    return SERVER_PATH.read_text()


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


def _extraction(calls=None, vr=None) -> ToolCallExtraction:
    return ToolCallExtraction(
        cleaned_text="",
        tool_calls=calls,
        cleaned_thinking="",
        validation_result=vr or ValidationResult(checks=[], passed=True),
    )


class TestPrerequisiteEndToEnd:
    def test_edit_without_read_flagged(self):
        checker = PrerequisiteChecker(
            {"edit_file": {"requires": ["read_file"]}}
        )
        ext = _extraction([_tc("edit_file", '{"path": "/x"}')])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        assert result.validation_result.passed is False
        prereq_checks = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ]
        assert len(prereq_checks) == 1
        assert prereq_checks[0].passed is False

    def test_edit_after_read_passes(self):
        checker = PrerequisiteChecker(
            {"edit_file": {"requires": ["read_file"]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": "{}"}}
                ],
            }
        ]
        ext = _extraction([_tc("edit_file")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=prior,
        )
        assert result.validation_result.passed is True

    def test_arg_matched_e2e(self):
        checker = PrerequisiteChecker(
            {
                "edit_file": {
                    "requires": [{"tool": "read_file", "match_arg": "path"}]
                }
            }
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/a"}',
                        }
                    }
                ],
            }
        ]
        ext = _extraction([_tc("edit_file", '{"path": "/b"}')])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=prior,
        )
        assert result.validation_result.passed is False

    def test_no_prereqs_declared_is_noop(self):
        checker = PrerequisiteChecker({})
        ext = _extraction([_tc("any_tool")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        prereq_checks = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ]
        assert prereq_checks == []

    def test_no_checker_is_noop(self):
        ext = _extraction([_tc("edit_file")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=None,
            prior_messages=None,
        )
        prereq_checks = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ]
        assert prereq_checks == []

    def test_prereq_check_detail_carries_missing_tool(self):
        checker = PrerequisiteChecker(
            {"deploy": {"requires": ["build", "test"]}}
        )
        ext = _extraction([_tc("deploy")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        prereq = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ][0]
        assert "build" in (prereq.detail or "")
        assert "test" in (prereq.detail or "")


class TestRespondToolEndToEnd:
    def test_inject_then_strip_pure_respond(self):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)
        assert any(
            t["function"]["name"] == RESPOND_TOOL_NAME for t in injected
        )

        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Done!"}'
                ),
            )
        ]
        real, text = strip_respond_calls(calls)
        assert real == []
        assert text == "Done!"

    def test_inject_then_strip_mixed(self):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)

        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Searching..."}'
                ),
            ),
            ToolCall(
                id="call_2",
                type="function",
                function=FunctionCall(
                    name="search", arguments='{"query": "test"}'
                ),
            ),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 1
        assert real[0].function.name == "search"
        assert text is None

    def test_full_pipeline_respond_becomes_text(self):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)

        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Hello!"}'
                ),
            )
        ]
        real_calls, text = strip_respond_calls(calls)
        assert real_calls == []
        assert text == "Hello!"

    def test_inject_preserves_original_tool_count_plus_one(self):
        tools = [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
        ]
        injected = inject_respond_tool(tools)
        assert len(injected) == 3


class TestRespondAndGuardrailPipeline:
    def test_respond_passes_validation_then_strips_to_text(self):
        tools = [
            {"type": "function", "function": {"name": "respond", "parameters": {}}},
        ]
        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Hi"}'
                ),
            )
        ]
        ext = _extraction(calls)
        result = apply_guardrails(
            ext, "auto", tools,
            validation_enabled=True,
        )
        assert result.validation_result.passed is True
        real, text = strip_respond_calls(result.tool_calls)
        assert real == []
        assert text == "Hi"


class TestPrerequisiteServerWiring:
    def test_prerequisite_checker_imported(self):
        src = _server_source()
        assert "PrerequisiteChecker" in src

    def test_enforce_mcp_prerequisites_referenced(self):
        src = _server_source()
        assert "enforce_mcp_prerequisites" in src

    def test_prerequisite_checker_passed_to_apply_guardrails(self):
        src = _server_source()
        lines = [
            line for line in src.splitlines()
            if "prerequisite_checker=" in line
        ]
        assert len(lines) >= 1, "prerequisite_checker not passed to apply_guardrails"

    def test_prior_messages_passed_to_apply_guardrails(self):
        src = _server_source()
        lines = [
            line for line in src.splitlines()
            if "prior_messages=" in line
        ]
        assert len(lines) >= 1, "prior_messages not passed to apply_guardrails"


class TestRespondToolServerWiring:
    def test_inject_respond_tool_imported(self):
        src = _server_source()
        assert "inject_respond_tool" in src

    def test_strip_respond_calls_imported(self):
        src = _server_source()
        assert "strip_respond_calls" in src

    def test_inject_respond_tool_called(self):
        src = _server_source()
        call_lines = [
            line for line in src.splitlines()
            if "inject_respond_tool(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 1, "inject_respond_tool not called"

    def test_strip_respond_calls_called(self):
        src = _server_source()
        call_lines = [
            line for line in src.splitlines()
            if "strip_respond_calls(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 1, "strip_respond_calls not called"

    def test_inject_respond_tool_setting_referenced(self):
        src = _server_source()
        assert "inject_respond_tool" in src
