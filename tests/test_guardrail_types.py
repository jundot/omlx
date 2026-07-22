# SPDX-License-Identifier: Apache-2.0
"""Unit tests for guardrails type definitions."""
from omlx.api.guardrails.types import (
    CheckResult,
    Nudge,
    ValidationResult,
    KIND_RETRY,
    KIND_UNKNOWN_TOOL,
    KIND_TOOL_ARG_VALIDATION,
    TOOL_CHANNEL_KINDS,
    TOOL_ERROR_KINDS,
)


class TestCheckResult:
    def test_basic_creation(self):
        cr = CheckResult(check="unknown_tool", passed=False, detail="bad tool")
        assert cr.check == "unknown_tool"
        assert cr.passed is False
        assert cr.detail == "bad tool"

    def test_passed_check_no_detail(self):
        cr = CheckResult(check="bare_text", passed=True)
        assert cr.detail is None

    def test_is_frozen(self):
        cr = CheckResult(check="bare_text", passed=True)
        try:
            cr.passed = False  # type: ignore
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # frozen dataclass raises AttributeError in 3.10


class TestNudge:
    def test_user_role_nudge(self):
        n = Nudge(role="user", content="try again", kind=KIND_RETRY)
        assert n.role == "user"
        assert n.kind == KIND_RETRY

    def test_tool_role_nudge(self):
        n = Nudge(role="tool", content="bad tool", kind=KIND_UNKNOWN_TOOL)
        assert n.role == "tool"

    def test_to_message(self):
        n = Nudge(role="tool", content="fix args", kind=KIND_TOOL_ARG_VALIDATION)
        msg = n.to_message()
        assert msg == {"role": "tool", "content": "fix args"}

    def test_is_frozen(self):
        n = Nudge(role="user", content="x", kind=KIND_RETRY)
        try:
            n.content = "y"  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestValidationResult:
    def test_passed_result(self):
        checks = [CheckResult(check="bare_text", passed=True)]
        vr = ValidationResult(checks=checks, nudge=None, passed=True)
        assert vr.passed is True
        assert vr.nudge is None

    def test_failed_result_with_nudge(self):
        checks = [CheckResult(check="unknown_tool", passed=False, detail="x")]
        nudge = Nudge(role="tool", content="retry", kind=KIND_UNKNOWN_TOOL)
        vr = ValidationResult(checks=checks, nudge=nudge, passed=False)
        assert vr.passed is False

    def test_to_dict_passed(self):
        checks = [CheckResult(check="bare_text", passed=True)]
        vr = ValidationResult(checks=checks, passed=True)
        d = vr.to_dict()
        assert d["passed"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0] == {"check": "bare_text", "passed": True, "detail": None}
        assert "nudge" not in d

    def test_to_dict_failed_with_nudge(self):
        checks = [
            CheckResult(check="unknown_tool", passed=False, detail="bad"),
        ]
        nudge = Nudge(role="tool", content="retry now", kind=KIND_UNKNOWN_TOOL)
        vr = ValidationResult(checks=checks, nudge=nudge, passed=False)
        d = vr.to_dict()
        assert d["passed"] is False
        assert d["nudge"] == {
            "role": "tool",
            "content": "retry now",
            "kind": KIND_UNKNOWN_TOOL,
            "tier": 0,
        }


class TestConstants:
    def test_kind_values(self):
        assert KIND_RETRY == "retry"
        assert KIND_UNKNOWN_TOOL == "unknown_tool"
        assert KIND_TOOL_ARG_VALIDATION == "tool_arg_validation"

    def test_tool_channel_kinds(self):
        assert KIND_UNKNOWN_TOOL in TOOL_CHANNEL_KINDS
        assert KIND_TOOL_ARG_VALIDATION in TOOL_CHANNEL_KINDS
        assert KIND_RETRY not in TOOL_CHANNEL_KINDS

    def test_tool_error_kinds(self):
        assert KIND_UNKNOWN_TOOL in TOOL_ERROR_KINDS
        assert KIND_TOOL_ARG_VALIDATION in TOOL_ERROR_KINDS
