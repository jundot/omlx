# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from omlx.api.guardrail_wiring import (
    apply_guardrails,
    guardrail_validation_payload,
)
from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.tool_calling import ToolCallExtraction

SERVER_PATH = Path(__file__).resolve().parent.parent / "omlx" / "server.py"


def _server_source() -> str:
    return SERVER_PATH.read_text()


def _mk_extraction(
    cleaned_text: str = "",
    tool_calls=None,
    validation_result: ValidationResult | None = None,
) -> ToolCallExtraction:
    return ToolCallExtraction(
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        cleaned_thinking="",
        tool_calls_from_thinking=False,
        validation_result=validation_result,
    )


class TestApplyGuardrailsPassthrough:
    def test_passthrough_when_validation_disabled(self):
        ext = _mk_extraction(cleaned_text="hi", validation_result=None)
        result = apply_guardrails(ext, "auto", None, validation_enabled=False)
        assert result is ext

    def test_passthrough_when_no_validation_result(self):
        vr = ValidationResult(checks=[], passed=True)
        ext = _mk_extraction(validation_result=vr)
        result = apply_guardrails(ext, "auto", None, validation_enabled=False)
        assert result is ext


class TestApplyGuardrailsEnforcement:
    def test_tool_choice_required_fails_when_no_calls(self):
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _mk_extraction(cleaned_text="hello", validation_result=vr)
        result = apply_guardrails(ext, "required", [], validation_enabled=True)
        check_names = [c.check for c in result.validation_result.checks]
        assert "tool_choice_enforcement" in check_names
        assert result.validation_result.passed is False

    def test_tool_choice_auto_passes(self):
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _mk_extraction(cleaned_text="hello", validation_result=vr)
        result = apply_guardrails(ext, "auto", [], validation_enabled=True)
        assert result.validation_result.passed is True

    def test_merged_check_preserves_existing(self):
        vr = ValidationResult(
            checks=[CheckResult(check="unknown_tool", passed=False)],
            nudge=None,
            passed=False,
        )
        ext = _mk_extraction(validation_result=vr)
        result = apply_guardrails(ext, "auto", None, validation_enabled=True)
        check_names = [c.check for c in result.validation_result.checks]
        assert "unknown_tool" in check_names
        assert "tool_choice_enforcement" in check_names
        assert result.validation_result.passed is False

    def test_preserves_tool_calls_and_text(self):
        vr = ValidationResult(checks=[], passed=True)
        ext = _mk_extraction(
            cleaned_text="output", validation_result=vr
        )
        result = apply_guardrails(ext, "auto", None, validation_enabled=True)
        assert result.cleaned_text == "output"
        assert result.tool_calls is None


class TestGuardrailValidationPayload:
    def test_returns_none_when_disabled(self):
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _mk_extraction(validation_result=vr)
        assert (
            guardrail_validation_payload(ext, include_validation_metadata=False)
            is None
        )

    def test_returns_dict_when_enabled(self):
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _mk_extraction(validation_result=vr)
        payload = guardrail_validation_payload(ext, include_validation_metadata=True)
        assert payload is not None
        assert "x_omlx_validation" in payload
        assert payload["x_omlx_validation"]["passed"] is True

    def test_returns_none_when_no_validation_result(self):
        ext = _mk_extraction(validation_result=None)
        assert (
            guardrail_validation_payload(ext, include_validation_metadata=True)
            is None
        )

    def test_includes_failed_check_detail(self):
        vr = ValidationResult(
            checks=[
                CheckResult(
                    check="unknown_tool",
                    passed=False,
                    detail="tool 'foo' not in registry",
                )
            ],
            passed=False,
        )
        ext = _mk_extraction(validation_result=vr)
        payload = guardrail_validation_payload(ext, include_validation_metadata=True)
        checks = payload["x_omlx_validation"]["checks"]
        assert any(c["check"] == "unknown_tool" and c["passed"] is False for c in checks)


class TestServerWiring:
    def test_extract_and_validate_tool_calls_imported(self):
        src = _server_source()
        assert "extract_and_validate_tool_calls" in src

    def test_apply_guardrails_imported(self):
        src = _server_source()
        assert "apply_guardrails" in src

    def test_guardrail_validation_payload_imported(self):
        src = _server_source()
        assert "guardrail_validation_payload" in src

    def test_no_bare_extract_tool_calls_with_thinking_calls_remain(self):
        src = _server_source()
        body_calls = [
            line.strip()
            for line in src.splitlines()
            if "extract_tool_calls_with_thinking(" in line
            and "def " not in line
            and "import" not in line
            and not line.strip().startswith("def ")
            and not line.strip().startswith("from ")
        ]
        assert body_calls == [], f"Remaining bare calls: {body_calls}"

    def test_wrapper_called_at_least_6_times(self):
        src = _server_source()
        call_lines = [
            line
            for line in src.splitlines()
            if "extract_and_validate_tool_calls(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 6, f"Expected >=6, found {len(call_lines)}"

    def test_guardrail_validation_payload_called(self):
        src = _server_source()
        call_lines = [
            line
            for line in src.splitlines()
            if "guardrail_validation_payload(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 1, "guardrail_validation_payload not called"

    def test_include_validation_metadata_referenced(self):
        src = _server_source()
        assert "include_validation_metadata" in src

    def test_apply_guardrails_called_at_sites(self):
        src = _server_source()
        call_lines = [
            line
            for line in src.splitlines()
            if "apply_guardrails(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 6, f"Expected >=6, found {len(call_lines)}"


class TestEndToEnd:
    def test_validation_metadata_in_response(self):
        pytest.skip("Requires running engine; covered in E2E tests")

    def test_no_validation_metadata_when_disabled(self):
        pytest.skip("Requires running engine; covered in E2E tests")
