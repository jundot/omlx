# SPDX-License-Identifier: Apache-2.0
"""Tests for ErrorBudget wiring through the validation pipeline.

Covers the end-to-end flow:
  GuardrailValidator(budget=...) -> ValidationResult.budget
  -> apply_guardrails carries budget forward
  -> guardrail_validation_payload serializes budget into x_omlx_validation
"""
from __future__ import annotations

from omlx.api.guardrail_wiring import (
    apply_guardrails,
    guardrail_validation_payload,
)
from omlx.api.guardrails.budget import ErrorBudget
from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.guardrails.validator import GuardrailValidator
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.tool_calling import ToolCallExtraction

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _make_extraction(
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


def _make_tool_call(name="search", arguments='{"query": "test"}', call_id="c1"):
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# GuardrailValidator budget attachment
# ---------------------------------------------------------------------------


class TestValidatorBudgetAttachment:
    def test_validator_without_budget_has_none(self):
        """Default validator produces ValidationResult.budget is None."""
        v = GuardrailValidator([SEARCH_TOOL])
        ext = _make_extraction(tool_calls=[_make_tool_call()])
        result = v.validate(ext, has_tools=True)
        assert result.budget is None

    def test_validator_with_budget_attaches_on_pass(self):
        """Budget is attached even when validation passes."""
        budget = ErrorBudget(max_retries=5, max_tool_errors=1)
        v = GuardrailValidator([SEARCH_TOOL], budget=budget)
        ext = _make_extraction(tool_calls=[_make_tool_call(arguments='{"query": "x"}')])
        result = v.validate(ext, has_tools=True)
        assert result.passed is True
        assert result.budget is budget

    def test_validator_with_budget_attaches_on_failure(self):
        """Budget is attached when validation fails (nudge selected)."""
        budget = ErrorBudget(max_retries=3, max_tool_errors=2)
        v = GuardrailValidator([SEARCH_TOOL], budget=budget)
        ext = _make_extraction(cleaned_text="I cannot help", tool_calls=None)
        result = v.validate(ext, has_tools=True)
        assert result.passed is False
        assert result.nudge is not None
        assert result.budget is budget

    def test_validator_default_budget_is_none_keyword(self):
        """Explicit budget=None keeps backward compatibility."""
        v = GuardrailValidator([SEARCH_TOOL], budget=None)
        ext = _make_extraction(tool_calls=[_make_tool_call()])
        result = v.validate(ext, has_tools=True)
        assert result.budget is None

    def test_validator_budget_is_keyword_only(self):
        """budget must be keyword-only, not positional."""
        budget = ErrorBudget()
        try:
            GuardrailValidator([SEARCH_TOOL], budget)  # type: ignore[misc]
        except TypeError:
            pass

    def test_validator_budget_uses_defaults(self):
        """Default ErrorBudget() when constructed inline."""
        v = GuardrailValidator([SEARCH_TOOL], budget=ErrorBudget())
        ext = _make_extraction(tool_calls=[_make_tool_call()])
        result = v.validate(ext, has_tools=True)
        assert result.budget is not None
        assert result.budget.max_retries == 3
        assert result.budget.max_tool_errors == 2


# ---------------------------------------------------------------------------
# apply_guardrails budget carry-forward
# ---------------------------------------------------------------------------


class TestApplyGuardrailsBudgetCarryForward:
    def test_budget_carried_forward_through_merge(self):
        """apply_guardrails preserves existing.budget in merged result."""
        budget = ErrorBudget(max_retries=5, max_tool_errors=1)
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
            budget=budget,
        )
        ext = _make_extraction(cleaned_text="hi", validation_result=vr)
        result = apply_guardrails(ext, "auto", None, validation_enabled=True)
        assert result.validation_result.budget is budget

    def test_budget_none_carried_forward(self):
        """When existing budget is None, merged budget stays None."""
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _make_extraction(validation_result=vr)
        result = apply_guardrails(ext, "auto", None, validation_enabled=True)
        assert result.validation_result.budget is None

    def test_budget_preserved_with_failed_merge(self):
        """Budget survives even when tool_choice enforcement fails."""
        budget = ErrorBudget(max_retries=10, max_tool_errors=5)
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
            budget=budget,
        )
        ext = _make_extraction(cleaned_text="hi", validation_result=vr)
        result = apply_guardrails(ext, "required", [], validation_enabled=True)
        assert result.validation_result.passed is False
        assert result.validation_result.budget is budget


# ---------------------------------------------------------------------------
# guardrail_validation_payload budget serialization
# ---------------------------------------------------------------------------


class TestPayloadBudgetSerialization:
    def test_default_budget_attached_when_enabled(self):
        """Payload gets default ErrorBudget(3, 2) when no kwargs given."""
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=False)],
            passed=False,
        )
        ext = _make_extraction(validation_result=vr)
        payload = guardrail_validation_payload(
            ext, include_validation_metadata=True
        )
        assert payload is not None
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 3
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 2
        assert payload["x_omlx_validation"]["budget"]["recommended_action"] == "retry"

    def test_custom_budget_via_kwargs(self):
        """max_retries/max_tool_errors kwargs construct custom budget."""
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=False)],
            passed=False,
        )
        ext = _make_extraction(validation_result=vr)
        payload = guardrail_validation_payload(
            ext,
            include_validation_metadata=True,
            max_retries=7,
            max_tool_errors=3,
        )
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 7
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 3

    def test_pre_existing_budget_not_overwritten(self):
        """If ValidationResult already has a budget, kwargs don't overwrite."""
        existing_budget = ErrorBudget(max_retries=99, max_tool_errors=88)
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=False)],
            passed=False,
            budget=existing_budget,
        )
        ext = _make_extraction(validation_result=vr)
        payload = guardrail_validation_payload(
            ext,
            include_validation_metadata=True,
            max_retries=1,
            max_tool_errors=1,
        )
        # Should use the pre-existing budget, not the kwargs
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 99
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 88

    def test_no_budget_when_disabled(self):
        """Payload is None when metadata disabled."""
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=False)],
            passed=False,
        )
        ext = _make_extraction(validation_result=vr)
        payload = guardrail_validation_payload(
            ext, include_validation_metadata=False, max_retries=5
        )
        assert payload is None

    def test_no_budget_when_no_validation_result(self):
        """Payload is None when extraction has no validation_result."""
        ext = _make_extraction(validation_result=None)
        payload = guardrail_validation_payload(
            ext, include_validation_metadata=True, max_retries=5
        )
        assert payload is None


# ---------------------------------------------------------------------------
# End-to-end pipeline: validator -> apply_guardrails -> payload
# ---------------------------------------------------------------------------


class TestEndToEndBudgetPipeline:
    def test_full_pipeline_failure_with_budget(self):
        """Validator budget survives through apply_guardrails into payload."""
        budget = ErrorBudget(max_retries=5, max_tool_errors=1)
        v = GuardrailValidator([SEARCH_TOOL], budget=budget)
        ext = _make_extraction(cleaned_text="no tools used", tool_calls=None)
        vr = v.validate(ext, has_tools=True)

        ext_with_vr = _make_extraction(
            cleaned_text="no tools used",
            validation_result=vr,
        )
        merged = apply_guardrails(
            ext_with_vr, "auto", [SEARCH_TOOL], validation_enabled=True
        )
        assert merged.validation_result.budget is budget

        # Serialize
        payload = guardrail_validation_payload(
            merged, include_validation_metadata=True, max_retries=99
        )
        assert payload is not None
        # Pre-existing budget wins over kwargs
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 5
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 1
        assert payload["x_omlx_validation"]["passed"] is False

    def test_full_pipeline_no_validator_budget_uses_payload_kwargs(self):
        """Without validator budget, payload kwargs fill in defaults."""
        v = GuardrailValidator([SEARCH_TOOL])
        ext = _make_extraction(cleaned_text="no tools", tool_calls=None)
        vr = v.validate(ext, has_tools=True)
        assert vr.budget is None

        ext_with_vr = _make_extraction(
            cleaned_text="no tools", validation_result=vr
        )
        merged = apply_guardrails(
            ext_with_vr, "auto", None, validation_enabled=True
        )
        payload = guardrail_validation_payload(
            merged,
            include_validation_metadata=True,
            max_retries=8,
            max_tool_errors=4,
        )
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 8
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 4

    def test_full_pipeline_passed_still_gets_budget(self):
        """Even passed validations get budget metadata when enabled."""
        budget = ErrorBudget(max_retries=3, max_tool_errors=2)
        v = GuardrailValidator([SEARCH_TOOL], budget=budget)
        ext = _make_extraction(tool_calls=[_make_tool_call(arguments='{"query": "x"}')])
        vr = v.validate(ext, has_tools=True)
        assert vr.passed is True
        assert vr.budget is budget

        ext_with_vr = _make_extraction(
            tool_calls=[_make_tool_call(arguments='{"query": "x"}')],
            validation_result=vr,
        )
        merged = apply_guardrails(
            ext_with_vr, "auto", None, validation_enabled=True
        )
        payload = guardrail_validation_payload(
            merged, include_validation_metadata=True
        )
        assert payload["x_omlx_validation"]["passed"] is True
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 3
