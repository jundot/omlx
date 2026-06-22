# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.tool_calling import ToolCallExtraction
from omlx.api.tool_choice import enforce_tool_choice


def apply_guardrails(
    extraction: ToolCallExtraction,
    tool_choice: Any,
    tools: Optional[list[dict]] = None,
    *,
    validation_enabled: bool = False,
) -> ToolCallExtraction:
    if not validation_enabled or extraction.validation_result is None:
        return extraction

    has_text = bool(extraction.cleaned_text.strip())
    _, tc_check = enforce_tool_choice(
        extraction.tool_calls, tool_choice, has_text, tools
    )

    existing: ValidationResult = extraction.validation_result
    merged = ValidationResult(
        checks=existing.checks + [tc_check],
        nudge=existing.nudge,
        passed=existing.passed and tc_check.passed,
    )

    return ToolCallExtraction(
        cleaned_text=extraction.cleaned_text,
        tool_calls=extraction.tool_calls,
        cleaned_thinking=extraction.cleaned_thinking,
        tool_calls_from_thinking=extraction.tool_calls_from_thinking,
        validation_result=merged,
    )


def guardrail_validation_payload(
    extraction: ToolCallExtraction,
    *,
    include_validation_metadata: bool = False,
) -> Optional[dict]:
    if not include_validation_metadata or extraction.validation_result is None:
        return None
    return {"x_omlx_validation": extraction.validation_result.to_dict()}
