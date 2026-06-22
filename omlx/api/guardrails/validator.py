# SPDX-License-Identifier: Apache-2.0
"""Stateless validator for parsed tool-call responses.

Runs 4 checks in priority order: bare_text → unknown_tool →
malformed_args → missing_required_params. All checks always run
(the design accumulates all failures), but the nudge is selected
from the highest-priority failure.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from omlx.api.guardrails.budget import ErrorBudget
from omlx.api.guardrails.nudge import (
    missing_params_nudge,
    retry_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
from omlx.api.guardrails.types import CheckResult, Nudge, ValidationResult

logger = logging.getLogger(__name__)

_NUDGE_PRIORITY = [
    "bare_text",
    "unknown_tool",
    "malformed_args",
    "missing_required_params",
]


class GuardrailValidator:
    """Validate parsed tool-call responses against the provided tool schemas."""

    def __init__(self, tools: list[dict] | None, *, budget: ErrorBudget | None = None):
        self._tool_schemas: dict[str, dict] = {}
        self._tool_names: set[str] = set()
        self._budget: ErrorBudget | None = budget
        if tools:
            for tool in tools:
                func = tool.get("function", tool)
                name = func.get("name", "")
                if name:
                    self._tool_names.add(name)
                    self._tool_schemas[name] = func.get("parameters", {})

    def validate(
        self,
        extraction: Any,
        tool_choice: Any = None,
        has_tools: bool = True,
    ) -> ValidationResult:
        """Run all 4 checks and return accumulated result."""
        try:
            checks: list[CheckResult] = []

            tool_calls = extraction.tool_calls or []

            checks.append(self._check_bare_text(extraction, tool_choice, has_tools))

            for tc in tool_calls:
                name = tc.function.name
                if name not in self._tool_names:
                    checks.append(
                        CheckResult(
                            check="unknown_tool",
                            passed=False,
                            detail=(
                                f"Tool '{name}' does not exist. "
                                f"Available: {', '.join(sorted(self._tool_names))}"
                            ),
                        )
                    )
                else:
                    checks.append(CheckResult(check="unknown_tool", passed=True))

            for tc in tool_calls:
                checks.append(self._check_malformed_args(tc))

            for tc in tool_calls:
                checks.append(self._check_missing_params(tc))

            passed = all(c.passed for c in checks) if checks else True
            nudge = self._select_nudge(checks, tool_calls) if not passed else None
            return ValidationResult(
                checks=checks, nudge=nudge, passed=passed, budget=self._budget
            )

        except Exception:
            logger.exception("GuardrailValidator.validate failed unexpectedly")
            return ValidationResult(
                checks=[], nudge=None, passed=True, budget=self._budget
            )

    def _check_bare_text(
        self, extraction: Any, tool_choice: Any, has_tools: bool
    ) -> CheckResult:
        has_tool_calls = bool(extraction.tool_calls)
        choice_is_none = tool_choice == "none"
        if not has_tool_calls and has_tools and not choice_is_none:
            return CheckResult(
                check="bare_text",
                passed=False,
                detail=(
                    "Model emitted text instead of tool calls when tools were expected."
                ),
            )
        return CheckResult(check="bare_text", passed=True)

    def _check_malformed_args(self, tc: Any) -> CheckResult:
        raw = tc.function.arguments
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            received_type = type(args).__name__ if args is not None else "NoneType"
            return CheckResult(
                check="malformed_args",
                passed=False,
                detail=(
                    f"Tool '{tc.function.name}' had malformed arguments. "
                    f"Got type: {received_type}. Required: JSON object (dict)."
                ),
            )
        return CheckResult(check="malformed_args", passed=True)

    def _check_missing_params(self, tc: Any) -> CheckResult:
        name = tc.function.name
        schema = self._tool_schemas.get(name, {})
        required = schema.get("required", []) if isinstance(schema, dict) else []

        if not isinstance(required, list):
            logger.warning(
                "Tool '%s' has non-list 'required' field (%s); skipping check.",
                name,
                type(required).__name__,
            )
            return CheckResult(check="missing_required_params", passed=True)

        if not required:
            return CheckResult(check="missing_required_params", passed=True)

        raw = tc.function.arguments
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            return CheckResult(check="missing_required_params", passed=True)

        missing = [p for p in required if p not in args]
        if missing:
            return CheckResult(
                check="missing_required_params",
                passed=False,
                detail=f"Tool '{name}' missing required params: {', '.join(missing)}",
            )
        return CheckResult(check="missing_required_params", passed=True)

    def _select_nudge(self, checks: list[CheckResult], tool_calls: list) -> Nudge | None:
        failed_checks = [c for c in checks if not c.passed]
        if not failed_checks:
            return None

        for check_name in _NUDGE_PRIORITY:
            for c in failed_checks:
                if c.check == check_name:
                    return self._build_nudge(c, tool_calls)
        return None

    def _build_nudge(self, check: CheckResult, tool_calls: list) -> Nudge:
        if check.check == "bare_text":
            return retry_nudge()

        if check.check == "unknown_tool":
            for tc in tool_calls:
                if tc.function.name not in self._tool_names:
                    return unknown_tool_nudge(tc.function.name, list(self._tool_names))
            return unknown_tool_nudge("unknown", list(self._tool_names))

        if check.check == "malformed_args":
            for tc in tool_calls:
                raw = tc.function.arguments
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = None
                if not isinstance(args, dict):
                    received_type = (
                        type(args).__name__ if args is not None else "NoneType"
                    )
                    return tool_arg_validation_nudge(
                        tc.function.name, str(raw)[:200], received_type
                    )
            return tool_arg_validation_nudge("unknown", "", "unknown")

        if check.check == "missing_required_params":
            for tc in tool_calls:
                name = tc.function.name
                schema = self._tool_schemas.get(name, {})
                required = schema.get("required", []) if isinstance(schema, dict) else []
                if not isinstance(required, list) or not required:
                    continue
                raw = tc.function.arguments
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = None
                if isinstance(args, dict):
                    missing = [p for p in required if p not in args]
                    if missing:
                        return missing_params_nudge(name, missing)
            return missing_params_nudge("unknown", [])

        return retry_nudge()
