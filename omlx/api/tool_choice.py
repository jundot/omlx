# SPDX-License-Identifier: Apache-2.0
"""tool_choice enforcement: applied after validation, before response construction.

Supports 4 modes:
  - "none":     suppress all tool calls
  - "auto":     pass through (no-op)
  - "required": if no tool calls + has_text, flag failure
  - {"type":"function","function":{"name":"X"}}: filter to named tool
"""
from __future__ import annotations

from typing import Any, Optional

from omlx.api.guardrails.types import CheckResult


def enforce_tool_choice(
    tool_calls: Optional[list],
    tool_choice: Any,
    has_text: bool,
    tools: Optional[list[dict]] = None,
) -> tuple[Optional[list], CheckResult]:
    """Enforce tool_choice semantics on parsed tool calls.

    Returns (possibly filtered tool_calls, check_result).
    The check_result.check is always 'tool_choice_enforcement'.
    """
    if tool_choice == "none":
        return None, CheckResult(
            check="tool_choice_enforcement",
            passed=True,
            detail="tool_choice='none': tool calls suppressed",
        )

    if tool_choice is None or tool_choice == "auto":
        return tool_calls, CheckResult(
            check="tool_choice_enforcement", passed=True
        )

    if tool_choice == "required":
        if not tool_calls and has_text:
            return tool_calls, CheckResult(
                check="tool_choice_enforcement",
                passed=False,
                detail="tool_choice='required' but model produced no tool calls",
            )
        return tool_calls, CheckResult(
            check="tool_choice_enforcement", passed=True
        )

    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        target_name = func.get("name", "")
        if target_name:
            if tool_calls:
                filtered = [
                    tc for tc in tool_calls if tc.function.name == target_name
                ]
                rejected = [
                    tc for tc in tool_calls if tc.function.name != target_name
                ]
                if rejected and not filtered:
                    return filtered, CheckResult(
                        check="tool_choice_enforcement",
                        passed=False,
                        detail=(
                            f"tool_choice requires '{target_name}' but model "
                            f"called other tools: "
                            f"{', '.join(tc.function.name for tc in rejected)}"
                        ),
                    )
                return filtered, CheckResult(
                    check="tool_choice_enforcement", passed=True
                )
            return tool_calls, CheckResult(
                check="tool_choice_enforcement",
                passed=False,
                detail=f"tool_choice requires '{target_name}' but no tool calls produced",
            )

    return tool_calls, CheckResult(
        check="tool_choice_enforcement", passed=True
    )
