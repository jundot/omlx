# SPDX-License-Identifier: Apache-2.0
"""Core data structures for guardrail validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from omlx.api.guardrails.budget import ErrorBudget

# ---------------------------------------------------------------------------
# Nudge kind constants
# ---------------------------------------------------------------------------
KIND_RETRY = "retry"
KIND_UNKNOWN_TOOL = "unknown_tool"
KIND_TOOL_ARG_VALIDATION = "tool_arg_validation"
KIND_STEP = "step"
KIND_PREREQUISITE = "prerequisite"

# Kinds that use the tool-result channel (role="tool").
TOOL_CHANNEL_KINDS = frozenset({KIND_UNKNOWN_TOOL, KIND_TOOL_ARG_VALIDATION})

# All tool-error kinds (used by the validator to pick roles).
TOOL_ERROR_KINDS = frozenset({KIND_UNKNOWN_TOOL, KIND_TOOL_ARG_VALIDATION})

CheckName = Literal[
    "bare_text",
    "unknown_tool",
    "malformed_args",
    "missing_required_params",
    "tool_choice_enforcement",
    "step",
    "prerequisite",
]


@dataclass(frozen=True)
class CheckResult:
    """Result of a single validation check."""

    check: CheckName
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class Nudge:
    """Corrective message for client to append before retry."""

    role: Literal["user", "tool"]
    content: str
    kind: Literal["retry", "unknown_tool", "tool_arg_validation", "step", "prerequisite"]
    tier: int = 0

    def to_message(self) -> dict:
        """Convert to chat message format for client retry."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ValidationResult:
    """Accumulated validation results for a tool-call response."""

    checks: list[CheckResult]
    nudge: Nudge | None = None
    passed: bool = False
    budget: ErrorBudget | None = None

    def to_dict(self) -> dict:
        """Serialize for x_omlx_validation response extension."""
        result: dict = {
            "passed": self.passed,
            "checks": [
                {"check": c.check, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }
        if self.nudge:
            result["nudge"] = {
                "role": self.nudge.role,
                "content": self.nudge.content,
                "kind": self.nudge.kind,
                "tier": self.nudge.tier,
            }
        if self.budget is not None:
            result["budget"] = self.budget.to_dict()
        return result
