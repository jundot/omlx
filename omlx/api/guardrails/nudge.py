# SPDX-License-Identifier: Apache-2.0
"""Nudge message generators for validation failures.

Each function returns a Nudge with the correct role and kind.
Message patterns follow Forge's prompts/nudges.py.
"""
from __future__ import annotations

from omlx.api.guardrails.types import (
    KIND_PREREQUISITE,
    KIND_RETRY,
    KIND_STEP,
    KIND_TOOL_ARG_VALIDATION,
    KIND_UNKNOWN_TOOL,
    Nudge,
)


def retry_nudge(tier: int = 0) -> Nudge:
    """Nudge for bare-text when tools were expected.

    Uses role='user' because bare-text correction is an instruction
    to the model, not a tool-result error.
    """
    return Nudge(
        role="user",
        content=(
            "You provided a text response instead of making a tool call. "
            "Please use the available tools to answer the request."
        ),
        kind=KIND_RETRY,
        tier=tier,
    )


def unknown_tool_nudge(
    tool_name: str, available_tools: list[str], tier: int = 0
) -> Nudge:
    """Nudge for calling a tool that does not exist.

    Uses role='tool' because models attend well to the 'tool call failed'
    wire shape.
    """
    tools_str = ", ".join(sorted(available_tools)) if available_tools else "(none)"
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' does not exist. "
            f"Available: {tools_str}. Call one of them."
        ),
        kind=KIND_UNKNOWN_TOOL,
        tier=tier,
    )


def tool_arg_validation_nudge(
    tool_name: str, args_repr: str, received_type: str, tier: int = 0
) -> Nudge:
    """Nudge for malformed (non-dict) arguments."""
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' had malformed arguments. "
            f"Got type: {received_type}. Required: JSON object (dict). "
            f"Received value: {args_repr[:200]}"
        ),
        kind=KIND_TOOL_ARG_VALIDATION,
        tier=tier,
    )


def missing_params_nudge(
    tool_name: str, missing_params: list[str], tier: int = 0
) -> Nudge:
    """Nudge for missing required parameters."""
    params_str = ", ".join(missing_params)
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' is missing required parameter(s): {params_str}. "
            f"Please provide all required parameters."
        ),
        kind=KIND_TOOL_ARG_VALIDATION,
        tier=tier,
    )


def prerequisite_nudge(
    tool_name: str, missing_prereqs: list[str], tier: int = 0
) -> Nudge:
    """Nudge for calling a tool without its declared prerequisites.

    Uses role='user' because this is workflow guidance (call the
    prerequisite first), not a tool-execution error. Adapted from
    Forge's prompts/nudges.py:prerequisite_nudge.
    """
    prereqs = ", ".join(missing_prereqs)
    return Nudge(
        role="user",
        content=(
            f"You cannot call {tool_name} yet. "
            f"You must first call: {prereqs}. "
            "Call the prerequisite tool now."
        ),
        kind=KIND_PREREQUISITE,
        tier=tier,
    )


def step_nudge(
    terminal_tool: str, pending_steps: list[str], tier: int = 1
) -> Nudge:
    """Escalating nudge for premature terminal tool attempts.

    Uses role='user' because this is workflow guidance (complete the
    required steps first), not a tool-execution error. Adapted from
    Forge's prompts/nudges.py:step_nudge.

    Args:
        terminal_tool: The terminal tool the model tried to call early.
        pending_steps: Required steps that must be completed first.
        tier: Escalation level (1=polite, 2=direct, 3=aggressive).
            Clamped to 1-3.
    """
    tier = max(1, min(3, tier))
    steps = ", ".join(pending_steps)
    if tier == 1:
        content = (
            f"You cannot call {terminal_tool} yet. "
            f"You must first complete these required steps: {steps}. "
            "Call one of them now."
        )
    elif tier == 2:
        content = (
            f"You must call one of these tools now: {steps}. "
            "Pick one."
        )
    else:
        content = (
            f"STOP. You MUST call one of: {steps}. "
            f"Do NOT call {terminal_tool}. "
            f"Your next response MUST be a tool call to one of: {steps}."
        )
    return Nudge(role="user", content=content, kind=KIND_STEP, tier=tier)
