# SPDX-License-Identifier: MIT
"""mlx-lm adapter for the official DeepSeek V4 DSpark/0731 encoder."""

from __future__ import annotations

import copy

from .vendor.encoding_dsv4 import (
    ASSISTANT_SP_TOKEN,
    encode_messages,
    eos_token,
    thinking_end_token,
    thinking_start_token,
)


def apply_chat_template(
    messages,
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
    **kwargs,
):
    """Adapt the official DSpark/0731 encoder to mlx-lm's template API."""
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "Only one of continue_final_message or add_generation_prompt can be True"
        )

    if "enable_thinking" in kwargs and "thinking_mode" not in kwargs:
        kwargs["thinking_mode"] = (
            "thinking" if kwargs.pop("enable_thinking") else "chat"
        )
    else:
        kwargs.pop("enable_thinking", None)
    kwargs.setdefault("thinking_mode", "thinking")

    # DeepSeek's reference API stores tools on the system/developer message,
    # while mlx-lm supplies them as a top-level template kwarg.
    tools = kwargs.pop("tools", None)
    messages = copy.deepcopy(messages)
    if tools:
        for message in messages:
            if message.get("role") in ("system", "developer"):
                message["tools"] = tools
                break
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})

    accepted = {
        "thinking_mode",
        "context",
        "drop_thinking",
        "add_default_bos_token",
        "reasoning_effort",
    }
    encode_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    out = encode_messages(messages, **encode_kwargs)

    last_role = messages[-1].get("role") if messages else None
    if not add_generation_prompt and last_role in ("user", "developer", "tool"):
        thinking_suffix = (
            thinking_start_token
            if encode_kwargs["thinking_mode"] == "thinking"
            else thinking_end_token
        )
        out = out.removesuffix(ASSISTANT_SP_TOKEN + thinking_suffix)
    if continue_final_message and last_role == "assistant":
        out = out.removesuffix(eos_token)
    return out
