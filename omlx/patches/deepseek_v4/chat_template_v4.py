# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
# ruff: noqa: N806, SIM114, UP006, UP007, UP035, UP045
"""DeepSeek V4 chat encoding based on the official 0731 reference.

The encoding core is vendored from ``deepseek-ai/DeepSeek-V4-Flash-0731``.
The small adapter at the end exposes mlx-lm's ``apply_chat_template`` API.
"""

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

# The open-weight 0731 encoding has no role delimiter for a system message
# after the conversation starts. oMLX exposes this capability explicitly so
# callers do not mistake raw marker ordering for model-level support.
supports_mid_system_messages = False

# Task special tokens for internal classification tasks
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

# ============================================================
# Templates
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_template: str = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"

tool_output_template: str = "<tool_result>{content}</tool_result>"

# Reasoning effort levels. In thinking mode, the prompt for the selected level is
# prepended at the very beginning of the conversation. `low` is the default and
# adds nothing.
REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"

# Clients send a wider vocabulary than this template's three levels (OpenAI
# uses minimal/medium/high; Hermes uses xhigh). Map aliases onto the nearest
# supported level instead of failing the request: a 400 here surfaces as an
# opaque provider error in agent clients. Unknown values fall back to the
# default rather than aborting the chat-template render.
REASONING_EFFORT_ALIASES: Dict[str, str] = {
    "none": "low",
    "off": "low",
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "moderate": "high",
    "high": "high",
    "xhigh": "max",
    "ultra": "max",
    "max": "max",
    "maximum": "max",
}


def normalize_reasoning_effort(value: Optional[str]) -> str:
    """Map a client-supplied reasoning effort to a supported level."""
    if not value:
        return DEFAULT_REASONING_EFFORT
    normalized = REASONING_EFFORT_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        logger.warning(
            "Unknown reasoning effort %r; falling back to %r",
            value,
            DEFAULT_REASONING_EFFORT,
        )
        return DEFAULT_REASONING_EFFORT
    return normalized

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ============================================================
# Utility Functions
# ============================================================


def to_json(value: Any) -> str:
    """Serialize a value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    """Extract function definitions from OpenAI-format tool list."""
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """Convert OpenAI-format tool calls to internal format."""
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def tool_calls_to_openai_format(tool_calls):
    """Convert internal tool calls to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    """
    Encode tool call arguments into DSML parameter format.

    Args:
        tool_call: Dict with "name" and "arguments" (JSON string) keys.

    Returns:
        DSML-formatted parameter string.
    """
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    P_dsml_strs = []

    raw_arguments = tool_call["arguments"]
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {"arguments": raw_arguments}

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        P_dsml_strs.append(p_dsml_str)

    return "\n".join(P_dsml_strs)


def decode_dsml_to_arguments(
    tool_name: str, tool_args: Dict[str, Tuple[str, str]]
) -> Dict[str, str]:
    """
    Decode DSML parameters back to a tool call dict.

    Args:
        tool_name: Name of the tool.
        tool_args: Dict mapping param_name -> (value, is_string_flag).

    Returns:
        Dict with "name" and "arguments" (JSON string) keys.
    """

    def _decode_value(key: str, value: str, string: str):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = (
        "{"
        + ", ".join(
            [_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]
        )
        + "}"
    )
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    """
    Render tool schemas into the system prompt format.

    Args:
        tools: List of tool schema dicts (each with name, description, parameters).

    Returns:
        Formatted tools section string.
    """
    tools_json = [to_json(t) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Find the index of the last user/developer message."""
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# Message Rendering
# ============================================================


def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Render a single message at the given index into its encoded string form.

    This is the core function that converts each message in the conversation
    into the DeepSeek-V4 format.

    Args:
        index: Index of the message to render.
        messages: Full list of messages in the conversation.
        thinking_mode: Either "chat" or "thinking".
        drop_thinking: Whether to drop reasoning content from earlier turns.
        reasoning_effort: Reasoning effort level, one of "low", "high", "max".
            None is treated as "low".

    Returns:
        Encoded string for this message.
    """
    assert 0 <= index < len(messages)
    assert thinking_mode in [
        "chat",
        "thinking",
    ], f"Invalid thinking_mode `{thinking_mode}`"

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    # Reasoning effort prefix (only at index 0 in thinking mode; "low" adds nothing)
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += content

        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN

        # Handle content blocks (tool results mixed with text)
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(
            content=content
        )

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v4 merges tool messages into user; please preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            tc_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        rc = reasoning_content or ""

        # Check if previous message has a task - if so, this is a task output (no thinking)
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (
                    thinking_template.format(reasoning_content=rc) + thinking_end_token
                )
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # Append transition tokens based on what follows
    if index + 1 < len(messages) and messages[index + 1].get("role") not in [
        "assistant",
        "latest_reminder",
    ]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        # Task special token for internal classification tasks
        assert (
            task in VALID_TASKS
        ), f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            # Non-action tasks: append task sp token directly after the message
            prompt += task_sp_token
        else:
            # Action task: append Assistant + thinking token + action sp token
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_end_token
                if thinking_mode != "thinking"
                else thinking_start_token
            )
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Preprocessing
# ============================================================


def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge tool messages into the preceding user message using content_blocks format.

    DeepSeek-V4 does not have a standalone "tool" role; instead, tool results
    are encoded as <tool_result> blocks within user messages.

    This function converts a standard OpenAI-format conversation (with separate
    "tool" role messages) into V4 format where tool results are merged into
    user messages.

    Args:
        messages: List of message dicts in OpenAI format.

    Returns:
        Processed message list with tool messages merged into user messages.
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Merge into previous message if it's already a user (merged tool)
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append(
                    {
                        "role": "user",
                        "content_blocks": [tool_block],
                    }
                )
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                # Preserve extra fields (task, wo_eos, mask, etc.)
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sort tool_result blocks within user messages by the order of tool_calls
    in the preceding assistant message.

    Args:
        messages: Preprocessed message list (after merge_tool_messages).

    Returns:
        Message list with sorted tool result blocks.
    """
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Main Encoding Function
# ============================================================


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Encode a list of messages into the DeepSeek-V4 prompt format.

    This is the main entry point for encoding conversations. It handles:
    - BOS token insertion
    - Thinking mode with optional reasoning content dropping
    - Tool message merging into user messages
    - Multi-turn conversation context

    Args:
        messages: List of message dicts to encode.
        thinking_mode: Either "chat" or "thinking".
        context: Optional preceding context messages (already encoded prefix).
        drop_thinking: If True, drop reasoning_content from earlier assistant turns
                      (only keep reasoning for messages after the last user message).
        add_default_bos_token: Whether to prepend BOS token at conversation start.
        reasoning_effort: Reasoning effort level, one of "low", "high", "max".
            Only takes effect in thinking mode. None is treated as "low".

    Returns:
        The encoded prompt string.
    """
    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    # Resolve drop_thinking: if any message has tools defined, don't drop thinking
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        # After dropping, recalculate how many messages to render
        # (context may have shrunk too)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop reasoning_content and non-essential messages before the last user message.

    Behavior:
    - Messages with role in ["user", "system", "tool", "latest_reminder"] are always kept.
    - Messages at or after the last user index are always kept.
    - Assistant messages before the last user get reasoning_content removed.
    - Developer messages before the last user are dropped entirely.
    """
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
        # developer and other roles before last_user_idx are dropped

    return result


# ============================================================
# mlx-lm Adapter
# ============================================================


def _attach_request_tools(
    messages: List[Dict[str, Any]],
    tools: Any,
    context: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Attach request-level tools where the reference encoder expects them."""
    copied_messages = [copy.deepcopy(message) for message in messages]
    copied_context = (
        [copy.deepcopy(message) for message in context] if context else context
    )
    if not tools:
        return copied_messages, copied_context

    # A context prefix that starts with system/developer is assumed to have
    # already encoded the tool definitions. Re-injecting them into the delta
    # would duplicate the schema and break append-only prompt caching.
    if copied_context and any(
        message.get("role") in {"system", "developer"} for message in copied_context
    ):
        return copied_messages, copied_context

    for message in copied_messages:
        if message.get("role") in {"system", "developer"}:
            message.setdefault("tools", tools)
            return copied_messages, copied_context

    copied_messages.insert(
        0,
        {"role": "system", "content": "", "tools": copy.deepcopy(tools)},
    )
    return copied_messages, copied_context


def _latest_reminder_content_as_text(content: Any) -> Optional[str]:
    """Return text-only system content, or None for unsupported content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type", "text") != "text":
            return None
        text = block.get("text", "")
        if text is not None and not isinstance(text, str):
            return None
        if text:
            parts.append(text)
    return "\n".join(parts)


def relocate_mid_system_messages(
    messages: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Move supported inline system runs to V4 ``latest_reminder`` context.

    Claude Code appends a volatile system message immediately after the user
    message it qualifies. The V4 reference encoder has no delimiter for that
    raw placement. Reclassify the system run as a new ``latest_reminder``
    context immediately before the same user, preserving append-only prompt
    caching across requests.

    Tool-call turns need the same treatment: the Anthropic adapter splits
    ``tool_result`` blocks into ``tool`` role messages, so Claude Code's
    periodic reminders arrive as ``tool -> system -> assistant`` runs.
    Those are reclassified as a ``latest_reminder`` before the contiguous
    tool run (the run merges into one user turn at encode time), which
    keeps the rendered prefix byte-stable across requests instead of
    falling back to front-consolidation that rewrites the whole prompt.

    Only ``(user | tool) -> system -> (assistant | end)`` shapes are
    rewritten. Other placements and non-leading developer messages return
    None so oMLX can use its configured strict or user-note fallback.
    """
    source = [copy.deepcopy(message) for message in messages]
    relocated: List[Dict[str, Any]] = []
    seen_non_system = False
    index = 0

    while index < len(source):
        message = source[index]
        if message.get("role") != "system":
            relocated.append(message)
            seen_non_system = True
            index += 1
            continue

        start = index
        parts = []
        while index < len(source) and source[index].get("role") == "system":
            text = _latest_reminder_content_as_text(source[index].get("content"))
            if text is None:
                return None
            if text:
                parts.append(text)
            index += 1

        if not seen_non_system:
            relocated.extend(source[start:index])
            continue

        next_role = source[index].get("role") if index < len(source) else None
        prev_role = relocated[-1].get("role") if relocated else None
        if prev_role not in {"user", "tool"} or next_role not in {None, "assistant"}:
            return None

        if prev_role == "user":
            associated_user = relocated.pop()
            if parts:
                relocated.append(
                    {
                        "role": "latest_reminder",
                        "content": "\n\n".join(parts),
                    }
                )
            relocated.append(associated_user)
        elif parts:
            # Tool-adjacent run: the tool messages merge into one user turn
            # whose template ends with the assistant primer, so the reminder
            # must move before the whole tool run to render ahead of that
            # user turn (mirroring the user-adjacent placement above).
            insert_at = len(relocated)
            while insert_at > 0 and relocated[insert_at - 1].get("role") == "tool":
                insert_at -= 1
            relocated.insert(
                insert_at,
                {
                    "role": "latest_reminder",
                    "content": "\n\n".join(parts),
                },
            )

    return relocated


def apply_chat_template(
    messages,
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
    **kwargs,
):
    """Expose the official encoder through mlx-lm's template callable API."""
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
    tools = kwargs.pop("tools", None)
    context = kwargs.get("context")
    prepared_messages, prepared_context = _attach_request_tools(
        messages,
        tools,
        context=context,
    )
    if context is not None:
        kwargs["context"] = prepared_context

    accepted = {
        "thinking_mode",
        "context",
        "drop_thinking",
        "add_default_bos_token",
        "reasoning_effort",
    }
    encode_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    out = encode_messages(prepared_messages, **encode_kwargs)

    if not add_generation_prompt:
        out = out.removesuffix(ASSISTANT_SP_TOKEN + thinking_start_token)
        out = out.removesuffix(ASSISTANT_SP_TOKEN + thinking_end_token)
    elif not (
        prepared_messages and prepared_messages[-1].get("task") is not None
    ) and not out.endswith(
        (
            ASSISTANT_SP_TOKEN + thinking_start_token,
            ASSISTANT_SP_TOKEN + thinking_end_token,
        )
    ):
        # encode_messages only emits the generation anchor after a
        # user/developer-final message. A system-final conversation (e.g. a
        # bare title-generation instruction, or a trailing workspace/system
        # note after the last user turn) otherwise ends without
        # <|Assistant|><think>, and the model free-runs as document
        # continuation until max_tokens.
        out += ASSISTANT_SP_TOKEN + (
            thinking_start_token
            if encode_kwargs.get("thinking_mode", "thinking") == "thinking"
            else thinking_end_token
        )
    if continue_final_message and messages and messages[-1].get("role") == "assistant":
        out = out.removesuffix(eos_token)
    return out


apply_chat_template.supports_mid_system_messages = (  # type: ignore[attr-defined]
    supports_mid_system_messages
)
apply_chat_template.relocate_mid_system_messages = (  # type: ignore[attr-defined]
    relocate_mid_system_messages
)
