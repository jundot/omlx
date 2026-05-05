# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 DSML chat template (derived from mlx-lm deepseek_v32).

This file is a near-verbatim copy of
``mlx_lm/chat_templates/deepseek_v32.py`` (Apple Inc., Apache 2.0). The
only edit is the outer DSML marker name: V3.2 wraps tool calls in
``<｜DSML｜function_calls>...</｜DSML｜function_calls>`` while V4 uses
``<｜DSML｜tool_calls>...</｜DSML｜tool_calls>`` (per vllm's
``DeepSeekV4ToolParser`` which subclasses ``DeepSeekV32ToolParser``
overriding only those two tokens). The inner ``<｜DSML｜invoke>`` /
``<｜DSML｜parameter>`` grammar is identical between V3.2 and V4.

omlx registers this module as ``mlx_lm.chat_templates.deepseek_v4`` so
mlx-lm's tokenizer_config ``chat_template_type`` lookup picks it up
transparently.
"""

import copy
import json
import re
from inspect import isfunction
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.utils.chat_template_utils import get_json_schema

TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}tool_calls>" block like the following as part of your reply to the user:
<{dsml_token}tool_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}tool_calls>
...
</{dsml_token}tool_calls>

<function_results>
...
</function_results>

{thinking_start_token}...thinking about results{thinking_end_token}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"
system_msg_template: str = "{content}"
user_msg_template: str = "<｜User｜>{content}<｜Assistant｜>"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"
thinking_template = "{reasoning_content}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_template: str = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
tool_calls_template = (
    "<{dsml_token}tool_calls>\n{tool_calls}\n</{dsml_token}tool_calls>"
)

tool_output_template: str = "\n<result>{content}</result>"


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    def normalize_tool(tool):
        if isfunction(tool):
            return get_json_schema(tool)
        return tool["function"]

    return [normalize_tool(tool) for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    p_dsml_template = """<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>"""
    P_dsml_strs = []

    arguments = json.loads(tool_call["arguments"])

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
    tools_json = [to_json(t) for t in tools]

    return TOOLS_SYSTEM_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    tools: Any = None,
) -> str:
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
    tools = tools or msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")

    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools_from_openai_format(tools))

        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"
        content_developer = ""
        if tools:
            content_developer += "\n\n" + render_tools(tools_from_openai_format(tools))

        if response_format:
            content_developer += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

        content_developer += "\n\n# The user's message is: {}".format(content)

        prompt += user_msg_template.format(content=content_developer)
        if index == last_user_idx and thinking_mode == "thinking":
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    elif role == "user":
        prompt += user_msg_template.format(content=content)

        if index == last_user_idx and thinking_mode == "thinking":
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    elif role == "tool":
        prev_assistant_idx = index - 1
        assistant_msg = messages[prev_assistant_idx]
        while prev_assistant_idx >= 0 and assistant_msg.get("role") == "tool":
            prev_assistant_idx -= 1
            assistant_msg = messages[prev_assistant_idx]

        assert (
            index == 0
            or prev_assistant_idx >= 0
            and assistant_msg.get("role") == "assistant"
        ), f"Invalid messages at {index}:\n{assistant_msg}"

        tool_call_order = index - prev_assistant_idx
        assistant_tool_calls = assistant_msg.get("tool_calls")
        assert (
            assistant_tool_calls and len(assistant_tool_calls) >= tool_call_order
        ), "No tool calls but found tool output"

        if tool_call_order == 1:
            prompt += "\n\n<function_results>"

        prompt += tool_output_template.format(content=content)

        if tool_call_order == len(assistant_tool_calls):
            prompt += "\n</function_results>"

            if index >= last_user_idx and thinking_mode == "thinking":
                prompt += "\n\n" + thinking_start_token
            else:
                prompt += "\n\n" + thinking_end_token

    elif role == "assistant":
        prev_assistant_idx = index
        thinking_part = ""

        tool_calls_content = ""
        if tool_calls:
            tool_calls = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tool_call.get("name"),
                    arguments=encode_arguments_to_dsml(tool_call),
                )
                for tool_call in tool_calls
            ]
            tool_calls_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token, tool_calls="\n".join(tool_calls)
            )

        summary_content = content or ""

        if thinking_mode == "thinking" and index > last_user_idx:
            assert (
                reasoning_content or tool_calls
            ), f"ThinkingMode: {thinking_mode}, invalid message without reasoning_content/tool_calls `{msg}` after last user message"
            thinking_part = (
                thinking_template.format(reasoning_content=reasoning_content or "")
                + thinking_end_token
            )

        prompt += assistant_msg_template.format(
            reasoning=thinking_part,
            content=summary_content,
            tool_calls=tool_calls_content,
        )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    return prompt


def drop_thinking_messages(
    messages: List[Dict[str, Any]], last_user_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    messages_wo_thinking: List[Dict[str, Any]] = []
    last_user_idx = (
        find_last_user_index(messages) if last_user_idx is None else last_user_idx
    )
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in ["user", "system", "tool"] or idx >= last_user_idx:
            messages_wo_thinking.append(msg)
            continue

        elif role == "assistant":
            msg_wo_thinking = copy.copy(msg)
            msg_wo_thinking.pop("reasoning_content", None)
            messages_wo_thinking.append(msg_wo_thinking)

    return messages_wo_thinking


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str = "thinking",
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    tools: Any = None,
) -> str:
    context = context if context else []
    full_messages = context + messages
    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    if thinking_mode == "thinking" and drop_thinking:
        full_messages = drop_thinking_messages(full_messages)

    for idx in range(len(messages)):
        prompt += render_message(
            idx + len(context),
            full_messages,
            thinking_mode=thinking_mode,
            tools=tools,
        )

    return prompt


def apply_chat_template(
    messages, continue_final_message=False, add_generation_prompt=False, **kwargs
):
    out = encode_messages(messages, **kwargs)
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "Only one of continue_final_message or add_generation_prompt can be True"
        )
    if not add_generation_prompt and messages[-1]["role"] == "user":
        out = out.removesuffix("<｜Assistant｜><think>")
    if continue_final_message and messages[-1]["role"] == "assistant":
        out = out.removesuffix(eos_token)
    return out
