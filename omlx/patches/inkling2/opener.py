# SPDX-License-Identifier: Apache-2.0
"""Opt-in generation prompt for Inkling 2.

The release chat template renders nothing for ``add_generation_prompt``: the
model writes its own block header, ``<|message_model|>`` and then a content
marker, before the first token a client can see. Each of those costs a decode
forward on top of the time to first token, and without a generation prompt the
prefix cache has no marker at which a prompt ends.

``chat_template_kwargs={"generation_opener": ...}`` (per request or in the model
settings) makes the template render that header as the generation prompt:

``"header"``
    ``<|message_model|>``. Every assistant block the template renders starts
    with it, and the release's response parser treats it as optional.
``"content"``
    The header plus the content marker the request decides: thinking when the
    effort is above zero, text when it is zero and no tool is declared (a tool
    call's header carries the tool name, so the header stays open with tools).
    This forces the first block's kind; the model no longer chooses it.

``generation_opener_answer`` (set by the server for a grammar that starts at the
answer) makes ``"content"`` open a text block whatever the effort.
"""

from __future__ import annotations

from typing import Any

_SENTINEL = "{#- omlx generation opener -#}"

# Runs after the release template, in its top-level scope: ``effort_map`` and
# ``tools`` are the template's own. An unknown effort has already raised.
_SUFFIX = _SENTINEL + """
{%- if add_generation_prompt and generation_opener is defined and generation_opener in ("header", "content") -%}
    {{- "<|message_model|>" -}}
    {%- if generation_opener == "content" -%}
        {%- set opener_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
        {%- if opener_effort is string -%}
            {%- set opener_effort = effort_map[opener_effort | trim] -%}
        {%- endif -%}
        {%- if generation_opener_answer is defined and generation_opener_answer -%}
            {{- "<|content_text|>" -}}
        {%- elif (opener_effort | float) > 0.0 -%}
            {{- "<|content_thinking|>" -}}
        {%- elif not tools -%}
            {{- "<|content_text|>" -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
"""


def install_generation_opener(tokenizer: Any) -> bool:
    """Teach the tokenizer's chat template the opt-in generation prompt.

    Leaves a template alone when it already renders a generation prompt or is
    not the release's (no ``effort_map``). Returns whether it was installed.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or _SENTINEL in template:
        return False
    if "add_generation_prompt" in template or "effort_map" not in template:
        return False
    tokenizer.chat_template = template + _SUFFIX
    return True

