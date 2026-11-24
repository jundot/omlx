# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 turn framing for grammar-constrained output.

The publisher template renders no generation prompt, so the model opens every
block of its turn itself::

    [<|message_model|><|content_thinking|>THOUGHT<|end_message|>]
    <|message_model|><|content_text|>ANSWER<|end_message|>

A grammar that starts at the answer masks those headers. The structural tag
built here puts the user's grammar inside the text block instead.
"""

from ...api.grammar import mark_grammar_thinking_phase

_TEXT_HEADER = "<|message_model|><|content_text|>"
_THINKING_HEADER = "<|message_model|><|content_thinking|>"
_END_MESSAGE = "<|end_message|>"
# Markers that would start another block inside the thinking text.
_BLOCK_MARKERS = [
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_xml|>",
    "<|content_invoke_tool_json|>",
]
_EFFORT_OFF = ("none", "off", "no_think")


def _thinking_enabled(chat_template_kwargs: dict) -> bool:
    """Return whether the template renders a thinking effort above zero.

    Follows the engine's mapping of the thinking switches to
    ``reasoning_effort``: an explicit effort wins over ``enable_thinking``.
    """
    effort = chat_template_kwargs.get("reasoning_effort")
    if effort is None:
        return chat_template_kwargs.get("enable_thinking") is not False
    if isinstance(effort, str):
        effort = effort.strip().lower()
        if effort in _EFFORT_OFF:
            return False
        try:
            effort = float(effort)
        except ValueError:
            return True
    return effort != 0


def _block(begin: str, content: dict) -> dict:
    return {"type": "tag", "begin": begin, "content": content, "end": _END_MESSAGE}


def structural_tag(fmt: dict, *, thinking: bool) -> dict:
    """Return the structural tag that frames ``fmt`` as the turn's answer."""
    answer = _block(_TEXT_HEADER, fmt)
    # A custom template can end the prompt with the text block header.
    turns = [answer, _block("", fmt)]
    if thinking:
        thought = _block(
            _THINKING_HEADER, {"type": "any_text", "excludes": _BLOCK_MARKERS}
        )
        turns.append({"type": "sequence", "elements": [thought, answer]})
    return {"type": "structural_tag", "format": {"type": "or", "elements": turns}}


def compile_answer_grammar(
    compiler, fmt: dict, chat_template_kwargs: dict | None, *, max_whitespace_cnt: int
):
    """Compile ``fmt`` inside the block structure of an Inkling 2 turn."""
    if fmt["type"] == "json_schema":
        fmt = {**fmt, "max_whitespace_cnt": max_whitespace_cnt}
    thinking = _thinking_enabled(chat_template_kwargs or {})
    compiled = compiler.compile_structural_tag(structural_tag(fmt, thinking=thinking))
    # The model opens the thinking block, so a budget waits for its header.
    return mark_grammar_thinking_phase(compiled, enabled=thinking, optional=thinking)
