# SPDX-License-Identifier: Apache-2.0
"""Grammar framing for Inkling 2, which writes its own block headers."""

from ...api.grammar import mark_grammar_thinking_phase

_TEXT_HEADER = "<|message_model|><|content_text|>"
_THINKING_HEADER = "<|message_model|><|content_thinking|>"
_END_MESSAGE = "<|end_message|>"
_BLOCK_MARKERS = [
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_xml|>",
    "<|content_invoke_tool_json|>",
]
_EFFORT_OFF = ("none", "off", "no_think")


def _thinking_enabled(chat_template_kwargs: dict) -> bool:
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
    header_in_prompt = _block("", fmt)
    turns = [answer, header_in_prompt]
    if thinking:
        thought = _block(
            _THINKING_HEADER, {"type": "any_text", "excludes": _BLOCK_MARKERS}
        )
        turns.append({"type": "sequence", "elements": [thought, answer]})
    return {"type": "structural_tag", "format": {"type": "or", "elements": turns}}


def compile_answer_grammar(
    compiler, fmt: dict, chat_template_kwargs: dict, *, max_whitespace_cnt: int
):
    """Compile ``fmt`` inside the block structure of an Inkling 2 turn."""
    if fmt["type"] == "json_schema":
        fmt = {**fmt, "max_whitespace_cnt": max_whitespace_cnt}
    thinking = _thinking_enabled(chat_template_kwargs)
    compiled = compiler.compile_structural_tag(structural_tag(fmt, thinking=thinking))
    return mark_grammar_thinking_phase(compiled, enabled=thinking, optional=thinking)
