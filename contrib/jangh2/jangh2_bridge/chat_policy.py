"""Explicit chat-template defaults and bounded reasoning-prefix detection.

No generated prose is rewritten or discarded here. The model's own template
chooses its thinking protocol; oMLX's existing output parser separates it.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class ChatPolicy:
    thinking: str = 'auto'
    reasoning_effort: str | None = None

    def __post_init__(self):
        if self.thinking not in ('auto', 'on', 'off'):
            raise ValueError('OMLX_JANGH_THINKING must be auto, on, or off')
        if self.reasoning_effort not in (None, 'low', 'medium', 'xhigh'):
            raise ValueError('OMLX_JANGH_REASONING_EFFORT must be low, medium, or xhigh')

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get('OMLX_JANGH_THINKING', 'auto'),
                   os.environ.get('OMLX_JANGH_REASONING_EFFORT') or None)

    def merge(self, options: Mapping[str, Any] | None, *, engine_thinking=None) -> dict:
        """Existing admin/request options beat engine settings, then env defaults."""
        if options is not None and not isinstance(options, Mapping):
            raise TypeError('chat_template_kwargs must be a mapping')
        result = dict(options or {})
        thinking = result.get('enable_thinking')
        if thinking is not None and type(thinking) is not bool:
            raise ValueError('enable_thinking must be a JSON boolean, not a string or number')
        if thinking is None:
            result.pop('enable_thinking', None)
            if engine_thinking is not None:
                if type(engine_thinking) is not bool:
                    raise ValueError('engine enable_thinking must be a boolean')
                result['enable_thinking'] = engine_thinking
            elif self.thinking != 'auto':
                result['enable_thinking'] = self.thinking == 'on'
        effort = result.get('reasoning_effort')
        if effort is not None and effort not in ('low', 'medium', 'xhigh'):
            raise ValueError('Qwen4 reasoning_effort must be low, medium, or xhigh')
        if effort is None:
            result.pop('reasoning_effort', None)
            if self.reasoning_effort is not None:
                result['reasoning_effort'] = self.reasoning_effort
        return result


def prompt_reasoning_state(prompt: str) -> bool | None:
    """State of the last ChatML assistant prefix, never of a prior user turn.

    None means the bounded tail lacks a recognised boundary. The scheduler then
    retains its own detection rather than guessing from arbitrary English prose.
    """
    if not isinstance(prompt, str):
        return None
    marker = '<|im_start|>'
    start = prompt.rfind(marker)
    if start < 0:
        return None
    tail = prompt[start + len(marker):]
    role, separator, body = tail.partition('\n')
    if not separator or role.strip() != 'assistant':
        return False
    if '<|im_end|>' in body or '<|endoftext|>' in body:
        return False
    opening = body.rfind('<think>')
    closing = body.rfind('</think>')
    return opening >= 0 and opening > closing


def token_reasoning_state(tokenizer, token_ids, *, max_tail_tokens=2048) -> bool | None:
    """Decode a small CPU token tail once per request; no model/GPU evaluation."""
    if type(max_tail_tokens) is not int or max_tail_tokens < 1:
        raise ValueError('max_tail_tokens must be positive')
    if token_ids is None or len(token_ids) == 0:
        return False
    decode = getattr(tokenizer, 'decode', None)
    if not callable(decode):
        return None
    tail = list(token_ids[-max_tail_tokens:])
    try:
        text = decode(tail, skip_special_tokens=False)
    except TypeError:
        # Some tokenizer wrappers expose decode(ids) only. If they discard the
        # role boundary, prompt_reasoning_state returns None rather than guessing.
        text = decode(tail)
    return prompt_reasoning_state(text)
