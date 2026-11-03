# SPDX-License-Identifier: Apache-2.0
"""Forced tool calls for OpenAI ``tool_choice`` (#2877).

``"required"`` and a named function make the reply a call (Anthropic's
``any`` and ``tool``, and the Responses API's choices, are mapped to these
first). The reply is started with the opening of a call in the model's own
format, as its chat template renders an assistant turn with calls, and the
model writes the rest. No grammar is needed, so this works without xgrammar.

The opening is a partial assistant turn, which every in-process engine
renders. Gemma 4's template drops the thought channel from assistant turns,
so its opening is appended to the rendered generation prompt instead, which
needs an engine that takes a ``reply_prefix``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import Any

from ..exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

# The opening of a call per tool parser, as the chat templates render it.
# Named: through the name, cut where the template's next token starts.
# Required: cut before the name, so the name's own tokens stay the model's
# (Qwen tokenizes "<function=get_weather" as "<function", "=get", "_weather").
# No opening ends in whitespace: templates trim the content of the partial
# assistant turn that carries it.
_OPENINGS = {
    "qwen3_coder": ("<tool_call>\n<function={name}>", "<tool_call>\n<function"),
    "json_tools": (
        '<tool_call>\n{{"name": "{name}", "arguments":',
        '<tool_call>\n{"name": "',
    ),
    "gemma4": ("<|tool_call>call:{name}{{", "<|tool_call>call:"),
}

# Formats whose model writes its own thought channel, with its raw open and
# close markers. Their template drops that channel from a partial assistant
# turn: Gemma 4's generation prompt ends with an empty one when thinking is
# off, and its model opens one first when thinking is on. So the opening, and
# the reasoning before it, follow the generation prompt as raw text.
_THOUGHT_CHANNELS = {"gemma4": ("<|channel>thought\n", "<channel|>")}

# How oMLX's output parsers write a reasoning block in the reply text.
_THINK_OPEN, _THINK_CLOSE = "<think>\n", "</think>"


@dataclass(frozen=True)
class ForcedToolCall:
    """A reply that must be a call.

    ``opening`` is the text the reply starts with: through the function's
    name when one is named or only one tool is offered. ``close`` is the
    marker that ends a call in this format. ``single``: a named function is
    called once, so the reply ends with its call; ``"required"`` lets the
    model make more calls after the first.

    ``thinks``: the model reasons before the call, in the reasoning block
    the prompt opens.

    ``thought``: the raw open and close markers of the thought channel, for
    formats whose reply starts after the generation prompt as raw text (the
    engine's ``reply_prefix``) instead of in a partial assistant turn.
    """

    opening: str
    close: str
    single: bool
    thinks: bool = False
    thought: tuple[str, str] | None = None

    @property
    def reply_prefix(self) -> str | None:
        """Raw text the reply starts with after the generation prompt.

        The opening, or, when the model reasons first, the open marker of its
        thought channel. None for formats that use a partial turn.
        """
        if self.thought is None:
            return None
        return self.thought[0] if self.thinks else self.opening


def _tool_names(tools: list[dict] | None) -> list[str]:
    names = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _field(choice: Any, key: str) -> Any:
    if isinstance(choice, dict):
        return choice.get(key)
    return getattr(choice, key, None)


def anthropic_tool_choice(choice: Any) -> Any:
    """An Anthropic ``tool_choice`` that forces a call, in OpenAI's form.

    ``{"type": "any"}`` is ``"required"``, and ``{"type": "tool", "name": n}``
    names the function ``n``. None for ``auto``, ``none`` and anything else.
    """
    kind = _field(choice, "type")
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": _field(choice, "name")}}
    return None


def responses_tool_choice(choice: Any) -> Any:
    """A Responses API ``tool_choice`` that forces a call, in OpenAI's form.

    ``"required"`` stays as it is, and ``{"type": "function", "name": n}``
    names the function ``n`` (the Chat Completions form is accepted too).
    None for ``auto``, ``none``, ``allowed_tools``, hosted tools and anything
    else.
    """
    if choice == "required":
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        return {"type": "function", "function": {"name": name or choice.get("name")}}
    return None


def _parser_format(tokenizer: Any) -> str | None:
    parser = getattr(tokenizer, "tool_parser", None)
    module = getattr(parser, "__module__", None) or ""
    kind = module.rsplit(".", 1)[-1]
    return kind if kind in _OPENINGS else None


def resolve_forced_tool_call(
    tool_choice: Any,
    tools: list[dict] | None,
    tokenizer: Any,
    *,
    reply_prefix: bool = False,
) -> ForcedToolCall | None:
    """Return the call a ``"required"`` or named ``tool_choice`` forces.

    None for ``"auto"``, ``"none"`` and anything else, which keep today's
    behaviour, and for tool-call formats this module does not know (logged),
    or whose reply needs a ``reply_prefix`` the engine does not take.
    Raises InvalidRequestError when the choice cannot be met by ``tools``.
    """
    if tool_choice == "required":
        name = None
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise InvalidRequestError(
                "tool_choice: a named choice needs the function's name",
                field="tool_choice",
            )
    else:
        return None
    names = _tool_names(tools)
    if not names:
        raise InvalidRequestError(
            "tool_choice: a required or named choice needs tools", field="tools"
        )
    if name is not None and name not in names:
        raise InvalidRequestError(
            f"tool_choice: function {name!r} is not in tools", field="tool_choice"
        )
    kind = _parser_format(tokenizer)
    thought = _THOUGHT_CHANNELS.get(kind)
    if kind is None or (thought is not None and not reply_prefix):
        reason = (
            "this model's tool-call format has no known call opening"
            if kind is None
            else "this engine cannot append the call's opening to the prompt"
        )
        logger.warning(
            'tool_choice %r is not enforced: %s; the model chooses as with "auto"',
            tool_choice if name is None else name,
            reason,
        )
        return None
    single = name is not None
    if name is None and len(names) == 1:
        name = names[0]
    named, required = _OPENINGS[kind]
    opening = named.format(name=name) if name is not None else required
    close = getattr(tokenizer, "tool_call_end", None) or "</tool_call>"
    return ForcedToolCall(opening=opening, close=close, single=single, thought=thought)


def forced_call_messages(
    messages: list[dict], forced: ForcedToolCall, reasoning: str | None = None
) -> list[dict]:
    """``messages`` plus the partial assistant turn that starts the reply.

    ``reasoning`` goes in the turn's ``reasoning_content``, which the chat
    templates of models that reason in the prompt's thinking block read (and
    Qwen3.8's is the only place it reads it from).
    """
    turn = {"role": "assistant", "content": forced.opening}
    if reasoning is not None:
        turn["reasoning_content"] = reasoning
    return [*messages, turn]


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


async def forced_tool_call_outputs(
    engine: Any,
    messages: list[dict],
    forced: ForcedToolCall,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Stream a reply that is the forced call, as ``engine.stream_chat`` does.

    When the model reasons first (``forced.thinks``), generation stops at the
    end of its reasoning, and a second request continues the same prompt with
    the reasoning, its close and the call's opening (the prefix cache holds
    the shared prompt). Otherwise ``messages`` already end with the partial
    turn holding the opening, or the engine appends it to the generation
    prompt (``forced.reply_prefix``).

    The texts are those the model would have written, so the thinking and
    tool-call parsers downstream see the whole call. Usage counts the
    client's prompt as prompt tokens and the reasoning, the opening and the
    call as completion tokens; without reasoning the opening is part of the
    prompt.
    """
    tokenizer = engine.tokenizer
    stop = list(kwargs.get("stop") or [])
    if forced.single:
        # A named function is called once: the reply ends after its call.
        stop.append(forced.close)
    if forced.thought is not None:
        kwargs = {**kwargs, "reply_prefix": forced.reply_prefix}
    head = forced.opening  # text before the engine's own output
    base = None  # phase 1's last output when the model reasoned first
    if forced.thinks:
        lead = ""  # text phase 1's output starts with
        if forced.thought is not None:
            # The prompt opens the model's thought channel. The scheduler
            # writes the open tag when it detects that; otherwise it is added
            # here, so the reply reads as an unforced one.
            think_end = forced.thought[1]
            lead = _THINK_OPEN
        else:
            think_end = getattr(tokenizer, "think_end", None) or "</think>"
        phase1 = {**kwargs, "stop": [think_end], "stop_skips_reasoning": False}
        last = None
        pending = lead
        async with aclosing(engine.stream_chat(messages=messages, **phase1)) as outputs:
            async for output in outputs:
                if lead and not output.text.startswith(lead):
                    output = replace(
                        output,
                        text=lead + output.text,
                        new_text=pending + (output.new_text or ""),
                    )
                pending = ""
                if last is not None:
                    yield replace(last, finished=False, finish_reason=None)
                last = output
        if last is None:
            return
        if last.finish_reason != "stop":
            yield last  # out of tokens (or failed) while reasoning: no call
            return
        yield replace(last, finished=False, finish_reason=None)
        reasoning = last.text[len(lead) :]
        if forced.thought is None:
            head = f"{think_end}\n\n{forced.opening}"
            added = head
            messages = forced_call_messages(messages, forced, reasoning.strip())
            kwargs = {**kwargs, "is_partial": True}
        else:
            body = reasoning.rstrip()
            if body.endswith(_THINK_CLOSE):
                # The output parser closed the block when generation stopped.
                reasoning = body[: -len(_THINK_CLOSE)]
            else:
                head = f"{_THINK_CLOSE}\n{forced.opening}"
            added = think_end + forced.opening
            kwargs = {
                **kwargs,
                "reply_prefix": forced.thought[0] + reasoning + added,
            }
        base = replace(
            last,
            completion_tokens=last.completion_tokens + _token_count(tokenizer, added),
        )
        kwargs.pop("_prepared_prompt", None)
        kwargs.pop("thinking_budget", None)
        kwargs["max_tokens"] = max(
            1, kwargs.get("max_tokens", 256) - base.completion_tokens
        )
        if kwargs.get("_request_id"):
            kwargs["_request_id"] = f"{kwargs['_request_id']}-call"
    kwargs["stop"] = stop
    prefix = (base.text if base is not None else "") + head
    pending = head
    async with aclosing(engine.stream_chat(messages=messages, **kwargs)) as outputs:
        async for output in outputs:
            text = prefix + output.text
            new_text = pending + (output.new_text or "")
            pending = ""
            if (
                output.finished
                and forced.single
                and output.finish_reason == "stop"
                and not text.rstrip().endswith(forced.close)
            ):
                # The stop string that ended the call is not in the text.
                text += forced.close
                new_text += forced.close
            if base is not None:
                output = replace(
                    output,
                    prompt_tokens=base.prompt_tokens,
                    completion_tokens=base.completion_tokens + output.completion_tokens,
                    cached_tokens=base.cached_tokens,
                    first_token_at=base.first_token_at,
                )
            yield replace(output, text=text, new_text=new_text)
