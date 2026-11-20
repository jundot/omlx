# SPDX-License-Identifier: Apache-2.0
"""Tests for forced tool calls (OpenAI ``tool_choice``, #2877)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from mlx_lm.tool_parsers import gemma4, json_tools, qwen3_coder

from omlx.api.anthropic_models import ToolChoice
from omlx.api.tool_choice import (
    ForcedToolCall,
    anthropic_tool_choice,
    forced_call_messages,
    forced_tool_call_outputs,
    resolve_forced_tool_call,
    responses_tool_choice,
)
from omlx.engine.base import GenerationOutput
from omlx.exceptions import InvalidRequestError


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


TOOLS = [_tool("get_weather"), _tool("get_time")]
NAMED = {"type": "function", "function": {"name": "get_time"}}


def _tokenizer(parser=qwen3_coder, start="<tool_call>", end="</tool_call>"):
    return SimpleNamespace(
        tool_parser=parser.parse_tool_call,
        tool_call_start=start,
        tool_call_end=end,
        think_end="</think>",
        encode=lambda text, add_special_tokens=False: list(text),
    )


class TestResolveForcedToolCall:
    @pytest.mark.parametrize("choice", [None, "auto", "none", "any", {"type": "x"}])
    def test_other_choices_keep_todays_behaviour(self, choice):
        assert resolve_forced_tool_call(choice, TOOLS, _tokenizer()) is None

    def test_named_function_opens_its_call_and_is_called_once(self):
        forced = resolve_forced_tool_call(NAMED, TOOLS, _tokenizer())
        assert forced == ForcedToolCall(
            opening="<tool_call>\n<function=get_time>",
            close="</tool_call>",
            single=True,
        )

    def test_required_leaves_the_name_to_the_model(self):
        forced = resolve_forced_tool_call("required", TOOLS, _tokenizer())
        # Cut before "=": Qwen tokenizes "=get" as one token.
        assert forced.opening == "<tool_call>\n<function"
        assert forced.single is False

    def test_required_with_one_tool_names_it(self):
        forced = resolve_forced_tool_call("required", TOOLS[:1], _tokenizer())
        assert forced.opening == "<tool_call>\n<function=get_weather>"
        assert forced.single is False

    def test_hermes_json_openings(self):
        tokenizer = _tokenizer(json_tools)
        forced = resolve_forced_tool_call(NAMED, TOOLS, tokenizer)
        assert forced.opening == '<tool_call>\n{"name": "get_time", "arguments":'
        assert forced.close == "</tool_call>"
        assert resolve_forced_tool_call("required", TOOLS, tokenizer).opening == (
            '<tool_call>\n{"name": "'
        )

    def test_gemma4_keeps_todays_behaviour(self, caplog):
        # Measured on Gemma 4 26B-A4B: with several tools or thinking on, the
        # model does not continue a prefilled call reliably.
        tokenizer = _tokenizer(gemma4, "<|tool_call>", "<tool_call|>")
        with caplog.at_level("WARNING"):
            assert resolve_forced_tool_call(NAMED, TOOLS, tokenizer) is None
        assert "not enforced" in caplog.text

    @pytest.mark.parametrize("choice", ["required", NAMED])
    def test_openings_never_end_in_whitespace(self, choice):
        # Chat templates trim the partial assistant turn that carries them.
        for parser in (qwen3_coder, json_tools):
            forced = resolve_forced_tool_call(choice, TOOLS, _tokenizer(parser))
            assert forced.opening == forced.opening.rstrip()

    def test_unknown_format_is_not_enforced(self, caplog):
        tokenizer = _tokenizer()
        tokenizer.tool_parser = lambda text, tools=None: {}
        with caplog.at_level("WARNING"):
            assert resolve_forced_tool_call(NAMED, TOOLS, tokenizer) is None
        assert "not enforced" in caplog.text

    @pytest.mark.parametrize(
        ("choice", "tools", "field"),
        [
            ({"type": "function", "function": {"name": "nope"}}, TOOLS, "tool_choice"),
            ({"type": "function", "function": {}}, TOOLS, "tool_choice"),
            ("required", None, "tools"),
            (NAMED, [], "tools"),
        ],
    )
    def test_unmet_choice_is_a_request_error(self, choice, tools, field):
        with pytest.raises(InvalidRequestError) as error:
            resolve_forced_tool_call(choice, tools, _tokenizer())
        assert error.value.field == field


class TestOtherApiChoices:
    """Anthropic's and the Responses API's choices, in OpenAI's form."""

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ({"type": "any"}, "required"),
            (ToolChoice(type="any"), "required"),
            ({"type": "tool", "name": "get_time"}, NAMED),
            (ToolChoice(type="tool", name="get_time"), NAMED),
            ({"type": "tool"}, {"type": "function", "function": {"name": None}}),
            ({"type": "auto"}, None),
            ({"type": "none"}, None),
            (None, None),
        ],
    )
    def test_anthropic(self, choice, expected):
        assert anthropic_tool_choice(choice) == expected

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("required", "required"),
            ({"type": "function", "name": "get_time"}, NAMED),
            (NAMED, NAMED),
            ("auto", None),
            ("none", None),
            ({"type": "allowed_tools", "mode": "required", "tools": []}, None),
            ({"type": "web_search_preview"}, None),
            (None, None),
        ],
    )
    def test_responses(self, choice, expected):
        assert responses_tool_choice(choice) == expected

    def test_anthropic_tool_without_a_name_is_a_request_error(self):
        choice = anthropic_tool_choice({"type": "tool"})
        with pytest.raises(InvalidRequestError) as error:
            resolve_forced_tool_call(choice, TOOLS, _tokenizer())
        assert error.value.field == "tool_choice"


class TestForcedCallMessages:
    FORCED = ForcedToolCall(opening="<tool_call>\n<function=f>", close="", single=True)

    def test_opening_is_a_partial_assistant_turn(self):
        messages = [{"role": "user", "content": "hi"}]
        assert forced_call_messages(messages, self.FORCED) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<tool_call>\n<function=f>"},
        ]

    def test_reasoning_goes_in_reasoning_content(self):
        turn = forced_call_messages([], self.FORCED, "why")[-1]
        assert turn == {
            "role": "assistant",
            "content": "<tool_call>\n<function=f>",
            "reasoning_content": "why",
        }


class FakeEngine:
    """Records stream_chat calls and replays one scripted stream per call."""

    def __init__(self, *streams):
        self.tokenizer = _tokenizer()
        self.streams = list(streams)
        self.calls = []

    async def stream_chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        for output in self.streams.pop(0):
            yield output


def _stream(pieces, *, finish_reason="stop", prompt_tokens=10, cached_tokens=0):
    """Engine outputs for ``pieces``, one token each."""
    outputs, text = [], ""
    for i, piece in enumerate(pieces, 1):
        text += piece
        last = i == len(pieces)
        outputs.append(
            GenerationOutput(
                text=text,
                new_text=piece,
                prompt_tokens=prompt_tokens,
                completion_tokens=i,
                cached_tokens=cached_tokens,
                finished=last,
                finish_reason=finish_reason if last else None,
                first_token_at=100.0 + i,
            )
        )
    return outputs


async def _collect(engine, messages, forced, **kwargs):
    return [
        o async for o in forced_tool_call_outputs(engine, messages, forced, **kwargs)
    ]


USER = [{"role": "user", "content": "Vienna?"}]
CALL = "\n<parameter=city>\nVienna\n</parameter>\n</function>\n"


class TestForcedToolCallOutputs:
    @pytest.mark.asyncio
    async def test_without_reasoning_outputs_start_with_the_opening(self):
        forced = resolve_forced_tool_call(NAMED, TOOLS, _tokenizer())
        messages = forced_call_messages(USER, forced)
        engine = FakeEngine(_stream([CALL[:20], CALL[20:]]))

        outputs = await _collect(
            engine, messages, forced, max_tokens=64, stop=["STOP"], _request_id="r1"
        )

        call = engine.calls[0]
        assert call["messages"] == messages
        # The named function is called once: generation stops at its close.
        assert call["stop"] == ["STOP", "</tool_call>"]
        assert call["_request_id"] == "r1"
        assert outputs[0].new_text == forced.opening + CALL[:20]
        # The stop string is not in the engine's text; the call gets it back.
        assert outputs[-1].text == forced.opening + CALL + "</tool_call>"
        assert "".join(o.new_text for o in outputs) == outputs[-1].text
        assert outputs[-1].completion_tokens == 2

    @pytest.mark.asyncio
    async def test_required_does_not_stop_after_the_first_call(self):
        forced = resolve_forced_tool_call("required", TOOLS, _tokenizer())
        engine = FakeEngine(_stream(["=get_time>", CALL, "</tool_call>"]))

        outputs = await _collect(engine, USER, forced, max_tokens=64)

        assert engine.calls[0]["stop"] == []
        assert outputs[-1].text == (
            "<tool_call>\n<function=get_time>" + CALL + "</tool_call>"
        )

    @pytest.mark.asyncio
    async def test_reasoning_comes_first_then_the_call(self):
        forced = replace(
            resolve_forced_tool_call(NAMED, TOOLS, _tokenizer()),
            thinks=True,
        )
        reasoning = ["The user", " asks the time.\n"]
        engine = FakeEngine(
            _stream(reasoning, prompt_tokens=10, cached_tokens=4),
            _stream([CALL], prompt_tokens=30, cached_tokens=10),
        )
        kwargs = {
            "max_tokens": 100,
            "stop": ["STOP"],
            "stop_skips_reasoning": True,
            "thinking_budget": 50,
            "_prepared_prompt": ("prompt", [1, 2]),
            "_request_id": "r1",
            "is_partial": False,
        }

        outputs = await _collect(engine, USER, forced, **kwargs)

        phase1, phase2 = engine.calls
        # Phase 1: the model reasons freely until it closes its reasoning.
        assert phase1["messages"] == USER
        assert phase1["stop"] == ["</think>"]
        assert phase1["stop_skips_reasoning"] is False
        assert phase1["thinking_budget"] == 50
        assert phase1["_prepared_prompt"] == ("prompt", [1, 2])
        # Phase 2 continues the same prompt with the reasoning and the opening.
        assert phase2["messages"] == [
            *USER,
            {
                "role": "assistant",
                "content": forced.opening,
                "reasoning_content": "The user asks the time.",
            },
        ]
        assert phase2["is_partial"] is True
        assert "_prepared_prompt" not in phase2
        assert "thinking_budget" not in phase2
        assert phase2["stop"] == ["STOP", "</tool_call>"]
        assert phase2["_request_id"] == "r1-call"
        bridge = "</think>\n\n" + forced.opening
        assert phase2["max_tokens"] == 100 - 2 - len(bridge)

        assert [o.finished for o in outputs] == [False, False, True]
        assert "".join(o.new_text for o in outputs) == outputs[-1].text
        assert outputs[-1].text == (
            "The user asks the time.\n" + bridge + CALL + "</tool_call>"
        )
        # Usage: the client's prompt, then reasoning + opening + call.
        assert outputs[-1].prompt_tokens == 10
        assert outputs[-1].cached_tokens == 4
        assert outputs[-1].completion_tokens == 2 + len(bridge) + 1
        assert outputs[-1].first_token_at == 102.0

    @pytest.mark.asyncio
    async def test_reasoning_that_runs_out_of_tokens_ends_the_reply(self):
        forced = replace(
            resolve_forced_tool_call(NAMED, TOOLS, _tokenizer()), thinks=True
        )
        engine = FakeEngine(_stream(["Hmm", " hmm"], finish_reason="length"))

        outputs = await _collect(engine, USER, forced, max_tokens=2)

        assert len(engine.calls) == 1
        assert outputs[-1].finish_reason == "length"
        assert outputs[-1].text == "Hmm hmm"
        assert "".join(o.new_text for o in outputs) == "Hmm hmm"
