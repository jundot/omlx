# SPDX-License-Identifier: Apache-2.0
"""Compare partial argument delivery with the native Chat completion path."""

import json
import random
from types import SimpleNamespace

import pytest
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

from omlx.api.openai_models import ChatCompletionRequest
from omlx.server import stream_chat_completion

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "number": {"type": "number"},
                    "flag": {"type": "boolean"},
                    "items": {"type": "array"},
                },
            },
        },
    }
]


def envelope(arguments, name="write"):
    return (
        "<tool_call>\n<function="
        + name
        + ">\n"
        + "".join(
            "<parameter="
            + key
            + ">\n"
            + (value if isinstance(value, str) else json.dumps(value))
            + "\n</parameter>\n"
            for key, value in (
                arguments.items() if isinstance(arguments, dict) else arguments
            )
        )
        + "</function>\n</tool_call>"
    )


class Engine:
    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=parse_tool_call,
    )

    def __init__(self, raw, size, capable=True):
        self.raw = raw
        self.size = size
        self.position = 0
        self.supports_early_tool_call_streaming = capable

    async def stream_chat(self, **kwargs):
        for position in range(0, len(self.raw), self.size):
            self.position = min(len(self.raw), position + self.size)
            yield SimpleNamespace(
                new_text=self.raw[position : self.position],
                tool_calls=None,
                finished=False,
                finish_reason="stop",
            )


async def run(raw, size, incremental, capable=True):
    engine = Engine(raw, size, capable)
    request = ChatCompletionRequest(
        model="model-alias",
        messages=[{"role": "user", "content": "fixture"}],
        stream=True,
        stream_options={"incremental_tool_arguments": incremental},
    )
    calls = {}
    positions = []
    errors = []
    finish = None
    content = []
    reasoning = []
    ordered = []
    async for event in stream_chat_completion(engine, [], request, tools=TOOLS):
        if event == "data: [DONE]\n\n":
            continue
        data = json.loads(event[6:])
        if data.get("error"):
            errors.append(data["error"])
        for choice in data.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            content.append(delta.get("content", ""))
            reasoning.append(delta.get("reasoning_content", ""))
            if delta.get("content"):
                ordered.append(("content", delta["content"]))
            for tc in delta.get("tool_calls", []):
                call = calls.setdefault(
                    tc["index"], {"name": "", "arguments": "", "id": ""}
                )
                call["id"] += tc.get("id", "")
                for key in ("name", "arguments"):
                    call[key] += tc["function"].get(key, "")
                if tc["function"].get("arguments"):
                    positions.append(engine.position)
                if tc["function"].get("name"):
                    ordered.append(("tool", tc["function"]["name"]))
    return dict(
        calls=list(calls.values()),
        positions=positions,
        errors=errors,
        finish=finish,
        content="".join(content),
        reasoning="".join(reasoning),
        ordered=ordered,
    )


def signature(result):
    return dict(
        calls=[
            {k: v for k, v in call.items() if k != "id"} for call in result["calls"]
        ],
        finish=result["finish"],
        content=result["content"],
        reasoning=result["reasoning"],
    )


VALUES = [
    "",
    "null",
    "NULL",
    " null",
    "\n",
    "\n\n",
    "  x  ",
    '"a"\\b\n🚀 café',
    "</tool_call>",
    "<parameter=x>",
    "\r\nend\n\n",
    "x" * 10000,
]
RNG = random.Random(192)
VALUES += [
    "".join(RNG.choice('abc <>/\\"\n\r\t🚀é') for _ in range(RNG.randrange(10, 600)))
    for _ in range(40)
]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 7, 13, 64, 1024])
@pytest.mark.parametrize("value", VALUES)
async def test_arguments_match_native_parser(value, size):
    raw = (
        "<think>planning</think>before\n"
        + envelope(
            {"content": value, "number": 3.0, "flag": True, "items": [1, {"x": "y"}]}
        )
        + "after"
    )
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    assert all(call["id"].startswith("call_") for call in actual["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 7, 13, 64, 1024])
@pytest.mark.parametrize(
    "parameters",
    [
        [("content", ""), ("path", "demo.txt"), ("content", "")],
        [("content", "same"), ("content", "same")],
        [("content", '"quoted"\\text\n'), ("content", '"quoted"\\text\n')],
        [("content", "🚀 café"), ("content", "🚀 café")],
        [("content", "x" * 5000), ("content", "x" * 5000)],
        [("content", "NULL"), ("content", "null")],
        [("number", "3.0"), ("number", "3")],
        [("flag", "TRUE"), ("flag", "true")],
        [("items", '[1, {"x": "y"}]'), ("items", '[1,{"x":"y"}]')],
        [("content", "same"), ("content", "different"), ("content", "same")],
    ],
)
async def test_repeated_parameters_match_native_without_duplicate_json_keys(
    parameters, size
):
    raw = envelope(parameters)
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    pairs = json.loads(actual["calls"][0]["arguments"], object_pairs_hook=list)
    names = [name for name, _ in pairs]
    assert len(names) == len(set(names))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 7, 13, 64, 1024])
@pytest.mark.parametrize(
    "parameters",
    [
        [("content", "first"), ("content", "second")],
        [("number", 1), ("number", 2)],
        [("items", [1]), ("items", [2])],
    ],
)
async def test_changed_final_parameter_cannot_validate_previously_emitted_value(
    parameters, size
):
    raw = envelope(parameters)
    native = await run(raw, size, False)
    assert not native["errors"] and native["finish"] == "tool_calls"
    actual = await run(raw, size, True)
    assert actual["errors"] and actual["finish"] is None
    with pytest.raises(json.JSONDecodeError):
        json.loads(actual["calls"][0]["arguments"])


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 7, 13, 64, 1024])
async def test_parameter_tracking_resets_for_the_next_call(size):
    raw = envelope([("content", ""), ("content", "")]) + envelope(
        {"content": "next"}
    )
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    assert len(actual["calls"]) == 2
    assert len({call["id"] for call in actual["calls"]}) == 2


@pytest.mark.asyncio
async def test_arguments_arrive_before_envelope_closes():
    raw = envelope({"path": "demo.txt", "content": "abcdefghij" * 5000})
    result = await run(raw, 13, True)
    assert not result["errors"]
    assert any(position < len(raw) // 2 for position in result["positions"])
    assert json.loads(result["calls"][0]["arguments"])["content"] == "abcdefghij" * 5000
    assert result["finish"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("capable,incremental", [(False, True), (True, False)])
async def test_explicit_opt_in_and_engine_capability_are_both_required(
    capable, incremental
):
    raw = envelope({"content": "abcdefghij" * 1000})
    result = await run(raw, 13, incremental, capable)
    assert not any(position < len(raw) // 2 for position in result["positions"])
    assert not result["errors"]


@pytest.mark.asyncio
async def test_coalesced_calls_preserve_order_and_occurrences():
    raw = (
        "before "
        + envelope({"content": "first"})
        + " between "
        + envelope({"content": "second"})
        + " after"
    )
    result = await run(raw, len(raw), True)
    assert not result["errors"]
    assert result["ordered"] == [
        ("content", "before "),
        ("tool", "write"),
        ("content", " between "),
        ("tool", "write"),
        ("content", " after"),
    ]
    assert [json.loads(call["arguments"])["content"] for call in result["calls"]] == [
        "first",
        "second",
    ]
    assert len({call["id"] for call in result["calls"]}) == 2


@pytest.mark.asyncio
async def test_unfinished_partial_call_never_completes_successfully():
    raw = envelope({"content": "x" * 1000})[:-20]
    result = await run(raw, 13, True)
    assert result["positions"]
    assert result["errors"]
    assert result["finish"] is None
    with pytest.raises(json.JSONDecodeError):
        json.loads(result["calls"][0]["arguments"])


@pytest.mark.asyncio
async def test_unsupported_json_dialect_keeps_native_fallback():
    raw = '<tool_call>{"name":"write","arguments":{"content":"example"}}</tool_call>'
    assert signature(await run(raw, 13, True)) == signature(await run(raw, 13, False))


@pytest.mark.asyncio
async def test_unknown_function_is_not_emitted_early():
    result = await run(envelope({"content": "x" * 1000}, "unknown"), 13, True)
    assert not result["errors"]
    assert not any(position < 500 for position in result["positions"])


def test_unfinished_header_has_bounded_buffer():
    from omlx.api.qwen_argument_stream import QwenArgumentStream

    stream = QwenArgumentStream(TOOLS)
    assert not stream.feed("<tool_call><function=" + "x" * 5000)
    assert not stream.enabled and not stream.buf


def test_partial_envelope_size_limit_fails_closed():
    from omlx.api.qwen_argument_stream import QwenArgumentStream

    stream = QwenArgumentStream(TOOLS)
    assert stream.feed("<tool_call><function=write><parameter=content>value")
    with pytest.raises(ValueError, match="character limit"):
        stream.feed("x" * stream.MAX_ENVELOPE_CHARS)


@pytest.mark.asyncio
async def test_cancellation_closes_the_producer():
    class Cancellable(Engine):
        closed = False

        async def stream_chat(self, **kwargs):
            try:
                async for item in super().stream_chat(**kwargs):
                    yield item
            finally:
                self.closed = True

    engine = Cancellable(envelope({"content": "x" * 10000}), 13)
    request = ChatCompletionRequest(
        model="model-alias",
        messages=[],
        stream=True,
        stream_options={"incremental_tool_arguments": True},
    )
    output = stream_chat_completion(engine, [], request, tools=TOOLS)
    async for event in output:
        if '"arguments"' in event:
            break
    await output.aclose()
    assert engine.closed
