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
                prompt_tokens=10,
                completion_tokens=25,
                cached_tokens=0,
            )


async def run(raw, size, incremental, capable=True, tools=TOOLS):
    engine = Engine(raw, size, capable and incremental)
    request = ChatCompletionRequest(
        model="model-alias",
        messages=[{"role": "user", "content": "fixture"}],
        stream=True,
    )
    calls = {}
    positions = []
    argument_lengths = []
    errors = []
    finish = None
    content = []
    reasoning = []
    ordered = []
    async for event in stream_chat_completion(engine, [], request, tools=tools):
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
                    argument_lengths.append(len(tc["function"]["arguments"]))
                if tc["function"].get("name"):
                    ordered.append(("tool", tc["function"]["name"]))
    return dict(
        calls=list(calls.values()),
        positions=positions,
        argument_lengths=argument_lengths,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 2, 3, 7, 128, 4096])
@pytest.mark.parametrize("content_first", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_docstring_body_streams_before_close(size, content_first, recover):
    value = '"""Module docstring: café 🚀 with \\"quotes\\"."""\n' + "x" * 65536
    params = [("content", value), ("path", "probe.py")]
    if not content_first:
        params.reverse()
    raw = envelope(params)
    if recover:
        raw = raw.removesuffix("</tool_call>")
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    # Require body bytes, not just a name/opening brace, during generation.
    halfway = raw.index(value) + len(value) // 2
    early = sum(
        n
        for p, n in zip(actual["positions"], actual["argument_lengths"])
        if p < halfway
    )
    assert early > 16000
    assert json.loads(actual["calls"][0]["arguments"])["content"] == value


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 2, 13, 4096])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        '"',
        '""',
        '"""',
        '""""',
        '"quoted"',
        '"escaped\\"quote"',
        '"\\u263a"',
        '"""doc"""\n',
        "'''doc'''\n",
        '\n"""doc"""',
        '  """doc"""',
        '\t"""doc"""',
        '\r\n"""doc"""',
        '\u00a0"""doc"""',
        '"""doc"""  ',
        '"""doc"""\n\n',
    ],
)
async def test_quote_and_whitespace_prefixes_keep_native_recovery_semantics(
    size, recover, value
):
    raw = envelope({"content": value, "path": "probe.py"})
    if recover:
        raw = raw.removesuffix("</tool_call>")
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 4096])
async def test_docstring_calls_keep_distinct_ids_and_append_only_arguments(size):
    value = '"""doc"""\n' + "x" * 4096
    raw = envelope({"content": value, "path": "one.py"}) + envelope(
        {"path": "two.py", "content": value}
    )
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    assert len({call["id"] for call in actual["calls"]}) == 2


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
    raw = envelope([("content", ""), ("content", "")]) + envelope({"content": "next"})
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
async def test_engines_without_raw_qwen_capability_keep_buffered_arguments(
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


def test_partial_envelope_size_limit_defers_final_validation():
    from omlx.api.qwen_argument_stream import QwenArgumentStream

    stream = QwenArgumentStream(TOOLS)
    assert stream.feed("<tool_call><function=write><parameter=content>value")
    before = stream.current["digest"].digest(), stream.current["length"]
    assert stream.feed("x" * stream.MAX_ENVELOPE_CHARS) == []
    assert not stream.enabled and not stream.buf
    assert before == (stream.current["digest"].digest(), stream.current["length"])


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
    )
    output = stream_chat_completion(engine, [], request, tools=TOOLS)
    async for event in output:
        if '"arguments"' in event:
            break
    await output.aclose()
    assert engine.closed


def test_emitted_string_body_is_not_retained():
    import tracemalloc
    from omlx.api.qwen_argument_stream import QwenArgumentStream

    stream = QwenArgumentStream(TOOLS)
    stream.feed("<tool_call><function=write><parameter=content>started")
    tracemalloc.start()
    try:
        for _ in range(1024):
            stream.feed("x" * 1024)
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert stream.current["length"] > 1024 * 1023
    assert retained < 65536
    assert peak < 131072


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 1024])
@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"description": "any value"},
        {"anyOf": [{"type": "string"}, {"type": "number"}]},
    ],
)
@pytest.mark.parametrize(
    "value", ['"quoted"', "42", " null ", '{"x": 1}', "  plain  ", "🚀\ntext"]
)
async def test_untyped_properties_match_final_qwen_wrapper(size, schema, value):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": schema},
                },
            },
        }
    ]
    raw = envelope({"content": value})
    native = await run(raw, size, False, tools=tools)
    actual = await run(raw, size, True, tools=tools)
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 1024])
@pytest.mark.parametrize(
    "value",
    [
        "  spaced  ",
        '"quoted"',
        "null ",
        "tail  ",
        "body\n\n",
        "body" + " " * 200 + "tail",
    ],
)
@pytest.mark.parametrize("recover", [False, True])
async def test_string_prefix_matches_both_normal_and_recovered_calls(
    size, value, recover
):
    raw = envelope({"content": value})
    if recover:
        raw = raw.removesuffix("</tool_call>")
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    if native["errors"]:
        assert actual["errors"] == native["errors"]
        for call in actual["calls"]:
            with pytest.raises(json.JSONDecodeError):
                json.loads(call["arguments"])
        return
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 1024])
@pytest.mark.parametrize(
    "header", [" content", "content ", "\tcontent", "content ignored"]
)
async def test_parameter_header_matches_native_name_and_value_boundary(size, header):
    raw = envelope({"content": "text"}).replace(
        "<parameter=content>", "<parameter=" + header + ">"
    )
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 4096])
@pytest.mark.parametrize("schema", [True, False, {}, {"description": "untyped"}])
async def test_fallback_schema_and_literal_parameter_marker_keep_final_semantics(
    size, schema
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": schema, "path": {"type": "string"}},
                },
            },
        }
    ]
    raw = envelope({"content": "a</parameter>b", "path": "file"})
    native = await run(raw, size, False, tools=tools)
    actual = await run(raw, size, True, tools=tools)
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
async def test_completed_queue_overflow_preserves_partial_id_and_later_calls():
    from omlx.api.tool_calling import ToolCallStreamFilter

    raw = "".join(
        envelope({"content": str(i)})
        for i in range(ToolCallStreamFilter._COMPLETED_ENVELOPE_MAX_COUNT + 2)
    )

    # Split the first call, then deliver enough siblings to overflow one FIFO.
    class Coalesced(Engine):
        async def stream_chat(self, **kwargs):
            for text in (raw[:70], raw[70:]):
                yield SimpleNamespace(
                    new_text=text,
                    tool_calls=None,
                    finished=False,
                    finish_reason="stop",
                    prompt_tokens=10,
                    completion_tokens=25,
                    cached_tokens=0,
                )

    engine = Coalesced(raw, 70)
    request = ChatCompletionRequest(model="test", messages=[], stream=True)
    calls = {}
    async for event in stream_chat_completion(engine, [], request, tools=TOOLS):
        if not event.startswith("data: {"):
            continue
        payload = json.loads(event[6:])
        assert "error" not in payload
        for choice in payload.get("choices", []):
            for tc in choice.get("delta", {}).get("tool_calls", []):
                call = calls.setdefault(tc["index"], {"id": "", "arguments": ""})
                call["id"] += tc.get("id", "")
                call["arguments"] += tc["function"].get("arguments", "")
    assert list(calls) == list(range(18))
    assert len({call["id"] for call in calls.values()}) == 18
    assert [
        json.loads(call["arguments"])["content"] for call in calls.values()
    ] == list(map(str, range(18)))


@pytest.mark.asyncio
async def test_large_valid_envelope_falls_back_after_incremental_buffer_limit():
    raw = envelope({"content": "x" * (2 * 1024 * 1024)})
    native = await run(raw, 4096, False)
    actual = await run(raw, 4096, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)
    assert any(p < len(raw) // 2 for p in actual["positions"])


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 4096])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "parameters",
    [
        {"undeclared": "42"},
        {"flag": " true "},
        {"content": "a</parameter>b"},
        {"content": "plain", "flag": " true ", "number": "invalid"},
        {"content": '"quoted"', "number": "invalid"},
    ],
)
async def test_each_emitted_value_is_settled_for_native_and_fallback(
    size, recover, parameters
):
    raw = envelope(parameters)
    if recover:
        raw = raw.removesuffix("</tool_call>")
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    if native["errors"]:
        assert actual["errors"] == native["errors"]
        for call in actual["calls"]:
            with pytest.raises(json.JSONDecodeError):
                json.loads(call["arguments"])
    else:
        assert not actual["errors"]
        assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 13, 4096])
@pytest.mark.parametrize("header", [" content", "\tcontent", "content!"])
async def test_native_only_parameter_names_defer_until_recovery_is_known(size, header):
    raw = (
        envelope({"content": "text"})
        .replace("<parameter=content>", "<parameter=" + header + ">")
        .removesuffix("</tool_call>")
    )
    native = await run(raw, size, False)
    actual = await run(raw, size, True)
    assert not actual["errors"]
    assert signature(actual) == signature(native)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,bad",
    [
        ("length", "<tool_call><function=write><parameter=content>partial"),
        ("stop", "<tool_call>malformed</tool_call>"),
    ],
)
async def test_overflow_complete_prefix_is_not_resent_when_a_later_call_fails(
    reason, bad
):
    from tests.integration.test_e2e_streaming import (
        _recovery_stream,
        _chat_argument_calls,
    )

    value = "x" * (2 * 1024 * 1024)
    events = await _recovery_stream(
        envelope({"content": value}) + bad, chunk_size=4096, finish_reason=reason
    )
    calls = _chat_argument_calls(events)
    assert len(calls) == 1
    assert json.loads(calls[0]["arguments"]) == {"content": value}
    starts = [
        tc
        for event in events
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", [])
        if "id" in tc
    ]
    assert len(starts) == 1 and starts[0]["index"] == 0
    if reason == "length":
        assert not any(e.get("error") for e in events)
        assert any(
            c.get("finish_reason") == "length"
            for e in events
            for c in e.get("choices", [])
        )
    else:
        assert events[-1]["error"]["code"] == "invalid_tool_call"
