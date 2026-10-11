# SPDX-License-Identifier: MIT
import json

import pytest
from test_output_parser import CohereTokenizer

from omlx.adapter.output_parser import detect_output_parser
from omlx.api.tool_calling import extract_tool_calls_with_thinking
from omlx.patches.deepseek_v41.tool_parser import parse_tool_call


class Tokenizer(CohereTokenizer):
    has_tool_calling = True
    tool_call_start = "<｜DSML｜ calls>"
    tool_call_end = "</｜DSML｜ calls>"
    tool_parser = staticmethod(parse_tool_call)


CALL = (
    '<｜DSML｜ calls>\n<｜DSML｜ invoke name="get">\n'
    '<｜DSML｜ parameter name="city" string="true">서울</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def replay(text, split, prefilled=False):
    chunks = [text[:split], text[split:]]
    tokenizer = Tokenizer(dict(enumerate(chunks)))
    factory = detect_output_parser("v41", tokenizer, {"model_type": "deepseek_v41"})
    assert factory.kind == "deepseek_v41"
    session = factory.create_session(tokenizer)
    if prefilled:
        session.notify_prefilled_thought()
    output = []
    stopped = []
    for i in range(len(chunks)):
        result = session.process_token(i)
        output.append(result.stream_text)
        stopped.append(result.is_stop)
    final = session.finalize()
    output.append(final.stream_text)
    return "".join(output), stopped, final


@pytest.mark.parametrize("prefilled", [False, True])
def test_reasoning_example_never_calls_or_stops_at_any_split(prefilled):
    text = ("" if prefilled else "<think>") + "Example: " + CALL + "</think>Answer"
    for split in range(len(text) + 1):
        visible, stops, final = replay(text, split, prefilled)
        assert visible == text, split
        assert not any(stops), split
        assert not final.tool_calls, split


def test_real_call_after_reasoning_at_any_split():
    reasoning = "<think>Example: " + CALL + "</think>"
    text = reasoning + CALL + "discard trailing text"
    for split in range(len(text) + 1):
        visible, stops, final = replay(text, split)
        assert visible == reasoning, split
        assert any(stops), split
        assert final.finish_reason == "tool_calls"
        assert len(final.tool_calls) == 1
        assert json.loads(final.tool_calls[0]["arguments"]) == {"city": "서울"}


@pytest.mark.parametrize(
    "text", [CALL[:-3], CALL.replace('string="true"', 'string="bad"')]
)
def test_incomplete_or_invalid_envelope_does_not_stop(text):
    for split in range(len(text) + 1):
        visible, stops, final = replay(text, split)
        assert visible == text, split
        assert not any(stops), split
        assert not final.tool_calls


def test_http_extraction_does_not_promote_reasoning_calls():
    tools = [{"type": "function", "function": {"name": "get"}}]
    result = extract_tool_calls_with_thinking(CALL, "Answer", Tokenizer({}), tools)
    assert result.cleaned_thinking == CALL
    assert not result.tool_calls
    assert not result.tool_calls_from_thinking


def test_server_thinking_filter_preserves_literal_dsml():
    from omlx.api.tool_calling import ToolCallStreamFilter

    stream = ToolCallStreamFilter(Tokenizer({}), consume_dsml_separator=False)
    assert "".join(stream.feed(char) for char in CALL) + stream.finish() == CALL


@pytest.mark.parametrize(
    "namespace", ["weather", {"name": "weather", "description": "Weather tools"}]
)
def test_namespace_survives_request_encoding_completion_and_history(namespace):
    from omlx.api.openai_models import ToolCall, ToolDefinition
    from omlx.api.tool_calling import parse_tool_calls
    from omlx.patches.deepseek_v41 import encoding

    raw = {
        "type": "function",
        "namespace": namespace,
        "function": {"name": "get", "parameters": {"type": "object"}},
    }
    tool = ToolDefinition.model_validate(raw).model_dump()
    assert tool["function"]["name"] == "weather::get"
    assert encoding.tools_from_openai_format(
        [tool]
    ) == encoding.tools_from_openai_format([raw])
    wire = CALL.replace('name="get"', 'name="weather::get"')
    _, calls = parse_tool_calls(wire, Tokenizer({}), [raw])
    assert calls[0].function.name == "weather::get"
    restored = ToolCall.model_validate(calls[0].model_dump()).model_dump()
    assert encoding.tool_calls_from_openai_format([restored]) == [
        {"namespace": "weather", "name": "get", "arguments": '{"city": "서울"}'}
    ]
    official_call = {
        "id": "call_1",
        "type": "function",
        "namespace": "weather",
        "function": {"name": "get", "arguments": "{}"},
    }
    assert ToolCall.model_validate(official_call).function.name == "weather::get"
    assert raw["function"]["name"] == "get"


def test_namespace_conflict_is_rejected_without_affecting_plain_tools():
    from omlx.api.openai_models import ToolDefinition

    plain = {"type": "function", "function": {"name": "get"}}
    assert ToolDefinition.model_validate(plain).model_dump() == plain
    with pytest.raises(ValueError, match="Conflicting"):
        ToolDefinition.model_validate(
            {
                "type": "function",
                "namespace": "weather",
                "function": {"name": "calendar::get"},
            }
        )


@pytest.mark.parametrize(
    "namespace,function",
    [
        ({"name": "weather", "description": 42}, {"name": "get"}),
        (
            {"name": "weather", "description": "Weather tools"},
            {"name": "get", "description": []},
        ),
        ("weather", {"name": "weather::"}),
    ],
)
def test_invalid_namespace_fields_raise_validation_errors(namespace, function):
    from pydantic import ValidationError

    from omlx.api.openai_models import ToolDefinition

    with pytest.raises(ValidationError):
        ToolDefinition.model_validate({"namespace": namespace, "function": function})


def test_same_bare_name_in_two_namespaces_stays_distinct():
    first = CALL.replace('name="get"', 'name="weather::get"')
    second = CALL.replace('name="get"', 'name="calendar::get"')
    wire = first.removesuffix(Tokenizer.tool_call_end) + second.removeprefix(
        Tokenizer.tool_call_start + "\n"
    )
    _, _, final = replay(wire, len(wire) // 2)
    assert [call["name"] for call in final.tool_calls] == [
        "weather::get",
        "calendar::get",
    ]


def test_invalid_json_cannot_end_tool_turn():
    wire = CALL.replace('string="true">서울', 'string="false">[broken')
    visible, stops, final = replay(wire, len(wire) // 2)
    assert visible == wire
    assert not any(stops)
    assert not final.tool_calls
