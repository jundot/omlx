# SPDX-License-Identifier: Apache-2.0
"""Client-visible K2 tool calls and request-isolated parse failures."""

import json
from types import SimpleNamespace

import pytest

from omlx.adapter.output_parser import _K2_MARKERS, K2HorizonOutputParserSession
from omlx.patches.k2_horizon.tool_parser import parse_tool_call


def session_for(text, tools):
    ids = {marker: i for i, marker in enumerate(_K2_MARKERS)}
    tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "",
        detokenizer=None,
        has_tool_calling=True,
        tool_call_start="<ifm|tool_calls>",
        tool_call_end="</ifm|tool_calls>",
        tool_parser=parse_tool_call,
    )
    session = K2HorizonOutputParserSession(tokenizer, ids, tools=tools)
    session._emit(text)
    return session


def test_unknown_name_reaches_client_without_renaming():
    tools = [{"type": "function", "function": {"name": "brave-search__search"}}]
    text = '<ifm|tool_calls><ifm|tool_call>{"name":"brave_search__search","arguments":{"query":"test"}}</ifm|tool_call></ifm|tool_calls>'
    result = session_for(text, tools).finalize()
    assert result.error is None
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["name"] == "brave_search__search"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "test"}


@pytest.mark.parametrize(
    "text",
    [
        '<ifm|tool_calls><ifm|tool_call>{"name":',
        '<ifm|tool_calls><ifm|tool_call>{"name":"search"}</ifm|tool_call></ifm|tool_calls>',
    ],
)
def test_malformed_call_returns_request_error(text):
    result = session_for(
        text, [{"type": "function", "function": {"name": "search"}}]
    ).finalize()
    assert result.error and "malformed tool call" in result.error
    assert result.tool_calls == []


def test_plain_answer_has_no_tool_error():
    result = session_for(
        "Hello!", [{"type": "function", "function": {"name": "search"}}]
    ).finalize()
    assert result.error is None


@pytest.mark.parametrize(
    "value", ["  indented\n", "\tline\r\n", "", " \t\n", " 42 ", " true ", ' {"a": 1} ']
)
@pytest.mark.parametrize("typed", [False, True])
def test_xml_string_arguments_preserve_exact_text(value, typed):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {} if typed else {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }
    ]
    type_tag = "<ifm|arg_type>string</ifm|arg_type>" if typed else ""
    text = (
        "<ifm|tool_calls><ifm|tool_call>edit"
        f"<ifm|arg_key>text</ifm|arg_key>{type_tag}"
        f"<ifm|arg_value>{value}</ifm|arg_value>"
        "<ifm|arg_key>count</ifm|arg_key><ifm|arg_value> 42 </ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    session = session_for("", tools)
    for character in text:
        session._emit(character)
    result = session.finalize()
    assert result.error is None
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value, "count": 42}


def test_xml_untyped_text_preserves_whitespace_without_schema():
    text = (
        "<ifm|tool_call>edit<ifm|arg_key>text</ifm|arg_key>"
        "<ifm|arg_value>  indented\n</ifm|arg_value></ifm|tool_call>"
    )
    assert parse_tool_call(text)[0]["arguments"] == {"text": "  indented\n"}


def test_one_malformed_group_does_not_disappear_beside_a_valid_call():
    good = '<ifm|tool_calls><ifm|tool_call>{"name":"read","arguments":{}}</ifm|tool_call></ifm|tool_calls>'
    bad = '<ifm|tool_calls><ifm|tool_call>{"name":"read"}</ifm|tool_call></ifm|tool_calls>'
    result = session_for(
        good + bad, [{"type": "function", "function": {"name": "read"}}]
    ).finalize()
    assert result.error
    assert result.tool_calls == []


def test_argument_may_mention_open_marker():
    text = '<ifm|tool_calls><ifm|tool_call>{"name":"read","arguments":{"text":"literal <ifm|tool_calls>"}}</ifm|tool_call></ifm|tool_calls>'
    result = session_for(
        text, [{"type": "function", "function": {"name": "read"}}]
    ).finalize()
    assert result.error is None
    assert (
        json.loads(result.tool_calls[0]["arguments"])["text"]
        == "literal <ifm|tool_calls>"
    )


@pytest.mark.parametrize("marker", ["</ifm|tool_call>", "</ifm|tool_calls>"])
def test_json_closing_markers_stay_inside_arguments(marker):
    from omlx.api.tool_calling import parse_tool_calls

    tools = [{"type": "function", "function": {"name": "edit"}}]
    values = [f'escaped "quote" \\ {marker} tail', "second " + marker, "third"]
    bodies = [
        "<ifm|tool_call>"
        + json.dumps({"name": "edit", "arguments": {"text": value}})
        + "</ifm|tool_call>"
        for value in values
    ]
    first = "<ifm|tool_calls> \n" + " \n".join(bodies[:2]) + " \n</ifm|tool_calls>"
    second = "<ifm|tool_calls>" + bodies[2] + "</ifm|tool_calls>"
    text = "before" + first + "between" + second + "after"
    assert [
        call["arguments"]["text"] for call in parse_tool_call("".join(bodies))
    ] == values

    session = session_for("", tools)
    visible = "".join(session._emit(character).visible_text for character in text)
    result = session.finalize()
    assert result.error is None
    assert visible + result.visible_text == "beforebetweenafter"
    assert [
        json.loads(call["arguments"])["text"] for call in result.tool_calls
    ] == values
    assert len({call["id"] for call in result.tool_calls}) == 3
    clean, calls = parse_tool_calls(text, session._tokenizer, tools)
    assert clean == "beforebetweenafter"
    assert [json.loads(call.function.arguments)["text"] for call in calls] == values


@pytest.mark.parametrize(
    "tail",
    [
        '<ifm|tool_calls><ifm|tool_call>{"name":',
        '<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{}}</ifm|tool_call>garbage</ifm|tool_calls>',
    ],
)
def test_marker_argument_does_not_hide_a_malformed_trailing_group(tail):
    good = '<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{"text":"literal </ifm|tool_calls>"}}</ifm|tool_call></ifm|tool_calls>'
    result = session_for(
        good + tail, [{"type": "function", "function": {"name": "edit"}}]
    ).finalize()
    assert result.error
    assert result.tool_calls == []


@pytest.mark.parametrize(
    "text",
    [
        "before<ifm|tool_calls>garbage",
        'before<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{"text":"literal </ifm|tool_calls> TAIL"}}</ifm|tool_call>garbage</ifm|tool_calls>after',
    ],
)
def test_malformed_group_content_is_not_streamed(text):
    session = session_for("", [{"type": "function", "function": {"name": "edit"}}])
    visible = "".join(
        session._emit(character).visible_text
        for character in text
    )
    result = session.finalize()
    assert visible + result.visible_text == "before"
    assert result.error
    assert result.tool_calls == []


def test_parser_error_stays_with_its_batch_request(mock_model, mock_tokenizer):
    from omlx.engine_core import _raise_request_output_error
    from omlx.request import Request, RequestStatus, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(model_name="test-model"),
    )
    scheduler._output_parser_factory = SimpleNamespace(
        kind="k2_horizon", stop_token_ids=set(), thinking_end_text=None
    )
    responses = []
    for uid, text in enumerate(
        ('<ifm|tool_calls><ifm|tool_call>{"name":', "Hello!"), start=1
    ):
        request_id = f"request-{uid}"
        request = Request(
            request_id=request_id,
            prompt="prompt",
            prompt_token_ids=[1, 3],
            num_prompt_tokens=2,
            sampling_params=SamplingParams(max_tokens=10),
            status=RequestStatus.RUNNING,
            batch_uid=uid,
        )
        scheduler.running[request_id] = scheduler.requests[request_id] = request
        scheduler.uid_to_request_id[uid] = request_id
        scheduler.request_id_to_uid[request_id] = uid
        scheduler._output_parser_sessions[request_id] = session_for(
            text, [{"type": "function", "function": {"name": "read"}}]
        )
        responses.append(
            SimpleNamespace(
                uid=uid, token=mock_tokenizer.eos_token_id, finish_reason="stop"
            )
        )
    outputs, finished = scheduler._process_batch_responses(responses)
    assert finished == {"request-1", "request-2"}
    with pytest.raises(RuntimeError, match="malformed tool call"):
        _raise_request_output_error(outputs[0])
    assert outputs[1].error is None
    assert outputs[1].finished
