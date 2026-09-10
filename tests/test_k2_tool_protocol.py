# SPDX-License-Identifier: Apache-2.0
"""K2 uses shared API parsing: valid calls are structured, failures remain text."""

import json
from types import SimpleNamespace

import pytest

from omlx.adapter.output_parser import _K2_MARKERS, K2HorizonOutputParserSession
from omlx.api.tool_calling import ToolCallStreamFilter, parse_tool_calls
from omlx.patches.k2_horizon.tool_parser import parse_tool_call


def k2_tokenizer():
    return SimpleNamespace(
        decode=lambda *_args, **_kwargs: "",
        detokenizer=None,
        has_tool_calling=True,
        tool_call_start="<ifm|tool_calls>",
        tool_call_end="</ifm|tool_calls>",
        tool_parser=parse_tool_call,
    )


def stream_text(text, chunk_size=1):
    stream_filter = ToolCallStreamFilter(k2_tokenizer())
    visible = "".join(
        stream_filter.feed(text[i : i + chunk_size])
        for i in range(0, len(text), chunk_size)
    )
    return visible + stream_filter.finish()


def test_unknown_name_reaches_client_without_renaming():
    tools = [{"type": "function", "function": {"name": "brave-search__search"}}]
    text = '<ifm|tool_calls><ifm|tool_call>{"name":"brave_search__search","arguments":{"query":"test"}}</ifm|tool_call></ifm|tool_calls>'
    clean, calls = parse_tool_calls(text, k2_tokenizer(), tools)
    assert clean == stream_text(text) == ""
    assert calls[0].function.name == "brave_search__search"
    assert json.loads(calls[0].function.arguments) == {"query": "test"}


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
@pytest.mark.parametrize(
    "text",
    [
        '<ifm|tool_calls><ifm|tool_call>{"name":',
        '<ifm|tool_calls><ifm|tool_call>{"name":"search"}</ifm|tool_call></ifm|tool_calls>',
        '<ifm|tool_calls><ifm|tool_call>{"name":"bash","arguments":{"command":"python3 verify.py</ifm|arg_value>\n</ifm|tool_call></ifm|tool_calls>',
        "before<ifm|tool_calls>garbage",
        "before<ifm|tool_calls><ifm|tool_calls><ifm|tool_call>search</ifm|tool_call></ifm|tool_calls>after",
        'before<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{"text":"literal </ifm|tool_calls> TAIL"}}</ifm|tool_call>garbage</ifm|tool_calls>after',
    ],
)
def test_malformed_output_reaches_client_as_text(text, chunk_size):
    tools = [{"type": "function", "function": {"name": "search"}}]
    clean, calls = parse_tool_calls(text, k2_tokenizer(), tools)
    assert clean == stream_text(text, chunk_size) == text
    assert calls is None


def test_plain_answer_streams_without_waiting_for_finalization():
    stream_filter = ToolCallStreamFilter(k2_tokenizer())
    assert stream_filter.feed("Hello!") == "Hello!"
    assert stream_filter.finish() == ""
    assert parse_tool_calls("Hello!", k2_tokenizer()) == ("Hello!", None)


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
    clean, calls = parse_tool_calls(text, k2_tokenizer(), tools)
    assert clean == stream_text(text) == ""
    assert json.loads(calls[0].function.arguments) == {"text": value, "count": 42}


def test_xml_untyped_text_preserves_whitespace_without_schema():
    text = (
        "<ifm|tool_call>edit<ifm|arg_key>text</ifm|arg_key>"
        "<ifm|arg_value>  indented\n</ifm|arg_value></ifm|tool_call>"
    )
    assert parse_tool_call(text)[0]["arguments"] == {"text": "  indented\n"}


@pytest.mark.parametrize(
    "marker", ["<ifm|tool_calls>", "</ifm|tool_call>", "</ifm|tool_calls>"]
)
def test_literal_markers_and_prose_between_valid_groups(marker):
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
    clean, calls = parse_tool_calls(text, k2_tokenizer(), tools)
    assert clean == stream_text(text) == "beforebetweenafter"
    assert [json.loads(call.function.arguments)["text"] for call in calls] == values
    assert len({call.id for call in calls}) == 3


@pytest.mark.parametrize("chunk_size", [1, 4096])
@pytest.mark.parametrize(
    "bad",
    [
        '<ifm|tool_calls><ifm|tool_call>{"name":',
        '<ifm|tool_calls><ifm|tool_call>{"name":"edit"}</ifm|tool_call></ifm|tool_calls>',
        '<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{}}</ifm|tool_call>garbage</ifm|tool_calls>',
    ],
)
def test_failed_group_parse_preserves_the_whole_attempt_without_partial_execution(
    bad, chunk_size
):
    good = '<ifm|tool_calls><ifm|tool_call>{"name":"edit","arguments":{"text":"literal </ifm|tool_calls>"}}</ifm|tool_call></ifm|tool_calls>'
    text = "before" + good + "between" + bad + "after"
    clean, calls = parse_tool_calls(text, k2_tokenizer())
    assert clean == stream_text(text, chunk_size) == text
    assert calls is None


def test_missing_grammar_backend_does_not_reject_k2_tools():
    # Keep this outside the xgrammar-dependent test module so a core-only
    # installation still tests the optional-dependency contract.
    from omlx.engine.batched import BatchedEngine
    from omlx.patches.k2_horizon.tool_grammar import compile_tool_grammar

    tools = [{"type": "function", "function": {"name": "read"}}]
    assert compile_tool_grammar(None, tools) is None
    existing = object()
    assert compile_tool_grammar(None, tools, existing) is existing
    engine = SimpleNamespace(model_type="k2_horizon", grammar_compiler=None)
    kwargs = {}
    BatchedEngine._prepare_k2_tool_grammar(engine, tools, kwargs)
    assert kwargs["compiled_grammar"] is None


def test_missing_grammar_does_not_apply_the_constrained_prefix_guard():
    from omlx.engine.batched import BatchedEngine
    from omlx.exceptions import InvalidRequestError

    engine = BatchedEngine("K2")
    engine._model = SimpleNamespace(args=SimpleNamespace(model_type="k2_horizon"))
    engine._tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt")
    engine._grammar_compiler_init_attempted = True
    tools = [{"type": "function", "function": {"name": "read"}}]
    messages = [{"role": "assistant", "content": "<ifm|tool_calls><ifm|tool_call>rea"}]
    assert engine._apply_chat_template(messages, tools, is_partial=True) == "prompt"
    engine._grammar_compiler = object()
    with pytest.raises(InvalidRequestError, match="prefix"):
        engine._apply_chat_template(messages, tools, is_partial=True)


def test_malformed_output_does_not_fail_its_batch_request(mock_model, mock_tokenizer):
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
    texts = ('<ifm|tool_calls><ifm|tool_call>{"name":', "Hello!")
    for uid, text in enumerate(texts, start=1):
        request_id = f"request-{uid}"
        request = Request(
            request_id=request_id,
            prompt="prompt",
            prompt_token_ids=[1, 3],
            num_prompt_tokens=2,
            sampling_params=SamplingParams(max_tokens=10),
            status=RequestStatus.RUNNING,
            batch_uid=uid,
            output_text=text,
        )
        scheduler.running[request_id] = scheduler.requests[request_id] = request
        scheduler.uid_to_request_id[uid] = request_id
        scheduler.request_id_to_uid[request_id] = uid
        scheduler._output_parser_sessions[request_id] = K2HorizonOutputParserSession(
            k2_tokenizer(), {marker: i for i, marker in enumerate(_K2_MARKERS)}
        )
        responses.append(
            SimpleNamespace(
                uid=uid, token=mock_tokenizer.eos_token_id, finish_reason="stop"
            )
        )
    outputs, finished = scheduler._process_batch_responses(responses)
    assert finished == {"request-1", "request-2"}
    assert all(output.error is None and output.finished for output in outputs)
    assert [output.output_text for output in outputs] == list(texts)
