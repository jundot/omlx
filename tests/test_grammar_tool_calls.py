# SPDX-License-Identifier: Apache-2.0
"""Structured final answers must leave investigation tools available."""

import importlib
import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.api.grammar import GrammarConstraintProcessor
from omlx.api.tool_calling import parse_tool_calls
from omlx.server import _compile_grammar_for_request, _json_instruction_for_request

xgr = pytest.importorskip("xgrammar")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["confirmed", "rejected", "uncertain"]},
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"body": {"type": "string"}},
                "required": ["body"],
                "additionalProperties": False,
            },
        },
    }
]
FINAL = '{"verdict":"confirmed","evidence":"checked source"}'
TOOL_CALL = (
    "<tool_call>\n<function=write_file>\n"
    "<parameter=body>\nhello\n</parameter>\n</function>\n</tool_call>"
)
VOCAB = [bytes([i]) for i in range(256)] + [b"</s>", b"\n\n", b"\n\n{"]


@pytest.fixture(scope="module")
def compiler():
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(VOCAB, stop_token_ids=[256]), max_threads=1
    )


def native_tokenizer(parser):
    module = importlib.import_module(f"mlx_lm.tool_parsers.{parser}")
    return SimpleNamespace(
        has_tool_calling=True,
        tool_parser=module.parse_tool_call,
        tool_call_start=module.tool_call_start,
        tool_call_end=module.tool_call_end,
    )


def compile_format(compiler, parser="qwen3_coder", **kwargs):
    engine = SimpleNamespace(
        grammar_compiler=compiler,
        tokenizer=native_tokenizer(parser),
    )
    options = {"structured_outputs": {"json": SCHEMA}, **kwargs}
    return _compile_grammar_for_request(engine, tools=TOOLS, **options)


def accepts(compiled, text):
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string(text) and matcher.accept_token(256)


@pytest.mark.parametrize("thinking", [None, False, True])
def test_schema_leaves_investigation_tools_available(compiler, thinking):
    compiled = compile_format(
        compiler,
        reasoning_parser="qwen_3_5" if thinking is not None else None,
        chat_template_kwargs={"enable_thinking": thinking},
    )
    prefix = "Checked the source.</think>\n\n" if thinking else ""
    for output in (FINAL, "\n\n" + FINAL + "\r\n", TOOL_CALL, TOOL_CALL * 2):
        assert accepts(compiled, prefix + output), output
    for invalid in (
        "plain text",
        '{"verdict":"invented","evidence":"none"}',
        '{"verdict":"confirmed"}',
        TOOL_CALL.replace("write_file", "unknown_tool"),
        "<tool_call>",
        "\n\n",
        " " * 33 + FINAL,
    ):
        assert not accepts(compiled, prefix + invalid), invalid


def test_newline_tokens_do_not_mask_the_json_branch(compiler):
    proc = GrammarConstraintProcessor(compile_format(compiler), len(VOCAB))
    scores = proc([], mx.zeros((1, len(VOCAB)))).tolist()[0]
    assert scores[258] == 0, "A token containing both newlines and '{' must be allowed"
    proc.accept_token(257)
    scores = proc([257], mx.zeros((1, len(VOCAB)))).tolist()[0]
    assert scores[ord("{")] == scores[ord("<")] == 0
    assert scores[ord("X")] == float("-inf")


@pytest.mark.parametrize(
    "parser,payload",
    [
        ("mistral", '[TOOL_CALLS]write_file[ARGS]{"body":"hello"}'),
        ("mistral", '[TOOL_CALLS][{"name":"write_file","arguments":{"body":"hello"}}]'),
        (
            "json_tools",
            '<tool_call>{"name":"write_file","arguments":{"body":"a </tool_call> b"}}</tool_call>',
        ),
        (
            "minimax_m2",
            '<minimax:tool_call><invoke name="write_file"><parameter name="body">hello</parameter></invoke></minimax:tool_call>',
        ),
    ],
)
def test_tool_protocols_keep_final_schema(compiler, parser, payload):
    calls = parse_tool_calls(payload, native_tokenizer(parser), TOOLS)[1]
    assert calls and calls[0].function.name == "write_file"
    if parser == "json_tools":
        assert json.loads(calls[0].function.arguments) == {"body": "a </tool_call> b"}
    compiled = compile_format(compiler, parser)
    assert accepts(compiled, payload)
    assert accepts(compiled, FINAL)
    assert not accepts(compiled, "unstructured final answer")


@pytest.mark.parametrize(
    "schema,output",
    [
        ({"type": "array", "items": {"type": "integer"}}, "[1,2]"),
        ({"type": "integer"}, "7"),
    ],
)
def test_non_object_json_roots(compiler, schema, output):
    compiled = compile_format(compiler, structured_outputs={"json": schema})
    assert accepts(compiled, "\n\t" + output + "\r\n")
    instruction = _json_instruction_for_request(structured_outputs={"json": schema})
    assert instruction and "JSON object" not in instruction


def test_non_json_output_keeps_its_grammar(compiler):
    compiled = compile_format(compiler, structured_outputs={"regex": "yes|no"})
    assert accepts(compiled, "yes")
    assert accepts(compiled, "\n" + TOOL_CALL)
    assert not accepts(compiled, "\nyes")
    assert (
        _json_instruction_for_request(
            structured_outputs={"regex": "yes|no"},
            response_format={"type": "json_schema", "json_schema": {"schema": SCHEMA}},
        )
        is None
    )


def test_harmony_routes_tools_outside_the_final_channel():
    # Harmony's call/return markers are stop tokens, not ordinary tag text.
    vocab = VOCAB + [b"<|return|>", b"<|call|>"]
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocab, stop_token_ids=[259, 260]), max_threads=1
    )
    engine = SimpleNamespace(grammar_compiler=compiler, tokenizer=SimpleNamespace())
    for tools in (None, TOOLS):
        compiled = _compile_grammar_for_request(
            engine,
            tools=tools,
            structured_outputs={"json": SCHEMA},
            reasoning_parser="harmony",
        )
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_string("<|channel|>final<|message|>" + FINAL)
        assert matcher.accept_token(259)
        call = (
            "<|channel|>commentary to=functions.write_file<|constrain|>json<|message|>"
            '{"body":"hello"}'
        )
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_string(call) == bool(tools)
        if tools:
            assert matcher.accept_token(260)
        assert not xgr.GrammarMatcher(compiled).accept_string(
            "<|channel|>final<|message|>unstructured"
        )
