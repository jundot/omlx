# SPDX-License-Identifier: Apache-2.0
"""Final-answer constraints must leave tool calls available."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.api.grammar import GrammarConstraintProcessor
from omlx.server import _compile_grammar_for_request

xgr = pytest.importorskip("xgrammar")
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
PARAMETERS = {
    "type": "object",
    "$defs": {"text": {"type": "string"}},
    "properties": {"first": {"$ref": "#/$defs/text"}, "second": {"type": "integer"}},
    "required": ["first", "second"],
}
TOOLS = [{"type": "function", "function": {"name": "check", "parameters": PARAMETERS}}]
FINAL = '{"answer":"yes"}'
CALL = (
    "<tool_call>\n<function=check>\n<parameter=second>\n1\n</parameter>\n"
    "<parameter=first>\nhello\n</parameter>\n</function>\n</tool_call>"
)
VOCAB = [bytes([i]) for i in range(256)] + [b"</s>", b"\n\n", b"\n\n{"]


@pytest.fixture(scope="module")
def compiler():
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(VOCAB, stop_token_ids=[256]), max_threads=1
    )


def compile_format(compiler, parser="qwen3_coder", **kwargs):
    module = importlib.import_module(f"mlx_lm.tool_parsers.{parser}")
    tokenizer = SimpleNamespace(
        tool_call_start=module.tool_call_start,
        tool_call_end=module.tool_call_end,
        tool_parser=module.parse_tool_call,
    )
    return _compile_grammar_for_request(
        SimpleNamespace(grammar_compiler=compiler, tokenizer=tokenizer),
        tools=TOOLS,
        structured_outputs={"json": SCHEMA},
        **kwargs,
    )


def accepts(grammar, text):
    matcher = xgr.GrammarMatcher(grammar)
    return matcher.accept_string(text) and matcher.accept_token(256)


@pytest.mark.parametrize(
    "parser,payload",
    [
        ("qwen3_coder", CALL),
        (
            "json_tools",
            '<tool_call>{"name":"check","arguments":{"second":1,"first":"a </tool_call> b"}}</tool_call>',
        ),
        ("mistral", '[TOOL_CALLS]check[ARGS]{"second":1,"first":"hello"}'),
    ],
)
def test_tool_arguments_do_not_change_final_schema(compiler, parser, payload):
    grammar = compile_format(compiler, parser)
    for valid in (payload, FINAL, "\n\n" + FINAL):
        assert accepts(grammar, valid)
    for invalid in (
        '{"answer":"invalid"}',
        "{}",
        "plain text",
        "\n\n",
        " " * 33 + FINAL,
    ):
        assert not accepts(grammar, invalid)


@pytest.mark.parametrize("thinking", [False, True])
def test_reasoning_keeps_tools_and_final_json(compiler, thinking):
    grammar = compile_format(
        compiler,
        reasoning_parser="qwen_3_5",
        chat_template_kwargs={"enable_thinking": thinking},
    )
    prefix = "Checked.</think>\n\n" if thinking else ""
    assert accepts(grammar, prefix + CALL * 2)
    assert accepts(grammar, prefix + FINAL)


def test_newline_tokens_keep_both_branches_available(compiler):
    proc = GrammarConstraintProcessor(compile_format(compiler), len(VOCAB))
    scores = proc([], mx.zeros((1, len(VOCAB)))).tolist()[0]
    assert scores[258] == 0
    proc.accept_token(257)
    scores = proc([257], mx.zeros((1, len(VOCAB)))).tolist()[0]
    assert scores[ord("{")] == scores[ord("<")] == 0
    assert scores[ord("X")] == float("-inf")


@pytest.mark.parametrize(
    "options",
    [
        {"structured_outputs": {"json": SCHEMA}},
        {"response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}},
    ],
)
def test_missing_markers_preserve_enforcement(compiler, options):
    grammar = _compile_grammar_for_request(
        SimpleNamespace(grammar_compiler=compiler, tokenizer=SimpleNamespace()),
        tools=TOOLS,
        **options,
    )
    assert accepts(grammar, FINAL)
    assert not accepts(grammar, '{"answer":"invalid"}')
