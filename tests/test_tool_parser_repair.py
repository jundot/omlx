# SPDX-License-Identifier: Apache-2.0
"""Generic parser selection and integration regression tests; no model weights."""

from types import SimpleNamespace

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tool_parsers import json_tools, qwen3_coder

from omlx.utils.tool_parser import repair_tool_parser

TEMPLATES = [
    ("<tool_call>\n<function=", "qwen3_coder"),
    (r"<tool_call>\n<function=", "qwen3_coder"),
    ("<minimax:tool_call>", "minimax_m2"),
    ("<|tool_call><tool_call|>", "gemma4"),
    ("<start_function_call>", "function_gemma"),
    ("<longcat_tool_call>", "longcat"),
    ("<arg_key>", "glm47"),
    ("<|tool_list_start|>", "pythonic"),
    ("<|tool_calls_section_begin|>", "kimi_k2"),
    ("[TOOL_CALLS]", "mistral"),
]


def wrapper(template, parser=json_tools.parse_tool_call):
    tokenizer = TokenizerWrapper.__new__(TokenizerWrapper)
    tokenizer._tokenizer = SimpleNamespace(
        chat_template=template,
        init_kwargs={"tool_parser_type": "json_tools"},
        encode=lambda text, **kwargs: list(text.encode()),
    )
    tokenizer._tool_parser = parser
    tokenizer._tool_call_start = "<tool_call>"
    tokenizer._tool_call_end = "</tool_call>"
    tokenizer._tool_call_start_tokens = (1,)
    tokenizer._tool_call_end_tokens = (2,)
    return tokenizer


@pytest.mark.parametrize("template, expected", TEMPLATES)
def test_supported_template_grammars_repair_generic_parser(template, expected):
    tokenizer = wrapper(template)
    assert repair_tool_parser(tokenizer) == expected
    assert tokenizer.tool_parser.__module__ == f"mlx_lm.tool_parsers.{expected}"
    assert tokenizer.tool_call_start_tokens == tuple(tokenizer.tool_call_start.encode())
    assert tokenizer.tool_call_end_tokens == tuple(tokenizer.tool_call_end.encode())
    assert tokenizer.init_kwargs == {"tool_parser_type": "json_tools"}
    assert repair_tool_parser(tokenizer) is None  # idempotent


@pytest.mark.parametrize(
    "template",
    [
        None,
        "",
        "ordinary template",
        "<tool_call>tool_call.name",
        {"default": "<arg_key>"},
    ],
)
def test_ambiguous_and_json_templates_unchanged(template):
    tokenizer = wrapper(template)
    assert repair_tool_parser(tokenizer) is None
    assert tokenizer.tool_parser is json_tools.parse_tool_call


@pytest.mark.parametrize(
    "parser", [qwen3_coder.parse_tool_call, lambda text: text, None]
)
def test_specific_custom_and_absent_parsers_unchanged(parser):
    tokenizer = wrapper("<arg_key>", parser)
    assert repair_tool_parser(tokenizer) is None
    assert tokenizer.tool_parser is parser


def test_non_mlx_wrappers_unchanged():
    assert repair_tool_parser(SimpleNamespace()) is None


def test_marker_failure_does_not_partially_change_parser():
    tokenizer = wrapper("<arg_key>")
    tokenizer._tokenizer.encode = lambda *a, **kw: []
    with pytest.raises(ValueError, match="nonempty"):
        repair_tool_parser(tokenizer)
    assert tokenizer.tool_parser is json_tools.parse_tool_call
    assert tokenizer.tool_call_start_tokens == (1,)


def test_qwen_xml_is_parsed_in_local_output_setup():
    from omlx.adapter.output_parser import detect_output_parser

    tokenizer = wrapper("<tool_call>\n<function=")
    detect_output_parser("generic-model", tokenizer)
    call = tokenizer.tool_parser(
        "<function=echo>\n<parameter=message>\nhello\n</parameter>\n</function>"
    )
    assert call == {"name": "echo", "arguments": {"message": "hello"}}


def test_distributed_protocol_setup_repairs_parser(tmp_path):
    from omlx.cluster.inference_worker import _install_distributed_model_protocol

    tokenizer = wrapper("<tool_call>\n<function=")
    assert _install_distributed_model_protocol(tokenizer, tmp_path) == "qwen3_coder"
    assert tokenizer.tool_parser is qwen3_coder.parse_tool_call


def test_custom_template_function_is_not_overridden():
    tokenizer = wrapper("<arg_key>")
    tokenizer._chat_template = lambda *args: "custom"
    assert repair_tool_parser(tokenizer) is None
    assert tokenizer.tool_parser is json_tools.parse_tool_call
