# SPDX-License-Identifier: Apache-2.0
"""Exact K2 names across token boundaries and speculative rollback."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx._torch_stub import install
from omlx.exceptions import InvalidRequestError
from omlx.patches.k2_horizon.tool_grammar import UnoToolConstraint, compile_tool_grammar
from omlx.patches.k2_horizon.uno_decode import acceptance_and_residual

install()
xgr = pytest.importorskip("xgrammar")

MARKERS = [
    "<ifm|tool_calls>",
    "</ifm|tool_calls>",
    "<ifm|tool_call>",
    "</ifm|tool_call>",
    "<ifm|arg_key>",
    "</ifm|arg_key>",
    "<ifm|arg_value>",
    "</ifm|arg_value>",
]
VOCAB = (
    [bytes([i]) for i in range(256)]
    + [s.encode() for s in MARKERS]
    + [b"</s>", b"read_file", b"read", b"brave-search", b"brave_search"]
)
STOP = 264


def tools_for(*names):
    return [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in names
    ]


@pytest.fixture(scope="module")
def compiler():
    info = xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, stop_token_ids=[STOP])
    return xgr.GrammarCompiler(info, cache_limit_bytes=8 * 1024**2)


def accept(matcher, text):
    while text:
        marker = next((m for m in MARKERS if text.startswith(m)), None)
        if marker:
            if not matcher.accept_token(256 + MARKERS.index(marker)):
                return False
            text = text[len(marker) :]
        else:
            char, text = text[0], text[1:]
            for token in char.encode():
                if not matcher.accept_token(token):
                    return False
    return True


@pytest.mark.parametrize(
    "name",
    ["brave-search", "brave_search", "read", "read_file", "tést_工具", 'quote"slash\\'],
)
@pytest.mark.parametrize("fmt", ["xml", "json", "json_reordered"])
def test_exact_names_and_argument_contents(compiler, name, fmt):
    compiled = compile_tool_grammar(compiler, tools_for(name, "read"))
    arguments = {
        "query": "brave_search read_file <ifm|tool_calls> is just text",
        "nested": {"name": "undeclared"},
    }
    if fmt == "xml":
        body = (
            name
            + "\n<ifm|arg_key>query</ifm|arg_key><ifm|arg_value>"
            + arguments["query"]
            + "</ifm|arg_value>"
        )
    else:
        value = {"name": name, "arguments": arguments}
        if fmt == "json_reordered":
            value = {"arguments": arguments, "name": name}
        body = json.dumps(value, ensure_ascii=False)
    matcher = xgr.GrammarMatcher(compiled)
    assert accept(
        matcher,
        "Thinking about undeclared_tools. <ifm|tool_calls>\n<ifm|tool_call>"
        + body
        + "</ifm|tool_call>\n</ifm|tool_calls>",
    )
    assert matcher.accept_token(STOP)


@pytest.mark.parametrize(
    "name", ["brave_search", "Brave-search", "brave-search-extra", "read_files"]
)
def test_undeclared_names_rejected(compiler, name):
    compiled = compile_tool_grammar(
        compiler, tools_for("brave-search", "read", "read_file")
    )
    matcher = xgr.GrammarMatcher(compiled)
    assert not accept(
        matcher,
        "<ifm|tool_calls><ifm|tool_call>" + name + "</ifm|tool_call></ifm|tool_calls>",
    )


def test_alternate_tokenizations_prefixes_and_request_isolation(compiler):
    compiled = compile_tool_grammar(compiler, tools_for("read", "read_file"))
    for pieces in ([266], [265], list(b"read_file")):
        matcher = xgr.GrammarMatcher(compiled)
        assert accept(matcher, "<ifm|tool_calls><ifm|tool_call>")
        assert all(matcher.accept_token(token) for token in pieces)
        assert accept(matcher, "</ifm|tool_call></ifm|tool_calls>")
    other = xgr.GrammarMatcher(
        compile_tool_grammar(compiler, tools_for("brave-search"))
    )
    assert not accept(other, "<ifm|tool_calls><ifm|tool_call>read</ifm|tool_call>")


def test_optional_constraints_fail_explicitly(compiler):
    existing = object()
    assert compile_tool_grammar(None, [], existing) is existing
    with pytest.raises(InvalidRequestError, match="xgrammar"):
        compile_tool_grammar(None, tools_for("read"))
    with pytest.raises(InvalidRequestError, match="together"):
        compile_tool_grammar(compiler, tools_for("read"), existing)


def test_uno_first_token_rejection_residual_and_rollback(compiler):
    compiled = compile_tool_grammar(compiler, tools_for("brave-search"))
    constraint = UnoToolConstraint(compiled, len(VOCAB))
    prefix = [256, 258]
    constraint.commit(prefix)
    logits = mx.zeros((3, len(VOCAB)))
    seed = constraint.seed(logits)
    assert mx.isneginf(seed[0, 268]).item()
    assert seed[0, 267].item() == 0
    assert mx.array_equal(seed[1:], logits[1:]).item()
    proposals = [ord("b"), ord("X"), ord("z")]
    target = constraint.verify(logits, proposals)
    assert mx.isneginf(target[0, ord("X")]).item()
    assert target[0, ord("r")].item() == 0
    assert mx.array_equal(target[1:], logits[1:]).item()
    p = mx.softmax(target[:1], axis=-1)
    q = mx.full(p.shape, 1 / len(VOCAB))
    flags, residual = acceptance_and_residual(
        p, q, mx.array([ord("X")]), mx.array([0.5])
    )
    assert not flags.item()
    assert residual[0, ord("X")].item() == 0
    constraint.commit(list(b"brave-search"))
    assert accept(constraint.matcher, "</ifm|tool_call></ifm|tool_calls>")
    assert constraint.matcher.accept_token(STOP)


def test_non_k2_engine_does_not_touch_grammar():
    from omlx.engine.batched import BatchedEngine

    for model_type in ("gemma4", "qwen3", "llama", None):
        engine = SimpleNamespace(model_type=model_type)
        kwargs = {"compiled_grammar": object()}
        existing = kwargs["compiled_grammar"]
        BatchedEngine._prepare_k2_tool_grammar(engine, tools_for("read"), kwargs)
        assert kwargs["compiled_grammar"] is existing


@pytest.mark.parametrize("temperature", [0, 1.0])
@pytest.mark.parametrize("block_size", [1, 2, 8])
def test_uno_commits_constrained_tokens_and_keeps_kv(compiler, temperature, block_size):
    from mlx_lm.models.cache import KVCache

    from omlx.patches.k2_horizon.uno_decode import UnoDecoder

    class Model:
        _uno_adapter_loaded = True
        args = SimpleNamespace(vocab_size=len(VOCAB))

        def make_cache(self):
            self.cache = [KVCache()]
            return self.cache

        def __call__(self, inputs, cache=None, lora_mask=None):
            values = inputs[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)
            return mx.broadcast_to(
                mx.where(mx.arange(len(VOCAB)) == ord("X"), 10.0, 0.0),
                (1, inputs.shape[1], len(VOCAB)),
            )

    model = Model()
    compiled = compiler.compile_grammar('root ::= "abc"')
    constraint = UnoToolConstraint(compiled, len(VOCAB))
    decoder = UnoDecoder(
        model,
        eos_token_ids=[STOP],
        block_size=block_size,
        temperature=temperature,
        constraint=constraint,
    )
    cycles = list(decoder.generate([10, 11], max_tokens=10))
    assert [token for cycle in cycles for token in cycle.tokens] == [97, 98, 99, STOP]
    assert cycles[-1].finish_reason == "stop"
    assert model.cache[0].state[0][0, 0, :, 0].tolist() == [10, 11, 97, 98, 99]


def test_partial_tool_prefix_is_explicitly_rejected():
    from omlx.patches.k2_horizon.tool_grammar import validate_tool_prefix

    messages = [{"role": "assistant", "content": "<ifm|tool_calls><ifm|tool_call>rea"}]
    with pytest.raises(InvalidRequestError, match="prefix"):
        validate_tool_prefix(messages, tools_for("read"), True)
    validate_tool_prefix(messages, [], True)
    validate_tool_prefix(messages, tools_for("read"), False)
    validate_tool_prefix(
        [{"role": "assistant", "content": "Let me think"}], tools_for("read"), True
    )
