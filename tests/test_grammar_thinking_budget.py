# SPDX-License-Identifier: Apache-2.0
"""Thinking budgets must not inject tokens into whole-output grammars."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx._torch_stub import install
from omlx.api.grammar import GrammarConstraintProcessor
from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler
from omlx.server import _compile_bare_grammar, _compile_with_structural_tag

install()
xgr = pytest.importorskip("xgrammar")
VOCAB = [bytes([i]) for i in range(256)] + [b"</think>", b"<think>", b"</s>"]
END, START, STOP = 256, 257, 258
VOCAB += [
    b"<|im_end|>",
    b"<ifm|tool_calls>",
    b"</ifm|tool_calls>",
    b"<ifm|tool_call>",
    b"</ifm|tool_call>",
    b"<ifm|arg_key>",
]


@pytest.fixture(scope="module")
def compiler():
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, stop_token_ids=[STOP])
    )


def processors_for(compiled):
    scheduler = MagicMock(spec=Scheduler)
    scheduler._xtc_special_tokens = set()
    scheduler._model_suppress_tokens = set()
    scheduler._get_model_vocab_size.return_value = len(VOCAB)
    scheduler._get_think_token_id.return_value = START
    scheduler._resolve_think_end_token_ids.return_value = [END]
    scheduler._resolve_think_close_pattern.return_value = (None, None)
    scheduler._resolve_output_parser_thinking_trailing_ids.return_value = None
    scheduler._get_output_parser_thinking_end_text.return_value = "</think>"
    scheduler._thinking_budget_token_to_piece.side_effect = lambda token: VOCAB[token]
    params = SamplingParams(temperature=0, thinking_budget=1, compiled_grammar=compiled)
    request = Request(
        request_id="grammar-budget", prompt="test", sampling_params=params
    )
    request.needs_think_prefix = True
    request.think_end_token_id = END
    return Scheduler._build_sampler_and_processors(scheduler, params, request)[1]


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ({"type": "grammar", "grammar": 'root ::= "abc"'}, "abc"),
        ({"type": "regex", "pattern": "abc"}, "abc"),
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"answer": {"const": "yes"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
            '{"answer":"yes"}',
        ),
    ],
)
def test_bare_grammar_remains_sampleable_past_thinking_budget(compiler, fmt, expected):
    compiled = _compile_bare_grammar(compiler, fmt)
    processors = processors_for(compiled)
    grammar = next(p for p in processors if isinstance(p, GrammarConstraintProcessor))
    history = []
    for token in [*expected.encode(), STOP]:
        logits = mx.full((len(VOCAB),), -10.0)
        logits[token] = 10.0
        for processor in processors:
            logits = processor(mx.array(history, dtype=mx.int32), logits)
        assert bool(mx.any(mx.isfinite(logits)).item())
        sampled = int(mx.argmax(logits).item())
        assert sampled == token
        grammar.accept_token(sampled)
        history.append(sampled)
    assert grammar.is_terminated
    assert not any(isinstance(p, ThinkingBudgetProcessor) for p in processors)


@pytest.mark.parametrize("reasoning", [False, True])
def test_structural_grammar_budget_follows_reasoning_phase(compiler, reasoning):
    compiled = _compile_with_structural_tag(
        compiler,
        {"type": "regex", "pattern": "abc"},
        "qwen_3_5",
        {"enable_thinking": reasoning},
    )
    processors = processors_for(compiled)
    assert any(isinstance(p, ThinkingBudgetProcessor) for p in processors) == reasoning


def test_unconstrained_output_keeps_thinking_budget():
    assert any(isinstance(p, ThinkingBudgetProcessor) for p in processors_for(None))


def test_compiler_cache_does_not_leak_reasoning_metadata(compiler):
    fmt = {"type": "regex", "pattern": "abc"}
    enabled = _compile_with_structural_tag(
        compiler, fmt, "qwen_3_5", {"enable_thinking": True}
    )
    disabled = _compile_with_structural_tag(
        compiler, fmt, "qwen_3_5", {"enable_thinking": False}
    )
    enabled_again = _compile_with_structural_tag(
        compiler, fmt, "qwen_3_5", {"enable_thinking": True}
    )
    assert any(isinstance(p, ThinkingBudgetProcessor) for p in processors_for(enabled))
    assert not any(
        isinstance(p, ThinkingBudgetProcessor) for p in processors_for(disabled)
    )
    assert any(
        isinstance(p, ThinkingBudgetProcessor) for p in processors_for(enabled_again)
    )


def test_k2_tool_grammar_keeps_thinking_budget(compiler):
    from omlx.patches.k2_horizon.tool_grammar import compile_tool_grammar

    compiled = compile_tool_grammar(compiler, [{"function": {"name": "read_file"}}])
    assert any(isinstance(p, ThinkingBudgetProcessor) for p in processors_for(compiled))


@pytest.mark.parametrize("reasoning", [False, True])
def test_structural_grammar_accepts_budget_close_and_constrained_answer(
    compiler, reasoning
):
    compiled = _compile_with_structural_tag(
        compiler,
        {"type": "regex", "pattern": "abc"},
        "qwen_3_5",
        {"enable_thinking": reasoning},
    )
    processors = processors_for(compiled)
    grammar = next(p for p in processors if isinstance(p, GrammarConstraintProcessor))
    expected = [END, ord("\n"), ord("\n")] if reasoning else []
    expected += [*b"abc", STOP]
    history = []
    for token in expected:
        logits = mx.full((len(VOCAB),), -10.0)
        # At the budget boundary the model would continue reasoning; the budget
        # must force the close while leaving at least one grammar-valid token.
        logits[ord("x") if token == END else token] = 10.0
        for processor in processors:
            logits = processor(mx.array(history, dtype=mx.int32), logits)
        assert bool(mx.any(mx.isfinite(logits)).item())
        sampled = int(mx.argmax(logits).item())
        assert sampled == token
        grammar.accept_token(sampled)
        history.append(sampled)
    assert grammar.is_terminated


def test_raw_cached_grammar_does_not_inherit_wrapper_metadata(compiler):
    from omlx.server import _patch_output_format

    fmt = {"type": "regex", "pattern": "abc"}
    _compile_with_structural_tag(compiler, fmt, "qwen_3_5", {"enable_thinking": True})
    tag = xgr.get_builtin_structural_tag("qwen_3_5", reasoning=True).model_dump()
    assert _patch_output_format(tag, fmt)
    raw = compiler.compile_structural_tag(tag)
    assert not any(isinstance(p, ThinkingBudgetProcessor) for p in processors_for(raw))
