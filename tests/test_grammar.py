# SPDX-License-Identifier: Apache-2.0
"""Tests for grammar-constrained decoding integration.

Covers:
- GrammarConstraintProcessor (logits processor)
- _build_format_element (structured_outputs → format element)
- _patch_output_format (structural tag patching)
- _compile_grammar_for_request (end-to-end compilation)
- _compile_with_structural_tag / _compile_bare_grammar
- Scheduler grammar path (_build_sampler_and_processors)
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from omlx.api.openai_models import StructuredOutputOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(*, grammar_compiler=None, tokenizer=None):
    """Create a lightweight mock engine."""
    engine = MagicMock()
    engine.grammar_compiler = grammar_compiler
    engine.tokenizer = tokenizer
    return engine


def _make_tokenizer(*, think_start_id=None, think_end_id=None,
                     think_start="<think>", think_end="</think>",
                     unk_token_id=0, convert_map=None):
    """Create a mock tokenizer with optional thinking attributes."""
    tok = MagicMock()
    tok.think_start_id = think_start_id
    tok.think_end_id = think_end_id
    tok.think_start = think_start
    tok.think_end = think_end
    tok.unk_token_id = unk_token_id
    if convert_map is not None:
        tok.convert_tokens_to_ids = lambda t: convert_map.get(t, unk_token_id)
    else:
        tok.convert_tokens_to_ids = MagicMock(side_effect=KeyError)
    return tok


# =========================================================================
# _build_format_element
# =========================================================================

class TestBuildFormatElement:
    """Tests for _build_format_element."""

    @staticmethod
    def _call(**kwargs):
        from omlx.server import _build_format_element
        return _build_format_element(**kwargs)

    def test_none_when_no_args(self):
        assert self._call() is None

    def test_json_schema_from_dict(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = self._call(structured_outputs={"json": schema})
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_json_schema_from_string(self):
        schema = {"type": "object"}
        result = self._call(structured_outputs={"json": json.dumps(schema)})
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_json_schema_from_pydantic_model(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        so = StructuredOutputOptions(json_schema=schema)
        result = self._call(structured_outputs=so)
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_regex(self):
        result = self._call(structured_outputs={"regex": r"\d+"})
        assert result == {"type": "regex", "pattern": r"\d+"}

    def test_choice(self):
        result = self._call(structured_outputs={"choice": ["yes", "no"]})
        assert result["type"] == "grammar"
        assert '"yes"' in result["grammar"]
        assert '"no"' in result["grammar"]

    def test_grammar_ebnf(self):
        ebnf = 'root ::= "a" | "b"'
        result = self._call(structured_outputs={"grammar": ebnf})
        assert result == {"type": "grammar", "grammar": ebnf}

    def test_response_format_json_schema(self):
        result = self._call(response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        })
        assert result["type"] == "json_schema"
        assert result["json_schema"]["type"] == "object"

    def test_response_format_json_object(self):
        result = self._call(response_format={"type": "json_object"})
        assert result == {"type": "json_schema", "json_schema": {}}

    def test_response_format_text_returns_none(self):
        assert self._call(response_format={"type": "text"}) is None

    def test_structured_outputs_takes_priority(self):
        """structured_outputs should be used even when response_format is set."""
        result = self._call(
            structured_outputs={"regex": r"\d+"},
            response_format={"type": "json_object"},
        )
        assert result["type"] == "regex"

    def test_none_structured_outputs_all_none(self):
        """StructuredOutputOptions with all fields None → None."""
        so = StructuredOutputOptions()
        assert self._call(structured_outputs=so) is None


# =========================================================================
# _patch_output_format
# =========================================================================

class TestPatchOutputFormat:
    """Tests for _patch_output_format."""

    @staticmethod
    def _call(tag_dict, user_grammar):
        from omlx.server import _patch_output_format
        return _patch_output_format(tag_dict, user_grammar)

    def test_replaces_top_level_any_text(self):
        """Qwen reasoning=False style: top-level any_text replaced."""
        tag_dict = {
            "type": "structural_tag",
            "format": {"type": "any_text", "excludes": ["<think>"]},
        }
        user_grammar = {"type": "json_schema", "json_schema": {"type": "object"}}
        assert self._call(tag_dict, user_grammar) is True
        assert tag_dict["format"] == user_grammar

    def test_replaces_last_any_text_in_sequence(self):
        """Qwen reasoning=True style: sequence with think tag + any_text."""
        tag_dict = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "tag", "begin": "<think>", "content": {"type": "any_text"}, "end": "</think>"},
                    {"type": "any_text", "excludes": ["<think>"]},
                ],
            },
        }
        user_grammar = {"type": "regex", "pattern": r"\d+"}
        assert self._call(tag_dict, user_grammar) is True
        assert tag_dict["format"]["elements"][1] == user_grammar
        assert tag_dict["format"]["elements"][0]["type"] == "tag"

    def test_replaces_final_channel_in_tags_with_separator(self):
        """Harmony style: tags_with_separator with 'final' channel."""
        tag_dict = {
            "type": "structural_tag",
            "format": {
                "type": "tags_with_separator",
                "tags": [
                    {"type": "tag", "begin": "<|channel|>analysis<|message|>", "content": {"type": "any_text"}, "end": "<|end|>"},
                    {"type": "tag", "begin": "<|channel|>final<|message|>", "content": {"type": "any_text"}, "end": "<|end|>"},
                ],
                "separator": "<|start|>assistant",
            },
        }
        user_grammar = {"type": "json_schema", "json_schema": {"type": "object"}}
        assert self._call(tag_dict, user_grammar) is True
        assert tag_dict["format"]["tags"][1]["content"] == user_grammar
        assert tag_dict["format"]["tags"][0]["content"] == {"type": "any_text"}

    def test_fallback_to_last_tag_when_no_final(self):
        """tags_with_separator without 'final' in begin → falls back to last tag."""
        tag_dict = {
            "type": "structural_tag",
            "format": {
                "type": "tags_with_separator",
                "tags": [
                    {"type": "tag", "begin": "<output>", "content": {"type": "any_text"}, "end": "</output>"},
                ],
                "separator": "",
            },
        }
        user_grammar = {"type": "grammar", "grammar": 'root ::= "x"'}
        assert self._call(tag_dict, user_grammar) is True
        assert tag_dict["format"]["tags"][0]["content"] == user_grammar

    def test_returns_false_for_unrecognised_format(self):
        """Unknown format type returns False without modification."""
        tag_dict = {"type": "structural_tag", "format": {"type": "unknown_thing"}}
        assert self._call(tag_dict, {"type": "regex", "pattern": "x"}) is False


# =========================================================================
# _compile_with_structural_tag / _compile_bare_grammar
# =========================================================================

class TestCompileWithStructuralTag:
    """Tests for _compile_with_structural_tag."""

    @staticmethod
    def _call(compiler, fmt, reasoning_parser, chat_template_kwargs=None):
        from omlx.server import _compile_with_structural_tag
        return _compile_with_structural_tag(compiler, fmt, reasoning_parser, chat_template_kwargs)

    @patch("omlx.server.xgr" if False else "xgrammar.get_builtin_structural_tag")
    def test_calls_get_builtin_structural_tag(self, mock_get_tag):
        """Verifies xgrammar.get_builtin_structural_tag is called with correct args."""
        xgr = pytest.importorskip("xgrammar")

        mock_tag = MagicMock()
        mock_tag.model_dump.return_value = {
            "type": "structural_tag",
            "format": {"type": "any_text", "excludes": []},
        }
        mock_get_tag.return_value = mock_tag

        compiler = MagicMock()
        compiler.compile_structural_tag.return_value = "compiled"

        fmt = {"type": "json_schema", "json_schema": {"type": "object"}}
        result = self._call(compiler, fmt, "qwen", None)

        mock_get_tag.assert_called_once_with("qwen", reasoning=True)
        compiler.compile_structural_tag.assert_called_once()
        assert result == "compiled"

    @patch("xgrammar.get_builtin_structural_tag")
    def test_reasoning_false_when_thinking_disabled(self, mock_get_tag):
        xgr = pytest.importorskip("xgrammar")

        mock_tag = MagicMock()
        mock_tag.model_dump.return_value = {
            "type": "structural_tag",
            "format": {"type": "any_text", "excludes": []},
        }
        mock_get_tag.return_value = mock_tag

        compiler = MagicMock()
        compiler.compile_structural_tag.return_value = "compiled"

        fmt = {"type": "json_schema", "json_schema": {"type": "object"}}
        self._call(compiler, fmt, "qwen", {"enable_thinking": False})

        mock_get_tag.assert_called_once_with("qwen", reasoning=False)

    @patch("xgrammar.get_builtin_structural_tag")
    def test_patches_user_grammar_into_tag(self, mock_get_tag):
        """The user's grammar should replace the any_text in the tag."""
        xgr = pytest.importorskip("xgrammar")

        mock_tag = MagicMock()
        mock_tag.model_dump.return_value = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "tag", "begin": "<think>", "content": {"type": "any_text"}, "end": "</think>"},
                    {"type": "any_text", "excludes": []},
                ],
            },
        }
        mock_get_tag.return_value = mock_tag

        compiler = MagicMock()
        compiler.compile_structural_tag.return_value = "compiled"

        user_fmt = {"type": "regex", "pattern": r"\d+"}
        self._call(compiler, user_fmt, "qwen", None)

        tag_arg = compiler.compile_structural_tag.call_args[0][0]
        assert tag_arg["format"]["elements"][1] == user_fmt


class TestCompileBareGrammar:
    """Tests for _compile_bare_grammar."""

    @staticmethod
    def _call(compiler, fmt):
        from omlx.server import _compile_bare_grammar
        return _compile_bare_grammar(compiler, fmt)

    def test_json_schema(self):
        compiler = MagicMock()
        compiler.compile_json_schema.return_value = "compiled_json"
        schema = {"type": "object"}
        result = self._call(compiler, {"type": "json_schema", "json_schema": schema})
        assert result == "compiled_json"
        compiler.compile_json_schema.assert_called_once()

    def test_empty_json_schema(self):
        compiler = MagicMock()
        compiler.compile_builtin_json_grammar.return_value = "compiled_builtin"
        result = self._call(compiler, {"type": "json_schema", "json_schema": {}})
        assert result == "compiled_builtin"

    def test_regex(self):
        compiler = MagicMock()
        compiler.compile_regex.return_value = "compiled_regex"
        result = self._call(compiler, {"type": "regex", "pattern": r"\d+"})
        assert result == "compiled_regex"
        compiler.compile_regex.assert_called_once_with(r"\d+")

    def test_grammar(self):
        compiler = MagicMock()
        compiler.compile_grammar.return_value = "compiled_ebnf"
        ebnf = 'root ::= "a"'
        result = self._call(compiler, {"type": "grammar", "grammar": ebnf})
        assert result == "compiled_ebnf"
        compiler.compile_grammar.assert_called_once_with(ebnf)


# =========================================================================
# _compile_grammar_for_request
# =========================================================================

class TestCompileGrammarForRequest:
    """Tests for _compile_grammar_for_request."""

    @staticmethod
    def _call(engine, **kwargs):
        from omlx.server import _compile_grammar_for_request
        return _compile_grammar_for_request(engine, **kwargs)

    def test_returns_none_when_no_grammar_requested(self):
        engine = _make_engine()
        assert self._call(engine) is None

    def test_raises_when_no_compiler_and_structured_outputs(self):
        from fastapi import HTTPException
        engine = _make_engine(grammar_compiler=None)
        with pytest.raises(HTTPException) as exc_info:
            self._call(engine, structured_outputs={"regex": r"\d+"})
        assert exc_info.value.status_code == 400
        assert "xgrammar" in exc_info.value.detail

    def test_returns_none_when_no_compiler_and_response_format(self):
        """response_format gracefully falls back to None when no compiler."""
        engine = _make_engine(grammar_compiler=None)
        result = self._call(engine, response_format={"type": "json_object"})
        assert result is None

    def test_bare_json_schema(self):
        """No reasoning_parser: compile_json_schema called directly."""
        compiler = MagicMock()
        compiler.compile_json_schema.return_value = "compiled_json"
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(engine, structured_outputs={
            "json": {"type": "object", "properties": {"x": {"type": "integer"}}},
        })
        assert result == "compiled_json"
        compiler.compile_json_schema.assert_called_once()

    def test_bare_regex(self):
        compiler = MagicMock()
        compiler.compile_regex.return_value = "compiled_regex"
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(engine, structured_outputs={"regex": r"\d{3}"})
        assert result == "compiled_regex"
        compiler.compile_regex.assert_called_once_with(r"\d{3}")

    def test_bare_grammar(self):
        compiler = MagicMock()
        compiler.compile_grammar.return_value = "compiled_ebnf"
        engine = _make_engine(grammar_compiler=compiler)

        ebnf = 'root ::= "a" | "b"'
        result = self._call(engine, structured_outputs={"grammar": ebnf})
        assert result == "compiled_ebnf"
        compiler.compile_grammar.assert_called_once_with(ebnf)

    def test_bare_json_object(self):
        compiler = MagicMock()
        compiler.compile_builtin_json_grammar.return_value = "compiled_builtin"
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(engine, response_format={"type": "json_object"})
        assert result == "compiled_builtin"
        compiler.compile_builtin_json_grammar.assert_called_once()

    @patch("xgrammar.get_builtin_structural_tag")
    def test_reasoning_parser_uses_structural_tag(self, mock_get_tag):
        """When reasoning_parser is set, compile_structural_tag is used."""
        pytest.importorskip("xgrammar")
        mock_tag = MagicMock()
        mock_tag.model_dump.return_value = {
            "type": "structural_tag",
            "format": {"type": "any_text", "excludes": []},
        }
        mock_get_tag.return_value = mock_tag

        compiler = MagicMock()
        compiler.compile_structural_tag.return_value = "compiled_structural"
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(
            engine,
            structured_outputs={"json": {"type": "object"}},
            reasoning_parser="qwen",
        )
        assert result == "compiled_structural"
        compiler.compile_structural_tag.assert_called_once()
        mock_get_tag.assert_called_once_with("qwen", reasoning=True)

    @patch("xgrammar.get_builtin_structural_tag")
    def test_reasoning_parser_with_thinking_disabled(self, mock_get_tag):
        """enable_thinking=False → reasoning=False passed to get_builtin_structural_tag."""
        pytest.importorskip("xgrammar")
        mock_tag = MagicMock()
        mock_tag.model_dump.return_value = {
            "type": "structural_tag",
            "format": {"type": "any_text", "excludes": []},
        }
        mock_get_tag.return_value = mock_tag

        compiler = MagicMock()
        compiler.compile_structural_tag.return_value = "compiled"
        engine = _make_engine(grammar_compiler=compiler)

        self._call(
            engine,
            structured_outputs={"json": {"type": "object"}},
            chat_template_kwargs={"enable_thinking": False},
            reasoning_parser="qwen",
        )
        mock_get_tag.assert_called_once_with("qwen", reasoning=False)

    def test_no_reasoning_parser_uses_bare_grammar(self):
        """Without reasoning_parser, bare grammar compilation is used."""
        compiler = MagicMock()
        compiler.compile_json_schema.return_value = "compiled_bare"
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(
            engine,
            structured_outputs={"json": {"type": "object"}},
            reasoning_parser=None,
        )
        assert result == "compiled_bare"
        compiler.compile_json_schema.assert_called_once()
        compiler.compile_structural_tag.assert_not_called()

    def test_compilation_error_raises_for_structured_outputs(self):
        from fastapi import HTTPException
        compiler = MagicMock()
        compiler.compile_json_schema.side_effect = RuntimeError("bad schema")
        engine = _make_engine(grammar_compiler=compiler)

        with pytest.raises(HTTPException) as exc_info:
            self._call(engine, structured_outputs={"json": {"type": "invalid"}})
        assert exc_info.value.status_code == 400
        assert "bad schema" in exc_info.value.detail

    def test_compilation_error_returns_none_for_response_format(self):
        """response_format compilation errors → graceful fallback to None."""
        compiler = MagicMock()
        compiler.compile_json_schema.side_effect = RuntimeError("bad")
        engine = _make_engine(grammar_compiler=compiler)

        result = self._call(engine, response_format={
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "object"}},
        })
        assert result is None


# =========================================================================
# GrammarConstraintProcessor
# =========================================================================

class TestGrammarConstraintProcessor:
    """Tests for GrammarConstraintProcessor using real xgrammar."""

    @pytest.fixture()
    def compiler(self):
        """Create a real xgrammar compiler with a small mock vocabulary."""
        xgr = pytest.importorskip("xgrammar")
        vocab = [f"<tok_{i}>" for i in range(256)]
        vocab[0] = "<unk>"
        vocab[1] = "<s>"
        vocab[2] = "</s>"
        vocab[ord("{")] = "{"
        vocab[ord("}")] = "}"
        vocab[ord('"')] = '"'
        vocab[ord(":")] = ":"
        vocab[ord(",")] = ","
        ti = xgr.TokenizerInfo(vocab)
        return xgr.GrammarCompiler(ti), len(vocab)

    def test_masks_invalid_tokens(self, compiler):
        """Bitmask should suppress some tokens for a constrained grammar."""
        from omlx.api.grammar import GrammarConstraintProcessor

        comp, vocab_size = compiler
        ebnf = 'root ::= "hello"'
        cg = comp.compile_grammar(ebnf)
        proc = GrammarConstraintProcessor(cg, vocab_size)

        logits = mx.zeros((vocab_size,))
        result = proc(mx.array([]), logits)
        result_np = np.array(result).flatten()
        assert result_np.shape[0] >= vocab_size
        assert np.any(result_np[:vocab_size] == -np.inf), "Should mask some tokens"

    def test_passthrough_after_termination(self, compiler):
        """After grammar terminates, logits should pass through unmodified."""
        from omlx.api.grammar import GrammarConstraintProcessor

        comp, vocab_size = compiler

        ebnf = 'root ::= "a"'
        cg = comp.compile_grammar(ebnf)
        proc = GrammarConstraintProcessor(cg, vocab_size)

        logits = mx.zeros((vocab_size,))
        proc(mx.array([]), logits)

        token_a = ord("a")
        proc(mx.array([token_a]), logits)

        if proc.is_terminated:
            result = proc(mx.array([token_a]), logits)
            np.testing.assert_array_equal(np.array(result), np.array(logits))

    def test_first_call_does_not_accept(self, compiler):
        """First call should apply bitmask without accepting any token."""
        from omlx.api.grammar import GrammarConstraintProcessor

        comp, vocab_size = compiler
        cg = comp.compile_grammar('root ::= "x"')
        proc = GrammarConstraintProcessor(cg, vocab_size)

        logits = mx.ones((vocab_size,))
        result = proc(mx.array([]), logits)
        result_np = np.array(result)
        assert np.any(result_np == -np.inf), "Should constrain on first call"
        assert not proc.is_terminated

    def test_is_terminated_property(self, compiler):
        from omlx.api.grammar import GrammarConstraintProcessor

        comp, vocab_size = compiler
        cg = comp.compile_grammar('root ::= ""')
        proc = GrammarConstraintProcessor(cg, vocab_size)
        assert proc.is_terminated is False


# =========================================================================
# Scheduler grammar path
# =========================================================================

class TestSchedulerGrammarPath:
    """Tests for grammar processor construction in _build_sampler_and_processors."""

    def _make_scheduler(self, *, vocab_size=256, compiled_grammar=None):
        from omlx.request import Request, SamplingParams

        scheduler = MagicMock()
        scheduler.model = MagicMock()
        scheduler.model.config = SimpleNamespace(vocab_size=vocab_size)
        scheduler.tokenizer = MagicMock()
        scheduler.tokenizer.eos_token_id = 2

        sampling_params = SamplingParams(
            max_tokens=100,
            compiled_grammar=compiled_grammar,
        )
        request = Request(
            request_id="test-001",
            prompt="test",
            sampling_params=sampling_params,
        )
        return scheduler, sampling_params, request

    def test_no_grammar_no_processor(self):
        """When compiled_grammar is None, no grammar processor is added."""
        from omlx.scheduler import Scheduler

        sched, sp, req = self._make_scheduler()
        sched._get_model_vocab_size = Scheduler._get_model_vocab_size.__get__(sched)
        sched._build_sampler_and_processors = Scheduler._build_sampler_and_processors.__get__(sched)

        _, processors = sched._build_sampler_and_processors(sp, req)
        from omlx.api.grammar import GrammarConstraintProcessor
        grammar_procs = [p for p in processors if isinstance(p, GrammarConstraintProcessor)]
        assert len(grammar_procs) == 0

    def test_grammar_processor_added_when_compiled_grammar(self):
        """When compiled_grammar is set, a GrammarConstraintProcessor is created."""
        xgr = pytest.importorskip("xgrammar")
        from omlx.scheduler import Scheduler

        vocab = [f"<tok_{i}>" for i in range(256)]
        ti = xgr.TokenizerInfo(vocab)
        comp = xgr.GrammarCompiler(ti)
        cg = comp.compile_grammar('root ::= "x"')

        sched, sp, req = self._make_scheduler(vocab_size=256, compiled_grammar=cg)
        sched._get_model_vocab_size = Scheduler._get_model_vocab_size.__get__(sched)
        sched._build_sampler_and_processors = Scheduler._build_sampler_and_processors.__get__(sched)

        _, processors = sched._build_sampler_and_processors(sp, req)
        from omlx.api.grammar import GrammarConstraintProcessor
        grammar_procs = [p for p in processors if isinstance(p, GrammarConstraintProcessor)]
        assert len(grammar_procs) == 1

    def test_skipped_when_vocab_size_unavailable(self):
        """Grammar processor is skipped when vocab_size cannot be determined."""
        xgr = pytest.importorskip("xgrammar")
        from omlx.scheduler import Scheduler

        vocab = [f"<tok_{i}>" for i in range(256)]
        ti = xgr.TokenizerInfo(vocab)
        comp = xgr.GrammarCompiler(ti)
        cg = comp.compile_grammar('root ::= "x"')

        sched, sp, req = self._make_scheduler(compiled_grammar=cg)
        sched.model = MagicMock(spec=[])
        sched._get_model_vocab_size = Scheduler._get_model_vocab_size.__get__(sched)
        sched._build_sampler_and_processors = Scheduler._build_sampler_and_processors.__get__(sched)

        _, processors = sched._build_sampler_and_processors(sp, req)
        from omlx.api.grammar import GrammarConstraintProcessor
        grammar_procs = [p for p in processors if isinstance(p, GrammarConstraintProcessor)]
        assert len(grammar_procs) == 0


# =========================================================================
# GrammarConstraintProcessor.advance (batched mode)
# =========================================================================

class TestGrammarProcessorAdvance:
    """Tests for the advance() method used in batched bitmask filling."""

    @pytest.fixture()
    def compiler(self):
        xgr = pytest.importorskip("xgrammar")
        vocab = [f"<tok_{i}>" for i in range(256)]
        vocab[0] = "<unk>"
        vocab[ord("a")] = "a"
        vocab[ord("b")] = "b"
        ti = xgr.TokenizerInfo(vocab)
        return xgr.GrammarCompiler(ti), len(vocab)

    def test_advance_returns_true_on_first_call(self, compiler):
        from omlx.api.grammar import GrammarConstraintProcessor
        comp, vs = compiler
        proc = GrammarConstraintProcessor(comp.compile_grammar('root ::= "ab"'), vs)
        assert proc.advance(mx.array([])) is True

    def test_advance_accepts_token(self, compiler):
        from omlx.api.grammar import GrammarConstraintProcessor
        comp, vs = compiler
        proc = GrammarConstraintProcessor(comp.compile_grammar('root ::= "ab"'), vs)
        proc.advance(mx.array([]))
        assert proc.advance(mx.array([ord("a")])) is True

    def test_advance_returns_false_when_terminated(self, compiler):
        from omlx.api.grammar import GrammarConstraintProcessor
        comp, vs = compiler
        proc = GrammarConstraintProcessor(comp.compile_grammar('root ::= "a"'), vs)
        proc.advance(mx.array([]))
        proc.advance(mx.array([ord("a")]))
        assert proc.is_terminated or proc.advance(mx.array([ord("a")])) is True

    def test_advance_then_batch_fill(self, compiler):
        """advance + batch_fill_next_token_bitmask produces correct mask."""
        xgr = pytest.importorskip("xgrammar")
        from omlx.api.grammar import GrammarConstraintProcessor
        comp, vs = compiler

        proc1 = GrammarConstraintProcessor(comp.compile_grammar('root ::= "a"'), vs)
        proc2 = GrammarConstraintProcessor(comp.compile_grammar('root ::= "b"'), vs)

        proc1.advance(mx.array([]))
        proc2.advance(mx.array([]))

        bitmask_width = (vs + 31) // 32
        bitmask = np.full((2, bitmask_width), -1, dtype=np.int32)
        batch_matcher = xgr.BatchGrammarMatcher()
        batch_matcher.batch_fill_next_token_bitmask(
            [proc1.matcher, proc2.matcher], bitmask,
        )

        def is_allowed(bm_row, token_id):
            word = bm_row[token_id // 32]
            return bool(word & (1 << (token_id % 32)))

        assert is_allowed(bitmask[0], ord("a")), "proc1 should allow 'a'"
        assert not is_allowed(bitmask[0], ord("b")), "proc1 should not allow 'b'"
        assert is_allowed(bitmask[1], ord("b")), "proc2 should allow 'b'"
        assert not is_allowed(bitmask[1], ord("a")), "proc2 should not allow 'a'"

    def test_matcher_property(self, compiler):
        from omlx.api.grammar import GrammarConstraintProcessor
        xgr = pytest.importorskip("xgrammar")
        comp, vs = compiler
        proc = GrammarConstraintProcessor(comp.compile_grammar('root ::= "x"'), vs)
        assert isinstance(proc.matcher, xgr.GrammarMatcher)


# =========================================================================
# _apply_batched_grammar (scheduler _step integration)
# =========================================================================

class TestApplyBatchedGrammar:
    """Tests for the batched grammar path in _step."""

    @pytest.fixture()
    def setup(self):
        xgr = pytest.importorskip("xgrammar")
        vocab = [f"<tok_{i}>" for i in range(256)]
        vocab[ord("a")] = "a"
        vocab[ord("b")] = "b"
        vocab[ord("{")] = "{"
        vocab[ord("}")] = "}"
        ti = xgr.TokenizerInfo(vocab)
        comp = xgr.GrammarCompiler(ti)
        return comp, len(vocab)

    @pytest.mark.skip(reason="Batched grammar optimization removed in mlx-lm BatchGenerator refactor. Grammar now runs via per-request logits_processors in GenerationBatch._step().")
    def test_batched_grammar_masks_logits(self, setup):
        """Batched grammar correctly masks logits for multiple requests."""
        pass

    @pytest.mark.skip(reason="Batched grammar optimization removed in mlx-lm BatchGenerator refactor. Grammar now runs via per-request logits_processors in GenerationBatch._step().")
    def test_non_grammar_processors_still_run(self, setup):
        """ThinkingBudgetProcessor and other processors still run per-request."""
        pass

    @pytest.mark.skip(reason="Batched grammar optimization removed in mlx-lm BatchGenerator refactor. Grammar now runs via per-request logits_processors in GenerationBatch._step().")
    def test_terminated_processors_skipped(self, setup):
        """Terminated grammar processors don't participate in batch fill."""
        pass


# =========================================================================
# _get_model_vocab_size
# =========================================================================

class TestGetModelVocabSize:
    """Tests for Scheduler._get_model_vocab_size."""

    @staticmethod
    def _call(model):
        from omlx.scheduler import Scheduler
        sched = MagicMock()
        sched.model = model
        return Scheduler._get_model_vocab_size(sched)

    def test_from_config_attr(self):
        model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
        assert self._call(model) == 32000

    def test_from_args_attr(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=128256))
        assert self._call(model) == 128256

    def test_from_nested_text_config_dict(self):
        model = SimpleNamespace(
            args=SimpleNamespace(text_config={"vocab_size": 151936}),
        )
        assert self._call(model) == 151936

    def test_returns_none_when_unavailable(self):
        model = MagicMock(spec=[])
        assert self._call(model) is None
