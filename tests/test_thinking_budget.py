# SPDX-License-Identifier: Apache-2.0
"""Tests for ThinkingBudgetProcessor logits processor."""

from unittest.mock import MagicMock

import pytest

# Lazy-import mlx.core — tests skip gracefully if unavailable.
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.adapter.output_parser import OutputParserFactory
from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.model_settings import ModelSettings
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(vocab_size: int = 100):
    """Create a dummy logits tensor [1, vocab_size]."""
    return mx.zeros((1, vocab_size))


def _make_tokens(*token_ids: int):
    """Create a tokens tensor from a list of token IDs."""
    return mx.array(list(token_ids))


# ---------------------------------------------------------------------------
# ThinkingBudgetProcessor unit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestThinkingBudgetProcessor:
    """Unit tests for the ThinkingBudgetProcessor."""

    THINK_END_ID = 42  # Dummy </think> token ID
    THINK_START_ID = 41  # Dummy <think> token ID

    NEWLINE_ID = 99  # Dummy \n token ID

    def _make_processor(self, budget: int = 5, end_ids=None, trailing_ids=None):
        return ThinkingBudgetProcessor(
            think_end_token_ids=end_ids or [self.THINK_END_ID],
            budget=budget,
            think_start_token_id=self.THINK_START_ID,
            trailing_token_ids=trailing_ids,
        )

    # --- Budget enforcement ---

    def test_forces_end_token_when_budget_exceeded(self):
        """After budget tokens, logits should force the end-think token."""
        proc = self._make_processor(budget=3)

        # First call (first_call flag skips state update)
        logits = proc(_make_tokens(10), _make_logits())
        assert not proc._forcing

        # Simulate token generation: each call = one decode step
        logits = proc(_make_tokens(10, 20), _make_logits())
        assert not proc._forcing

        logits = proc(_make_tokens(10, 20, 30), _make_logits())
        # Budget=3, third token should trigger forcing
        assert proc._forcing or proc._done

        # The forced logits should have -inf everywhere except target
        target_logit = logits[0, self.THINK_END_ID].item()
        other_logit = logits[0, 0].item()
        assert target_logit == 0.0
        assert other_logit == float("-inf")

    def test_done_after_forced_sequence(self):
        """After forcing the close sequence, processor should become a no-op."""
        proc = self._make_processor(budget=1)

        # Call 1 (first_call): budget=1, forcing starts → forces THINK_END_ID
        forced_logits = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert forced_logits[0, self.THINK_END_ID].item() == 0.0

        # Call 2: force_sequence has only [THINK_END_ID], so the processor is done.
        logits = proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._done
        assert not proc._forcing
        assert mx.array_equal(logits, _make_logits())

    def test_trailing_tokens_forced_after_end(self):
        """Trailing tokens (e.g. \\n) should be forced after </think>."""
        trailing = [self.NEWLINE_ID]
        proc = self._make_processor(budget=1, trailing_ids=trailing)
        # _force_sequence = [THINK_END_ID, NEWLINE_ID]

        # Call 1: budget hit, forces THINK_END_ID
        logits0 = proc(_make_tokens(10), _make_logits())
        assert logits0[0, self.THINK_END_ID].item() == 0.0

        # Call 2: _force_idx advances to 1, forces NEWLINE_ID
        logits1 = proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._forcing
        assert logits1[0, self.NEWLINE_ID].item() == 0.0

        # Call 3: _force_idx advances to 2 == len([42, 99]) → done
        logits2 = proc(_make_tokens(10, self.THINK_END_ID, self.NEWLINE_ID), _make_logits())
        assert proc._done
        assert mx.array_equal(logits2, _make_logits())

    def test_natural_end_before_budget(self):
        """If model produces </think> naturally, processor becomes no-op."""
        proc = self._make_processor(budget=100)

        # First call
        proc(_make_tokens(10), _make_logits())

        # Second call — model naturally produced </think>
        proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._done

        # Subsequent call should be no-op
        original = _make_logits()
        result = proc(_make_tokens(10, self.THINK_END_ID, 50), original)
        assert mx.array_equal(result, original)

    def test_first_call_skips_state_update(self):
        """First call should not check tokens[-1] for state transitions."""
        proc = self._make_processor(budget=100)

        # Simulate prompt ending with </think> token (shouldn't happen but edge case)
        proc(_make_tokens(self.THINK_END_ID), _make_logits())

        # Should still be in thinking mode (first call skipped state update)
        assert proc._in_thinking
        assert not proc._done

    # --- Multi-token end sequence ---

    def test_multi_token_forcing(self):
        """Multi-token </think> should be forced one token at a time."""
        end_ids = [50, 51, 52]  # e.g. "</" + "think" + ">"
        proc = self._make_processor(budget=1, end_ids=end_ids)

        # Call 1 (first_call): budget hit, forcing starts at _force_idx=0 → token 50
        logits0 = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert logits0[0, 50].item() == 0.0

        # Call 2: _update_state advances _force_idx to 1 → forces token 51
        logits1 = proc(_make_tokens(10, 50), _make_logits())
        assert proc._forcing
        assert logits1[0, 51].item() == 0.0

        # Call 3: _force_idx advances to 2 → forces token 52
        logits2 = proc(_make_tokens(10, 50, 51), _make_logits())
        assert proc._forcing
        assert logits2[0, 52].item() == 0.0

        # Call 4: _force_idx advances to 3 == len(end_ids), then becomes done.
        logits3 = proc(_make_tokens(10, 50, 51, 52), _make_logits())
        assert proc._done
        assert not proc._forcing
        assert mx.array_equal(logits3, _make_logits())

    def test_waits_for_utf8_completion_before_forcing(self):
        """Budget exhaustion waits until the current token piece is UTF-8 complete."""
        pieces = {
            20: b"\xe2",
            21: b"\x82",
            22: b"\xac",
        }
        proc = ThinkingBudgetProcessor(
            think_end_token_ids=[self.THINK_END_ID],
            budget=2,
            think_start_token_id=self.THINK_START_ID,
            token_to_piece=lambda token_id: pieces.get(token_id, "x"),
        )

        proc(_make_tokens(10), _make_logits())
        logits = proc(_make_tokens(10, 20), _make_logits())
        assert proc._waiting_utf8
        assert not proc._forcing
        assert mx.array_equal(logits, _make_logits())

        logits = proc(_make_tokens(10, 20, 21), _make_logits())
        assert proc._waiting_utf8
        assert not proc._forcing
        assert mx.array_equal(logits, _make_logits())

        logits = proc(_make_tokens(10, 20, 21, 22), _make_logits())
        assert proc._forcing
        assert logits[0, self.THINK_END_ID].item() == 0.0

    def test_multi_token_natural_detection(self):
        """Sliding window should detect multi-token </think> naturally."""
        end_ids = [50, 51]
        proc = self._make_processor(budget=100, end_ids=end_ids)

        proc(_make_tokens(10), _make_logits())  # First call

        # Generate tokens that match the end sequence
        proc(_make_tokens(10, 50), _make_logits())
        assert not proc._done

        proc(_make_tokens(10, 50, 51), _make_logits())
        assert proc._done

    # --- Edge cases ---

    def test_zero_budget(self):
        """Budget=0 should force on the very first thinking token."""
        proc = self._make_processor(budget=0)

        # First call — budget is 0, so _thinking_tokens (0) >= budget (0)
        logits = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert logits[0, self.THINK_END_ID].item() == 0.0

    def test_large_budget_no_forcing(self):
        """With a very large budget, no forcing should happen."""
        proc = self._make_processor(budget=10000)

        # Use token IDs 100+ to avoid colliding with THINK_END_ID (42) or THINK_START_ID (41)
        for i in range(50):
            proc(_make_tokens(*range(100, 100 + i + 1)), _make_logits())

        assert not proc._forcing
        assert not proc._done
        assert proc._in_thinking


# ---------------------------------------------------------------------------
# ModelSettings serialization
# ---------------------------------------------------------------------------


class TestModelSettingsThinkingBudget:
    """Test thinking_budget fields in ModelSettings."""

    def test_to_dict_includes_thinking_budget(self):
        settings = ModelSettings(thinking_budget_enabled=True, thinking_budget_tokens=4096)
        d = settings.to_dict()
        assert d["thinking_budget_enabled"] is True
        assert d["thinking_budget_tokens"] == 4096

    def test_from_dict_with_thinking_budget(self):
        data = {"thinking_budget_enabled": True, "thinking_budget_tokens": 2048}
        settings = ModelSettings.from_dict(data)
        assert settings.thinking_budget_enabled is True
        assert settings.thinking_budget_tokens == 2048

    def test_defaults(self):
        settings = ModelSettings()
        assert settings.thinking_budget_enabled is False
        assert settings.thinking_budget_tokens is None

    def test_to_dict_excludes_none(self):
        settings = ModelSettings()
        d = settings.to_dict()
        assert "thinking_budget_tokens" not in d


class TestParserBackedThinkingBudgetWiring:
    """Scheduler wiring for parsers that own reasoning protocol markers."""

    def _make_scheduler(self, factory, encode_map):
        scheduler = MagicMock(spec=Scheduler)
        scheduler._output_parser_factory = factory
        scheduler._xtc_special_tokens = set()
        scheduler._model_suppress_tokens = set()
        scheduler._get_think_token_id = Scheduler._get_think_token_id.__get__(
            scheduler, Scheduler
        )
        scheduler._get_output_parser_thinking_end_text = (
            Scheduler._get_output_parser_thinking_end_text.__get__(scheduler, Scheduler)
        )
        scheduler._encode_thinking_marker = Scheduler._encode_thinking_marker.__get__(
            scheduler, Scheduler
        )
        scheduler._token_piece_to_bytes = Scheduler._token_piece_to_bytes.__get__(
            scheduler, Scheduler
        )
        scheduler._resolve_output_parser_thinking_trailing_ids = (
            Scheduler._resolve_output_parser_thinking_trailing_ids.__get__(
                scheduler, Scheduler
            )
        )
        scheduler._resolve_think_end_token_ids = (
            Scheduler._resolve_think_end_token_ids.__get__(scheduler, Scheduler)
        )
        scheduler._resolve_think_close_pattern = MagicMock(return_value=(None, None))
        scheduler._build_sampler_and_processors = (
            Scheduler._build_sampler_and_processors.__get__(scheduler, Scheduler)
        )

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: encode_map[
            text
        ]
        scheduler.tokenizer = tokenizer
        return scheduler

    def _make_request(self):
        request = Request(
            request_id="parser-thinking-budget",
            prompt="test",
            sampling_params=SamplingParams(thinking_budget=512),
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
        )
        request.needs_think_prefix = False
        return request

    def test_gemma4_uses_parser_thinking_close_marker(self):
        factory = OutputParserFactory(
            kind="gemma4",
            create_session=MagicMock(),
            thinking_end_text="<channel|>",
        )
        scheduler = self._make_scheduler(factory, {"<channel|>": [101]})
        request = self._make_request()

        _, processors = scheduler._build_sampler_and_processors(
            request.sampling_params, request
        )

        budget_processors = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        assert len(budget_processors) == 1
        assert budget_processors[0]._think_end_ids == [101]

    def test_parser_marker_ignores_none_tokenizer_think_end(self):
        factory = OutputParserFactory(
            kind="gemma4",
            create_session=MagicMock(),
            thinking_end_text="<channel|>",
        )
        scheduler = self._make_scheduler(factory, {"<channel|>": [101]})
        scheduler._resolve_think_close_pattern = (
            Scheduler._resolve_think_close_pattern.__get__(scheduler, Scheduler)
        )
        scheduler.tokenizer.think_end = None
        scheduler._get_chat_template_text = MagicMock(return_value="no close marker")
        request = self._make_request()

        _, processors = scheduler._build_sampler_and_processors(
            request.sampling_params, request
        )

        budget_processors = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        assert len(budget_processors) == 1
        assert budget_processors[0]._think_end_ids == [101]

    def test_token_piece_to_bytes_handles_sentencepiece_byte_fallback(self):
        scheduler = self._make_scheduler(None, {})
        assert scheduler._token_piece_to_bytes("<0xE2><0x82><0xAC>") == "€".encode()

    def test_harmony_uses_parser_thinking_close_and_final_header(self):
        final_header = "<|start|>assistant<|channel|>final<|message|>"
        factory = OutputParserFactory(
            kind="harmony",
            create_session=MagicMock(),
            thinking_end_text="<|end|>",
            thinking_end_trailing_text=final_header,
        )
        scheduler = self._make_scheduler(
            factory,
            {
                "<|end|>": [200],
                final_header: [201, 202, 203, 204, 205],
            },
        )
        request = self._make_request()
        request.is_harmony_model = True

        _, processors = scheduler._build_sampler_and_processors(
            request.sampling_params, request
        )

        budget_processors = [
            p for p in processors if isinstance(p, ThinkingBudgetProcessor)
        ]
        assert len(budget_processors) == 1
        assert budget_processors[0]._think_end_ids == [200]
        assert budget_processors[0]._force_sequence == [200, 201, 202, 203, 204, 205]


# ---------------------------------------------------------------------------
# _resolve_thinking_budget (server.py helper)
# ---------------------------------------------------------------------------


class TestResolveThinkingBudget:
    """Test the _resolve_thinking_budget helper function."""

    def _import_resolve(self):
        from omlx.server import _resolve_thinking_budget
        return _resolve_thinking_budget

    def test_request_override_takes_priority(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        req.thinking_budget = 1024
        result = resolve(req, None)
        assert result == 1024

    def test_anthropic_budget_tokens(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        thinking = MagicMock(spec=[])
        thinking.budget_tokens = 2048
        req.thinking = thinking
        result = resolve(req, None)
        assert result == 2048

    def test_returns_none_when_disabled(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        result = resolve(req, None)
        assert result is None


class TestCompletionsThinkingBudget:
    """The /v1/completions surface carries thinking_budget like chat."""

    def test_completion_request_accepts_thinking_budget(self):
        from omlx.api.openai_models import CompletionRequest

        req = CompletionRequest(model="m", prompt="<think>\n", thinking_budget=300)
        assert req.thinking_budget == 300

    def test_completion_request_thinking_budget_defaults_to_none(self):
        from omlx.api.openai_models import CompletionRequest

        req = CompletionRequest(model="m", prompt="p")
        assert req.thinking_budget is None

    def test_resolve_thinking_budget_reads_completion_request(self):
        from omlx.api.openai_models import CompletionRequest
        from omlx.server import _resolve_thinking_budget

        req = CompletionRequest(model="m", prompt="p", thinking_budget=128)
        assert _resolve_thinking_budget(req, None) == 128

    @staticmethod
    def _engine_call_passes_budget(handler_name: str, engine_method: str) -> bool:
        """True when ``handler_name`` threads a ``thinking_budget`` resolved from
        ``_resolve_thinking_budget`` into ``<obj>.<engine_method>(...)``.

        Accepts both wirings: the inline ``thinking_budget=_resolve_thinking_budget(...)``
        keyword and the ``**gen_kwargs`` dict-unpack pattern the chat path uses
        (#1844), where the handler sets ``gen_kwargs["thinking_budget"]`` from the
        resolved value and unpacks the dict into the engine call.

        Structural AST check: immune to reformatting, wrappers, and comments,
        unlike substring counting."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "server.py"
        ).read_text()

        def _is_resolve_call(value) -> bool:
            return (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "_resolve_thinking_budget"
            )

        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == handler_name
            ):
                continue

            # Locals bound directly to _resolve_thinking_budget(...), e.g.
            #     thinking_budget = _resolve_thinking_budget(request, request.model)
            resolved_locals = {
                t.id
                for n in ast.walk(node)
                if isinstance(n, ast.Assign) and _is_resolve_call(n.value)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            # Dicts that get a "thinking_budget" entry from the resolved value, e.g.
            #     gen_kwargs["thinking_budget"] = thinking_budget
            budget_dicts = set()
            for n in ast.walk(node):
                if not isinstance(n, ast.Assign):
                    continue
                for t in n.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "thinking_budget"
                        and (
                            _is_resolve_call(n.value)
                            or (
                                isinstance(n.value, ast.Name)
                                and n.value.id in resolved_locals
                            )
                        )
                    ):
                        budget_dicts.add(t.value.id)

            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr == engine_method):
                    continue
                for keyword in call.keywords:
                    # inline: engine.generate(..., thinking_budget=_resolve_thinking_budget(...))
                    if keyword.arg == "thinking_budget" and _is_resolve_call(keyword.value):
                        return True
                    # dict-unpack: engine.generate(..., **gen_kwargs)
                    if (
                        keyword.arg is None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id in budget_dicts
                    ):
                        return True
                return False
            return False
        raise AssertionError(f"{handler_name} not found in server.py")

    def test_non_streaming_completion_path_resolves_the_budget(self):
        """The field alone is useless if the handler stops threading it to
        the engine — which was the original bug. See #1825."""
        assert self._engine_call_passes_budget("create_completion", "generate"), (
            "/v1/completions (non-streaming) must pass "
            "thinking_budget=_resolve_thinking_budget(...) to engine.generate; "
            "dropping it silently disables the budget again. See #1825."
        )

    def test_streaming_completion_path_resolves_the_budget(self):
        assert self._engine_call_passes_budget("stream_completion", "stream_generate"), (
            "/v1/completions (streaming) must pass "
            "thinking_budget=_resolve_thinking_budget(...) to "
            "engine.stream_generate; dropping it silently disables the "
            "budget again. See #1825."
        )

    def test_negative_thinking_budget_is_rejected_on_completions(self):
        """A negative budget has no semantics anywhere in the enforcement
        chain; reject it at the API boundary instead of accepting it
        silently."""
        import pytest
        from pydantic import ValidationError

        from omlx.api.openai_models import CompletionRequest

        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="p", thinking_budget=-1)

    def test_negative_thinking_budget_is_rejected_on_chat(self):
        import pytest
        from pydantic import ValidationError

        from omlx.api.openai_models import ChatCompletionRequest

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                thinking_budget=-1,
            )

    def test_zero_thinking_budget_is_accepted(self):
        """Zero is meaningful (thinking off), keep it valid."""
        from omlx.api.openai_models import CompletionRequest

        req = CompletionRequest(model="m", prompt="p", thinking_budget=0)
        assert req.thinking_budget == 0


class TestCompletionsStreamThinkPrefixParity:
    """Raw completions are a continuation of the prompt: when the prompt
    opens the thinking block itself, the synthetic ``<think>\\n`` opener the
    scheduler prepends for chat streams must not leak into the completions
    stream — the non-streaming path never returns it."""

    def test_synthetic_prefix_is_stripped(self):
        from omlx.server import _strip_synthetic_think_prefix

        assert (
            _strip_synthetic_think_prefix("<think>\n</think>\n\nHi", "<think>")
            == "</think>\n\nHi"
        )

    def test_chunk_without_prefix_is_untouched(self):
        from omlx.server import _strip_synthetic_think_prefix

        assert _strip_synthetic_think_prefix("Hello", "<think>") == "Hello"

    def test_bare_tag_without_newline_is_untouched(self):
        """Only the exact synthetic shape (tag + newline) is synthetic;
        anything else is model output and must pass through."""
        from omlx.server import _strip_synthetic_think_prefix

        assert _strip_synthetic_think_prefix("<think>data", "<think>") == "<think>data"

    def test_prompt_detection_uses_tokenizer_over_text_suffix(self):
        """A textual ``<think>`` suffix is not enough: completions should only
        strip when the engine would actually add the synthetic opener."""
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            think_start_id = 41
            think_end_id = 42

            def encode(self, prompt, add_special_tokens=False):
                return [10, 11, 12]

        opens, tag = prompt_opens_thinking(Tokenizer(), "literal <think>\n")

        assert (opens, tag) == (False, "<think>")

    def test_prompt_detection_handles_tokenized_template_suffix(self):
        """Mirror Scheduler._detect_needs_think_prefix: a prompt can need the
        synthetic opener when the think-start token is in the final token tail,
        even if the raw text does not literally end with the tag string."""
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            think_start_id = 41
            think_end_id = 42

            def encode(self, prompt, add_special_tokens=False):
                return [100, 41, 99]

        opens, tag = prompt_opens_thinking(Tokenizer(), "templated suffix")

        assert (opens, tag) == (True, "<think>")

    def test_prompt_detection_reuses_precomputed_prompt_ids(self):
        """The streaming presentation guard should use the same prompt ids as
        context validation instead of re-encoding with different tokenizer
        options."""
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            think_start_id = 41
            think_end_id = 42

            def encode(self, prompt, add_special_tokens=False):
                raise AssertionError("prompt ids should already be available")

        opens, tag = prompt_opens_thinking(
            Tokenizer(), "templated suffix", prompt_token_ids=[100, 41, 99]
        )

        assert (opens, tag) == (True, "<think>")

    def test_prompt_detection_rejects_disabled_thinking_pattern(self):
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            think_start_id = 41
            think_end_id = 42

            def encode(self, prompt, add_special_tokens=False):
                return [41, 42]

        opens, tag = prompt_opens_thinking(Tokenizer(), "<think></think>")

        assert (opens, tag) == (False, "<think>")

    def test_prompt_detection_rejects_multi_token_disabled_thinking_pattern(self):
        """Mirror the scheduler's encode(think_end) fallback: when the close
        marker is multi-token, seeing its first token after <think> still means
        the prompt disabled thinking."""
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            think_start_id = 41
            think_end = "</think>"
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return self.unk_token_id

            def encode(self, prompt, add_special_tokens=False):
                if prompt == self.think_end:
                    return [42, 43]
                return [41, 42]

        opens, tag = prompt_opens_thinking(Tokenizer(), "<think></think>")

        assert (opens, tag) == (False, "<think>")

    def test_prompt_detection_rejects_text_suffix_when_think_id_is_unavailable(self):
        """If a tokenizer is present but cannot resolve the think-start id,
        mirror the scheduler and do not assume a synthetic opener exists."""
        from omlx.api.thinking import prompt_opens_thinking

        class Tokenizer:
            think_start = "<think>"
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return self.unk_token_id

            def encode(self, prompt, add_special_tokens=False):
                return [10, 11, 12]

        opens, tag = prompt_opens_thinking(Tokenizer(), "literal <think>\n")

        assert (opens, tag) == (False, "<think>")

    def test_prompt_detection_keeps_text_fallback_without_tokenizer(self):
        from omlx.api.thinking import prompt_opens_thinking

        assert prompt_opens_thinking(None, "literal <think>\n") == (True, "<think>")

    def test_stream_completion_wires_the_strip(self):
        """Structural guard: the streaming handler must call the strip
        helper, or the prefix leaks back on the first chunk."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "server.py"
        ).read_text()
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "stream_completion"
            ):
                called = {
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                assert "prompt_opens_thinking" in called, (
                    "stream_completion must use the tokenizer-backed prompt "
                    "detector so it only strips when the engine would add the "
                    "synthetic opener."
                )
                prompt_detector_calls = [
                    call
                    for call in ast.walk(node)
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "prompt_opens_thinking"
                    )
                ]
                assert any(
                    keyword.arg == "prompt_token_ids"
                    for call in prompt_detector_calls
                    for keyword in call.keywords
                ), (
                    "stream_completion must pass the validation prompt ids "
                    "into prompt_opens_thinking so both paths use the same "
                    "tokenizer defaults."
                )
                assert "_strip_synthetic_think_prefix" in called, (
                    "stream_completion must strip the synthetic think opener "
                    "from the first chunk when the prompt opens the thinking "
                    "block; the non-streaming path never returns it. See #1825."
                )
                return
        raise AssertionError("stream_completion not found in server.py")

    def test_create_completion_threads_validation_prompt_ids_to_streaming(self):
        """The completion endpoint should reuse the prompt ids it already
        computed for context-window validation on the stream path."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "server.py"
        ).read_text()
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "create_completion"
            ):
                stream_calls = [
                    call
                    for call in ast.walk(node)
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "stream_completion"
                    )
                ]
                assert any(
                    keyword.arg == "prompt_token_ids"
                    for call in stream_calls
                    for keyword in call.keywords
                ), (
                    "create_completion must thread the validation prompt ids "
                    "to stream_completion instead of making the strip guard "
                    "re-encode the prompt."
                )
                return
        raise AssertionError("create_completion not found in server.py")

@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestSoftThinkingBudget:
    """Unit tests for the progressive soft budget zone (vllm#38277 port)."""

    THINK_END_ID = 42
    THINK_START_ID = 41
    BUDGET = 10
    FRAC = ThinkingBudgetProcessor._SOFT_ZONE_START_FRAC
    FACTOR = ThinkingBudgetProcessor._SOFT_BIAS_FACTOR
    SOFT_START = int(BUDGET * FRAC)
    SPAN = BUDGET - SOFT_START

    def _make_processor(self, budget: int = None, soft_budget: bool = True, end_ids=None):
        return ThinkingBudgetProcessor(
            think_end_token_ids=end_ids or [self.THINK_END_ID],
            budget=self.BUDGET if budget is None else budget,
            think_start_token_id=self.THINK_START_ID,
            soft_budget=soft_budget,
        )

    def _ramp_logits(self, vocab_size: int = 100, top_id: int = 7, top: float = 5.0):
        """Logits with a known top value and zeros elsewhere."""
        logits = mx.zeros((1, vocab_size))
        logits[0, top_id] = top
        return logits

    def _step(self, proc, n_tokens: int, logits=None):
        """Run proc for a history of n_tokens generated tokens."""
        tokens = _make_tokens(*range(10, 10 + n_tokens))
        return proc(tokens, logits if logits is not None else _make_logits())

    def _expected_boost(self, step: int, top: float = 5.0) -> float:
        """target - end_logit for a zeroed end logit at a given think step."""
        if step <= self.SOFT_START:
            return 0.0
        progress = (step - self.SOFT_START) / max(1, self.SPAN)
        return self.FACTOR * max(top, 1.0) * progress

    def test_no_bias_before_soft_zone(self):
        """Up to the zone boundary, logits pass through unchanged."""
        proc = self._make_processor()
        for step in range(1, self.SOFT_START + 1):
            logits = self._step(proc, step, self._ramp_logits())
            assert logits[0, self.THINK_END_ID].item() == 0.0
        assert not proc._forcing

    def test_soft_zone_boosts_end_logit_progressively(self):
        """Inside the soft zone the close-think logit ramps toward the top."""
        proc = self._make_processor()
        boosts = []
        for step in range(1, self.BUDGET):
            logits = self._step(proc, step, self._ramp_logits(top=5.0))
            boosts.append(logits[0, self.THINK_END_ID].item())
        assert boosts[self.SOFT_START - 1] == 0.0  # boundary step: progress=0
        for step in range(self.SOFT_START + 1, self.BUDGET):
            assert boosts[step - 1] == pytest.approx(self._expected_boost(step))
        assert boosts[self.BUDGET - 2] > boosts[self.SOFT_START]

    def test_soft_zone_end_dominates_at_high_progress(self):
        """Past 1/FACTOR of the soft zone the close-think logit exceeds the top."""
        proc = self._make_processor()
        logits = None
        for step in range(1, self.BUDGET):
            logits = self._step(proc, step, self._ramp_logits(top=5.0))
        assert logits[0, self.THINK_END_ID].item() > 5.0

    def test_hard_force_still_applies_at_budget(self):
        """The 100% hard force stays as the safety net."""
        proc = self._make_processor()
        logits = None
        for step in range(1, self.BUDGET + 1):
            logits = self._step(proc, step, self._ramp_logits())
        assert proc._forcing
        assert logits[0, self.THINK_END_ID].item() == 0.0
        assert logits[0, 0].item() == float("-inf")

    def test_natural_close_in_soft_zone_stops_biasing(self):
        """Sampling </think> naturally inside the soft zone ends the ramp."""
        proc = self._make_processor()
        in_zone = self.SOFT_START + 1
        for step in range(1, in_zone):
            self._step(proc, step, self._ramp_logits())
        tokens = _make_tokens(*range(10, 10 + in_zone - 1), self.THINK_END_ID)
        logits = proc(tokens, self._ramp_logits())
        assert proc._done
        assert logits[0, self.THINK_END_ID].item() == 0.0  # no further bias

    def test_soft_budget_disabled_keeps_hard_only(self):
        """soft_budget=False restores the previous hard-cut-only behavior."""
        proc = self._make_processor(soft_budget=False)
        for step in range(1, self.BUDGET):
            logits = self._step(proc, step, self._ramp_logits())
            assert logits[0, self.THINK_END_ID].item() == 0.0
        assert not proc._forcing
        self._step(proc, self.BUDGET, self._ramp_logits())
        assert proc._forcing

    def test_multi_token_end_boosts_only_first_id(self):
        """Without a partial close prefix, only the first marker id is boosted.

        Boosting later ids too could make the model sample them out of
        order and leak a close-marker fragment into the thinking text.
        """
        end_ids = [42, 43]
        proc = self._make_processor(end_ids=end_ids)
        logits = None
        for step in range(1, self.BUDGET):
            logits = self._step(proc, step, self._ramp_logits(top=5.0))
        target = self._expected_boost(self.BUDGET - 1)
        assert logits[0, 42].item() == pytest.approx(target)
        assert logits[0, 43].item() == 0.0

    def test_multi_token_partial_prefix_biases_next_id(self):
        """After the model emits the first marker id, the second is boosted."""
        end_ids = [42, 43]
        proc = self._make_processor(end_ids=end_ids)
        in_zone = self.SOFT_START + 2
        for step in range(1, in_zone):
            self._step(proc, step, self._ramp_logits(top=5.0))
        # Model naturally samples the first id of the close marker.
        tokens = _make_tokens(*range(10, 10 + in_zone - 1), end_ids[0])
        logits = proc(tokens, self._ramp_logits(top=5.0))
        assert not proc._done  # marker not complete yet
        assert logits[0, 43].item() > 0.0  # continuation id boosted
        assert logits[0, 42].item() == 0.0  # first id no longer targeted

    def test_min_gap_floor_when_end_already_near_top(self):
        """gap is floored at 1.0 so the ramp still progresses on flat logits."""
        proc = self._make_processor()
        logits = None
        for step in range(1, self.BUDGET):
            logits = self._step(proc, step, _make_logits())  # all zeros, gap->1.0
        progress = (self.BUDGET - 1 - self.SOFT_START) / max(1, self.SPAN)
        assert logits[0, self.THINK_END_ID].item() == pytest.approx(self.FACTOR * 1.0 * progress)

    def test_multi_token_wrong_prefix_falls_back_to_first_id(self):
        """A generated tail that matches a LATER marker id (not a proper
        prefix) must not be treated as a partial close: bias ids[0]."""
        end_ids = [42, 43]
        proc = self._make_processor(end_ids=end_ids)
        in_zone = self.SOFT_START + 2
        for step in range(1, in_zone):
            self._step(proc, step, self._ramp_logits(top=5.0))
        # Model emits the SECOND marker id out of order: not a close prefix.
        tokens = _make_tokens(*range(10, 10 + in_zone - 1), end_ids[1])
        logits = proc(tokens, self._ramp_logits(top=5.0))
        assert not proc._done
        assert logits[0, 42].item() > 0.0  # first id targeted
        assert logits[0, 43].item() == 0.0  # out-of-order id not boosted

    @pytest.mark.parametrize(
        ("end_ids", "recent", "expected"),
        [
            # Repeated leading id [A, A, B]: one trailing A is a 1-prefix, a
            # double A a 2-prefix; an [A, B] tail is NOT a prefix of
            # [A, A, B], so the bias falls back to ids[0].
            ([1, 1, 2], [1], 1),
            ([1, 1, 2], [1, 1], 2),
            ([1, 1, 2], [1, 1, 1], 2),
            ([1, 1, 2], [1, 2], 1),
            ([1, 1, 2], [2, 1], 1),
            # Overlapping marker [A, B, A]: a completed [A, B] tail targets
            # the closing ids[2]; a bare trailing A re-matches the 1-prefix
            # and targets ids[1].
            ([1, 2, 1], [1], 2),
            ([1, 2, 1], [1, 2], 1),
            ([1, 2, 1], [2, 1], 2),
            ([1, 2, 1], [1, 2, 1], 2),
        ],
    )
    def test_next_close_token_id_on_repeated_and_overlapping_markers(
        self, end_ids, recent, expected
    ):
        """Pin the longest-proper-prefix resolution on markers with repeated
        or overlapping ids: the biased token must always keep the generated
        tail a valid marker prefix, never leak a fragment."""
        proc = self._make_processor(end_ids=end_ids)
        proc._recent_tokens = list(recent)
        assert proc._next_close_token_id() == expected

    def test_soft_zone_policy_constants(self):
        """The zone boundaries are product choices, not incidental values:
        the soft zone covers the last 50% of the budget, and FACTOR=2 makes
        the close-think logit overtake the top halfway through the zone.
        Changing either changes when models stop thinking; update the PR
        narrative (and the vllm#38277 port) together with this test."""
        assert ThinkingBudgetProcessor._SOFT_ZONE_START_FRAC == 0.5
        assert ThinkingBudgetProcessor._SOFT_BIAS_FACTOR == 2.0

    def test_degenerate_budgets_keep_the_hard_cut_contract(self):
        """Tiny budgets must degrade to the hard cut without a crash or a
        dead zone: 0 and 1 force at the very first step, 2 has no usable
        soft step before the wall."""
        for budget in (0, 1):
            proc = self._make_processor(budget=budget)
            logits = self._step(proc, 1, self._ramp_logits())
            assert proc._forcing
            assert logits[0, self.THINK_END_ID].item() == 0.0
            assert logits[0, 0].item() == float("-inf")

        proc = self._make_processor(budget=2)
        logits = self._step(proc, 1, self._ramp_logits())
        assert not proc._forcing
        assert logits[0, self.THINK_END_ID].item() == 0.0  # no bias yet
        self._step(proc, 2, self._ramp_logits())
        assert proc._forcing

    def test_budget_three_gets_one_soft_step(self):
        """budget=3 is the smallest budget with a live soft step: step 2
        runs the ramp at progress 1/2 (boost = FACTOR * gap * 1/2), step 3
        is the hard wall."""
        proc = self._make_processor(budget=3)
        logits = self._step(proc, 1, _make_logits())
        assert logits[0, self.THINK_END_ID].item() == 0.0
        # Flat logits: the gap floors at 1.0, so the boost is the pure ramp.
        logits = self._step(proc, 2, _make_logits())
        assert not proc._forcing
        assert logits[0, self.THINK_END_ID].item() == pytest.approx(self.FACTOR * 1.0 * 0.5)
        self._step(proc, 3, _make_logits())
        assert proc._forcing
