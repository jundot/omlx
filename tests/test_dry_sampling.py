# SPDX-License-Identifier: Apache-2.0
"""Tests for the DRY repetition penalty processor (omlx/utils/dry.py)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.request import DryParams, SamplingParams
from omlx.utils.dry import MATCH_HEADROOM, DryProcessor, find_breaker_token_ids

VOCAB = 64


def _reference(history, breakers, multiplier, base, allowed, cap):
    """Host-side DRY: walk back from every earlier occurrence of the last token."""
    penalties: dict[int, float] = {}
    last = len(history) - 1
    if last < 0 or history[last] in breakers:
        return penalties
    for i in range(last):
        if history[i] != history[last]:
            continue
        n = 1
        while (
            n < cap
            and i - n >= 0
            and history[i - n] == history[last - n]
            and history[i - n] not in breakers
        ):
            n += 1
        if n >= allowed:
            follower = int(history[i + 1])
            penalties[follower] = max(
                penalties.get(follower, 0.0), multiplier * base ** (n - allowed)
            )
    return penalties


def _penalties(processor, tokens):
    out = processor(tokens, mx.zeros((1, VOCAB)))
    values = -np.array(out[0])
    return {int(i): float(values[i]) for i in np.flatnonzero(values)}


class TestDryMatching:
    @pytest.mark.parametrize("allowed", [1, 2, 4])
    def test_matches_reference_on_random_histories(self, allowed):
        rng = np.random.default_rng(allowed)
        breakers = {10, 11}
        processor = DryProcessor(
            0.8, allowed_length=allowed, breaker_token_ids=sorted(breakers)
        )
        for trial in range(150):
            high = 14 if trial % 2 else 40
            history = rng.integers(8, high, rng.integers(1, 200)).tolist()
            expected = _reference(
                history, breakers, 0.8, 1.75, allowed, allowed + MATCH_HEADROOM
            )
            got = _penalties(processor, mx.array(history))
            assert got.keys() == expected.keys(), (trial, history)
            for token, value in expected.items():
                assert got[token] == pytest.approx(value, rel=1e-4)

    def test_worked_example(self):
        # the cat sat on the mat . the cat sat on  ->  penalise "the"
        the, cat, sat, on, mat, dot = 1, 2, 3, 4, 5, 6
        history = [the, cat, sat, on, the, mat, dot, the, cat, sat, on]
        processor = DryProcessor(0.8, breaker_token_ids=[dot])
        assert _penalties(processor, mx.array(history)) == {
            the: pytest.approx(0.8 * 1.75**2)
        }

    def test_short_repeat_is_free(self):
        processor = DryProcessor(0.8, allowed_length=3)
        assert _penalties(processor, mx.array([1, 2, 9, 1, 2])) == {}

    def test_breaker_resets_match(self):
        history = [1, 2, 3, 7, 1, 2, 3]
        assert _penalties(DryProcessor(0.8), mx.array(history)) == {
            7: pytest.approx(0.8 * 1.75)
        }
        broken = DryProcessor(0.8, breaker_token_ids=[2])
        assert _penalties(broken, mx.array(history)) == {}

    def test_accepts_python_list_history(self):
        history = [1, 2, 3, 7, 1, 2, 3]
        processor = DryProcessor(0.8)
        assert _penalties(processor, history) == _penalties(
            processor, mx.array(history)
        )

    def test_lazy_draft_prefix_matches_materialised(self):
        # Lightning MTP hands processors a concatenation of lazy arrays.
        processor = DryProcessor(0.8)
        lazy = mx.concatenate(
            [mx.array([1, 2, 3, 7]), mx.array([1, 2], dtype=mx.uint32)]
        )
        assert _penalties(processor, lazy) == _penalties(processor, [1, 2, 3, 7, 1, 2])


class TestDryScope:
    def test_prompt_only_repeat_is_not_penalised(self):
        prompt = [1, 2, 3, 4, 5, 6]
        generated = [1, 2, 3]
        processor = DryProcessor(0.8, prompt_length=len(prompt))
        assert _penalties(processor, mx.array(prompt + generated)) == {}

    def test_loop_in_generated_output_is_penalised(self):
        prompt = [1, 2, 3, 4, 5, 6]
        generated = [20, 21, 22, 23, 20, 21, 22]
        processor = DryProcessor(0.8, prompt_length=len(prompt))
        assert _penalties(processor, mx.array(prompt + generated)) == {
            23: pytest.approx(0.8 * 1.75)
        }

    def test_answer_may_copy_from_reasoning(self):
        think_end = 50
        reasoning = [20, 21, 22, 23, 24]
        processor = DryProcessor(0.8, think_end_token_id=think_end)
        history = reasoning + [think_end] + [20, 21, 22]
        assert _penalties(processor, mx.array(history)) == {}

    def test_loop_inside_reasoning_is_penalised(self):
        processor = DryProcessor(0.8, think_end_token_id=50)
        assert _penalties(processor, mx.array([20, 21, 22, 23, 20, 21, 22])) == {
            23: pytest.approx(0.8 * 1.75)
        }

    def test_loop_after_reasoning_is_penalised(self):
        processor = DryProcessor(0.8, think_end_token_id=50)
        history = [20, 21, 22, 23, 50, 30, 31, 32, 33, 30, 31, 32]
        assert _penalties(processor, mx.array(history)) == {
            33: pytest.approx(0.8 * 1.75)
        }

    def test_penalty_last_n_bounds_the_search(self):
        history = [1, 2, 3, 7] + [40 + i for i in range(10)] + [1, 2, 3]
        assert _penalties(DryProcessor(0.8), mx.array(history)) != {}
        assert _penalties(DryProcessor(0.8, penalty_last_n=8), mx.array(history)) == {}
        assert _penalties(DryProcessor(0.8, penalty_last_n=-1), mx.array(history)) != {}


class TestDryExcludeReasoning:
    OPEN, CLOSE = 49, 50
    LOOP = [20, 21, 22, 23, 20, 21, 22]

    def _processor(self, **kwargs):
        kwargs.setdefault("think_end_token_id", self.CLOSE)
        kwargs.setdefault("think_start_token_id", self.OPEN)
        return DryProcessor(0.8, exclude_reasoning=True, **kwargs)

    def test_no_penalty_when_prompt_opened_reasoning(self):
        processor = self._processor(starts_in_reasoning=True)
        assert _penalties(processor, mx.array(self.LOOP)) == {}

    def test_no_penalty_after_generated_think_start(self):
        processor = self._processor()
        assert _penalties(processor, mx.array([self.OPEN] + self.LOOP)) == {}

    def test_penalty_resumes_after_reasoning_closes(self):
        processor = self._processor(starts_in_reasoning=True)
        history = self.LOOP + [self.CLOSE] + [30, 31, 32, 33, 30, 31, 32]
        assert _penalties(processor, mx.array(history)) == {
            33: pytest.approx(0.8 * 1.75)
        }

    def test_close_think_outside_the_window_still_counts(self):
        processor = self._processor(starts_in_reasoning=True, penalty_last_n=8)
        answer = [40 + i for i in range(6)] + [30, 31, 32, 33, 30, 31, 32]
        history = [1, 2, 3] + [self.CLOSE] + answer
        assert _penalties(processor, mx.array(history)) == {
            33: pytest.approx(0.8 * 1.75)
        }

    def test_reasoning_scoped_after_the_prompt(self):
        prompt = [self.CLOSE, 5, 6]  # a close-think in the prompt is history
        processor = self._processor(starts_in_reasoning=True, prompt_length=len(prompt))
        assert _penalties(processor, mx.array(prompt + self.LOOP)) == {}

    def test_non_thinking_response_is_still_penalised(self):
        processor = self._processor()
        assert _penalties(processor, mx.array(self.LOOP)) == {
            23: pytest.approx(0.8 * 1.75)
        }

    def test_without_close_think_token_exclusion_is_off(self):
        processor = DryProcessor(0.8, exclude_reasoning=True, starts_in_reasoning=True)
        assert not processor.exclude_reasoning
        assert _penalties(processor, mx.array(self.LOOP)) != {}

    def test_accepts_python_list_history(self):
        processor = self._processor(starts_in_reasoning=True)
        history = self.LOOP + [self.CLOSE] + [30, 31, 32, 33, 30, 31, 32]
        assert _penalties(processor, history) == _penalties(
            processor, mx.array(history)
        )


class TestDryProcessorContract:
    def test_disabled_returns_logits_untouched(self):
        logits = mx.zeros((1, VOCAB))
        assert DryProcessor(0.0)([1, 2, 1, 2, 1], logits) is logits
        assert DryProcessor(0.8, penalty_last_n=0)([1, 2, 1, 2, 1], logits) is logits

    def test_preserves_logits_dtype_and_shape(self):
        logits = mx.zeros((1, VOCAB), dtype=mx.float16)
        out = DryProcessor(0.8)(mx.array([1, 2, 3, 7, 1, 2, 3]), logits)
        assert out.dtype == mx.float16
        assert out.shape == logits.shape

    def test_breaker_ids_beyond_logit_vocab_are_ignored(self):
        processor = DryProcessor(0.8, breaker_token_ids=[2, VOCAB + 100])
        assert _penalties(processor, mx.array([1, 2, 3, 7, 1, 2, 3])) == {}

    def test_is_stateless_for_speculative_paths(self):
        from omlx.speculative.processing_sampler import supports_vlm_mtp_processing

        processor = DryProcessor(0.8)
        assert supports_vlm_mtp_processing(processor)
        history = [1, 2, 3, 7, 1, 2, 3]
        first = _penalties(processor, history)
        processor([9, 9, 9, 9, 9, 9], mx.zeros((1, VOCAB)))  # a rejected draft
        processor.restore_state(processor.snapshot_state())
        assert _penalties(processor, history) == first

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"multiplier": -1},
            {"multiplier": 1, "base": 0.5},
            {"multiplier": 1, "allowed_length": 0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            DryProcessor(**kwargs)


class _Tokenizer:
    pieces = ["a", "b\n", ":", ' "x', "cd", "*"]

    def get_vocab(self):
        return {p: i for i, p in enumerate(self.pieces)}

    def batch_decode(self, batches):
        return [self.pieces[b[0]] for b in batches]


def test_find_breaker_token_ids_matches_on_substring():
    assert find_breaker_token_ids(_Tokenizer(), ("\n", ":", '"', "*")) == [1, 2, 3, 5]
    assert find_breaker_token_ids(_Tokenizer(), ()) == []


def test_sampling_params_dry_defaults_to_disabled():
    assert SamplingParams().dry is None
    dry = DryParams(multiplier=0.8)
    assert (dry.base, dry.allowed_length, dry.penalty_last_n) == (1.75, 2, 4096)
    assert dry.sequence_breakers == ("\n", ":", '"', "*", "/")


class TestResolveDryParams:
    @pytest.fixture
    def resolve(self, monkeypatch):
        from types import SimpleNamespace

        import omlx.server as server

        def run(request: dict, model: dict | None = None, force: bool = False):
            settings = (
                SimpleNamespace(force_sampling=force, **model)
                if model is not None
                else None
            )
            monkeypatch.setattr(
                server, "get_model_settings_for_request", lambda model_id: settings
            )
            return server._resolve_dry_params(SimpleNamespace(**request), "m")

        return run

    def test_disabled_by_default(self, resolve):
        assert resolve({}) is None
        assert resolve({"dry_multiplier": 0.0}, {"dry_multiplier": None}) is None

    def test_request_fields(self, resolve):
        dry = resolve({"dry_multiplier": 0.8, "dry_allowed_length": 4})
        assert dry == DryParams(multiplier=0.8, allowed_length=4)

    def test_model_default_applies_and_request_overrides_per_field(self, resolve):
        model = {"dry_multiplier": 0.6, "dry_base": 2.0, "dry_penalty_last_n": 1024}
        assert resolve({}, model) == DryParams(0.6, base=2.0, penalty_last_n=1024)
        assert resolve({"dry_base": 1.5}, model) == DryParams(
            0.6, base=1.5, penalty_last_n=1024
        )

    def test_request_zero_multiplier_turns_model_default_off(self, resolve):
        assert resolve({"dry_multiplier": 0.0}, {"dry_multiplier": 0.6}) is None

    def test_exclude_reasoning(self, resolve):
        assert resolve({"dry_multiplier": 0.8}).exclude_reasoning is False
        dry = resolve({}, {"dry_multiplier": 0.8, "dry_exclude_reasoning": True})
        assert dry.exclude_reasoning is True
        dry = resolve(
            {"dry_exclude_reasoning": False},
            {"dry_multiplier": 0.8, "dry_exclude_reasoning": True},
        )
        assert dry.exclude_reasoning is False

    def test_none_clears_sequence_breakers(self, resolve):
        dry = resolve({"dry_multiplier": 0.8, "dry_sequence_breakers": ["none"]})
        assert dry.sequence_breakers == ()
        dry = resolve({"dry_multiplier": 0.8, "dry_sequence_breakers": ["##"]})
        assert dry.sequence_breakers == ("##",)

    def test_force_sampling_prefers_model_settings(self, resolve):
        model = {"dry_multiplier": 0.6, "dry_base": 2.0}
        request = {"dry_multiplier": 0.0, "dry_base": 1.5, "dry_allowed_length": 5}
        assert resolve(request, model, force=True) == DryParams(
            0.6, base=2.0, allowed_length=5
        )

    def test_llama_cpp_disabled_values_do_not_error(self, resolve):
        assert resolve({"dry_multiplier": 0.8, "dry_base": 0.0}) is None
        assert resolve({"dry_multiplier": 0.8, "dry_allowed_length": 0}) == DryParams(
            0.8, allowed_length=1
        )


class TestSchedulerWiring:
    """``Scheduler._make_dry_processor`` builds the processor from request state."""

    def _scheduler(self, think_end_ids=(50,), think_start_id=49):
        from types import SimpleNamespace

        return SimpleNamespace(
            tokenizer=_Tokenizer(),
            _dry_breaker_token_ids={},
            _resolve_think_end_token_ids=lambda: list(think_end_ids),
            _get_think_token_id=lambda attr: think_start_id,
        )

    def _build(self, scheduler, dry, **request):
        from types import SimpleNamespace

        from omlx.scheduler import Scheduler

        return Scheduler._make_dry_processor(scheduler, dry, SimpleNamespace(**request))

    def test_disabled_builds_nothing(self):
        assert self._build(self._scheduler(), None) is None
        assert self._build(self._scheduler(), DryParams(multiplier=0.0)) is None

    def test_request_state_reaches_the_processor(self):
        processor = self._build(
            self._scheduler(),
            DryParams(multiplier=0.8, exclude_reasoning=True),
            prompt_token_ids=[1, 2, 3],
            needs_think_prefix=True,
        )
        assert processor.prompt_length == 3
        assert processor.exclude_reasoning
        assert processor.starts_in_reasoning
        assert (processor.think_end_token_id, processor.think_start_token_id) == (
            50,
            49,
        )

    def test_breaker_ids_are_cached_per_breaker_set(self):
        scheduler = self._scheduler()
        self._build(scheduler, DryParams(multiplier=0.8), prompt_token_ids=[1])
        assert scheduler._dry_breaker_token_ids == {
            ("\n", ":", '"', "*", "/"): [1, 2, 3, 5]
        }

    def test_multi_token_close_think_is_not_a_boundary(self):
        processor = self._build(
            self._scheduler(think_end_ids=(7, 8)),
            DryParams(multiplier=0.8, exclude_reasoning=True),
            prompt_token_ids=[1],
            needs_think_prefix=True,
        )
        assert processor.think_end_token_id is None
        assert not processor.exclude_reasoning
