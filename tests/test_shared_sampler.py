# SPDX-License-Identifier: Apache-2.0
"""Requests with equal sampling parameters share one sampler object.

mlx-lm's GenerationBatch._step samples a decode batch in one call only when
every row's sampler is the same object. These tests pin the sharing, its
exceptions (XTC, the opt-out) and that a shared call over a batch filters each
row exactly as a one-row call would. CPU only.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.request import SamplingParams
from omlx.utils import sampling


@pytest.fixture(autouse=True, scope="module")
def _cpu_device():
    # CPU only, but restore the previous default device: a module-level
    # set_default_device would move every later test in the session.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(sampling, "_shared_samplers", type(sampling._shared_samplers)())
    monkeypatch.setattr(sampling, "_SHARED_SAMPLERS_ENABLED", True)


def test_equal_parameters_share_one_sampler():
    a = sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=20)
    b = sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=20)
    assert a is b
    assert a.temp == 0.7 and a.top_p == 0.95 and a.top_k == 20


def test_different_parameters_get_different_samplers():
    a = sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=20)
    assert sampling.make_shared_sampler(temp=0.6, top_p=0.95, top_k=20) is not a
    assert sampling.make_shared_sampler(temp=0.7, top_p=0.9, top_k=20) is not a
    assert sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=40) is not a
    assert (
        sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=20, min_p=0.05)
        is not a
    )
    assert sampling.make_shared_sampler(temp=0.0) is sampling.make_shared_sampler(
        temp=0.0
    )


def test_xtc_is_never_shared():
    kwargs = dict(temp=1.0, xtc_probability=0.5, xtc_threshold=0.1)
    assert sampling.make_shared_sampler(**kwargs) is not sampling.make_shared_sampler(
        **kwargs
    )


def test_opt_out_builds_one_sampler_per_call(monkeypatch):
    monkeypatch.setattr(sampling, "_SHARED_SAMPLERS_ENABLED", False)
    kwargs = dict(temp=0.7, top_p=0.95, top_k=20)
    assert sampling.make_shared_sampler(**kwargs) is not sampling.make_shared_sampler(
        **kwargs
    )


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(sampling, "_SHARED_SAMPLERS_MAX", 4)
    for k in range(1, 10):
        sampling.make_shared_sampler(temp=0.7, top_k=k)
    assert len(sampling._shared_samplers) == 4


def _logprobs(rows: int, vocab: int = 128 * 80, seed: int = 0) -> mx.array:
    logits = mx.random.normal((rows, vocab), key=mx.random.key(seed)) * 4.0
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(temp=0.7, top_p=0.95, top_k=20),
        dict(temp=0.7, top_p=0.9),
        dict(temp=0.7, min_p=0.05, top_k=50),
    ],
)
def test_batched_filter_matches_one_row_calls(kwargs):
    sampler = sampling.make_shared_sampler(**kwargs)
    logprobs = _logprobs(8)
    batched = sampler._mtp_sampling_logits(logprobs)
    rows = mx.concatenate(
        [sampler._mtp_sampling_logits(logprobs[e : e + 1]) for e in range(8)], axis=0
    )
    assert mx.array_equal(batched, rows).item()


def test_greedy_batched_call_matches_one_row_calls():
    sampler = sampling.make_shared_sampler(temp=0.0)
    logprobs = _logprobs(8, seed=1)
    rows = mx.concatenate([sampler(logprobs[e : e + 1]) for e in range(8)], axis=0)
    assert mx.array_equal(sampler(logprobs), rows).item()


def test_batched_draws_stay_inside_each_rows_support():
    sampler = sampling.make_shared_sampler(temp=0.7, top_p=0.95, top_k=20)
    logprobs = _logprobs(8, seed=2)
    support = sampler._mtp_sampling_logits(logprobs) > -float("inf")
    for _ in range(16):
        tokens = sampler(logprobs)
        assert tokens.shape == (8,)
        assert mx.take_along_axis(support, tokens[:, None], axis=-1).all().item()


def test_scheduler_requests_with_equal_parameters_share_a_sampler():
    from omlx.scheduler import Scheduler

    fake = SimpleNamespace(_xtc_special_tokens=[], _model_suppress_tokens=set())
    params = SamplingParams(temperature=0.7, top_p=0.95, top_k=20)
    first, _ = Scheduler._build_sampler_and_processors(fake, params)
    second, _ = Scheduler._build_sampler_and_processors(
        fake, SamplingParams(temperature=0.7, top_p=0.95, top_k=20, seed=3)
    )
    other, _ = Scheduler._build_sampler_and_processors(
        fake, SamplingParams(temperature=0.7, top_p=0.95, top_k=40)
    )
    assert first is second
    assert other is not first
    # mlx-lm GenerationBatch._step's one-call condition.
    samplers = [first, second]
    assert all(s is samplers[0] for s in samplers)
