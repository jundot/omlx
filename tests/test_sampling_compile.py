# SPDX-License-Identifier: Apache-2.0
"""Tests for the compiled RNG-free sampling filters in omlx.utils.sampling.

apply_top_p / apply_min_p / apply_top_k are wrapped in mx.compile (they never
touch the RNG, so the frozen-RNG-state bug that forced mlx-lm's
@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state) off
cannot recur). categorical_sampling / apply_xtc use the RNG and must stay
uncompiled. These tests pin:

(a) numerical equivalence of the compiled filters against plain uncompiled
    reference implementations, across shapes, dtypes, and edge cases
    (top_p=1.0 / ~0, top_k=1 / >= vocab, flat min_p inputs);
(b) the RNG-diversity regression guard — repeated sampling at temp > 0 on
    identical logits must still produce varied tokens;
(c) determinism under a fixed seed.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.utils.sampling import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
    make_sampler,
)

# Plain uncompiled reference implementations (same algorithms, eager ops).


def _ref_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    return mx.where(cumulative_probs > 1 - top_p, logprobs, -float("inf"))


def _ref_min_p(logprobs: mx.array, min_p: float) -> mx.array:
    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    tokens_to_remove = logprobs < top_logprobs + math.log(min_p)
    return mx.where(tokens_to_remove, -float("inf"), logprobs)


def _ref_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )


def _make_logprobs(rows: int, vocab: int, seed: int, dtype=mx.float32) -> mx.array:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((rows, vocab)).astype(np.float32)
    logprobs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return mx.array(logprobs).astype(dtype)


def _capture_rng() -> tuple:
    """Materialize the global RNG state so it can be compared across calls."""
    s = mx.random.state[0]
    mx.eval(s)
    return tuple(np.asarray(s).tolist())


def _as_np(arr: mx.array) -> np.ndarray:
    return np.asarray(arr.astype(mx.float32))


def _assert_filter_equiv(
    out: mx.array, ref: mx.array, logprobs: mx.array, boundary_dist: np.ndarray
):
    """Compiled vs eager reference: tokens kept by both must carry identical
    values; mask flips are allowed only within float-rounding distance of the
    decision boundary (kernel fusion can round a boundary comparison the other
    way — semantically irrelevant ties)."""
    out_np, ref_np = _as_np(out), _as_np(ref)
    kept_out, kept_ref = np.isfinite(out_np), np.isfinite(ref_np)
    both = kept_out & kept_ref
    assert np.array_equal(out_np[both], ref_np[both])
    flips = kept_out ^ kept_ref
    if flips.any():
        # Few flips, all essentially on the boundary (f32 rounding distance).
        assert flips.sum() <= max(1, flips.size // 1000)
        assert (boundary_dist[flips] <= 1e-4).all()


# (a) Numerical equivalence compiled vs plain reference.


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("rows,vocab", [(1, 8), (1, 1000), (3, 50000), (2, 131072)])
def test_apply_top_p_matches_reference(dtype, rows, vocab):
    logprobs = _make_logprobs(rows, vocab, seed=0, dtype=dtype)
    lp64 = _as_np(logprobs).astype(np.float64)
    for top_p in (1.0, 0.95, 0.5, 0.1, 1e-4):
        out = apply_top_p(logprobs, top_p)
        ref = _ref_top_p(logprobs, top_p)
        mx.eval(out, ref)
        # Per-token distance to the top-p decision boundary in float64.
        order = np.argsort(lp64, axis=-1)
        cum = np.cumsum(np.take_along_axis(np.exp(lp64), order, axis=-1), axis=-1)
        inv = np.empty_like(order)
        np.put_along_axis(inv, order, np.arange(order.shape[-1]), axis=-1)
        cum = np.take_along_axis(cum, inv, axis=-1)
        _assert_filter_equiv(out, ref, logprobs, np.abs(cum - (1 - top_p)))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("rows,vocab", [(1, 8), (1, 1000), (3, 50000), (2, 131072)])
def test_apply_min_p_matches_reference(dtype, rows, vocab):
    logprobs = _make_logprobs(rows, vocab, seed=1, dtype=dtype)
    lp64 = _as_np(logprobs).astype(np.float64)
    for min_p in (1.0, 0.1, 0.01, 1e-3):
        out = apply_min_p(logprobs, min_p)
        ref = _ref_min_p(logprobs, min_p)
        mx.eval(out, ref)
        thr = lp64.max(axis=-1, keepdims=True) + math.log(min_p)
        _assert_filter_equiv(out, ref, logprobs, np.abs(lp64 - thr))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("rows,vocab", [(1, 8), (1, 1000), (3, 50000), (2, 131072)])
def test_apply_top_k_matches_reference(dtype, rows, vocab):
    logprobs = _make_logprobs(rows, vocab, seed=2, dtype=dtype)
    lp64 = _as_np(logprobs).astype(np.float64)
    for top_k in (1, 2, vocab // 2, vocab - 1):
        out = apply_top_k(logprobs, top_k)
        ref = _ref_top_k(logprobs, top_k)
        mx.eval(out, ref)
        kth = np.sort(lp64, axis=-1)[..., [-top_k]]
        _assert_filter_equiv(out, ref, logprobs, np.abs(lp64 - kth))


# (a) Edge-case semantics.


def test_top_p_one_keeps_everything():
    logprobs = _make_logprobs(2, 1000, seed=3)
    out = apply_top_p(logprobs, 1.0)
    mx.eval(out)
    assert np.array_equal(_as_np(out), _as_np(logprobs))


def test_top_p_near_zero_keeps_only_top_tokens():
    # One dominant logit: with p ~ 0 only the argmax may survive.
    raw = mx.array([[10.0, 1.0, 0.0, -1.0, -2.0]])
    logprobs = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
    out = apply_top_p(logprobs, 1e-4)
    mx.eval(out)
    out_np = np.asarray(out)
    assert out_np[0, 0] == np.asarray(logprobs)[0, 0]
    assert all(np.isinf(out_np[0, i]) for i in range(1, 5))


def test_top_k_one_keeps_only_argmax():
    logprobs = _make_logprobs(2, 1000, seed=4)
    out = apply_top_k(logprobs, 1)
    mx.eval(out)
    out_np = _as_np(out)
    kept = np.isfinite(out_np)
    assert kept.sum() == 2  # exactly one token per row
    assert np.array_equal(
        np.argmax(_as_np(logprobs), axis=-1), np.argmax(kept, axis=-1)
    )


def test_top_k_vocab_minus_one_masks_only_minimum():
    vocab = 1000
    logprobs = _make_logprobs(1, vocab, seed=5)
    out = apply_top_k(logprobs, vocab - 1)
    mx.eval(out)
    out_np = _as_np(out)
    assert np.isinf(out_np).sum() == 1
    assert int(np.argmin(out_np)) == int(np.argmin(_as_np(logprobs)))


@pytest.mark.parametrize("top_k", [0, -1, 8, 100])
def test_top_k_out_of_range_raises(top_k):
    logprobs = _make_logprobs(1, 8, seed=6)
    with pytest.raises(ValueError):
        apply_top_k(logprobs, top_k)


def test_min_p_flat_distribution_keeps_everything():
    # Uniform logits: every token has the max probability, none is removable.
    logprobs = mx.full((2, 500), -math.log(500))
    out = apply_min_p(logprobs, 0.5)
    mx.eval(out)
    assert np.isfinite(_as_np(out)).all()


def test_min_p_one_keeps_only_max():
    logprobs = _make_logprobs(1, 1000, seed=7)
    out = apply_min_p(logprobs, 1.0)
    mx.eval(out)
    out_np = _as_np(out)
    assert np.isfinite(out_np).sum() == 1
    assert int(np.argmax(out_np)) == int(np.argmax(_as_np(logprobs)))


def test_min_p_invalid_raises():
    logprobs = _make_logprobs(1, 8, seed=8)
    with pytest.raises(ValueError):
        apply_min_p(logprobs, 1.5)
    with pytest.raises(ValueError):
        apply_min_p(logprobs, -0.1)
    with pytest.raises(ValueError):
        apply_min_p(logprobs, 0.5, min_tokens_to_keep=0)


# (b) RNG-diversity regression guards.


def test_compiled_filters_do_not_touch_rng_state():
    """The compiled filters must leave the global RNG state bit-identical —
    this is the mechanism behind the original frozen-state bug."""
    mx.random.seed(0)
    logprobs = _make_logprobs(4, 50000, seed=9)
    mx.eval(logprobs)

    pre = _capture_rng()
    outs = [
        apply_top_p(logprobs, 0.9),
        apply_min_p(logprobs, 0.1),
        apply_top_k(logprobs, 50),
    ]
    mx.eval(*outs)
    assert _capture_rng() == pre, "compiled filters advanced the RNG state"


def test_compiled_chain_advances_rng_state_each_call():
    """make_sampler with all compiled filters active must advance the RNG on
    every call — the original bug must not reappear via the filter path."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 5000))
    mx.eval(logits)

    sampler = make_sampler(temp=1.0, top_p=0.9, min_p=0.02, top_k=100)
    states = [_capture_rng()]
    for _ in range(10):
        out = sampler(logits)
        mx.eval(out)
        states.append(_capture_rng())

    advanced = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    assert advanced == 10, f"RNG advanced only {advanced}/10 times"


def test_compiled_chain_produces_diverse_tokens():
    """Repeated sampling at temp > 0 on identical logits must still vary."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 5000))
    mx.eval(logits)

    sampler = make_sampler(temp=1.0, top_p=0.95, min_p=0.02, top_k=200)
    results = set()
    for _ in range(30):
        out = sampler(logits)
        mx.eval(out)
        results.add(out.item())

    assert len(results) > 5, f"sampler produced only {len(results)} unique tokens"


# (c) Determinism under a fixed seed.


def _sample_sequence(sampler, logprobs, n):
    tokens = []
    for _ in range(n):
        out = sampler(logprobs)
        mx.eval(out)
        tokens.append(out.item())
    return tokens


def test_compiled_chain_deterministic_under_fixed_seed():
    logits = mx.array(
        np.random.default_rng(11).standard_normal((1, 5000)).astype(np.float32)
    )
    mx.eval(logits)
    sampler = make_sampler(temp=0.8, top_p=0.95, min_p=0.02, top_k=100)

    mx.random.seed(42)
    first = _sample_sequence(sampler, logits, 16)
    mx.random.seed(42)
    second = _sample_sequence(sampler, logits, 16)
    assert first == second


def test_categorical_sampling_still_uncompiled_and_stochastic():
    """Direct guard: categorical_sampling advances RNG state on every call."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 1000))
    mx.eval(logits)

    states = [_capture_rng()]
    for _ in range(5):
        out = categorical_sampling(logits, 1.0)
        mx.eval(out)
        states.append(_capture_rng())
    assert all(states[i] != states[i - 1] for i in range(1, len(states)))


# Public contract.


def test_make_sampler_attribute_contract():
    """make_sampler must keep exposing the sampling params on the callable."""
    sampler = make_sampler(
        temp=0.7, top_p=0.9, min_p=0.05, min_tokens_to_keep=2, top_k=64
    )
    assert sampler.temp == 0.7
    assert sampler.top_p == 0.9
    assert sampler.min_p == 0.05
    assert sampler.min_tokens_to_keep == 2
    assert sampler.top_k == 64

    greedy = make_sampler(temp=0.0)
    assert greedy.temp == 0.0
    logits = mx.random.normal(shape=(1, 100))
    mx.eval(logits)
    out = greedy(logits)
    mx.eval(out)
    assert out.item() == mx.argmax(logits, axis=-1).item()


def test_compiled_filters_accept_kwargs_and_defaults():
    """Public signatures are unchanged: kwargs and default args still work."""
    raw = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    logprobs = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
    a = apply_top_p(logprobs=logprobs, top_p=0.5)
    b = apply_min_p(logprobs, 0.1)
    c = apply_min_p(logprobs=logprobs, min_p=0.1, min_tokens_to_keep=1)
    d = apply_top_k(logprobs=logprobs, top_k=2)
    mx.eval(a, b, c, d)
    assert np.array_equal(np.asarray(b), np.asarray(c))
