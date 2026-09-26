# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.sampling.

The mlx-lm samplers wrap categorical_sampling and apply_* with
@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state). In the
omlx server environment that decorator stops advancing the global RNG state
after the first call, so identical prompts produce identical output. This
module re-implements the samplers without the decorator. These tests guard
against regression — RNG state must advance on every call and identical
inputs must produce non-trivial diversity at temperature > 0.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.utils.sampling import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    apply_xtc,
    categorical_sampling,
    make_sampler,
)


def _capture_rng() -> tuple:
    """Materialize the global RNG state so it can be compared across calls."""
    s = mx.random.state[0]
    mx.eval(s)
    return tuple(np.asarray(s).tolist())


def test_temp_zero_returns_argmax():
    """At temperature 0 make_sampler should be deterministic and return argmax."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 1000)) * 3.0
    mx.eval(logits)

    sampler = make_sampler(temp=0.0)
    out = sampler(logits)
    mx.eval(out)
    assert out.item() == mx.argmax(logits, axis=-1).item()


def test_categorical_advances_rng_state_each_call():
    """categorical_sampling must advance the global RNG state on every call.

    This is the regression we are guarding against: with the mlx-lm
    @partial(mx.compile, ...) decorator the state stops advancing after call 1.
    """
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 1000)) * 3.0
    mx.eval(logits)

    states = []
    for _ in range(5):
        states.append(_capture_rng())
        out = categorical_sampling(logits, 1.0)
        mx.eval(out)
    states.append(_capture_rng())

    for i in range(1, len(states)):
        assert states[i] != states[i - 1], f"RNG did not advance at step {i}"


def test_make_sampler_is_stochastic_with_top_p():
    """make_sampler(temp=1.0, top_p=0.95) should produce diverse outputs across
    repeated calls with the same logits."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 5000))
    mx.eval(logits)

    sampler = make_sampler(temp=1.0, top_p=0.95)
    results = set()
    for _ in range(30):
        out = sampler(logits)
        mx.eval(out)
        results.add(out.item())

    # With diverse logits and top_p=0.95 we expect plenty of variation
    assert len(results) > 5, f"sampler produced only {len(results)} unique tokens"


def test_apply_top_p_masks_tail_tokens():
    """apply_top_p should set masked tokens to -inf and keep top-mass tokens.

    The function expects logprobs (log of softmaxed probs), so feed it a
    log_softmax of raw logits.
    """
    raw = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    logprobs = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
    out = apply_top_p(logprobs, 0.5)
    mx.eval(out)
    out_np = np.asarray(out)
    logprobs_np = np.asarray(logprobs)
    # Token at index 4 has the highest logprob; it must survive
    assert out_np[0, 4] == logprobs_np[0, 4]
    # The lowest-logprob token must be masked to -inf with top_p=0.5
    assert np.isinf(out_np[0, 0]) and out_np[0, 0] < 0


def test_apply_top_k_keeps_only_k_tokens():
    """apply_top_k should mask all but the top-k highest logits."""
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = apply_top_k(logits, 2)
    mx.eval(out)
    out_np = np.asarray(out)
    # Top 2 are indices 3 and 4
    assert out_np[0, 4] == 5.0
    assert out_np[0, 3] == 4.0
    # The others must be -inf
    assert all(np.isinf(out_np[0, i]) and out_np[0, i] < 0 for i in (0, 1, 2))


def test_apply_min_p_masks_below_threshold():
    """apply_min_p should mask tokens below max(p) * min_p."""
    # Logits engineered so top token has prob ~ 0.99, others negligible
    logits = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
    out = apply_min_p(logits, min_p=0.1)
    mx.eval(out)
    out_np = np.asarray(out)
    assert out_np[0, 0] == 10.0
    # Tail tokens should be filtered
    assert all(np.isinf(out_np[0, i]) and out_np[0, i] < 0 for i in range(1, 5))


def test_apply_xtc_advances_rng_state():
    """apply_xtc uses mx.random.uniform internally, so it must also advance RNG."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 1000))
    mx.eval(logits)

    pre = _capture_rng()
    out = apply_xtc(logits, xtc_probability=0.5, xtc_threshold=0.1, xtc_special_tokens=[])
    mx.eval(out)
    post = _capture_rng()
    assert pre != post, "apply_xtc did not advance RNG"


def test_make_sampler_chain_advances_rng_state_each_call():
    """End-to-end: make_sampler with top_p must advance RNG on every call.

    This is the most direct guard for the regression: per-call state delta
    must be non-zero for at least the majority of calls.
    """
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 5000))
    mx.eval(logits)

    sampler = make_sampler(temp=1.0, top_p=0.9)
    states = [_capture_rng()]
    for _ in range(10):
        out = sampler(logits)
        mx.eval(out)
        states.append(_capture_rng())

    advanced = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    assert advanced == 10, f"RNG advanced only {advanced}/10 times"


@pytest.mark.parametrize("top_p", [0.0, 0.5, 0.9, 0.99])
def test_make_sampler_runs_with_various_top_p(top_p):
    """Sanity check: sampler should not crash for a range of top_p values."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 1000))
    mx.eval(logits)

    sampler = make_sampler(temp=1.0, top_p=top_p)
    out = sampler(logits)
    mx.eval(out)
    token = out.item()
    assert 0 <= token < 1000


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "params",
    [
        {"temp": 1.0},
        {"temp": 0.6, "top_p": 0.95, "top_k": 20},
        {"temp": 1.0, "top_p": 0.95, "top_k": 20},
        {"temp": 0.8, "min_p": 0.1},
    ],
)
def test_shared_draft_filter_preserves_draw_density_and_rng(dtype, params):
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _accept_lp_for,
        _sample_draft_with_logprobs,
    )

    mx.random.seed(731)
    logits = (mx.random.normal((4, 257)) * 3).astype(dtype)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    sampler = make_sampler(**params)
    for seed in range(5):
        mx.random.seed(seed)
        token = sampler(lp)
        density = _accept_lp_for(sampler, lp)
        mx.eval(token, density)
        rng = _capture_rng()
        mx.random.seed(seed)
        shared_token, shared_density = _sample_draft_with_logprobs(sampler, lp)
        mx.eval(shared_token, shared_density)
        assert mx.array_equal(token, shared_token).item()
        assert mx.array_equal(density, shared_density).item()
        assert _capture_rng() == rng


def test_shared_draft_filter_keeps_custom_and_greedy_sampler_contract():
    from omlx.patches.mlx_lm_mtp.batch_generator import _sample_draft_with_logprobs

    lp = mx.array([[-2.0, -1.0, -3.0]])
    calls = []

    def custom(values):
        calls.append(values)
        return mx.array([2])

    token, density = _sample_draft_with_logprobs(custom, lp)
    assert token.item() == 2
    assert len(calls) == 1
    assert density is lp
    greedy = make_sampler(temp=0)
    token, density = _sample_draft_with_logprobs(greedy, lp)
    assert token.item() == 1
    assert density is lp
    assert not hasattr(make_sampler(temp=1, xtc_probability=0.5), "sample_with_logprobs")


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("temp", [0.6, 1.0])
def test_shared_verify_filter_preserves_packet_and_rng(dtype, depth, temp):
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _accept_lp_for,
        _stochastic_verify_tokens,
    )

    mx.random.seed(719)
    lp = (mx.random.normal((depth + 1, 257)) * 3).astype(dtype)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    sampler = make_sampler(temp=temp, top_p=0.95, top_k=20)
    draft_lp = lp[:depth] + mx.random.normal((depth, 257)).astype(dtype)
    draft_lp = draft_lp - mx.logsumexp(draft_lp, axis=-1, keepdims=True)
    q = _accept_lp_for(sampler, draft_lp)
    drafts = mx.argmax(q, axis=-1)
    qs = [q[index] for index in range(depth)]
    mx.eval(lp, q, drafts)
    callback = sampler._mtp_sampling_logits
    for seed in range(3):
        del sampler._mtp_sampling_logits
        mx.random.seed(seed)
        expected = _stochastic_verify_tokens(sampler, lp, drafts, qs)
        mx.eval(expected)
        rng = _capture_rng()
        sampler._mtp_sampling_logits = callback
        mx.random.seed(seed)
        actual = _stochastic_verify_tokens(sampler, lp, drafts, qs)
        mx.eval(actual)
        assert mx.array_equal(expected, actual).item()
        assert _capture_rng() == rng


@pytest.mark.parametrize("scale", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("top_p, top_k", [(0.9, 20), (0.5, 50), (0.99, 5)])
def test_top_p_top_k_matches_sequential_filters(scale, top_p, top_k):
    from omlx.utils.sampling import apply_top_p_top_k

    mx.random.seed(11)
    logits = mx.random.normal((6, 512)) * scale
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    expected = apply_top_k(apply_top_p(lp, top_p), top_k)
    actual = apply_top_p_top_k(lp, top_p, top_k)
    kept = ~mx.isinf(expected)
    assert mx.array_equal(kept, ~mx.isinf(actual)).item()
    assert mx.array_equal(mx.where(kept, expected, 0), mx.where(kept, actual, 0)).item()


@pytest.mark.parametrize("vocab, top_k", [(16384, 20), (16384, 64), (1000, 20)])
def test_top_k_indices_matches_full_sort(vocab, top_k):
    from omlx.utils.sampling import top_k_indices

    mx.random.seed(5)
    values = mx.random.normal((3, vocab)) * 4
    expected = mx.sort(mx.argsort(-values, axis=-1)[:, :top_k], axis=-1)
    actual = mx.sort(top_k_indices(values, top_k), axis=-1)
    assert mx.array_equal(expected, actual).item()
