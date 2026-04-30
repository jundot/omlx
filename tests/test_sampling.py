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
