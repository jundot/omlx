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

import random

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
    make_shared_sampler,
    select_logprobs,
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


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_apply_top_p_keeps_the_top_token_at_tiny_thresholds(dtype):
    """A low-precision cumulative sum used to mask every token (mlx-lm #1912)."""
    mx.random.seed(0)
    logits = mx.random.normal((2, 4096)) * 3
    logprobs = (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(dtype)
    for top_p in (1e-8, 1e-4, 1e-3):
        filtered = apply_top_p(logprobs, top_p)
        assert ((filtered > -float("inf")).sum(axis=-1) > 0).all().item()
        assert filtered.max(axis=-1).tolist() == logprobs.max(axis=-1).tolist()


def test_apply_xtc_threshold_is_per_row():
    """A batch row must not take its cutoff from another row's probabilities."""
    logits = mx.log(mx.array([[0.5, 0.3, 0.2], [0.05, 0.15, 0.8]]))
    out = apply_xtc(
        logits, xtc_probability=1.0, xtc_threshold=0.1, xtc_special_tokens=[]
    )
    # Each row keeps only its own lowest above-threshold token and below.
    assert (out > -float("inf")).tolist() == [[False, False, True], [True, True, False]]


def test_apply_min_p_keeps_min_tokens():
    logprobs = mx.log(mx.array([[0.9, 0.0, 0.0, 0.1]]))
    out = apply_min_p(logprobs, 0.5, min_tokens_to_keep=2)
    assert (out > -float("inf")).tolist() == [[True, False, False, True]]


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


def test_make_shared_sampler_reuses_one_object_per_parameter_set():
    sampler = make_shared_sampler(temp=0.7, top_p=0.95, top_k=20)
    assert make_shared_sampler(temp=0.7, top_p=0.95, top_k=20) is sampler
    assert make_shared_sampler(temp=0.7, top_p=0.95, top_k=40) is not sampler
    xtc = dict(temp=1.0, xtc_probability=0.5, xtc_threshold=0.1)
    assert make_shared_sampler(**xtc) is not make_shared_sampler(**xtc)


@pytest.mark.parametrize(
    "kwargs",
    [dict(temp=0.7, top_p=0.95, top_k=20), dict(temp=0.7, min_p=0.05, top_k=50)],
)
def test_shared_sampler_filters_a_batch_like_one_row_calls(kwargs):
    sampler = make_shared_sampler(**kwargs)
    logits = mx.random.normal((8, 128 * 80), key=mx.random.key(0)) * 4.0
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    rows = mx.concatenate(
        [sampler._mtp_sampling_logits(logprobs[e : e + 1]) for e in range(8)], axis=0
    )
    assert mx.array_equal(sampler._mtp_sampling_logits(logprobs), rows).item()


def _reference_no_repeat_ngram(sequence, ngram_size, window):
    """Unlimited-OCR's SlidingWindowNoRepeatNgramProcessor for one row.

    ``window=0`` searches the whole sequence, which is transformers'
    ``no_repeat_ngram_size``.
    """
    if len(sequence) < ngram_size:
        return set()
    search_start = max(0, len(sequence) - window) if window else 0
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return set()
    current_prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
    banned = set()
    for idx in range(search_start, search_end):
        ngram = sequence[idx : idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return banned


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
def test_no_repeat_ngram_matches_reference(dtype):
    from omlx.utils.sampling import make_no_repeat_ngram_processor

    rng = random.Random(0)
    for _ in range(400):
        vocab = rng.choice([6, 64])
        alphabet = rng.randint(1, 4)
        sequence = [rng.randrange(alphabet) for _ in range(rng.randint(0, 48))]
        ngram_size = rng.choice([1, 2, 3, 5])
        # Windows shorter and longer than the sequence, and none at all.
        window = rng.choice([0, ngram_size, ngram_size + 4, 16, 100])
        logits = mx.random.normal((1, vocab)).astype(mx.bfloat16)

        processor = make_no_repeat_ngram_processor(ngram_size, window)
        out = processor(mx.array(sequence, dtype=dtype), logits)

        expected = _reference_no_repeat_ngram(sequence, ngram_size, window)
        assert out.dtype == logits.dtype
        banned = mx.isinf(out[0]).tolist()
        assert {i for i, b in enumerate(banned) if b} == expected
        kept = mx.logical_not(mx.isinf(out))
        assert mx.array_equal(mx.where(kept, out, 0), mx.where(kept, logits, 0)).item()


@pytest.mark.parametrize(
    ("dtype", "tol"), [(mx.bfloat16, 0.07), (mx.float16, 0.01), (mx.float32, 1e-5)]
)
def test_select_logprobs_reads_batch_rows_on_host(dtype, tol):
    rng = np.random.default_rng(0)
    logits = mx.array(
        np.stack([rng.permutation(64) * 0.25 - 8 for _ in range(3)]), dtype=dtype
    )
    rows = list(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    mx.eval(rows)
    ref = logits.astype(mx.float32)
    ref = np.array(ref - mx.logsumexp(ref, axis=-1, keepdims=True))

    chosen, top = select_logprobs(rows[1], 5, 3)

    assert chosen == pytest.approx(ref[1, 5], abs=tol)
    assert [token for token, _ in top] == np.argsort(-ref[1])[:3].tolist()
    assert [value for _, value in top] == pytest.approx(
        np.sort(ref[1])[::-1][:3], abs=tol
    )
    assert select_logprobs(rows[1], 5, 0)[1] == []
    forced = mx.array([-float("inf"), 0.0, -float("inf")])
    assert select_logprobs(forced, 1, 3) == (0.0, [(1, 0.0)])


# ---- top-p-only MTP verify on the top-512 candidates (top_k == 0) ----------


def _nucleus_rows(rows, vocab, dtype, scale, seed):
    mx.random.seed(seed)
    logits = (mx.random.normal((rows, vocab)) * scale).astype(dtype)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp


def _nucleus_inputs(sampler, lp, depth, seed):
    from omlx.patches.mlx_lm_mtp.batch_generator import _accept_lp_for

    mx.random.seed(seed)
    noise = mx.random.normal(lp[:depth].shape) * 0.5
    draft_lp = lp[:depth].astype(mx.float32) + noise
    draft_lp = draft_lp - mx.logsumexp(draft_lp, axis=-1, keepdims=True)
    q = _accept_lp_for(sampler, draft_lp)
    drafts = mx.random.categorical(q).astype(mx.uint32)
    qs = [q[row] for row in range(depth)]
    mx.eval(q, drafts)
    return drafts, qs


@pytest.mark.parametrize("vocab", [65536, 40001])  # chunked / plain candidates
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("temp,top_p", [(0.6, 0.95), (1.0, 0.9)])
def test_nucleus_support_is_the_dense_top_p_density(vocab, dtype, temp, top_p):
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _accept_lp_for,
        _nucleus_support,
    )

    lp = _nucleus_rows(5, vocab, dtype, scale=6.0, seed=11)
    sampler = make_sampler(temp=temp, top_p=top_p)
    ids, logp, exact = _nucleus_support(sampler, lp)
    dense = _accept_lp_for(sampler, lp)
    sparse = mx.put_along_axis(mx.full(dense.shape, -float("inf")), ids, logp, axis=-1)
    mx.eval(exact, dense, sparse)
    assert exact.item() == 1
    assert mx.array_equal(mx.isinf(sparse), mx.isinf(dense)).item()
    keep = ~mx.isinf(dense)
    diff = mx.abs(mx.where(keep, sparse - dense, 0.0))
    assert diff.max().item() < 1e-5


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_nucleus_verify_packet_matches_dense(dtype):
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _settled_verify,
        _stochastic_verify_tokens,
        _verify_stochastic,
    )

    depth = 4
    lp = _nucleus_rows(depth + 1, 65536, dtype, scale=6.0, seed=3)
    sampler = make_sampler(temp=0.6, top_p=0.95)
    drafts, qs = _nucleus_inputs(sampler, lp, depth, seed=4)
    accepted = set()
    for seed in range(40):
        mx.random.seed(seed)
        dense = _stochastic_verify_tokens(sampler, lp, drafts, qs).tolist()
        mx.random.seed(seed)
        fast, redone = _settled_verify(
            _verify_stochastic(sampler, lp, drafts, qs).tolist(),
            sampler,
            lp,
            drafts,
            qs,
        )
        assert not redone
        # Same uniform draws, same densities: same acceptance and drafts.
        assert fast[: depth + 1] == dense[: depth + 1]
        accepted.add(dense[0])
    assert len(accepted) > 1  # the comparison saw both accepts and rejects


def test_nucleus_draft_outside_nucleus_rejects_and_residual_stays_inside():
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _accept_lp_for,
        _settled_verify,
        _verify_stochastic,
    )

    lp = _nucleus_rows(3, 65536, mx.float32, scale=6.0, seed=5)
    sampler = make_sampler(temp=1.0, top_p=0.9)
    dense = _accept_lp_for(sampler, lp)
    outside = mx.argmin(dense, axis=-1)[:2].astype(mx.uint32)  # p == 0
    q_peak = mx.full((2, 65536), -30.0).astype(mx.float32)
    q_peak = mx.put_along_axis(
        q_peak, outside[:, None].astype(mx.int32), mx.array([[0.0], [0.0]]), axis=-1
    )
    q_peak = q_peak - mx.logsumexp(q_peak, axis=-1, keepdims=True)
    qs = [q_peak[0], q_peak[1]]
    nucleus = [
        set(np.flatnonzero(np.asarray(~mx.isinf(dense[r]))).tolist()) for r in range(3)
    ]
    for seed in range(25):
        mx.random.seed(seed)
        host, redone = _settled_verify(
            _verify_stochastic(sampler, lp, outside, qs).tolist(),
            sampler,
            lp,
            outside,
            qs,
        )
        assert not redone
        assert host[0] == 0  # p(d) = 0: rejected at the first draft
        assert host[3] in nucleus[0]  # residual of row 0
        assert host[5] in nucleus[2]  # bonus


def test_nucleus_wider_than_candidates_is_redone_densely():
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _settled_verify,
        _stochastic_verify_tokens,
        _verify_stochastic,
    )

    depth = 3
    lp = _nucleus_rows(depth + 1, 65536, mx.float32, scale=0.3, seed=8)  # flat
    sampler = make_sampler(temp=1.0, top_p=0.95)
    drafts, qs = _nucleus_inputs(sampler, lp, depth, seed=9)
    mx.random.seed(1)
    flagged = _verify_stochastic(sampler, lp, drafts, qs).tolist()
    assert flagged[-1] == 0
    mx.random.seed(2)
    host, redone = _settled_verify(flagged, sampler, lp, drafts, qs)
    mx.random.seed(2)
    dense = _stochastic_verify_tokens(sampler, lp, drafts, qs).tolist()
    assert redone and host == dense


def test_nucleus_cutoff_tie_across_the_candidate_window_is_redone(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    monkeypatch.setattr(bg, "_NUCLEUS_CANDIDATES", 4)
    # One token holds 0.5; ten more tie at 0.05. top_p 0.7 cuts inside the tie,
    # so every tied token is kept, including those outside the 4 candidates.
    probs = np.array([0.5] + [0.05] * 10, dtype=np.float32)
    probs = np.concatenate([probs, np.zeros(2048 - 11, dtype=np.float32) + 1e-12])
    lp = mx.log(mx.array(probs / probs.sum()))[None, :]
    sampler = make_sampler(temp=1.0, top_p=0.7)
    _, _, exact = bg._nucleus_support(sampler, lp)
    assert exact.item() == 0


def test_nucleus_fast_path_is_limited_to_top_p_only_targets():
    from omlx.patches.mlx_lm_mtp.batch_generator import _nucleus_verify_eligible

    assert _nucleus_verify_eligible(make_sampler(temp=0.7, top_p=0.95), 248320)
    for params in (
        {"temp": 0.7, "top_p": 0.95, "top_k": 20},
        {"temp": 0.7, "top_p": 0.95, "min_p": 0.05},
        {"temp": 0.7, "top_p": 1.0},
        {"temp": 0.0, "top_p": 0.95},
    ):
        assert not _nucleus_verify_eligible(make_sampler(**params), 248320)
    assert not _nucleus_verify_eligible(make_sampler(temp=0.7, top_p=0.95), 1024)


def test_nucleus_mixed_rows_in_advance_group_settle_per_row(monkeypatch):
    """A row whose nucleus overflows the candidates is redone densely while its
    peaked neighbour keeps the fast host packet (fused_batch._advance_group)."""
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.mlx_lm_mtp import batched_head, fused_batch

    depth, vocab = 3, 2048
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=0)
    lps = [
        _nucleus_rows(depth + 1, vocab, mx.float32, scale=6.0, seed=21),  # peaked
        _nucleus_rows(depth + 1, vocab, mx.float32, scale=0.3, seed=22),  # flat
    ]
    assert [bg._nucleus_support(sampler, lp)[2].item() for lp in lps] == [1, 0]
    logits = mx.stack(lps)
    inputs = [
        _nucleus_inputs(sampler, lp, depth, seed=23 + i) for i, lp in enumerate(lps)
    ]
    rows, received = [], {}
    for i in range(2):
        state = SimpleNamespace(
            uid=i,
            drafts=inputs[i][0],
            draft_accept_lps=inputs[i][1],
            next_main=mx.array([0], dtype=mx.int32),
        )
        row = SimpleNamespace(
            samplers=[sampler],
            fallback_sampler=sampler,
            logits_processors=[None],
            prompt_cache=None,
            _token_context=[[0]],
        )
        rows.append((i, row, state))
    batch = SimpleNamespace(
        model=SimpleNamespace(),
        uids=[0, 1],
        _token_context=[[0], [0]],
        _omlx_mtp_batch_state=None,
    )

    def chain(row, state, **kwargs):
        received[state.uid] = kwargs["stochastic_result"]

    monkeypatch.setattr(bg, "_run_verify_cycle_chain", chain)
    monkeypatch.setattr(bg, "_drafter_for", lambda model: None)
    monkeypatch.setattr(bg, "_drafter_capture_ids", lambda model: None)
    monkeypatch.setattr(bg, "_set_batched_mrope_deltas", lambda batch, uids: None)
    monkeypatch.setattr(bg, "_set_singleton_mrope_delta", lambda row: None)
    monkeypatch.setattr(bg, "_slice_captured", lambda captured, idx: None)
    monkeypatch.setattr(bg, "_clear_rollback", lambda cache: None)
    monkeypatch.setattr(
        bg,
        "_call_backbone_captured",
        lambda model, inputs, cache, **kw: (
            logits,
            mx.zeros((2, depth + 1, 4)),
            None,
            None,
        ),
    )
    monkeypatch.setattr(batched_head, "eligible", lambda batch, rows: False)
    monkeypatch.setattr(batched_head, "flush", lambda batch_state: None)

    cache = []
    mx.random.seed(5)
    fused_batch._advance_group(batch, depth, rows, {}, cache=cache)

    # Same draws in the same order: both fast packets, then the dense redo of
    # row 1 only. Row 1's packet must be the dense one, row 0's the fast one.
    mx.random.seed(5)
    lps_in = [bg._logprobs(logits[i]) for i in range(2)]
    fast = [
        bg._verify_stochastic(sampler, lps_in[i], inputs[i][0], inputs[i][1]).tolist()
        for i in range(2)
    ]
    dense1, redone = bg._settled_verify(
        fast[1], sampler, lps_in[1], inputs[1][0], inputs[1][1]
    )
    assert redone
    assert dense1 != fast[1][:-1]  # the fixture makes the redo observable
    assert received[0] == fast[0][:-1]
    assert received[1] == dense1
