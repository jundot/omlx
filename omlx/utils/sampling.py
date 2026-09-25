# SPDX-License-Identifier: Apache-2.0
"""omlx sampling utilities — mx.compile-free re-implementation of mlx-lm samplers.

mlx-lm 0.31.x decorates ``categorical_sampling`` and the apply_* helpers with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``. In
the omlx server environment the decorator stops advancing the RNG state after
the first call: all subsequent samples reuse the same state, so identical
prompts produce character-identical output even at temperature > 1. Direct
calls to the underlying primitives advance the state correctly.

This module mirrors the mlx-lm implementation but drops the ``mx.compile``
wrappers, keeping behavior identical otherwise. ``make_sampler`` matches
``mlx_lm.sample_utils.make_sampler`` so it can replace the import in scheduler
without further changes.
"""

from __future__ import annotations

import math
from typing import Callable, List

import mlx.core as mx


def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Top-p (nucleus) filtering — keep the smallest set of tokens whose
    cumulative probability mass is at least ``top_p``. The most likely token
    is always kept."""
    # Sum in float32: a bfloat16 running sum over a large vocabulary stops
    # growing once each new term is below its rounding step.
    probs = mx.exp(logprobs.astype(mx.float32))
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

    # With a tiny top_p, 1 - top_p rounds to (or above) the float32 total, so
    # no token passes the threshold. Keep the most likely one regardless.
    keep = cumulative_probs > 1 - top_p
    keep = keep | (logprobs == mx.max(logprobs, axis=-1, keepdims=True))
    return mx.where(keep, logprobs, -float("inf"))


def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """Min-p filtering — drop tokens with probability below
    ``max(p) * min_p``, while always keeping ``min_tokens_to_keep`` tokens."""
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            False,
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


# The k largest entries of a row lie in the k chunks with the largest maxima,
# so ranking V / chunk maxima and k * chunk candidates finds them exactly.
# MLX argpartition sorts the whole row, which costs more at vocab scale.
_TOP_K_CHUNK = 128


def top_k_indices(values: mx.array, k: int) -> mx.array:
    """Indices of the ``k`` largest entries along the last axis, unordered."""
    vocab = values.shape[-1]
    if k > 64 or vocab % _TOP_K_CHUNK or vocab < 64 * _TOP_K_CHUNK:
        return mx.argpartition(-values, kth=k - 1, axis=-1)[..., :k]
    lead = values.shape[:-1]
    blocks = values.reshape(-1, vocab // _TOP_K_CHUNK, _TOP_K_CHUNK)
    best = mx.argpartition(-blocks.max(axis=-1), kth=k - 1, axis=-1)[:, :k]
    cand = mx.take_along_axis(blocks, best[:, :, None], axis=1)
    pick = mx.argpartition(-cand.reshape(-1, k * _TOP_K_CHUNK), kth=k - 1, axis=-1)[
        :, :k
    ]
    chunk = mx.take_along_axis(best, pick // _TOP_K_CHUNK, axis=-1)
    return (chunk * _TOP_K_CHUNK + pick % _TOP_K_CHUNK).reshape(*lead, k)


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Top-k filtering — keep only the ``top_k`` highest-probability tokens."""
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    keep = top_k_indices(logprobs, top_k)
    out = mx.full(logprobs.shape, -float("inf"), dtype=logprobs.dtype)
    return mx.put_along_axis(
        out, keep, mx.take_along_axis(logprobs, keep, axis=-1), axis=-1
    )


def apply_top_p_top_k(logprobs: mx.array, top_p: float, top_k: int) -> mx.array:
    """``apply_top_k(apply_top_p(logprobs, top_p), top_k)`` without a vocab sort.

    Top-p keeps a prefix of the descending order and top-k keeps the first k
    survivors, so only the k highest tokens can remain. For those, the top-p
    test needs the mass of higher-ranked tokens, which are also among the k.
    ``logprobs`` must be normalized over the vocabulary.
    """
    vocab_size = logprobs.shape[-1]
    if top_k >= vocab_size:
        return apply_top_p(logprobs, top_p)
    idx = top_k_indices(logprobs, top_k)
    vals = mx.take_along_axis(logprobs, idx, axis=-1).astype(mx.float32)
    order = mx.argsort(-vals, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    probs = mx.exp(vals)
    mass_above = mx.cumsum(probs, axis=-1) - probs
    vals = mx.where(mass_above < top_p, vals, -float("inf"))
    out = mx.full(logprobs.shape, -float("inf"), dtype=logprobs.dtype)
    return mx.put_along_axis(out, idx, vals.astype(logprobs.dtype), axis=-1)


def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> mx.array:
    """XTC sampling — with ``xtc_probability``, mask out all but the lowest
    above-threshold token to encourage diversity."""
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )

    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min()
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


# Below this temperature, sampling is argmax in practice: a 1e-3 logprob gap
# already becomes a factor of e^100. make_sampler treats these as greedy, which
# also keeps 1 / temp finite.
_MIN_SAMPLING_TEMP = 1e-5


def scale_by_temperature(logits: mx.array, temp: float) -> mx.array:
    """Return ``logits / temp``, scaling float16 input in float32.

    float16 tops out at 65504, so a small temperature turns every entry of a
    flat row into -inf and the draw is no longer from the requested
    distribution. Other dtypes are kept so seeded draws do not move.
    """
    if logits.dtype == mx.float16:
        logits = logits.astype(mx.float32)
    return logits * (1 / temp)


def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    """Sample a token id from the categorical distribution defined by
    ``logits / temp``. RNG state is advanced through ``mx.random.categorical``."""
    return mx.random.categorical(scale_by_temperature(logits, temp))


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]:
    """Build a sampler callable matching ``mlx_lm.sample_utils.make_sampler``.

    Returns ``argmax`` when ``temp == 0`` (or below ``_MIN_SAMPLING_TEMP``);
    otherwise composes optional top-p / min-p / xtc / top-k filters and
    finishes with categorical sampling.
    """
    if 0 < temp < _MIN_SAMPLING_TEMP:
        temp = 0.0
    if temp == 0:
        sampler = lambda x: mx.argmax(x, axis=-1)
    else:
        sampling_methods = []
        if 0 < top_p < 1.0 and top_k > 0 and min_p == 0.0 and xtc_probability <= 0.0:
            sampling_methods.append(lambda x: apply_top_p_top_k(x, top_p, top_k))
        else:
            if top_p > 0 and top_p < 1.0:
                sampling_methods.append(lambda x: apply_top_p(x, top_p))
            if min_p != 0.0:
                sampling_methods.append(
                    lambda x: apply_min_p(x, min_p, min_tokens_to_keep)
                )
            if xtc_probability > 0.0:
                sampling_methods.append(
                    lambda x: apply_xtc(
                        x, xtc_probability, xtc_threshold, xtc_special_tokens
                    )
                )
            if top_k > 0:
                sampling_methods.append(lambda x: apply_top_k(x, top_k))

        def sampler(logprobs: mx.array) -> mx.array:
            for method in sampling_methods:
                logprobs = method(logprobs)
            return categorical_sampling(logprobs, temp)

    if temp > 0 and xtc_probability == 0:

        def sampling_logits(logprobs: mx.array):
            for method in sampling_methods:
                logprobs = method(logprobs)
            return scale_by_temperature(logprobs, temp)

        def sample_with_logprobs(logprobs: mx.array, *, rowwise: bool = False):
            # Draft sampling and its acceptance density share the same filters.
            scaled = sampling_logits(logprobs)
            if rowwise:
                token = mx.concatenate(
                    [
                        mx.random.categorical(scaled[row : row + 1])
                        for row in range(scaled.shape[0])
                    ]
                )
            else:
                token = mx.random.categorical(scaled)
            density = scaled.astype(mx.float32)
            density = density - mx.logsumexp(density, axis=-1, keepdims=True)
            return token, density

        sampler._mtp_sampling_logits = sampling_logits
        sampler.sample_with_logprobs = sample_with_logprobs
        if 0 < top_p < 1 or min_p > 0 or top_k > 0:
            sampler._mtp_batch_sampling_key = (
                temp,
                top_p,
                min_p,
                min_tokens_to_keep,
                top_k,
            )

    # Expose sampling params on the returned callable so downstream code
    # (e.g. MTP acceptance check) can rebuild the filtered distribution
    # without re-plumbing the params through the BatchGenerator contract.
    # Lambda functions accept attribute assignment in CPython.
    sampler.temp = temp
    sampler.top_p = top_p
    sampler.min_p = min_p
    sampler.top_k = top_k
    sampler.min_tokens_to_keep = min_tokens_to_keep
    sampler.xtc_probability = xtc_probability
    return sampler
