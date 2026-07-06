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
    cumulative probability mass is at least ``top_p``."""
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

    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


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


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Top-k filtering — keep only the ``top_k`` highest-probability tokens."""
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


def _top_k_top_p_candidates(
    logprobs: mx.array, top_k: int, top_p: float
) -> tuple[mx.array, mx.array]:
    """Shared step for the fused top-k+top-p fast path: gather the top_k
    logprobs via ``argpartition`` (no full-vocab sort), then run the same
    ascending-sort/cumsum recipe ``apply_top_p`` uses — restricted to just
    those top_k candidates instead of the whole vocabulary.

    Top-p and top-k are both *prefix* filters over the same probability
    ranking (highest logprob first), so composing them (in either order)
    keeps exactly the top ``min(top_k, nucleus_size)`` tokens. Any token
    ranked above a top-k candidate is, by construction, also inside the
    top-k candidate set, so the "probability mass above this token" needed
    for the top-p threshold is exactly computable from the top_k subset
    alone. It just isn't relative to a subset sum of 1 the way the
    full-vocab version is, since the top_k candidates don't carry the
    whole probability mass — hence the ``subset_total`` shift below instead
    of the ``1`` used in ``apply_top_p``.

    Everything downstream (scatter for ``_fused_top_k_top_p``, sampling for
    ``_fused_top_k_top_p_sample``) only needs a consistent (index, value)
    pairing, not any particular order, so this stays in ascending-sorted
    order throughout rather than remapping back to argpartition's order —
    one fewer top_k-sized permutation to build.

    Returns ``(sorted_idx, filtered_sorted_vals)``, both in ascending
    logprob order: the top_k candidate indices into the vocab, and their
    logprobs with non-surviving entries set to -inf.
    """
    # Must negate (matching apply_top_k's own argpartition call) rather than
    # use kth=-top_k on the un-negated array: the two are only equivalent
    # when there are no exact ties. At this vocab size in bf16, ties at the
    # top_k boundary are common, and argpartition's tie-breaking differs
    # between the two forms — using the un-negated form would occasionally
    # select a different (equally-probable, but different-id) token than
    # apply_top_k does, breaking exact parity with the composition this
    # fast path replaces.
    idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., :top_k]
    vals = mx.take_along_axis(logprobs, idx, axis=-1)

    order = mx.argsort(vals, axis=-1)
    sorted_idx = mx.take_along_axis(idx, order, axis=-1)
    sorted_vals = mx.take_along_axis(vals, order, axis=-1)
    cumulative_probs = mx.cumsum(mx.exp(sorted_vals), axis=-1)
    subset_total = cumulative_probs[..., -1:]

    filtered_sorted_vals = mx.where(
        cumulative_probs > subset_total - top_p, sorted_vals, -float("inf")
    )
    return sorted_idx, filtered_sorted_vals


def _fused_top_k_top_p(logprobs: mx.array, top_k: int, top_p: float) -> mx.array:
    """Fused top-k + top-p filtering that avoids a full-vocab argsort.

    Equivalent to ``apply_top_p(logprobs, top_p)`` followed by
    ``apply_top_k(..., top_k)`` — the order ``make_sampler`` composes them
    in when both are the only active filters — but computed from just the
    top_k candidates (see ``_top_k_top_p_candidates``) instead of sorting
    the whole vocabulary. Scatters the filtered candidates back into a
    vocab-sized -inf array, matching apply_top_p/apply_top_k's contract.
    """
    sorted_idx, filtered_sorted_vals = _top_k_top_p_candidates(logprobs, top_k, top_p)
    out = mx.full(logprobs.shape, -float("inf"), dtype=logprobs.dtype)
    return mx.put_along_axis(out, sorted_idx, filtered_sorted_vals, axis=-1)


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


def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    """Sample a token id from the categorical distribution defined by
    ``logits / temp``. RNG state is advanced through ``mx.random.categorical``."""
    return mx.random.categorical(logits * (1 / temp))


def _fused_top_k_top_p_sample(
    logprobs: mx.array, top_k: int, top_p: float, temp: float
) -> mx.array:
    """Sample directly from the fused top-k+top-p candidates instead of
    materializing a vocab-sized filtered array first.

    ``_fused_top_k_top_p`` still has to scatter its ``top_k`` survivors
    into a vocab-sized -inf array purely so a subsequent
    ``categorical_sampling`` call can index into the vocab axis. Sampling
    from the ``top_k``-sized candidates directly and mapping the result
    back to a vocab id via the gathered indices skips that vocab-sized
    scatter entirely — this is what ``make_sampler`` actually calls on its
    fast path.
    """
    sorted_idx, filtered_sorted_vals = _top_k_top_p_candidates(logprobs, top_k, top_p)
    local_idx = categorical_sampling(filtered_sorted_vals, temp)
    return mx.take_along_axis(sorted_idx, local_idx[..., None], axis=-1).squeeze(-1)


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

    Returns ``argmax`` when ``temp == 0``; otherwise composes optional
    top-p / min-p / xtc / top-k filters and finishes with categorical sampling.
    """
    if temp == 0:
        sampler = lambda x: mx.argmax(x, axis=-1)
    else:
        top_p_active = top_p > 0 and top_p < 1.0
        top_k_active = top_k > 0
        if top_p_active and top_k_active and min_p == 0.0 and xtc_probability == 0.0:
            # Fast path: top-p and top-k are the only active filters, so
            # they can be fused into a single top_k-sized computation
            # instead of a full-vocab argsort, sampling directly from the
            # top_k candidates (see _fused_top_k_top_p_sample).
            def sampler(logprobs: mx.array) -> mx.array:
                return _fused_top_k_top_p_sample(logprobs, top_k, top_p, temp)

        else:
            sampling_methods = []
            if top_p_active:
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
            if top_k_active:
                sampling_methods.append(lambda x: apply_top_k(x, top_k))

            def sampler(logprobs: mx.array) -> mx.array:
                for method in sampling_methods:
                    logprobs = method(logprobs)
                return categorical_sampling(logprobs, temp)

    # Expose sampling params on the returned callable so downstream code
    # (e.g. MTP acceptance check) can rebuild the filtered distribution
    # without re-plumbing the params through the BatchGenerator contract.
    # Lambda functions accept attribute assignment in CPython.
    sampler.temp = temp
    sampler.top_p = top_p
    sampler.min_p = min_p
    sampler.top_k = top_k
    sampler.min_tokens_to_keep = min_tokens_to_keep
    return sampler
