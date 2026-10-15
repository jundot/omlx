# SPDX-License-Identifier: Apache-2.0
"""Draft proposals and their q distributions from the drafter's logits."""

from __future__ import annotations

import math

import mlx.core as mx

from ...patches.mlx_lm_mtp import batch_generator as bg
from ...utils.sampling import top_k_indices
from .kernels import select_kernel


def greedy_proposals(logits: mx.array) -> mx.array:
    # DFlash v1 samples proposals from this callable; the DFlash2 selector
    # only looks for ``sample_proposal`` on it and otherwise takes argmax.
    return mx.argmax(logits, axis=-1)


def sample_positions(logits: mx.array, samplers: list) -> list[tuple]:
    """Independent per-position proposals (DFlash v1) with their q rows."""
    out = []
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for index, sampler in enumerate(samplers):
        if sampler is None:
            out.append((mx.argmax(logits[index : index + 1], axis=-1), []))
            continue
        tokens, accept = [], []
        for position in range(lp.shape[1]):
            token, accept_lp = bg._sample_draft_with_logprobs(
                sampler, lp[index, position : position + 1]
            )
            tokens.append(token.reshape(1))
            accept.append(accept_lp.reshape(-1))
        out.append((mx.concatenate(tokens)[None], accept))
    return out


def sample_candidates(scores: mx.array, sampler) -> tuple[mx.array, mx.array]:
    """Draw from ``sampler``'s filters applied to candidate scores ``(1, C)``.

    Returns the chosen column and the log density over the candidates. The
    density is the proposal q that acceptance reads, so the filters only have
    to match the request's in spirit, not bit for bit.
    """
    lp = scores.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    count = lp.shape[-1]
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    temp = float(getattr(sampler, "temp", 1.0) or 1.0)
    if 0 < top_k < count:
        kth = mx.sort(lp, axis=-1)[..., count - top_k : count - top_k + 1]
        lp = mx.where(lp >= kth, lp, -float("inf"))
    if 0.0 < top_p < 1.0:
        order = mx.argsort(-lp, axis=-1)
        ranked = mx.take_along_axis(lp, order, axis=-1)
        probs = mx.exp(ranked)
        keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
        keep = mx.put_along_axis(
            mx.zeros(keep.shape, dtype=mx.bool_), order, keep, axis=-1
        )
        lp = mx.where(keep, lp, -float("inf"))
    if min_p > 0.0:
        floor = lp.max(axis=-1, keepdims=True) + math.log(min_p)
        lp = mx.where(lp >= floor, lp, -float("inf"))
    scaled = lp * (1.0 / temp)
    pick = mx.random.categorical(scaled)
    return pick, scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def fused_select_eligible(selector, hidden) -> bool:
    pcb = getattr(selector.predecessor_codebook, "weight", None)
    scb = getattr(selector.successor_codebook, "weight", None)
    return (
        selector.top_k == 16
        and isinstance(pcb, mx.array)
        and isinstance(scb, mx.array)
        and not hasattr(selector.predecessor_codebook, "scales")
        and not hasattr(selector.successor_codebook, "scales")
        and pcb.shape[-1] % 32 == 0
        and hidden.shape[1] <= 32
    )


def select_fused(selector, hidden, logits, anchors, samplers) -> list[tuple]:
    """``select_sampled`` in one launch with sparse q over the candidates."""
    batch, length, vocab = logits.shape
    candidates = top_k_indices(logits, selector.top_k).astype(mx.int32)
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden)
    params = mx.array(
        [
            (
                [0.0, 0.0, 0.0, 0.0]
                if sampler is None
                else [
                    float(getattr(sampler, "temp", 1.0) or 1.0),
                    float(getattr(sampler, "top_p", 0.0) or 0.0),
                    float(int(getattr(sampler, "top_k", 0) or 0)),
                    float(getattr(sampler, "min_p", 0.0) or 0.0),
                ]
            )
            for sampler in samplers
        ],
        dtype=mx.float32,
    )
    pcb = selector.predecessor_codebook.weight
    tokens, logq = select_kernel()(
        inputs=[
            candidates,
            unary,
            projected,
            anchors.reshape(-1).astype(mx.int32),
            pcb,
            selector.successor_codebook.weight,
            params,
            mx.random.uniform(shape=(batch, length)),
        ],
        template=[("P", length), ("R", pcb.shape[-1])],
        grid=(32 * selector.top_k * batch, 1, 1),
        threadgroup=(32 * selector.top_k, 1, 1),
        output_shapes=[(batch, length), (batch, length, selector.top_k)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return [
        (
            tokens[index : index + 1],
            (
                []
                if sampler is None
                else bg.SparseDraftQ(candidates[index], logq[index], vocab)
            ),
        )
        for index, sampler in enumerate(samplers)
    ]


def q_arrays(accept_lps) -> list:
    if isinstance(accept_lps, bg.SparseDraftQ):
        return accept_lps.arrays()
    return list(accept_lps)


def select_sampled(selector, hidden, logits, anchors, samplers) -> list[tuple]:
    """DFlash2 selector path with per-row sampling over the candidate set.

    Mirrors ``CandidateSelector.select``: top-k candidates per position, a
    predecessor/successor edge score, then the next predecessor is the
    chosen token. Sampled rows draw from their sampler's filters over the
    candidate scores and expose that sparse distribution as q.
    """
    batch, length, vocab = logits.shape
    candidates = top_k_indices(logits, selector.top_k)
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden)
    predecessor = anchors.reshape(-1)
    tokens = [[] for _ in range(batch)]
    densities = [[] for _ in range(batch)]
    for position in range(length):
        edges = mx.sum(
            selector.predecessor_codebook(predecessor)[:, None]
            * projected[:, position, None]
            * selector.successor_codebook(candidates[:, position]),
            axis=-1,
        )
        scores = unary[:, position] + edges  # (batch, top_k)
        chosen = []
        for index, sampler in enumerate(samplers):
            if sampler is None:
                pick = mx.argmax(scores[index], keepdims=True)
            else:
                pick, density = sample_candidates(scores[index : index + 1], sampler)
                densities[index].append(density)
            token = candidates[index, position][pick]
            chosen.append(token.reshape(1))
            tokens[index].append(token.reshape(1))
        predecessor = mx.concatenate(chosen).astype(mx.int32)
    accept = [[] for _ in range(batch)]
    for index in range(batch):
        if not densities[index]:
            continue
        rows = mx.concatenate(densities[index])  # (length, top_k)
        q = mx.full((length, vocab), -float("inf"), dtype=mx.float32)
        q = mx.put_along_axis(q, candidates[index], rows, axis=-1)
        accept[index] = [q[position] for position in range(length)]
    return [
        (mx.concatenate(tokens[index])[None].astype(mx.int32), accept[index])
        for index in range(batch)
    ]
