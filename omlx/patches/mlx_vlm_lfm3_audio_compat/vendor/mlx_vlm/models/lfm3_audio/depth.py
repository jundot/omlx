# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Audio checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0
"""Depth decoder: one speech frame (12 codebooks) from a backbone hidden state.

The codebooks of a frame are revealed by MaskGIT in a random order over a few passes;
each pass is a bidirectional LFM3 trunk over 12 slots, one per codebook.
"""

import mlx.core as mx
import mlx.nn as nn

from ...sample_utils import apply_min_p, apply_top_k, apply_top_p
from ..lfm3.language import DecoderLayer


class DepthModel(nn.Module):
    """The LFM3 trunk without embed_tokens, attending bidirectionally (mask=None)."""

    def __init__(self, config):
        super().__init__()
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.embedding_norm(x)


class HeadRMSNorm(nn.Module):
    """RMSNorm that scales by the weight in fp32 and rounds once, as the trained heads do."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        y = mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32), self.eps
        )
        return y.astype(x.dtype)


class CodebookEmbedding(nn.Module):
    """Input embedding and tied output head of one codebook."""

    def __init__(self, codebook_size, dims, eps):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, dims)
        self.embedding_norm = HeadRMSNorm(dims, eps)

    def logits(self, x):
        return self.embedding.as_linear(self.embedding_norm(x))


class DepthDecoder(nn.Module):
    def __init__(self, config, input_size, num_codebooks, codebook_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.hidden_size = config.hidden_size
        self.depth_linear = nn.Linear(input_size, num_codebooks * config.hidden_size)
        self.mask_embedding = mx.zeros((config.hidden_size,))
        self.model = DepthModel(config)
        self.depth_embeddings = [
            CodebookEmbedding(codebook_size, config.hidden_size, config.norm_eps)
            for _ in range(num_codebooks)
        ]

    def __call__(self, hidden, codes, masked):
        """hidden [B, input_size], codes [B, K], masked [B, K] (True = hidden) -> logits [B, K, V]."""
        x = self.depth_linear(hidden).reshape(
            *hidden.shape[:-1], self.num_codebooks, self.hidden_size
        )
        embeds = mx.stack(
            [e.embedding(codes[..., i]) for i, e in enumerate(self.depth_embeddings)],
            axis=-2,
        )
        mask = self.mask_embedding.astype(embeds.dtype)
        x = self.model(x + mx.where(masked[..., None], mask, embeds))
        return mx.stack(
            [e.logits(x[..., i, :]) for i, e in enumerate(self.depth_embeddings)],
            axis=-2,
        )

    @staticmethod
    def sanitize(weights):
        # to_logits is a stored copy of the tied embedding.
        return {k: v for k, v in weights.items() if ".to_logits." not in k}


def sample(
    logits,
    temperature=None,
    top_k=None,
    repetition_penalty=None,
    history=None,
    suppress=(),
    top_p=None,
    min_p=None,
):
    """Sample ids from logits [..., V] like the reference ``_sample``.

    ``history`` [..., n] holds the ids to penalize per row; the penalty applies
    even when decoding greedily. Greedy when ``temperature`` is 0, or None
    without ``top_k``.
    """
    scores = logits.astype(mx.float32)
    vocab = scores.shape[-1]
    if repetition_penalty not in (None, 1.0) and history is not None:
        ids = mx.broadcast_to(history, scores.shape[:-1] + history.shape[-1:])
        picked = mx.take_along_axis(scores, ids, axis=-1)
        picked = mx.where(
            picked < 0, picked * repetition_penalty, picked / repetition_penalty
        )
        scores = mx.put_along_axis(scores, ids, picked, axis=-1)
    if suppress:
        scores[..., mx.array(list(suppress))] = -mx.inf
    if temperature == 0 or (temperature is None and not top_k):
        return mx.argmax(scores, axis=-1)
    if temperature:
        scores = scores / temperature
    scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    if top_k and top_k < vocab:
        scores = apply_top_k(scores, int(top_k))
    if top_p is not None and top_p < 1.0:
        scores = apply_top_p(scores, top_p)
    if min_p:
        scores = apply_min_p(scores, min_p)
    return mx.random.categorical(scores)


def sample_frame(
    depth,
    hidden,
    schedule=(4, 4, 4),
    temperature=None,
    top_k=None,
    repetition_penalty=None,
    history=None,
    order=None,
):
    """MaskGIT one frame per row: hidden [B, D] -> raw codes [B, K] (before EOS handling).

    Pass p reveals the codebooks ``order[:, s:s + schedule[p]]``; ``order`` [B, K] is
    random per row unless given. ``history`` [..., n_frames, K] holds earlier frames
    for the per-codebook repetition penalty.
    """
    batch, k = hidden.shape[0], depth.num_codebooks
    if order is None:
        order = mx.stack([mx.random.permutation(k) for _ in range(batch)])
    if history is not None:
        history = history.swapaxes(-1, -2)
    codes = mx.zeros((batch, k), mx.int32)
    revealed = mx.zeros((batch, k), mx.bool_)
    start = 0
    for size in schedule:
        pick = mx.put_along_axis(
            mx.zeros((batch, k), mx.bool_),
            order[:, start : start + size],
            mx.array(True),
            axis=-1,
        )
        logits = depth(hidden, codes, ~revealed)
        sampled = sample(logits, temperature, top_k, repetition_penalty, history)
        codes = mx.where(pick, sampled.astype(mx.int32), codes)
        revealed = revealed | pick
        start += size
    return codes


def normalize_eos(codes, eos_code):
    """A frame with any EOS code becomes all EOS."""
    return mx.where((codes == eos_code).any(-1, keepdims=True), eos_code, codes)


def embed_frames(table, codes):
    """Sum of the codebook embeddings of each frame: codes [..., K] -> [..., D]."""
    k = codes.shape[-1]
    offsets = mx.arange(k) * (table.weight.shape[0] // k)
    return table(codes + offsets).sum(-2)
