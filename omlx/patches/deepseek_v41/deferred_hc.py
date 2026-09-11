# SPDX-License-Identifier: Apache-2.0
"""Deferred Hyper-Connection mixes for DeepSeek-V4.1.

V4 applies ``HyperConnection.__call__`` which collapses with the *same*
stream's freshly computed ``pre``. V4.1 instead:

- Attention collapses with the **previous** sublayer's ``pre_mix`` (FFN of
  prior block, or identity at layer 0).
- FFN collapses with **this** attention's ``pre``.
- Each sublayer still computes post/comb from the current residual and
  returns its ``pre`` for the next consumer.

See official ``Block.forward`` in inference/model.py.
"""
from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.deepseek_v4.decode_consistency import matmul as decode_matmul
from omlx.patches.deepseek_v4.hyper_connection import (
    _hc_split_sinkhorn_ops,
    hc_expand,
)


def make_identity_pre_mix(x: mx.array, hc_mult: int) -> mx.array:
    """One-hot pre_mix selecting stream 0. Shape [B, L, hc_mult]."""
    B, L = x.shape[0], x.shape[1]
    pre = mx.zeros((B, L, hc_mult), dtype=mx.float32)
    # set index 0 to 1
    ones = mx.ones((B, L, 1), dtype=mx.float32)
    zeros = mx.zeros((B, L, hc_mult - 1), dtype=mx.float32) if hc_mult > 1 else None
    if zeros is None:
        return ones
    return mx.concatenate([ones, zeros], axis=-1)


def hc_collapse(x: mx.array, pre_mix: mx.array) -> mx.array:
    """[B,L,hc,D] x [B,L,hc] -> [B,L,D]."""
    y = (pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32)).sum(axis=2)
    return y.astype(x.dtype)


class DeferredHyperConnection(nn.Module):
    """Stores HC parameters; exposes mix computation without collapsing."""

    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def mixes(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Return (pre, post, comb) from residual stream ``x`` [B,L,hc,D]."""
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = decode_matmul(z, self.fn.T)
        return _hc_split_sinkhorn_ops(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )


__all__ = [
    "DeferredHyperConnection",
    "hc_collapse",
    "hc_expand",
    "make_identity_pre_mix",
]
