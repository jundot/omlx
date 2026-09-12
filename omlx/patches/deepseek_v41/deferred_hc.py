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
    _hc_sinkhorn_only,
    _hc_split_sinkhorn_ops,
    hc_expand,
)


def _make_hc_collapse_kernel():
    """Vectorized HC collapse for HC=4: [B,L,4,D] x [B,L,4] -> [B,L,D]."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        // Avoid address-space casts: mlx may place small `pre` in constant space.
        const float p0 = pre[row * HC + 0];
        const float p1 = pre[row * HC + 1];
        const float p2 = pre[row * HC + 2];
        const float p3 = pre[row * HC + 3];
        uint x_base = row * HC * D;
        uint out_base = row * D;
        constexpr uint D4 = (uint)D / 4;
        for (uint d4 = tid; d4 < D4; d4 += 256) {
            uint d = d4 * 4;
            float4 xv0 = float4(
                float(x_in[x_base + 0 * D + d + 0]), float(x_in[x_base + 0 * D + d + 1]),
                float(x_in[x_base + 0 * D + d + 2]), float(x_in[x_base + 0 * D + d + 3]));
            float4 xv1 = float4(
                float(x_in[x_base + 1 * D + d + 0]), float(x_in[x_base + 1 * D + d + 1]),
                float(x_in[x_base + 1 * D + d + 2]), float(x_in[x_base + 1 * D + d + 3]));
            float4 xv2 = float4(
                float(x_in[x_base + 2 * D + d + 0]), float(x_in[x_base + 2 * D + d + 1]),
                float(x_in[x_base + 2 * D + d + 2]), float(x_in[x_base + 2 * D + d + 3]));
            float4 xv3 = float4(
                float(x_in[x_base + 3 * D + d + 0]), float(x_in[x_base + 3 * D + d + 1]),
                float(x_in[x_base + 3 * D + d + 2]), float(x_in[x_base + 3 * D + d + 3]));
            float4 acc = fma(float4(p0), xv0,
                         fma(float4(p1), xv1,
                         fma(float4(p2), xv2, float4(p3) * xv3)));
            out[out_base + d + 0] = T(acc.x);
            out[out_base + d + 1] = T(acc.y);
            out[out_base + d + 2] = T(acc.z);
            out[out_base + d + 3] = T(acc.w);
        }
    """
    return mx.fast.metal_kernel(
        name="hc_collapse_only",
        input_names=["x_in", "pre"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_collapse_kernel = _make_hc_collapse_kernel()



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
    if (
        _hc_collapse_kernel is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and x.ndim == 4
        and x.shape[2] == 4
        and x.shape[3] % 4 == 0
        and pre_mix.shape[-1] == 4
    ):
        B, L, HC, D = x.shape
        flat_x = x.reshape(B * L, HC, D)
        flat_pre = pre_mix.astype(mx.float32).reshape(B * L, HC)
        (out,) = _hc_collapse_kernel(
            inputs=[flat_x, flat_pre],
            template=[("T", x.dtype), ("HC", HC), ("D", D)],
            grid=(B * L * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B * L, D)],
            output_dtypes=[x.dtype],
        )
        return out.reshape(B, L, D)
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
        # Training / CPU: exact MLX sinkhorn. Decode GPU: Metal sinkhorn-only
        # (collapse stays separate — V4.1 uses a deferred pre_mix).
        if self.training:
            return _hc_split_sinkhorn_ops(
                mixes,
                self.scale,
                self.base,
                self.hc_mult,
                self.sinkhorn_iters,
                self.hc_eps,
            )
        return _hc_sinkhorn_only(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
            batch_shape=tuple(x.shape[:2]),
        )


__all__ = [
    "DeferredHyperConnection",
    "hc_collapse",
    "hc_expand",
    "make_identity_pre_mix",
]
