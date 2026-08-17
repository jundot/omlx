# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused Qwen3.5/3.6 GDN verify prework kernel.

The fused kernel must be BIT-exact to the composed chain (conv-state concat
+ depthwise conv1d + SiLU + split + ones-weight RMS norms + scalar scales +
next conv-state slice) at every verify width it claims (S in 3..9).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.qwen35_gdn_prework import gdn_prework_fused

HK, HV, DK, DV = 16, 48, 128, 128
C = 2 * HK * DK + HV * DV
KEY_DIM = HK * DK


def _composed(qkv, conv_state, conv1d):
    B, S, _ = qkv.shape
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -3:, :])
    co = nn.silu(conv1d(conv_input))
    q, k, v = mx.split(co, [KEY_DIM, 2 * KEY_DIM], -1)
    q = q.reshape(B, S, HK, DK)
    k = k.reshape(B, S, HK, DK)
    v = v.reshape(B, S, HV, DV)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    return q, k, v, new_state


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("seq", [3, 4, 5, 7, 9])
def test_fused_prework_bit_exact(seq):
    mx.random.seed(11)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((1, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    inv = DK**-0.5
    q_scale = mx.array(inv * inv, dtype=mx.bfloat16)
    k_scale = mx.array(inv, dtype=mx.bfloat16)

    ref = _composed(qkv, state, conv1d)
    got = gdn_prework_fused(qkv, state, conv_w, q_scale, k_scale, HK, HV, DK, DV)
    for name, r, g in zip(("q", "k", "v", "conv_state"), ref, got):
        assert r.shape == g.shape, name
        assert bool((r == g).all().item()), f"{name} not bit-exact at S={seq}"
