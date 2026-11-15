# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the MMA multi-row verify attention kernel.

Oracle: fp32 attention with end-aligned causal (row j attends keys
[0 .. N - L + j]) over the first N positions of the padded KV buffers.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import verify_mma_kernel as vmk

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)


def _ref_attention(q, k_buf, v_buf, n_keys, scale):
    B, hq, L, D = q.shape
    gqa = hq // k_buf.shape[1]
    k = mx.repeat(k_buf[:, :, :n_keys, :].astype(mx.float32), gqa, axis=1)
    v = mx.repeat(v_buf[:, :, :n_keys, :].astype(mx.float32), gqa, axis=1)
    scores = (q.astype(mx.float32) * scale) @ k.transpose(0, 1, 3, 2)
    rows = mx.arange(L).reshape(L, 1)
    cols = mx.arange(n_keys).reshape(1, n_keys)
    scores = mx.where(cols <= (n_keys - L + rows), scores, mx.array(-1e30))
    return mx.softmax(scores, axis=-1) @ v


@pytest.mark.parametrize(
    "B,hq,hkv,L,n_keys,cap,dtype",
    [
        (1, 32, 8, 2, 513, 640, mx.bfloat16),  # Qwen3-4B geometry, 2 rows
        (1, 32, 8, 4, 4096, 4160, mx.bfloat16),  # 16 rows = two row tiles
        (1, 32, 8, 8, 1000, 1024, mx.bfloat16),  # 32 rows = four row tiles
        (1, 32, 8, 3, 300, 320, mx.float16),  # 12 rows: padded row tile
        (1, 32, 8, 5, 21, 64, mx.bfloat16),  # fewer keys than tiles x blocks
        (1, 16, 2, 4, 777, 800, mx.bfloat16),  # gqa 8, key tail not x8
        (2, 32, 8, 2, 129, 192, mx.bfloat16),  # batch > 1
        (1, 8, 8, 4, 100, 128, mx.bfloat16),  # gqa 1 (no sharing)
    ],
)
def test_parity_vs_reference(B, hq, hkv, L, n_keys, cap, dtype):
    mx.random.seed(7)
    D = 128
    q = (mx.random.normal((B, hq, L, D)) * 0.5).astype(dtype)
    k_buf = (mx.random.normal((B, hkv, cap, D)) * 0.5).astype(dtype)
    v_buf = (mx.random.normal((B, hkv, cap, D)) * 0.5).astype(dtype)
    scale = D**-0.5

    got = vmk.mma_verify_sdpa(q, k_buf, v_buf, n_keys, scale)
    want = _ref_attention(q, k_buf, v_buf, n_keys, scale)
    mx.eval(got, want)

    assert got.shape == q.shape and got.dtype == dtype
    diff = mx.abs(got.astype(mx.float32) - want).max().item()
    assert diff / max(mx.abs(want).max().item(), 1e-6) < 2e-2


def test_rejects_more_rows_than_one_mma_group():
    q = mx.zeros((1, 32, 9, 128), dtype=mx.bfloat16)  # 4 x 9 rows > 32
    kv = mx.zeros((1, 8, 64, 128), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        vmk.mma_verify_sdpa(q, kv, kv, 32, 1.0)
