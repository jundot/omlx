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
    # Per row (row = one query position of one head), so a row that is wrong
    # cannot hide behind the largest value of the whole tensor.
    row_diff = mx.abs(got.astype(mx.float32) - want).max(axis=-1)
    row_peak = mx.maximum(mx.abs(want).max(axis=-1), 1e-6)
    assert (row_diff / row_peak).max().item() < 2e-2


def test_rejects_more_rows_than_one_mma_group():
    q = mx.zeros((1, 32, 9, 128), dtype=mx.bfloat16)  # 4 x 9 rows > 32
    kv = mx.zeros((1, 8, 64, 128), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        vmk.mma_verify_sdpa(q, kv, kv, 32, 1.0)


def test_each_row_sees_its_own_key_and_no_later_key():
    # Row j is aligned with one dimension; only key N-T+j (its own) lights it up,
    # and the later keys of the block light it up too but are masked for row j.
    # The output of row j must be the value of its own key: a kernel that drops
    # the own key or lets a later key in is off by far more than any tolerance.
    D, T, N, cap = 128, 4, 4096, 4160
    k = mx.zeros((1, 1, cap, D))
    v = (mx.random.normal((1, 1, cap, D)) * 0.01)
    q = mx.zeros((1, 1, T, D))
    for j in range(T):
        q[0, 0, j, j] = 400.0
        for key in range(N - T + j, N):
            k[0, 0, key, j] = 1.0
        v[0, 0, N - T + j] = mx.random.normal((D,))  # the needle value of row j
        for key in range(N - T + j + 1, N):
            v[0, 0, key] = 50.0  # a later key: must not leak into row j
    q, k, v = (a.astype(mx.bfloat16) for a in (q, k, v))
    got = vmk.mma_verify_sdpa(q, k, v, N, D**-0.5)
    mx.eval(got)
    for j in range(T):
        want = v[0, 0, N - T + j].astype(mx.float32)
        err = mx.abs(got[0, 0, j].astype(mx.float32) - want).max().item()
        assert err < 5e-2 * max(mx.abs(want).max().item(), 1.0), (j, err)


def test_scale_is_applied_in_fp32_not_in_the_query_dtype():
    # fp16 q = 300 times scale 300 is 9e4, beyond fp16: scaling the queries in
    # fp16 before the kernel turns them to inf. Stock SDPA scales in fp32.
    D, T, N, cap = 128, 3, 256, 256
    mx.random.seed(3)
    q = (mx.random.normal((1, 8, T, D)) * 300).astype(mx.float16)
    k = (mx.random.normal((1, 8, cap, D)) * 1e-3).astype(mx.float16)
    v = (mx.random.normal((1, 8, cap, D)) * 0.5).astype(mx.float16)
    got = vmk.mma_verify_sdpa(q, k, v, N, 300.0)
    want = _ref_attention(q, k, v, N, 300.0)
    mx.eval(got, want)
    assert mx.isfinite(got).all().item()
    row_diff = mx.abs(got.astype(mx.float32) - want).max(axis=-1)
    assert (row_diff / mx.maximum(mx.abs(want).max(axis=-1), 1e-6)).max().item() < 3e-3


def test_tiny_shape_whose_partials_metal_binds_as_constants():
    # MLX passes arrays of a few elements in the constant address space; the
    # second pass must compile for them too (one head, 3 rows, 16 keys).
    D, T, N = 128, 3, 16
    mx.random.seed(5)
    q, k, v = (mx.random.normal(sh_).astype(mx.bfloat16) for sh_ in ((1, 1, T, D), (1, 1, N, D), (1, 1, N, D)))
    got = vmk.mma_verify_sdpa(q, k, v, N, D**-0.5)
    want = _ref_attention(q, k, v, N, D**-0.5)
    mx.eval(got, want)
    assert (mx.abs(got.astype(mx.float32) - want).max(axis=-1) / mx.abs(want).max(axis=-1)).max().item() < 2e-2
