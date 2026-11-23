# SPDX-License-Identifier: Apache-2.0
"""MMA split-K attention for BF16/FP16 Inkling 2 single-row verify blocks."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.inkling2 import kernels
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _case(heads, kv_heads, length, keys, rel_extent, dtype, dim=128, batch=1, seed=0):
    mx.random.seed(seed)
    # Match model layout: q transposed from [B, L, H, D], k/v views into a buffer.
    q = (mx.random.normal((batch, length, heads, dim)) * 0.4).astype(dtype)
    q = q.transpose(0, 2, 1, 3)
    kbuf = (mx.random.normal((batch, kv_heads, keys + 19, dim)) * 0.4).astype(dtype)
    vbuf = (mx.random.normal((batch, kv_heads, keys + 19, dim)) * 0.4).astype(dtype)
    k, v = kbuf[:, :, 5 : 5 + keys], vbuf[:, :, 5 : 5 + keys]
    rel = (mx.random.normal((batch, length, heads, rel_extent)) * 0.5).astype(dtype)
    sink = mx.random.normal((batch, length, heads)).astype(dtype)
    return q, k, v, rel, sink


@pytest.mark.parametrize(
    "heads,kv_heads,dim,length,keys,window,rel_extent",
    [
        (24, 8, 128, 2, 1200, 1024, 1024),  # Inkling 2 Mini sliding
        (24, 8, 128, 4, 4200, 0, 1024),  # Inkling 2 Mini full
        (24, 8, 128, 8, 4200, 0, 1024),
        (48, 8, 128, 4, 1200, 1024, 1024),  # Inkling 2 Small sliding
        (48, 8, 128, 8, 4200, 0, 1024),  # Inkling 2 Small full
        (6, 2, 40, 3, 256, 64, 64),  # Non-32-multiple head dim (fold tail)
        (2, 1, 128, 2, 16, 0, 1),  # Tiny rel (<8 elems -> constant address space)
        (6, 2, 128, 8, 193, 0, 64),  # Active row in split > 0 with zero visible keys
        (6, 2, 128, 5, 10, 0, 16),  # Short context (< BK)
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
def test_mma_verify_matches_masked_attention(
    heads, kv_heads, dim, length, keys, window, rel_extent, dtype
):
    q, k, v, rel, sink = _case(heads, kv_heads, length, keys, rel_extent, dtype, dim)
    scale = dim**-0.5
    expected = kernels._masked_attention(q, k, v, rel, sink, scale, window, None)
    got = kernels.mma_verify_attention(q, k, v, rel, sink, scale=scale, window=window)
    routed = kernels.banded_sink_attention(
        q, k, v, rel, sink, scale=scale, window=window
    )
    assert mx.array_equal(routed, got)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    tol = 1.5e-2 if dtype == mx.bfloat16 else 5e-3
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max().item()
    assert diff < tol


def test_mma_verify_batch_and_left_padding():
    q, k, v, rel, sink = _case(12, 4, 4, 300, 64, mx.bfloat16, dim=128, batch=2, seed=5)
    lp = mx.array([0, 113], dtype=mx.int32)
    scale = 128**-0.5
    expected = kernels._masked_attention(q, k, v, rel, sink, scale, 0, lp)
    got = kernels.mma_verify_attention(
        q, k, v, rel, sink, scale=scale, window=0, left_padding=lp
    )
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max().item()
    assert diff < 1.5e-2


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
def test_mma_verify_handles_strided_and_unaligned_kv(dtype):
    q, _, _, rel, sink = _case(24, 8, 4, 300, 64, dtype, dim=64, seed=7)
    scale = 64**-0.5
    # Unaligned row stride and unaligned base address both take the scalar loader.
    for width, row, col in ((71, 8, 0), (72, 7, 1)):
        k = (mx.random.normal((1, 8, 344, width)) * 0.3).astype(dtype)[
            :, :, row : row + 300, col : col + 64
        ]
        v = (mx.random.normal((1, 8, 344, width)) * 0.3).astype(dtype)[
            :, :, row : row + 300, col : col + 64
        ]
        got = kernels.mma_verify_attention(q, k, v, rel, sink, scale=scale, window=0)
        packed = kernels.mma_verify_attention(
            q, mx.contiguous(k), mx.contiguous(v), rel, sink, scale=scale, window=0
        )
        assert mx.array_equal(got, packed)
    # Strided last dimension also falls back to scalar staging.
    k = (mx.random.normal((1, 8, 300, 64)) * 0.3).astype(dtype)
    v = (mx.random.normal((1, 8, 300, 64)) * 0.3).astype(dtype)
    wide_k = mx.repeat(k, 2, axis=-1)[..., ::2]
    wide_v = mx.repeat(v, 2, axis=-1)[..., ::2]
    assert mx.array_equal(
        kernels.mma_verify_attention(q, wide_k, wide_v, rel, sink, scale=scale),
        kernels.mma_verify_attention(q, k, v, rel, sink, scale=scale),
    )


def test_mma_verify_routing_and_fallbacks(monkeypatch):
    scale = 128**-0.5
    calls = []
    orig = kernels.mma_verify_attention

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return orig(*args, **kwargs)

    monkeypatch.setattr(kernels, "mma_verify_attention", spy)

    # B=1, L=4 non-row-exact takes MMA verify.
    args = _case(24, 8, 4, 256, 64, mx.bfloat16)
    kernels.banded_sink_attention(*args, scale=scale)
    assert len(calls) == 1

    # L=1 decode, B=2 batch, row-exact verify, left-padded rows, L=9, FP32 stay off MMA.
    kernels.banded_sink_attention(*_case(24, 8, 1, 256, 64, mx.bfloat16), scale=scale)
    kernels.banded_sink_attention(
        *_case(24, 8, 4, 256, 64, mx.bfloat16, batch=2), scale=scale
    )
    set_verify_qmm_armed(True, row_exact=True)
    try:
        kernels.banded_sink_attention(*args, scale=scale)
    finally:
        set_verify_qmm_armed(False)
    kernels.banded_sink_attention(
        *args, scale=scale, left_padding=mx.array([0], dtype=mx.int32)
    )
    kernels.banded_sink_attention(*_case(24, 8, 9, 256, 64, mx.bfloat16), scale=scale)
    kernels.banded_sink_attention(*_case(24, 8, 4, 256, 64, mx.float32), scale=scale)
    assert len(calls) == 1

    # Mismatched rel dtype or wide heads past 32 KB threadgroup budget fall back cleanly.
    q, k, v, rel, sink = args
    assert not kernels._use_mma_verify(q, k, v, rel.astype(mx.float32))
    wide = _case(2, 1, 4, 64, 32, mx.bfloat16, dim=512)
    assert not kernels._use_mma_verify(wide[0], wide[1], wide[2], wide[3])
    assert mx.array_equal(
        kernels.banded_sink_attention(*wide, scale=512**-0.5),
        kernels._masked_attention(*wide, 512**-0.5, 0, None),
    )

    # Threadgroup rejection falls back to masked SDPA.
    monkeypatch.setattr(kernels, "_fits_threadgroup", lambda *a, **k: False)
    assert not kernels._use_mma_verify(q, k, v, rel)
    assert len(calls) == 1

    # Direct calls reject out-of-range query counts.
    with pytest.raises(ValueError, match="1..8"):
        orig(*_case(24, 8, 9, 64, 32, mx.bfloat16), scale=scale)
