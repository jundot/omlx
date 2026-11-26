# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 row-exact verify attention in one launch: each row keeps the bits
of its token's one-row decode step."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.inkling2 import kernels, verify_attention
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_HQ, _HKV, _D, _R = 24, 8, 128, 1024


def _inputs(dtype, keys, rows, seed, hq=_HQ):
    mx.random.seed(seed)
    # The model hands q over as a transpose of [B, L, H, D] and the keys and
    # values as views into a longer cache buffer.
    q = mx.random.normal((1, rows, hq, _D)).astype(dtype).transpose(0, 2, 1, 3)
    kbuf = mx.random.normal((1, _HKV, keys + 21, _D)).astype(dtype)
    vbuf = mx.random.normal((1, _HKV, keys + 21, _D)).astype(dtype)
    k, v = kbuf[:, :, 3 : 3 + keys], vbuf[:, :, 3 : 3 + keys]
    rel = (mx.random.normal((1, rows, hq, _R)) * 2).astype(dtype)
    sink = mx.random.normal((1, rows, hq)).astype(dtype)
    return q, k, v, rel, sink


def _per_row(q, k, v, rel, sink, window):
    s, length = k.shape[2], q.shape[2]
    rows = [
        kernels._decode_attention(
            q[:, :, r : r + 1],
            k[:, :, : s - length + 1 + r],
            v[:, :, : s - length + 1 + r],
            rel[:, r : r + 1],
            sink[:, r : r + 1],
            _D**-0.5,
            window,
        )
        for r in range(length)
    ]
    return mx.concatenate(rows, axis=2)


# Every row sees at least 4096 keys, where MLX takes its two-pass kernel on
# every GPU.
@pytest.mark.parametrize(
    "keys, rows, window, dtype",
    [
        (4200, 2, 0, mx.bfloat16),
        (4200, 5, 0, mx.float16),
        (4200, 16, 0, mx.bfloat16),
        (20000, 5, 0, mx.bfloat16),
        (20000, 16, 0, mx.float16),
        (4200, 5, 4096, mx.bfloat16),
        (4200, 16, 4096, mx.bfloat16),
        (20000, 2, 4096, mx.float16),
    ],
)
def test_rows_match_decode_steps_bit_for_bit(dtype, window, rows, keys):
    q, k, v, rel, sink = _inputs(dtype, keys, rows, seed=keys + rows)
    out = verify_attention.rows_attention(
        q, k, v, rel, sink, scale=_D**-0.5, window=window
    )
    assert out is not None
    assert mx.array_equal(out, _per_row(q, k, v, rel, sink, window))


def test_rows_at_gqa_6_fit_one_threadgroup():
    q, k, v, rel, sink = _inputs(mx.bfloat16, 16399, 16, seed=16399, hq=48)
    out = verify_attention.rows_attention(q, k, v, rel, sink, scale=_D**-0.5, window=0)
    assert mx.array_equal(out, _per_row(q, k, v, rel, sink, 0))


def test_banded_sink_attention_takes_the_fused_rows(monkeypatch):
    fused = verify_attention.rows_attention
    taken = []

    def spy(*args, **kwargs):
        out = fused(*args, **kwargs)
        taken.append(out is not None)
        return out

    monkeypatch.setattr(verify_attention, "rows_attention", spy)
    q, k, v, rel, sink = _inputs(mx.bfloat16, 4100, 4, seed=3)
    set_verify_qmm_armed(True, row_exact=True)
    try:
        out = kernels.banded_sink_attention(
            q, k, v, rel, sink, scale=_D**-0.5, window=0
        )
    finally:
        set_verify_qmm_armed(False)
    assert taken == [True]
    assert mx.array_equal(out, _per_row(q, k, v, rel, sink, 0))


def test_short_caches_keep_the_per_row_loop():
    # Below MLX's two-pass threshold a decode row runs its one-pass kernel,
    # whose arithmetic this module does not reproduce.
    q, k, v, rel, sink = _inputs(mx.float16, 300, 4, seed=5)
    assert (
        verify_attention.rows_attention(q, k, v, rel, sink, scale=_D**-0.5, window=0)
        is None
    )
