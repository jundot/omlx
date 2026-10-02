# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused multi-row verify attention kernel (gemma4, D=512).

Parity oracle: fp32 manual attention with end-aligned causal (row j attends
keys [0 .. N - L + j]) over the first N positions of the padded KV buffers.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import gemma4_verify_kernel as gvk

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)


def _ref_attention(q, k_buf, v_buf, n_keys, scale):
    B, hq, L, D = q.shape
    hkv = k_buf.shape[1]
    gqa = hq // hkv
    k = mx.repeat(k_buf[:, :, :n_keys, :].astype(mx.float32), gqa, axis=1)
    v = mx.repeat(v_buf[:, :, :n_keys, :].astype(mx.float32), gqa, axis=1)
    scores = (q.astype(mx.float32) * scale) @ k.transpose(0, 1, 3, 2)
    rows = mx.arange(L).reshape(L, 1)
    cols = mx.arange(n_keys).reshape(1, n_keys)
    scores = mx.where(
        cols <= (n_keys - L + rows), scores, mx.array(-1e30)
    )
    return mx.softmax(scores, axis=-1) @ v


@pytest.mark.parametrize(
    "B,hq,hkv,L,n_keys,cap,dtype",
    [
        (1, 32, 4, 2, 513, 640, mx.float16),  # 31B geometry, single group
        (1, 32, 4, 3, 300, 320, mx.bfloat16),  # odd L -> front pad
        (1, 32, 4, 4, 1024, 1088, mx.bfloat16),  # two row groups
        (1, 16, 2, 5, 200, 256, mx.bfloat16),  # 26B geometry, pad + groups
        (1, 32, 4, 9, 150, 192, mx.bfloat16),  # rows_cap chunking (8 + 1)
        (2, 16, 2, 3, 96, 128, mx.bfloat16),  # batch > 1
        (1, 32, 4, 2, 17, 64, mx.float16),  # N < _BLOCKS edge
    ],
)
def test_parity_vs_reference(B, hq, hkv, L, n_keys, cap, dtype):
    mx.random.seed(7)
    D = 512
    q = (mx.random.normal((B, hq, L, D)) * 0.3).astype(dtype)
    k_buf = (mx.random.normal((B, hkv, cap, D)) * 0.3).astype(dtype)
    v_buf = (mx.random.normal((B, hkv, cap, D)) * 0.3).astype(dtype)

    got = gvk.fused_verify_sdpa(q, k_buf, v_buf, n_keys, 1.0)
    want = _ref_attention(q, k_buf, v_buf, n_keys, 1.0)
    mx.eval(got, want)

    diff = mx.abs(got.astype(mx.float32) - want).max().item()
    denom = max(mx.abs(want).max().item(), 1e-6)
    assert diff / denom < 2e-2


def test_scale_applied():
    mx.random.seed(11)
    q = (mx.random.normal((1, 8, 2, 512)) * 0.3).astype(mx.float16)
    kv = (mx.random.normal((1, 1, 64, 512)) * 0.3).astype(mx.float16)
    got = gvk.fused_verify_sdpa(q, kv, kv, 40, 0.25)
    want = _ref_attention(q, kv, kv, 40, 0.25)
    mx.eval(got, want)
    assert mx.abs(got.astype(mx.float32) - want).max().item() < 1e-2


def test_is_available_probe():
    assert gvk.is_available() is True
    # Cached: second call must not re-probe (same object identity semantics).
    assert gvk.is_available() is True


def test_row_chunking_under_constrained_threadgroup_budget(monkeypatch):
    # Virtualized/low-end GPUs cap the pass-1 pipeline below 32 * gqa * S
    # threads (the CI runner allows 448). With the budget forced to one
    # row group, the host must cover any L with 2-row dispatches and still
    # match the reference.
    monkeypatch.setitem(gvk._tg_thread_cap, mx.bfloat16, 256)
    assert gvk.kernel_max_rows(8, mx.bfloat16) == 2

    mx.random.seed(13)
    q = (mx.random.normal((1, 32, 5, 512)) * 0.3).astype(mx.bfloat16)
    k_buf = (mx.random.normal((1, 4, 256, 512)) * 0.3).astype(mx.bfloat16)
    v_buf = (mx.random.normal((1, 4, 256, 512)) * 0.3).astype(mx.bfloat16)

    got = gvk.fused_verify_sdpa(q, k_buf, v_buf, 200, 1.0)
    want = _ref_attention(q, k_buf, v_buf, 200, 1.0)
    mx.eval(got, want)
    diff = mx.abs(got.astype(mx.float32) - want).max().item()
    denom = max(mx.abs(want).max().item(), 1e-6)
    assert diff / denom < 2e-2


def test_infeasible_geometry_reports_zero_rows(monkeypatch):
    monkeypatch.setitem(gvk._tg_thread_cap, mx.bfloat16, 0)
    assert gvk.kernel_max_rows(8, mx.bfloat16) == 0
