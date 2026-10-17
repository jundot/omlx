# SPDX-License-Identifier: Apache-2.0
"""M1 route of the FA-256 patch: query-chunked MLX SDPA."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import qwen35_fa256_attention as patch

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _reference(q, k, v, scale):
    """Dense end-aligned causal attention in fp32 (GQA by head repeat)."""
    q32, k32, v32 = (t.astype(mx.float32) for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    k32 = mx.repeat(k32, rep, axis=1)
    v32 = mx.repeat(v32, rep, axis=1)
    scores = (q32 * scale) @ k32.transpose(0, 1, 3, 2)
    q_len, kv_len = q.shape[2], k.shape[2]
    rows = mx.arange(q_len)[:, None] + (kv_len - q_len)
    cols = mx.arange(kv_len)[None, :]
    scores = mx.where(cols <= rows, scores, -mx.inf)
    return mx.softmax(scores, axis=-1) @ v32


@pytest.mark.parametrize(
    "q_len,kv_len,rows",
    [(256, 1024, 64), (200, 777, 64), (128, 128, 32), (96, 2048, 96)],
)
@pytest.mark.parametrize("heads", [(8, 2), (6, 1)])
def test_chunked_sdpa_matches_dense_causal(q_len, kv_len, rows, heads):
    hq, hkv = heads
    mx.random.seed(0)
    q = mx.random.normal((1, hq, q_len, 256)).astype(mx.float16)
    k = mx.random.normal((1, hkv, kv_len, 256)).astype(mx.float16)
    v = mx.random.normal((1, hkv, kv_len, 256)).astype(mx.float16)
    scale = 256**-0.5
    out = patch._m1_chunked_sdpa(q, k, v, scale, rows)
    ref = _reference(q, k, v, scale)
    assert out.shape == q.shape
    assert mx.max(mx.abs(out.astype(mx.float32) - ref)).item() < 2e-2


def test_query_rows_respect_work_budget_and_floor():
    # 2048-row chunk, 24 q heads: shrinks with depth, never below 64 rows
    assert patch._m1_query_rows(24, 2048, 16384) == 256
    assert patch._m1_query_rows(24, 2048, 32768) == 128
    assert patch._m1_query_rows(24, 2048, 65536) == 64
    assert patch._m1_query_rows(24, 2048, 98304) == 64
    # ~128k keys: even 64 rows exceed the max dispatch work -> steel kernel
    assert patch._m1_query_rows(24, 2048, 131072) == 0
    # 16-head MoE layout keeps the M1 route a little deeper
    assert patch._m1_query_rows(16, 2048, 131072) == 64
    # short calls run in one dispatch
    assert patch._m1_query_rows(24, 512, 4096) == 512


def test_route_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OMLX_FA256_M1_SDPA", "0")
    assert not patch._m1_sdpa_enabled()


def test_d512_route_shape_gates(monkeypatch):
    monkeypatch.setattr(patch, "_m1_sdpa_enabled_cached", lambda: True)
    q = mx.zeros((1, 16, 64, 512), dtype=mx.float16)
    short = mx.zeros((1, 2, 1024, 512), dtype=mx.float16)
    # below the key threshold, wrong head_dim, array mask, sinks: not routed
    assert patch._m1_d512_attention(q, short, short, None, 0.04, "causal", None) is None
    q256 = mx.zeros((1, 16, 64, 256), dtype=mx.float16)
    k256 = mx.zeros((1, 2, 40000, 256), dtype=mx.float16)
    assert (
        patch._m1_d512_attention(q256, k256, k256, None, 0.04, "causal", None) is None
    )
    monkeypatch.setattr(patch, "_M1_D512_MIN_KV", 256)
    k = mx.zeros((1, 2, 300, 512), dtype=mx.float16)
    assert (
        patch._m1_d512_attention(q, k, k, None, 0.04, mx.zeros((64, 300)), None) is None
    )
    assert (
        patch._m1_d512_attention(q, k, k, None, 0.04, "causal", mx.zeros((16,))) is None
    )
    monkeypatch.setattr(patch, "_m1_sdpa_enabled_cached", lambda: False)
    assert patch._m1_d512_attention(q, k, k, None, 0.04, "causal", None) is None


def test_d512_route_matches_dense_causal(monkeypatch):
    monkeypatch.setattr(patch, "_m1_sdpa_enabled_cached", lambda: True)
    monkeypatch.setattr(patch, "_M1_D512_MIN_KV", 256)
    monkeypatch.setattr(patch, "_M1_SDPA_WORK", 16 * 32 * 300)  # force several blocks
    mx.random.seed(3)
    q = mx.random.normal((1, 16, 96, 512)).astype(mx.float16)
    k = mx.random.normal((1, 2, 300, 512)).astype(mx.float16)
    out = patch._m1_d512_attention(q, k, k, None, 512**-0.5, "causal", None)
    assert out is not None and out.shape == q.shape
    ref = _reference(q, k, k, 512**-0.5)
    assert mx.max(mx.abs(out.astype(mx.float32) - ref)).item() < 2e-2
