# SPDX-License-Identifier: Apache-2.0
"""Tests for the head_dim=256 long-context prefill SDPA patch.

Covers (without needing the full Qwen3.6 model):
  - the flash kernel matches mx.fast.scaled_dot_product_attention numerically
    (square causal, chunked-prefill non-square causal, and decode shapes);
  - the route gate engages only for head_dim=256 / qL>1 / causal / long kv;
  - the patched SDPA passes through unchanged for non-256 / decode / short kv;
  - the memory-monitor estimator switches head_dim=256 prefill to O(L) once
    registered, and stays O(L^2) otherwise.
"""

import math

import mlx.core as mx
import pytest

SCALE_256 = 1.0 / math.sqrt(256)


def _qkv(Lq, Lk, n_q=24, n_kv=4, D=256, dtype=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, n_q, Lq, D)).astype(dtype)
    k = mx.random.normal((1, n_kv, Lk, D)).astype(dtype)
    v = mx.random.normal((1, n_kv, Lk, D)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


# --- kernel correctness --------------------------------------------------

@pytest.mark.parametrize("L", [256, 1024, 4096])
def test_flash_sdpa256_square_causal_matches_reference(L):
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, k, v = _qkv(L, L)
    out = _flash_sdpa256(q, k, v, SCALE_256, "causal")
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


@pytest.mark.parametrize("Lq,Lk", [(1, 4096), (128, 4096), (2048, 8192)])
def test_flash_sdpa256_chunked_prefill_offset_causal(Lq, Lk):
    """Chunked prefill: Lq queries over a longer cached context (Lk). MLX
    'causal' aligns queries to the END of the key axis — the kernel must match."""
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, _, _ = _qkv(Lq, Lq)
    _, k, v = _qkv(Lk, Lk)
    out = _flash_sdpa256(q, k, v, SCALE_256, "causal")
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


def test_flash_sdpa256_memory_is_sub_quadratic():
    """Peak memory must grow ~O(L), not O(L^2). Over an 8K->32K span (4x in L)
    O(L^2) would grow ~16x; we require < 6x (O(L) is ~4x), a sharp signal."""
    if not hasattr(mx, "reset_peak_memory"):
        return  # peak-memory API unavailable on this MLX build; skip
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    peaks = []
    for L in (8192, 32768):
        q, k, v = _qkv(L, L)
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        mx.reset_peak_memory()
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        peaks.append(mx.get_peak_memory())
    assert peaks[1] < 6 * peaks[0]


# --- route gate ----------------------------------------------------------

def test_should_route_gate():
    from omlx.patches import sdpa256_attention as sdpa256

    q, k, _ = _qkv(2048, 16384)  # 256, prefill, long
    assert sdpa256._should_route(q, k, None, "causal", None) is True
    assert sdpa256._should_route(q, k, None, None, None) is True
    # decode (qL==1) -> fused vector kernel handles 256
    qd, kd, _ = _qkv(1, 16384)
    assert sdpa256._should_route(qd, kd, None, "causal", None) is False
    # short kv -> keep the faster fallback
    qs, ks, _ = _qkv(2048, 4096)
    assert sdpa256._should_route(qs, ks, None, "causal", None) is False
    # wrong head_dim
    qh, kh, _ = _qkv(2048, 16384, D=128)
    assert sdpa256._should_route(qh, kh, None, "causal", None) is False
    # array mask / sinks -> passthrough
    assert sdpa256._should_route(q, k, None, mx.zeros((2048, 16384)), None) is False
    assert sdpa256._should_route(q, k, None, "causal", mx.zeros((4,))) is False

    # quantized KV cache (has .bits) -> passthrough to the quant-aware SDPA
    class _QuantCache:
        bits = 4
    assert sdpa256._should_route(q, k, _QuantCache(), "causal", None) is False


# --- patched dispatcher passthrough vs route -----------------------------

def test_patch_routes_256_and_passes_through_others(monkeypatch):
    from mlx_lm.models import base as mlx_base

    import omlx.patches.sdpa256_attention as sdpa256

    # Force a fresh install regardless of prior test state.
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    original = mlx_base.scaled_dot_product_attention
    calls = {"orig": 0, "flash": 0}

    def counting_original(q, k, v, cache, scale, mask, sinks=None):
        calls["orig"] += 1
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    monkeypatch.setattr(mlx_base, "scaled_dot_product_attention", counting_original)

    real_flash = sdpa256._flash_sdpa256

    def counting_flash(q, k, v, scale, mask):
        calls["flash"] += 1
        return real_flash(q, k, v, scale, mask)

    monkeypatch.setattr(sdpa256, "_flash_sdpa256", counting_flash)

    assert sdpa256.apply_sdpa256_attention_patch() is True
    patched = mlx_base.scaled_dot_product_attention
    try:
        # head_dim 256, long prefill -> flash kernel, and output matches MLX.
        q, k, v = _qkv(2048, 16384)
        out = patched(q, k, v, None, SCALE_256, "causal")
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
        mx.eval(out, ref)
        assert calls["flash"] == 1
        assert _max_abs(out, ref) < 2e-2

        # decode (qL=1) -> passthrough to original.
        qd, kd, vd = _qkv(1, 16384)
        mx.eval(patched(qd, kd, vd, None, SCALE_256, "causal"))
        assert calls["orig"] >= 1

        # head_dim 128 -> passthrough.
        q2, k2, v2 = _qkv(2048, 16384, D=128)
        before = calls["orig"]
        mx.eval(patched(q2, k2, v2, None, 1.0 / math.sqrt(128), "causal"))
        assert calls["orig"] == before + 1
    finally:
        monkeypatch.setattr(mlx_base, "scaled_dot_product_attention", original)


# --- estimator lockstep --------------------------------------------------

def test_estimator_switches_to_ol_when_registered():
    from omlx import memory_monitor as mm

    monitor = mm.MemoryMonitor.__new__(mm.MemoryMonitor)
    monitor._head_dim = 256
    monitor._num_attention_heads = 24
    monitor._num_kv_heads = 4
    monitor._score_dtype_size = 2

    chunk, kv = 2048, 200_000
    # Ensure not registered first (isolate from import-time state).
    mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    quadratic = monitor._estimate_sdpa_activation_bytes(chunk, kv)

    mm.register_tiled_prefill_head_dim(256, min_kv_len=8192, kv_tile=1024)
    try:
        linear = monitor._estimate_sdpa_activation_bytes(chunk, kv)
        # O(L^2) charges the full [n_q, chunk, kv] score matrix; O(L) charges
        # only output + one kv tile -> dramatically smaller at 200K context.
        assert linear < quadratic / 10
        # And short kv still uses the fallback estimate (no regression of the
        # short-prefill accounting).
        short = monitor._estimate_sdpa_activation_bytes(2048, 4096)
        scores = 24 * 2048 * 4096 * 2
        assert short >= scores
    finally:
        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
