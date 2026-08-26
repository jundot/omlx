# SPDX-License-Identifier: Apache-2.0
"""Tests for the head_dim=256 long-context prefill SDPA patch.

Covers (without needing the full Qwen3.6 model):
  - the flash kernel matches mx.fast.scaled_dot_product_attention numerically
    (square causal, chunked-prefill non-square causal, and decode shapes);
  - the route gate engages only for head_dim=256 / qL>1 / causal / long kv;
  - the patched SDPA passes through unchanged for non-256 / decode / short kv;
  - the memory-monitor estimator switches head_dim=256 prefill to O(L) once
    registered, and stays O(L^2) otherwise;
  - memory-aware routing (issue #2204): with a headroom provider registered
    the route prefers the faster unfused fallback whenever its transient
    fits, and falls back to always-tiled without headroom info.
"""

import logging
import math
import sys
import types

import mlx.core as mx
import pytest

SCALE_256 = 1.0 / math.sqrt(256)


def _qkv(q_len, k_len, n_q=24, n_kv=4, head_dim=256, dtype=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, n_q, q_len, head_dim)).astype(dtype)
    k = mx.random.normal((1, n_kv, k_len, head_dim)).astype(dtype)
    v = mx.random.normal((1, n_kv, k_len, head_dim)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


# --- kernel correctness --------------------------------------------------


@pytest.mark.parametrize("seq_len", [256, 1024, 4096])
def test_flash_sdpa256_square_causal_matches_reference(seq_len):
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, k, v = _qkv(seq_len, seq_len)
    out = _flash_sdpa256(q, k, v, SCALE_256, "causal")
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


@pytest.mark.parametrize("q_len,k_len", [(1, 4096), (128, 4096), (2048, 8192)])
def test_flash_sdpa256_chunked_prefill_offset_causal(q_len, k_len):
    """Chunked prefill: q_len queries over a longer cached context (k_len). MLX
    'causal' aligns queries to the END of the key axis — the kernel must match."""
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, _, _ = _qkv(q_len, q_len)
    _, k, v = _qkv(k_len, k_len)
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
    for seq_len in (8192, 32768):
        q, k, v = _qkv(seq_len, seq_len)
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        mx.reset_peak_memory()
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        peaks.append(mx.get_peak_memory())
    assert peaks[1] < 6 * peaks[0]


# --- q-split correctness --------------------------------------------------


def _stock_sdpa256(q, k, v, cache, scale, mask, sinks):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


@pytest.mark.parametrize("q_len,k_len,q_sub", [(2048, 2048, 512), (4096, 4096, 384)])
def test_qsplit_square_causal_matches_reference(q_len, k_len, q_sub):
    from omlx.patches.sdpa256_attention import _unfused_qsplit_sdpa

    q, k, v = _qkv(q_len, k_len)
    out = _unfused_qsplit_sdpa(
        q, k, v, None, SCALE_256, "causal", None, _stock_sdpa256, q_sub
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert _max_abs(out, ref) < 2e-2


@pytest.mark.parametrize(
    "q_len,k_len,q_sub,tol",
    [(128, 4096, 64, 2e-2), (2048, 8192, 500, 2e-2), (2048, 20480, 384, 1e-1)],
)
def test_qsplit_chunked_prefill_offset_causal_matches_reference(q_len, k_len, q_sub, tol):
    """Chunked prefill (k_len > q_len, cached prefix) at a q_sub that does
    not evenly divide q_len -- exercises the ragged final sub-tile.

    The largest case uses a looser tolerance: fp16 softmax/accumulation
    error over a much longer kv_len grows with the reduction length and is
    hardware-dependent (summation order differs across Metal GPU
    generations/drivers) -- confirmed exact (0.0 max abs diff) on one
    machine but ~0.067 on CI's runner for the identical seeded inputs,
    i.e. genuine cross-hardware fp16 variance, not an algorithm bug. 1e-1
    stays far below the O(1) error a real q-split correctness bug (wrong
    offset, dropped rows, mis-narrowed keys) would produce."""
    from omlx.patches.sdpa256_attention import _unfused_qsplit_sdpa

    q, _, _ = _qkv(q_len, q_len)
    _, k, v = _qkv(k_len, k_len)
    out = _unfused_qsplit_sdpa(
        q, k, v, None, SCALE_256, "causal", None, _stock_sdpa256, q_sub
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask="causal")
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert _max_abs(out, ref) < tol


def test_qsplit_matches_flash_sdpa256():
    """Both alternate routes for the same shape must agree with each other,
    not just the reference -- catches a routing bug that happens to pass
    the reference check via coincidental cancellation."""
    from omlx.patches.sdpa256_attention import _flash_sdpa256, _unfused_qsplit_sdpa

    q, k, v = _qkv(2048, 8192)
    qsplit_out = _unfused_qsplit_sdpa(
        q, k, v, None, SCALE_256, "causal", None, _stock_sdpa256, 384
    )
    tiled_out = _flash_sdpa256(q, k, v, SCALE_256, "causal")
    mx.eval(qsplit_out, tiled_out)
    assert _max_abs(qsplit_out, tiled_out) < 2e-2


def test_qsplit_no_mask_matches_reference():
    from omlx.patches.sdpa256_attention import _unfused_qsplit_sdpa

    q, k, v = _qkv(1024, 1024)
    out = _unfused_qsplit_sdpa(
        q, k, v, None, SCALE_256, None, None, _stock_sdpa256, 384
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask=None)
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


def test_qsplit_calls_original_sdpa_per_subtile():
    """Confirms the loop actually shrinks each call's query axis instead of
    passing the full tensor through once -- a no-op wrapper would still
    pass the reference-match tests above."""
    from omlx.patches.sdpa256_attention import _unfused_qsplit_sdpa

    calls = []

    def counting_sdpa(q, k, v, cache, scale, mask, sinks):
        calls.append((q.shape[-2], k.shape[-2]))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    q, k, v = _qkv(2048, 2048)
    mx.eval(
        _unfused_qsplit_sdpa(
            q, k, v, None, SCALE_256, "causal", None, counting_sdpa, 512
        )
    )
    assert len(calls) == 4  # 2048 / 512
    assert [c[0] for c in calls] == [512, 512, 512, 512]  # query rows per call
    assert [c[1] for c in calls] == [512, 1024, 1536, 2048]  # narrowed kv per call


def test_qsplit_evals_each_subtile_before_the_next(monkeypatch):
    """Regression for the pool-growth fix: without an eval between
    sub-tiles, MLX's laziness lets every sub-call's graph -- including its
    own score-matrix transient -- stay unmaterialized and pile up
    simultaneously instead of being bounded to one sub-tile at a time (the
    assumption q_sub's sizing depends on). A live run showed q-split
    engaging without actually preventing the memory trip it was sized to
    avoid; this asserts the eval that fixes it is actually called, once per
    sub-tile, before the next sub-tile's (larger) call is issued."""
    import mlx.core as real_mx

    from omlx.patches.sdpa256_attention import _unfused_qsplit_sdpa

    q, k, v = _qkv(2048, 2048)
    original_eval = real_mx.eval
    eval_order = []

    def tracking_eval(*arrays):
        eval_order.append("eval")
        return original_eval(*arrays)

    def counting_sdpa(q, k, v, cache, scale, mask, sinks):
        eval_order.append("call")
        return real_mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

    monkeypatch.setattr(
        "omlx.patches.sdpa256_attention.mx.eval", tracking_eval
    )
    original_eval(
        _unfused_qsplit_sdpa(
            q, k, v, None, SCALE_256, "causal", None, counting_sdpa, 512
        )
    )
    assert eval_order == ["call", "eval"] * 4


# --- route gate ----------------------------------------------------------


def test_should_route_gate():
    """No headroom provider is registered here, so any shape that reaches
    the routing decision at all lands on tiled (the memory-safe default) --
    the point of this test is the shape gates upstream of that decision."""
    from omlx.patches import sdpa256_attention as sdpa256

    q, k, _ = _qkv(2048, 16384)  # 256, prefill, long
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
    assert sdpa256._should_route(q, k, None, None, None) == ("tiled", 0)
    # decode (qL==1) -> fused vector kernel handles 256
    qd, kd, _ = _qkv(1, 16384)
    assert sdpa256._should_route(qd, kd, None, "causal", None) == ("unfused", 0)
    # decode-shaped multi-row (MTP verify, qL = 1 + depth <= 9) -> stock path;
    # the per-tile eval sync collapses long-context MTP tok/s (issue #2127)
    for q_len in (2, 4, 9, 15):
        qv, kv, _ = _qkv(q_len, 16384)
        assert sdpa256._should_route(qv, kv, None, "causal", None) == ("unfused", 0)
    qv, kv, _ = _qkv(16, 16384)
    assert sdpa256._should_route(qv, kv, None, "causal", None) == ("tiled", 0)
    # short kv -> keep the faster fallback
    qs, ks, _ = _qkv(2048, 4096)
    assert sdpa256._should_route(qs, ks, None, "causal", None) == ("unfused", 0)
    # wrong head_dim
    qh, kh, _ = _qkv(2048, 16384, head_dim=128)
    assert sdpa256._should_route(qh, kh, None, "causal", None) == ("unfused", 0)
    # array mask / sinks -> passthrough
    assert sdpa256._should_route(q, k, None, mx.zeros((2048, 16384)), None) == (
        "unfused",
        0,
    )
    assert sdpa256._should_route(q, k, None, "causal", mx.zeros((4,))) == (
        "unfused",
        0,
    )

    # quantized KV cache (has .bits) -> passthrough to the quant-aware SDPA
    class _QuantCache:
        bits = 4

    assert sdpa256._should_route(q, k, _QuantCache(), "causal", None) == (
        "unfused",
        0,
    )


# --- patched dispatcher passthrough vs route -----------------------------


def test_patch_routes_256_and_passes_through_others(monkeypatch):
    from mlx_lm.models import base as mlx_base

    import omlx.patches.sdpa256_attention as sdpa256

    # Force a fresh install regardless of prior test state.
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(
        sdpa256,
        "_SDPA256_MIN_KV_LEN",
        sdpa256._SDPA256_MIN_KV_LEN,
        raising=False,
    )
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

    assert sdpa256.apply_sdpa256_attention_patch(min_kv_len=512) is True
    patched = mlx_base.scaled_dot_product_attention
    try:
        # head_dim 256 routed prefill -> flash kernel. Kernel numerical
        # correctness is covered above; keep this dispatcher test small so it
        # does not re-run the O(L^2) MLX reference path under full-suite memory
        # pressure.
        q, k, v = _qkv(128, 512)
        out = patched(q, k, v, None, SCALE_256, "causal")
        mx.eval(out)
        assert calls["flash"] == 1
        assert out.shape == q.shape
        assert out.dtype == q.dtype

        # decode (qL=1) -> passthrough to original.
        qd, kd, vd = _qkv(1, 512)
        mx.eval(patched(qd, kd, vd, None, SCALE_256, "causal"))
        assert calls["orig"] >= 1

        # head_dim 128 -> passthrough.
        q2, k2, v2 = _qkv(128, 512, head_dim=128)
        before = calls["orig"]
        mx.eval(patched(q2, k2, v2, None, 1.0 / math.sqrt(128), "causal"))
        assert calls["orig"] == before + 1
    finally:
        monkeypatch.setattr(mlx_base, "scaled_dot_product_attention", original)
        from omlx import memory_monitor as mm

        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)


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


def test_unfused_call_bytes_shared_with_guard_estimator():
    """The route gate and the guard must price the unfused path identically:
    the guard's unfused branch is the shared module function."""
    from omlx import memory_monitor as mm

    monitor = mm.MemoryMonitor.__new__(mm.MemoryMonitor)
    monitor._head_dim = 256
    monitor._num_attention_heads = 24
    monitor._num_kv_heads = 4
    monitor._score_dtype_size = 2

    mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    assert monitor._estimate_sdpa_activation_bytes(2048, 200_000) == (
        mm.estimate_unfused_sdpa_call_bytes(24, 2048, 200_000, 256, 2)
    )


# --- memory-aware routing (issue #2204) -----------------------------------


class _HeadroomOwner:
    """Stand-in for the Scheduler side of set_unfused_headroom_provider."""

    def __init__(self, value):
        self.value = value

    def headroom(self, kv_len):
        return self.value


@pytest.fixture
def _sdpa256_provider_reset(monkeypatch):
    """Isolate the module-level provider/override state and restore it."""
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None, raising=False)
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None, raising=False)
    monkeypatch.setattr(sdpa256, "_QSPLIT_ENABLED", True, raising=False)
    monkeypatch.setattr(sdpa256, "_LAST_ROUTE_DECISION_NO_CACHE", None, raising=False)
    return sdpa256


def test_route_prefers_stock_when_unfused_fits(_sdpa256_provider_reset):
    sdpa256 = _sdpa256_provider_reset
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)  # ~1 TB headroom: unfused clearly fits
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("unfused", 0)

    # Exactly at the estimated transient the unfused path still fits...
    need = estimate_unfused_sdpa_call_bytes(24, 2048, 16384, 256, q.dtype.size)
    owner.value = need
    assert sdpa256._should_route(q, k, None, "causal", None) == ("unfused", 0)
    # ...one byte short -> the full call doesn't fit, but a narrower q-split
    # slice still does, so it stays on the fast kernel instead of falling
    # all the way to tiled.
    owner.value = need - 1
    route, q_sub = sdpa256._should_route(q, k, None, "causal", None)
    assert route == "qsplit"
    assert sdpa256._QSPLIT_MIN_Q <= q_sub < 2048

    # Headroom below even the q-split floor (128 rows) -> tiled.
    per_row = need // 2048
    owner.value = sdpa256._QSPLIT_MIN_Q * per_row - 1
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)

    # Negative headroom = no active ceiling -> memory-safe default.
    owner.value = -1
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)


def test_route_qsplit_disabled_falls_straight_to_tiled(_sdpa256_provider_reset):
    """OMLX_SDPA256_QSPLIT=0 restores pre-q-split behavior: once the full
    unfused call doesn't fit, go straight to tiled without trying a
    narrower slice."""
    sdpa256 = _sdpa256_provider_reset
    sdpa256._QSPLIT_ENABLED = False
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

    q, k, _ = _qkv(2048, 16384)
    need = estimate_unfused_sdpa_call_bytes(24, 2048, 16384, 256, q.dtype.size)
    owner = _HeadroomOwner(need - 1)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)


def test_route_hysteresis_never_climbs_back_to_unfused(_sdpa256_provider_reset):
    """Once a request's cache has needed q-split or tiled, later chunks must
    not flip back to full unfused even if that chunk's own estimate looks
    like it fits -- kv_len is monotone within a request, so a route shed
    earlier reflects pressure that cannot have relaxed. Regression for a
    live run that flickered qsplit->unfused 16 seconds before the reactive
    memory guard aborted the request."""
    sdpa256 = _sdpa256_provider_reset

    class _Cache:
        pass

    cache = _Cache()
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)  # ample headroom: fast path every time
    sdpa256.set_unfused_headroom_provider(owner.headroom)

    # First call: plenty of headroom, stays unfused, no downgrade recorded.
    assert sdpa256._should_route(q, k, cache, "causal", None) == ("unfused", 0)
    assert getattr(cache, "_sdpa256_q_sub_ceiling", None) is None

    # Headroom tightens (simulating real pressure) and the route sheds to
    # tiled; the cache now carries a permanent downgrade marker.
    owner.value = 0
    assert sdpa256._should_route(q, k, cache, "causal", None) == ("tiled", 0)
    assert cache._sdpa256_q_sub_ceiling == 0

    # Headroom "recovers" (e.g. a momentarily generous estimate) -- without
    # hysteresis this would flip back to unfused. It must not.
    owner.value = 1 << 40
    assert sdpa256._should_route(q, k, cache, "causal", None) != ("unfused", 0)

    # A fresh request (new cache object) is unaffected by the first
    # request's downgrade.
    fresh_cache = _Cache()
    assert sdpa256._should_route(q, k, fresh_cache, "causal", None) == (
        "unfused",
        0,
    )


def test_route_hysteresis_caps_q_sub_not_just_the_route_label(
    _sdpa256_provider_reset,
):
    """A request downgraded to qsplit must not have q_sub float back up to
    q_len (a full-size transient, byte-for-byte identical to unfused) just
    because a later chunk's headroom estimate looks momentarily generous
    again -- capping only the route *label* while leaving q_sub uncapped
    was verified live to reproduce exactly that: a single qsplit sub-call
    the same size as the plain unfused call it was supposed to avoid."""
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

    sdpa256 = _sdpa256_provider_reset

    class _Cache:
        pass

    cache = _Cache()
    q, k, _ = _qkv(2048, 16384)
    need = estimate_unfused_sdpa_call_bytes(24, 2048, 16384, 256, q.dtype.size)
    per_row = need // 2048

    # First call: headroom fits ~1024 rows, not the full 2048 -> qsplit,
    # ceiling latches to that q_sub.
    owner = _HeadroomOwner(1024 * per_row)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    route, q_sub = sdpa256._should_route(q, k, cache, "causal", None)
    assert route == "qsplit"
    assert q_sub == 1024
    assert cache._sdpa256_q_sub_ceiling == 1024

    # Headroom "recovers" to ample -- without capping q_sub itself, this
    # would compute q_sub=2048 (== q_len), a full-size transient.
    owner.value = 1 << 40
    route, q_sub = sdpa256._should_route(q, k, cache, "causal", None)
    assert route == "qsplit"
    assert q_sub <= 1024
    assert cache._sdpa256_q_sub_ceiling <= 1024

    # Headroom tightens further -> ceiling ratchets down, never back up.
    owner.value = 256 * per_row
    route, q_sub = sdpa256._should_route(q, k, cache, "causal", None)
    assert route == "qsplit"
    assert q_sub == 256
    assert cache._sdpa256_q_sub_ceiling == 256

    # Recovers again -- still held at the smallest q_sub ever proven safe.
    owner.value = 1 << 40
    route, q_sub = sdpa256._should_route(q, k, cache, "causal", None)
    assert route == "qsplit"
    assert q_sub <= 256


def test_route_defaults_to_tiled_when_provider_owner_dies(_sdpa256_provider_reset):
    import gc

    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("unfused", 0)
    del owner
    gc.collect()
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)


def test_route_defaults_to_tiled_when_provider_raises(_sdpa256_provider_reset):
    sdpa256 = _sdpa256_provider_reset

    class _Boom:
        def headroom(self, kv_len):
            raise RuntimeError("no headroom info")

    boom = _Boom()
    sdpa256.set_unfused_headroom_provider(boom.headroom)
    q, k, _ = _qkv(2048, 16384)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)


def test_force_tiled_override(_sdpa256_provider_reset, monkeypatch):
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    # 1: always tiled even though unfused fits.
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", True, raising=False)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
    # 0: never tiled (or q-split) even without headroom info.
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None, raising=False)
    assert sdpa256._should_route(q, k, None, "causal", None) == ("unfused", 0)


def test_parse_force_tiled_env(monkeypatch):
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.delenv("OMLX_SDPA256_TILED", raising=False)
    assert sdpa256._parse_force_tiled_env() is None
    monkeypatch.setenv("OMLX_SDPA256_TILED", "1")
    assert sdpa256._parse_force_tiled_env() is True
    monkeypatch.setenv("OMLX_SDPA256_TILED", "0")
    assert sdpa256._parse_force_tiled_env() is False


def test_parse_qsplit_env(monkeypatch):
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.delenv("OMLX_SDPA256_QSPLIT", raising=False)
    assert sdpa256._parse_qsplit_env() is True
    monkeypatch.setenv("OMLX_SDPA256_QSPLIT", "1")
    assert sdpa256._parse_qsplit_env() is True
    monkeypatch.setenv("OMLX_SDPA256_QSPLIT", "0")
    assert sdpa256._parse_qsplit_env() is False


def test_max_q_sub_for_headroom():
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes
    from omlx.patches.sdpa256_attention import _max_q_sub_for_headroom

    n_q, kv_len, hd, dtype_size = 24, 16384, 256, 2.0
    # Headroom for exactly N rows must accept N and reject N+1.
    for n_rows in (1, 37, 512, 2048):
        headroom = estimate_unfused_sdpa_call_bytes(
            n_q, n_rows, kv_len, hd, dtype_size
        )
        assert _max_q_sub_for_headroom(n_q, kv_len, hd, dtype_size, headroom) == n_rows
        assert (
            _max_q_sub_for_headroom(n_q, kv_len, hd, dtype_size, headroom - 1)
            == n_rows - 1
        )
    assert _max_q_sub_for_headroom(n_q, kv_len, hd, dtype_size, 0) == 0
    assert _max_q_sub_for_headroom(n_q, kv_len, hd, dtype_size, -1) == 0


# --- tiled-route engagement logging (issue #2283) --------------------------


def _route_log_records(caplog, route=None):
    records = [
        r
        for r in caplog.records
        if r.levelname == "INFO" and r.getMessage().startswith("sdpa256: route -> ")
    ]
    if route is None:
        return records
    return [r for r in records if r.getMessage().startswith(f"sdpa256: route -> {route}")]


def test_tiled_route_logs_once_when_no_provider(_sdpa256_provider_reset, caplog):
    """Guard-off servers land on the tiled path silently (issue #2283); the
    first engagement must say so at INFO, repeats must stay quiet."""
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
        records = _route_log_records(caplog)
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "no guard headroom provider" in msg
        assert "OMLX_SDPA256_TILED" in msg
        # Second engagement, same decision: no new record (transition-only).
        assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
        assert len(_route_log_records(caplog)) == 1


def test_tiled_route_logs_headroom_numbers(_sdpa256_provider_reset, caplog):
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1)  # 1 byte of headroom: not even q-split fits
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
    records = _route_log_records(caplog, "tiled")
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "exceeds" in msg
    assert "kv_len=16384" in msg
    assert "MiB" in msg


def test_route_dedup_is_keyed_per_cache_not_module_global(
    _sdpa256_provider_reset, caplog
):
    """E5: the model passes one cache per full-attention layer. A single
    module-global last-decision meant interleaved calls from different
    layers (or different concurrent requests) flapped the transition log on
    every call instead of only on real transitions -- each cache's own
    decision now dedups independently, keyed on that cache object.
    See docs/qwen35-hardening-and-optimization.md E5."""
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)  # unfused clearly fits
    sdpa256.set_unfused_headroom_provider(owner.headroom)

    class _FakeCache:
        pass

    layer_a = _FakeCache()
    layer_b = _FakeCache()

    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        # Interleaved calls from two different layers, same decision each
        # time -- a module-global would see this as alternating (A, B, A,
        # B, ...) and log every single call. Per-cache keying logs each
        # cache's FIRST call only.
        assert sdpa256._should_route(q, k, layer_a, "causal", None) == ("unfused", 0)
        assert sdpa256._should_route(q, k, layer_b, "causal", None) == ("unfused", 0)
        assert sdpa256._should_route(q, k, layer_a, "causal", None) == ("unfused", 0)
        assert sdpa256._should_route(q, k, layer_b, "causal", None) == ("unfused", 0)

    records = _route_log_records(caplog, "unfused")
    assert len(records) == 2  # one per cache, not one per call


def test_tiled_route_logs_forced_env(_sdpa256_provider_reset, caplog, monkeypatch):
    sdpa256 = _sdpa256_provider_reset
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", True, raising=False)
    q, k, _ = _qkv(2048, 16384)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        assert sdpa256._should_route(q, k, None, "causal", None) == ("tiled", 0)
    records = _route_log_records(caplog, "tiled")
    assert len(records) == 1
    assert "OMLX_SDPA256_TILED=1" in records[0].getMessage()


def test_unfused_route_logs_nothing_extra(_sdpa256_provider_reset, caplog):
    """The unfused route still logs its own transition (useful for
    confirming a request stayed on the fast path), but must not produce a
    *tiled*-labeled record."""
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)  # ample headroom: fast path
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        assert sdpa256._should_route(q, k, None, "causal", None) == ("unfused", 0)
    assert _route_log_records(caplog, "tiled") == []
    assert len(_route_log_records(caplog, "unfused")) == 1


def test_qsplit_route_logs_transition(_sdpa256_provider_reset, caplog):
    sdpa256 = _sdpa256_provider_reset
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

    q, k, _ = _qkv(2048, 16384)
    need = estimate_unfused_sdpa_call_bytes(24, 2048, 16384, 256, q.dtype.size)
    owner = _HeadroomOwner(need - 1)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        route, q_sub = sdpa256._should_route(q, k, None, "causal", None)
        assert route == "qsplit"
    records = _route_log_records(caplog, "qsplit")
    assert len(records) == 1
    msg = records[0].getMessage()
    assert f"q_sub={q_sub}" in msg
    assert "kv_len=16384" in msg


def _sdpa256_headroom_fake(gib, usage_bytes=None):
    """Shared fake scaffold for _sdpa256_unfused_headroom tests. usage_bytes
    fixed (usage doesn't vary with the credit flag) unless a subclass
    overrides _current_usage_bytes to record calls."""
    from omlx.scheduler import Scheduler

    class _Fake:
        _memory_hard_limit_bytes = 0
        _memory_hard_watermark_bytes = 0
        _memory_abort_limit_bytes = 0
        _memory_limits_propagated = False
        _prefill_memory_guard = False
        _sdpa256_unguarded_logged = False
        _sdpa256_last_kv_len = None
        _prefill_headroom_safety = 0.90
        _PREFILL_HEADROOM_SAFETY = 0.90
        _prefill_abort_margin = 0.95
        _prefill_abort_cap = Scheduler._prefill_abort_cap
        _admission_limit_bytes = Scheduler._admission_limit_bytes

        def _current_usage_bytes(self, *, credit_cache_memory=False):
            return usage_bytes

    return _Fake()


def test_scheduler_headroom_provider_math():
    """_sdpa256_unfused_headroom targets _admission_limit_bytes (the same
    hard-watermark line the reactive enforcer kills at) x headroom safety,
    minus usage, then charges 2x by halving the result (see
    test_sdpa256_headroom_charges_2x_transient for that specifically).
    Usage is held fixed regardless of the credit flag -- the credit-toggle
    behavior itself is covered separately by
    test_sdpa256_headroom_credits_pool_only_on_repeated_shape."""
    from omlx.scheduler import _SDPA256_UNBOUNDED_HEADROOM, Scheduler

    gib = 1024**3
    fake = _sdpa256_headroom_fake(gib, usage_bytes=10 * gib)
    kv_len = 16384
    # Nothing propagated yet: guard state is unknown, so the negative
    # sentinel keeps the tiled default even though the flag reads False.
    assert Scheduler._sdpa256_unfused_headroom(fake, kv_len) == -1

    # Enforcer has spoken and the guard is explicitly off: the user opted
    # out of memory management, so the route gets unbounded headroom and
    # keeps the unfused fast path (#2283).
    fake._memory_limits_propagated = True
    assert (
        Scheduler._sdpa256_unfused_headroom(fake, kv_len)
        == _SDPA256_UNBOUNDED_HEADROOM
    )

    # Guard on but the ceiling has not landed yet (startup race): stay on
    # the memory-safe default.
    fake._prefill_memory_guard = True
    assert Scheduler._sdpa256_unfused_headroom(fake, kv_len) == -1

    # Watermark not propagated yet: _admission_limit_bytes falls back to
    # the hard limit alone.
    fake._memory_hard_limit_bytes = 100 * gib
    target = int(100 * gib * 0.90)
    assert Scheduler._sdpa256_unfused_headroom(fake, kv_len) == (
        target - 10 * gib
    ) // 2

    # Watermark binds once propagated and lower than the hard limit -- the
    # same line the reactive enforcer actually kills at.
    fake._memory_hard_watermark_bytes = 85 * gib
    target = int(85 * gib * 0.90)
    assert Scheduler._sdpa256_unfused_headroom(fake, kv_len) == (
        target - 10 * gib
    ) // 2


def test_sdpa256_headroom_charges_2x_transient():
    """A call only clears the route gate when its transient fits HALF the
    raw (target - usage) headroom, not the whole thing -- pricing in the
    same-size predecessor transient that kv_len monotonicity guarantees is
    still sitting unreclaimed in the pool (chunk k-1's transient can never
    be reused by chunk k's strictly larger one, and reclaim can't run
    mid-chunk). Proven necessary, not just defensive, by a live run where
    the 1x-charge version still rode the fast path to the same real-memory
    abort point as the original bug."""
    from omlx.scheduler import Scheduler

    gib = 1024**3
    fake = _sdpa256_headroom_fake(gib, usage_bytes=10 * gib)
    fake._memory_limits_propagated = True
    fake._prefill_memory_guard = True
    fake._memory_hard_limit_bytes = 100 * gib

    raw_headroom = int(100 * gib * 0.90) - 10 * gib
    charged = Scheduler._sdpa256_unfused_headroom(fake, 16384)
    assert charged == raw_headroom // 2
    assert charged < raw_headroom


def test_sdpa256_headroom_credits_pool_only_on_repeated_shape():
    """Regression for the pool-growth memory regression found via a live
    258k-token run: crediting mx.get_cache_memory() unconditionally let the
    router keep choosing the big-transient unfused/qsplit route as the
    retained pool inflated, so the *uncredited* reactive hard-watermark
    guard tripped abruptly instead of the old code's clean predictive 400.
    Within one chunk every full-attention layer shares one (kv_len, q_len)
    shape, so only a same-kv_len repeat call is guaranteed a same-size freed
    transient to reuse -- the first call at a new, larger kv_len is not."""
    from omlx.scheduler import Scheduler

    gib = 1024**3

    class _Fake:
        _memory_hard_limit_bytes = 100 * gib
        _memory_hard_watermark_bytes = 0
        _memory_abort_limit_bytes = 0
        _memory_limits_propagated = True
        _prefill_memory_guard = True
        _sdpa256_unguarded_logged = False
        _sdpa256_last_kv_len = None
        _prefill_headroom_safety = 0.90
        _PREFILL_HEADROOM_SAFETY = 0.90
        _prefill_abort_margin = 0.95
        _prefill_abort_cap = Scheduler._prefill_abort_cap
        _admission_limit_bytes = Scheduler._admission_limit_bytes

        def __init__(self):
            self.credit_calls = []

        def _current_usage_bytes(self, *, credit_cache_memory=False):
            self.credit_calls.append(credit_cache_memory)
            return 10 * gib

    fake = _Fake()
    # First call at kv_len=A: no same-shape call has run yet -> uncredited.
    Scheduler._sdpa256_unfused_headroom(fake, 16384)
    # Second call, same kv_len=A (e.g. layer 2 of the same chunk): a
    # same-size transient was already freed by the first call -> credited.
    Scheduler._sdpa256_unfused_headroom(fake, 16384)
    # Third call, new larger kv_len=B (next chunk): nothing this size has
    # been freed yet -> uncredited again, even though the pool has entries.
    Scheduler._sdpa256_unfused_headroom(fake, 18432)
    # Fourth call, repeats kv_len=B -> credited.
    Scheduler._sdpa256_unfused_headroom(fake, 18432)

    assert fake.credit_calls == [False, True, False, True]


def test_current_usage_bytes_credits_cache_memory_on_the_phys_side(monkeypatch):
    """credit_cache_memory=True subtracts mx.get_cache_memory() from the
    phys-footprint side only -- it must never pull the result below
    mx.get_active_memory() (which never included pooled memory), and must
    have no effect at all when active already wins the max()."""
    from omlx.scheduler import Scheduler

    gib = 1024**3

    class _Fake:
        _last_mlx_active_memory_bytes = 0
        _hot_cache_cpu_bytes = staticmethod(lambda: 0)

    fake = _Fake()

    monkeypatch.setattr("omlx.scheduler.get_phys_footprint", lambda: 40 * gib)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 5 * gib)

    # phys (40GiB) dominates active (5GiB); a pool of 15GiB should credit
    # back, landing well above active but below the uncredited phys.
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 15 * gib)
    uncredited = Scheduler._current_usage_bytes(fake, credit_cache_memory=False)
    credited = Scheduler._current_usage_bytes(fake, credit_cache_memory=True)
    assert uncredited == 40 * gib
    assert credited == 25 * gib

    # An oversized pool credit must floor at active, not go negative or
    # below the true active-memory sample.
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 100 * gib)
    assert Scheduler._current_usage_bytes(fake, credit_cache_memory=True) == 5 * gib

    # active already dominates phys: crediting has nothing to do.
    monkeypatch.setattr(mx, "get_active_memory", lambda: 45 * gib)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 15 * gib)
    assert Scheduler._current_usage_bytes(fake, credit_cache_memory=True) == 45 * gib


def test_unguarded_fast_path_logs_once(caplog):
    """Guard-off fast routing runs without a memory ceiling, which is the
    one state worth a breadcrumb (#2283): exactly one INFO naming the OOM
    trade and the recovery levers, then silence."""
    from omlx.scheduler import _SDPA256_UNBOUNDED_HEADROOM, Scheduler

    class _Fake:
        _memory_hard_limit_bytes = 0
        _memory_limits_propagated = True
        _prefill_memory_guard = False
        _sdpa256_unguarded_logged = False

    fake = _Fake()
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        assert (
            Scheduler._sdpa256_unfused_headroom(fake, 16384)
            == _SDPA256_UNBOUNDED_HEADROOM
        )
        assert (
            Scheduler._sdpa256_unfused_headroom(fake, 16384)
            == _SDPA256_UNBOUNDED_HEADROOM
        )
    records = [
        r for r in caplog.records if "memory guard disabled" in r.getMessage()
    ]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "OMLX_SDPA256_TILED=1" in msg


def test_scheduler_init_registers_headroom_provider(_sdpa256_provider_reset):
    """Constructing a Scheduler must wire the provider (the production seam:
    a rename that silently skips registration would revert #2204 to
    always-tiled)."""
    from unittest.mock import MagicMock

    from omlx.scheduler import Scheduler, SchedulerConfig

    sdpa256 = _sdpa256_provider_reset
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(paged_cache_block_size=0),
    )
    ref = sdpa256._HEADROOM_PROVIDER
    assert ref is not None
    bound = ref()
    assert bound is not None
    assert bound.__self__ is scheduler
    # Ceiling not propagated yet -> negative sentinel keeps the tiled default.
    assert bound(16384) == -1


# --- mlx-vlm coverage (issue: VLM engine head-256 prefill unprotected) ----


def _install_fake_vlm_tree(monkeypatch):
    """Fake mlx-vlm namespace mirroring the production import pattern:
    ``qwen3_5.language`` copies base's SDPA reference at import time."""
    root = types.ModuleType("mlx_vlm")
    models = types.ModuleType("mlx_vlm.models")
    base = types.ModuleType("mlx_vlm.models.base")
    language = types.ModuleType("mlx_vlm.models.qwen3_5.language")

    calls = {"vlm_orig": 0}

    def original(q, k, v, cache, scale, mask=None, sinks=None):
        calls["vlm_orig"] += 1
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    base.scaled_dot_product_attention = original
    language.scaled_dot_product_attention = original
    root.models = models
    models.base = base

    for name, module in {
        "mlx_vlm": root,
        "mlx_vlm.models": models,
        "mlx_vlm.models.base": base,
        "mlx_vlm.models.qwen3_5.language": language,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return base, language, original, calls


def _snapshot_lm_sdpa():
    snap = {}
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("mlx_lm.models."):
            continue
        fn = getattr(mod, "scaled_dot_product_attention", None)
        if fn is not None:
            snap[name] = fn
    return snap


def _restore_lm_sdpa(snap):
    for name, fn in snap.items():
        mod = sys.modules.get(name)
        if mod is not None:
            mod.scaled_dot_product_attention = fn


def test_vlm_submodule_rebind_covers_copied_reference(
    _sdpa256_provider_reset, monkeypatch
):
    """The patch must rebind mlx-vlm model modules that copied base's SDPA at
    import time — assigning to mlx_vlm.models.base alone never reaches them."""
    sdpa256 = _sdpa256_provider_reset
    base, language, original, calls = _install_fake_vlm_tree(monkeypatch)
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_SDPA256_MIN_KV_LEN", 512, raising=False)

    flash_calls = {"n": 0}
    real_flash = sdpa256._flash_sdpa256

    def counting_flash(q, k, v, scale, mask):
        flash_calls["n"] += 1
        return real_flash(q, k, v, scale, mask)

    monkeypatch.setattr(sdpa256, "_flash_sdpa256", counting_flash)

    lm_snap = _snapshot_lm_sdpa()
    try:
        assert sdpa256.apply_sdpa256_attention_patch(min_kv_len=512) is True
        assert language.scaled_dot_product_attention is not original
        assert (
            base.scaled_dot_product_attention
            is language.scaled_dot_product_attention
        )

        # Routed shape through the module the VLM model actually calls.
        q, k, v = _qkv(128, 512)
        mx.eval(language.scaled_dot_product_attention(q, k, v, None, SCALE_256, "causal"))
        assert flash_calls["n"] == 1

        # Decode shape passes through to the mlx-vlm original, not mlx-lm's.
        qd, kd, vd = _qkv(1, 512)
        mx.eval(language.scaled_dot_product_attention(qd, kd, vd, None, SCALE_256, "causal"))
        assert calls["vlm_orig"] == 1
    finally:
        _restore_lm_sdpa(lm_snap)
        from omlx import memory_monitor as mm

        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)


def test_production_install_order_covers_vlm_language(
    _sdpa256_provider_reset, monkeypatch
):
    """Both engines install sdpa256 first and fa256 second. fa256 captures
    whatever mlx_vlm.models.base holds at that point as its "original", so
    sdpa256 must have already rebound the submodules — otherwise the identity
    sweep misses qwen3_5.language and the VLM engine keeps the unfused path
    (the baseline defect this suite pins)."""
    sdpa256 = _sdpa256_provider_reset
    import omlx.patches.qwen35_fa256_attention as fa256

    base, language, original, calls = _install_fake_vlm_tree(monkeypatch)
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_SDPA256_MIN_KV_LEN", 512, raising=False)
    monkeypatch.setattr(fa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(fa256, "is_nax_available", lambda: False)
    monkeypatch.setattr(fa256, "_auto_dispatch_budget", lambda *a, **k: 0)
    monkeypatch.delenv("OMLX_FA256_STEEL", raising=False)

    steel_calls = {"n": 0}

    def fake_kernel(q, k, v, scale, causal=True, **kwargs):
        steel_calls["n"] += 1
        return q

    monkeypatch.setattr(fa256, "_native_kernel", lambda: fake_kernel)

    lm_snap = _snapshot_lm_sdpa()
    try:
        assert sdpa256.apply_sdpa256_attention_patch(min_kv_len=512) is True
        assert fa256.apply_qwen35_fa256_attention_patch(min_kv_len=512) is True

        # qwen3_5.language must have been carried through both rebinds.
        assert language.scaled_dot_product_attention is not original
        assert (
            language.scaled_dot_product_attention
            is base.scaled_dot_product_attention
        )

        # Steel-eligible prefill through the VLM call site hits the kernel.
        q, k, v = _qkv(128, 2048, dtype=mx.bfloat16)
        out = language.scaled_dot_product_attention(
            q, k, v, None, SCALE_256, "causal"
        )
        mx.eval(out)
        assert steel_calls["n"] == 1

        # Decode still reaches the true mlx-vlm original at the chain's end.
        qd, kd, vd = _qkv(1, 2048, dtype=mx.bfloat16)
        mx.eval(
            language.scaled_dot_product_attention(
                qd, kd, vd, None, SCALE_256, "causal"
            )
        )
        assert calls["vlm_orig"] == 1
    finally:
        _restore_lm_sdpa(lm_snap)
        from omlx import memory_monitor as mm

        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
