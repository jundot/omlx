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

import math

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


# --- route gate ----------------------------------------------------------


def test_should_route_gate():
    from omlx.patches import sdpa256_attention as sdpa256

    q, k, _ = _qkv(2048, 16384)  # 256, prefill, long
    assert sdpa256._should_route(q, k, None, "causal", None) is True
    assert sdpa256._should_route(q, k, None, None, None) is True
    # decode (qL==1) -> fused vector kernel handles 256
    qd, kd, _ = _qkv(1, 16384)
    assert sdpa256._should_route(qd, kd, None, "causal", None) is False
    # decode-shaped multi-row (MTP verify, qL = 1 + depth <= 9) -> stock path;
    # the per-tile eval sync collapses long-context MTP tok/s (issue #2127)
    for q_len in (2, 4, 9, 15):
        qv, kv, _ = _qkv(q_len, 16384)
        assert sdpa256._should_route(qv, kv, None, "causal", None) is False
    qv, kv, _ = _qkv(16, 16384)
    assert sdpa256._should_route(qv, kv, None, "causal", None) is True
    # short kv -> keep the faster fallback
    qs, ks, _ = _qkv(2048, 4096)
    assert sdpa256._should_route(qs, ks, None, "causal", None) is False
    # wrong head_dim
    qh, kh, _ = _qkv(2048, 16384, head_dim=128)
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

    def headroom(self):
        return self.value


@pytest.fixture
def _sdpa256_provider_reset(monkeypatch):
    """Isolate the module-level provider/override state and restore it."""
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None, raising=False)
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None, raising=False)
    return sdpa256


def test_route_prefers_stock_when_unfused_fits(_sdpa256_provider_reset):
    sdpa256 = _sdpa256_provider_reset
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)  # ~1 TB headroom: unfused clearly fits
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    assert sdpa256._should_route(q, k, None, "causal", None) is False

    # Exactly at the estimated transient the unfused path still fits...
    need = estimate_unfused_sdpa_call_bytes(24, 2048, 16384, 256, q.dtype.size)
    owner.value = need
    assert sdpa256._should_route(q, k, None, "causal", None) is False
    # ...one byte short -> tiled.
    owner.value = need - 1
    assert sdpa256._should_route(q, k, None, "causal", None) is True

    # Negative headroom = no active ceiling -> memory-safe default.
    owner.value = -1
    assert sdpa256._should_route(q, k, None, "causal", None) is True


def test_route_defaults_to_tiled_when_provider_owner_dies(_sdpa256_provider_reset):
    import gc

    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    assert sdpa256._should_route(q, k, None, "causal", None) is False
    del owner
    gc.collect()
    assert sdpa256._should_route(q, k, None, "causal", None) is True


def test_route_defaults_to_tiled_when_provider_raises(_sdpa256_provider_reset):
    sdpa256 = _sdpa256_provider_reset

    class _Boom:
        def headroom(self):
            raise RuntimeError("no headroom info")

    boom = _Boom()
    sdpa256.set_unfused_headroom_provider(boom.headroom)
    q, k, _ = _qkv(2048, 16384)
    assert sdpa256._should_route(q, k, None, "causal", None) is True


def test_force_tiled_override(_sdpa256_provider_reset, monkeypatch):
    sdpa256 = _sdpa256_provider_reset
    q, k, _ = _qkv(2048, 16384)
    owner = _HeadroomOwner(1 << 40)
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    # 1: always tiled even though unfused fits.
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", True, raising=False)
    assert sdpa256._should_route(q, k, None, "causal", None) is True
    # 0: never tiled even without headroom info.
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER", None, raising=False)
    assert sdpa256._should_route(q, k, None, "causal", None) is False


def test_parse_force_tiled_env(monkeypatch):
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.delenv("OMLX_SDPA256_TILED", raising=False)
    assert sdpa256._parse_force_tiled_env() is None
    monkeypatch.setenv("OMLX_SDPA256_TILED", "1")
    assert sdpa256._parse_force_tiled_env() is True
    monkeypatch.setenv("OMLX_SDPA256_TILED", "0")
    assert sdpa256._parse_force_tiled_env() is False


def test_scheduler_headroom_provider_math():
    """_sdpa256_unfused_headroom mirrors the adaptive throttle target:
    hard ceiling x headroom safety, clamped by the abort cap, minus usage."""
    from omlx.scheduler import Scheduler

    gib = 1024**3

    class _Fake:
        _memory_hard_limit_bytes = 0
        _memory_abort_limit_bytes = 0
        _prefill_headroom_safety = 0.90
        _PREFILL_HEADROOM_SAFETY = 0.90
        _prefill_abort_margin = 0.95
        _prefill_abort_cap = Scheduler._prefill_abort_cap

        def _current_usage_bytes(self):
            return 10 * gib

    fake = _Fake()
    # No ceiling propagated yet -> negative sentinel (keep the tiled default).
    assert Scheduler._sdpa256_unfused_headroom(fake) == -1

    # Throttle target binds: abort cap (100 * 0.95) > target (100 * 0.90).
    fake._memory_hard_limit_bytes = 100 * gib
    assert Scheduler._sdpa256_unfused_headroom(fake) == int(100 * gib * 0.90) - 10 * gib

    # Abort cap binds when lower than the throttle target.
    fake._memory_abort_limit_bytes = 80 * gib
    assert Scheduler._sdpa256_unfused_headroom(fake) == int(80 * gib * 0.95) - 10 * gib


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
    assert bound() == -1
