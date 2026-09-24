# SPDX-License-Identifier: Apache-2.0
"""SDPA256 bounded routing, numerical fallback, and memory registration tests."""

import logging
import math
import sys
import types

import mlx.core as mx
import pytest


@pytest.fixture
def _sdpa256_reset():
    """Hand out the patch module with its process-wide route state reset."""
    from omlx import memory_monitor as mm
    from omlx.patches import sdpa256_attention as sdpa256

    saved_routes = dict(mm._SDPA_TILED_PREFILL_HEAD_DIMS)
    saved_force = sdpa256._FORCE_TILED
    saved_logged = set(sdpa256._TILED_ROUTE_LOGGED)
    sdpa256._TILED_ROUTE_LOGGED.clear()
    sdpa256._FORCE_TILED = None
    try:
        yield sdpa256
    finally:
        mm._SDPA_TILED_PREFILL_HEAD_DIMS.clear()
        mm._SDPA_TILED_PREFILL_HEAD_DIMS.update(saved_routes)
        sdpa256._FORCE_TILED = saved_force
        sdpa256._TILED_ROUTE_LOGGED.clear()
        sdpa256._TILED_ROUTE_LOGGED.update(saved_logged)


def _tiled_log_records(caplog):
    return [r for r in caplog.records if "memory-bounded path" in r.getMessage()]


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


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_flash_sdpa256_memory_is_sub_quadratic(dtype):
    """Peak memory must grow ~O(L), not O(L^2). Over an 8K->32K span (4x in L)
    O(L^2) would grow ~16x; we require < 6x (O(L) is ~4x), a sharp signal."""
    if not hasattr(mx, "reset_peak_memory"):
        return  # peak-memory API unavailable on this MLX build; skip
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    peaks = []
    for seq_len in (8192, 32768):
        baseline = mx.get_active_memory()
        q, k, v = _qkv(seq_len, seq_len, n_q=6, n_kv=1, dtype=dtype)
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        mx.reset_peak_memory()
        mx.eval(_flash_sdpa256(q, k, v, SCALE_256, "causal"))
        peaks.append(mx.get_peak_memory() - baseline)
        del q, k, v
    assert peaks[0] > 0 and peaks[1] < 6 * peaks[0], peaks


def test_metal_bounded_path_forces_mlx0322_fused_kernel(monkeypatch):
    from omlx.patches import sdpa256_attention as sdpa256

    calls = []

    def fake_sdpa(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(sdpa256.mx.fast, "scaled_dot_product_attention", fake_sdpa)
    q = types.SimpleNamespace(shape=(1, 4, 16, 256), dtype=mx.float16)
    k = types.SimpleNamespace(shape=(1, 2, 32, 256), dtype=mx.float16)
    v = types.SimpleNamespace(shape=(1, 2, 32, 256), dtype=mx.float16)
    assert sdpa256._flash_sdpa256(q, k, v, SCALE_256, "causal") is q
    assert calls == [
        {
            "scale": SCALE_256,
            "mask": "causal",
            "sinks": None,
            "force_fused": True,
        }
    ]


@pytest.mark.parametrize("case", ["boolean", "additive", "sinks"])
def test_bounded_path_preserves_array_masks_and_sinks(case):
    """Array masks route to the bounded portable path on Metal too (native
    fused array-mask support is unproven and may silently unfuse); causal/
    no-mask cases exercise the real MLX 0.32.2 fused call when available.
    All cases stay numerically pinned against the reference SDPA."""
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, k, v = _qkv(16, 32, n_q=4, n_kv=2)
    mask = None
    sinks = None
    if case == "boolean":
        mask = mx.arange(32)[None, None, None, :] >= 8
    elif case == "additive":
        allowed = mx.arange(32)[None, None, None, :] >= 8
        mask = mx.where(allowed, 0.0, -1e4).astype(mx.float16)
    else:
        sinks = mx.array([-0.5, 0.0, 0.5, 1.0], dtype=mx.float16)

    out = _flash_sdpa256(q, k, v, SCALE_256, mask, sinks)
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=SCALE_256, mask=mask, sinks=sinks
    )
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
def test_metal_array_masks_never_reach_native_fused(mask_kind, monkeypatch):
    """On Metal, an explicit array mask must go straight to the bounded
    array-tiled kernel — never to mx.fast.scaled_dot_product_attention with
    force_fused=True, whose array-mask handling could silently unfuse into
    the O(L^2) fp32 score matrix this patch exists to bound."""
    from omlx.patches import sdpa256_attention as sdpa256

    calls = []

    def boom(*args, **kwargs):
        raise AssertionError("native fused SDPA must not see an array mask")

    def tiled(q, k, v, scale, mask, sinks=None):
        calls.append((mask, sinks))
        return q

    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        sdpa256.mx.fast, "scaled_dot_product_attention", boom
    )
    monkeypatch.setattr(sdpa256, "_array_tiled_sdpa256", tiled)

    q, k, v = _qkv(16, 32, n_q=4, n_kv=2)
    if mask_kind == "boolean":
        mask = mx.arange(32)[None, None, None, :] >= 8
    else:
        allowed = mx.arange(32)[None, None, None, :] >= 8
        mask = mx.where(allowed, 0.0, -1e4).astype(mx.float16)
    sinks = mx.array([-0.5, 0.0, 0.5, 1.0], dtype=mx.float16)

    out = sdpa256._flash_sdpa256(q, k, v, SCALE_256, mask, sinks)
    assert out is q
    # Mask and sinks forwarded unchanged to the bounded kernel.
    assert len(calls) == 1
    assert calls[0][0] is mask
    assert calls[0][1] is sinks


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("mask", ["causal", None], ids=["causal", "none"])
def test_metal_causal_and_none_keep_native_fused(mask, dtype, monkeypatch):
    """The router dtype fix must not push the proven causal/no-mask paths off
    the native fused kernel."""
    from omlx.patches import sdpa256_attention as sdpa256

    calls = []

    def fake_sdpa(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    def tiled(*args, **kwargs):
        raise AssertionError("causal/no-mask native shape must stay fused")

    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(sdpa256.mx.fast, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(sdpa256, "_array_tiled_sdpa256", tiled)

    q, k, v = _qkv(16, 32, n_q=4, n_kv=2, dtype=dtype)
    out = sdpa256._flash_sdpa256(q, k, v, SCALE_256, mask)
    assert out is q
    assert calls == [
        {"scale": SCALE_256, "mask": mask, "sinks": None, "force_fused": True}
    ]


@pytest.mark.parametrize("fp32_input", ["all", "keys", "values", "mixed_16"])
@pytest.mark.parametrize("mask", ["causal", None])
def test_fp32_bounded_prefill_matches_reference(fp32_input, mask):
    from omlx.patches.sdpa256_attention import _flash_sdpa256

    q, k, v = _qkv(32, 8192, n_q=4, n_kv=2, dtype=mx.float32)
    if fp32_input != "all":
        q = q.astype(mx.float16)
        if fp32_input == "mixed_16":
            k = k.astype(mx.bfloat16)
            v = v.astype(mx.float16)
        elif fp32_input == "keys":
            v = v.astype(mx.float16)
        else:
            k = k.astype(mx.float16)
    out = _flash_sdpa256(q, k, v, SCALE_256, mask)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask=mask)
    mx.eval(out, ref)
    assert out.dtype == ref.dtype == mx.float32
    assert _max_abs(out, ref) < 2e-5


@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
def test_portable_array_mask_matches_reference(mask_kind, monkeypatch):
    """CUDA's bounded fallback must preserve both MLX array-mask forms."""
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(sdpa256, "_Q_TILE", 16)
    monkeypatch.setattr(sdpa256, "_KV_TILE", 32)
    q, k, v = _qkv(32, 96, n_q=4, n_kv=2)
    allowed = mx.arange(96)[None, None, None, :] >= 24
    if mask_kind == "boolean":
        mask = allowed
    else:
        mask = mx.where(allowed, 0.0, -1e4).astype(mx.float16)

    out = sdpa256._flash_sdpa256(q, k, v, SCALE_256, mask)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, mask=mask)
    mx.eval(out, ref)
    assert _max_abs(out, ref) < 2e-2


def test_portable_sinks_and_value_dimension_match_reference(monkeypatch):
    """The portable path must cover sink models and non-square value heads."""
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.setattr(sdpa256.mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(sdpa256, "_Q_TILE", 16)
    monkeypatch.setattr(sdpa256, "_KV_TILE", 32)
    mx.random.seed(1)
    q = mx.random.normal((1, 4, 32, 256)).astype(mx.float16)
    k = mx.random.normal((1, 2, 96, 256)).astype(mx.float16)
    v = mx.random.normal((1, 2, 96, 128)).astype(mx.float16)
    sinks = mx.array([-0.5, 0.0, 0.5, 1.0], dtype=mx.float16)
    mx.eval(q, k, v, sinks)

    out = sdpa256._flash_sdpa256(q, k, v, SCALE_256, None, sinks)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE_256, sinks=sinks)
    mx.eval(out, ref)
    assert out.shape == (1, 4, 32, 128)
    assert _max_abs(out, ref) < 2e-2


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
    # tiny-query fused routing is already handled by MLX's vector kernel
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
    # Boolean/additive masks and attention sinks remain memory-bounded.
    bool_mask = mx.ones((1, 1, 1, 16384), dtype=mx.bool_)
    additive_mask = mx.zeros((1, 1, 1, 16384), dtype=mx.float16)
    sinks = mx.zeros((24,), dtype=mx.float32)
    assert sdpa256._should_route(q, k, None, bool_mask, None) is True
    assert sdpa256._should_route(q, k, None, additive_mask, None) is True
    assert sdpa256._should_route(q, k, None, "causal", sinks) is True

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

    def counting_flash(q, k, v, scale, mask, sinks=None):
        calls["flash"] += 1
        return real_flash(q, k, v, scale, mask, sinks)

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
    monitor._tiled_prefill_override = None

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


def test_estimator_keeps_registered_route_thresholds_independent():
    """Two bounded kernels must not create coverage neither one provides."""
    from omlx import memory_monitor as mm

    monitor = mm.MemoryMonitor.__new__(mm.MemoryMonitor)
    monitor._head_dim = 256
    monitor._num_attention_heads = 24
    monitor._num_kv_heads = 4
    monitor._score_dtype_size = 2

    mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    mm.register_tiled_prefill_head_dim(
        256, min_query_len=16, min_kv_len=8192, kv_tile=1024
    )
    mm.register_tiled_prefill_head_dim(
        256, min_query_len=64, min_kv_len=2048, kv_tile=512
    )
    try:
        # q=16 / kv=2048 satisfies one threshold from each registration but
        # neither complete route, so the estimate must remain quadratic.
        estimate = monitor._estimate_sdpa_activation_bytes(16, 2048)
        assert estimate == mm.estimate_unfused_sdpa_call_bytes(24, 16, 2048, 256, 2)
        assert monitor._estimate_sdpa_activation_bytes(64, 2048) < estimate * 4
    finally:
        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)


def test_unfused_call_bytes_shared_with_guard_estimator():
    """The guard must include the score matrix and FP32 output allocation."""
    from omlx import memory_monitor as mm

    monitor = mm.MemoryMonitor.__new__(mm.MemoryMonitor)
    monitor._head_dim = 256
    monitor._num_attention_heads = 24
    monitor._num_kv_heads = 4
    monitor._score_dtype_size = 2
    monitor._tiled_prefill_override = None

    mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    assert monitor._estimate_sdpa_activation_bytes(2048, 200_000) == (
        mm.estimate_unfused_sdpa_call_bytes(24, 2048, 200_000, 256, 2)
    )


# --- bounded routing overrides -------------------------------------------


def test_parse_force_tiled_env(monkeypatch):
    from omlx.patches import sdpa256_attention as sdpa256

    monkeypatch.delenv("OMLX_SDPA256_TILED", raising=False)
    assert sdpa256._parse_force_tiled_env() is None
    monkeypatch.setenv("OMLX_SDPA256_TILED", "1")
    assert sdpa256._parse_force_tiled_env() is True
    monkeypatch.setenv("OMLX_SDPA256_TILED", "0")
    assert sdpa256._parse_force_tiled_env() is False


def test_force_off_does_not_publish_a_bounded_memory_route(monkeypatch):
    """The O(L^2) benchmark override must keep conservative admission math."""
    from omlx import memory_monitor as mm
    from omlx.patches import sdpa256_attention as sdpa256

    mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", False)
    assert sdpa256._register_bounded_route(8192) is False
    assert 256 not in mm._SDPA_TILED_PREFILL_HEAD_DIMS

    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None)
    assert sdpa256._register_bounded_route(8192) is True
    try:
        routes = mm._SDPA_TILED_PREFILL_HEAD_DIMS[256]
        assert len(routes) == 1
        assert routes[0].supports_array_mask is True
    finally:
        mm._SDPA_TILED_PREFILL_HEAD_DIMS.pop(256, None)


# --- bounded-route engagement logging (issue #2283) ------------------------


def test_tiled_route_logs_forced_env(_sdpa256_reset, caplog, monkeypatch):
    sdpa256 = _sdpa256_reset
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", True, raising=False)
    q, k, _ = _qkv(2048, 16384)
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        assert sdpa256._should_route(q, k, None, "causal", None) is True
    records = _tiled_log_records(caplog)
    assert len(records) == 1
    assert "OMLX_SDPA256_TILED=1" in records[0].getMessage()


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
    _sdpa256_reset, monkeypatch
):
    """The patch must rebind mlx-vlm model modules that copied base's SDPA at
    import time — assigning to mlx_vlm.models.base alone never reaches them."""
    sdpa256 = _sdpa256_reset
    base, language, original, calls = _install_fake_vlm_tree(monkeypatch)
    monkeypatch.setattr(sdpa256, "_PATCHED", False, raising=False)
    monkeypatch.setattr(sdpa256, "_SDPA256_MIN_KV_LEN", 512, raising=False)

    flash_calls = {"n": 0}
    real_flash = sdpa256._flash_sdpa256

    def counting_flash(q, k, v, scale, mask, sinks=None):
        flash_calls["n"] += 1
        return real_flash(q, k, v, scale, mask, sinks)

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
    _sdpa256_reset, monkeypatch
):
    """Both engines install sdpa256 first and fa256 second. fa256 captures
    whatever mlx_vlm.models.base holds at that point as its "original", so
    sdpa256 must have already rebound the submodules — otherwise the identity
    sweep misses qwen3_5.language and the VLM engine keeps the unfused path
    (the baseline defect this suite pins)."""
    sdpa256 = _sdpa256_reset
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
