# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Tests for the Qwen3.5/3.6 GDN prefill Metal patch."""

from __future__ import annotations

import sys
import types

import mlx.core as mx
import pytest


class _Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def _install_fake_qwen35(monkeypatch):
    root = types.ModuleType("mlx_vlm")
    models = types.ModuleType("mlx_vlm.models")
    qwen = types.ModuleType("mlx_vlm.models.qwen3_5")
    gd = types.ModuleType("mlx_vlm.models.qwen3_5.gated_delta")
    lang = types.ModuleType("mlx_vlm.models.qwen3_5.language")

    def original(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        return "original", state

    gd.gated_delta_update = original
    lang.gated_delta_update = original
    gd._compute_g_beta = lambda A_log, a, b, dt_bias: ("g", "beta")

    root.models = models
    models.qwen3_5 = qwen
    qwen.gated_delta = gd
    qwen.language = lang

    for name, module in {
        "mlx_vlm": root,
        "mlx_vlm.models": models,
        "mlx_vlm.models.qwen3_5": qwen,
        "mlx_vlm.models.qwen3_5.gated_delta": gd,
        "mlx_vlm.models.qwen3_5.language": lang,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return gd, lang


@pytest.fixture(autouse=True)
def _fresh_gdn_patch(monkeypatch):
    import omlx.patches.qwen35_gdn_chunked as patch

    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)


def test_prefill_patch_routes_default_pipelined(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, lang = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    calls = []

    def pipelined(q, k, v, g, beta, state):
        calls.append(("pipelined", g, beta, state))
        return "pipelined_y", "pipelined_state"

    monkeypatch.setattr(kernels, "gated_delta_pipelined", pipelined)
    monkeypatch.setattr(
        kernels,
        "gated_delta_blocked_seq",
        lambda *args: pytest.fail("blocked kernel should not be the default"),
    )

    assert patch.apply_qwen35_gdn_prefill_patch() is True
    assert lang.gated_delta_update is gd.gated_delta_update

    q = _Tensor((1, 128, 16, 128))
    k = _Tensor((1, 128, 16, 128))
    v = _Tensor((1, 128, 48, 128))
    a = _Tensor((1, 128, 48))
    assert gd.gated_delta_update(q, k, v, a, object(), object(), object()) == (
        "pipelined_y",
        "pipelined_state",
    )
    assert calls == [("pipelined", "g", "beta", None)]


def test_prefill_patch_blocked_seq_impl_opt_in(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, _ = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_IMPL", "blocked_seq")
    calls = []
    monkeypatch.setattr(
        kernels,
        "gated_delta_blocked_seq",
        lambda *args: calls.append("blocked") or ("blocked_y", "blocked_state"),
    )
    monkeypatch.setattr(
        kernels,
        "gated_delta_pipelined",
        lambda *args: pytest.fail("pipelined kernel should not be routed"),
    )

    assert patch.apply_qwen35_gdn_prefill_patch() is True
    q = _Tensor((1, 128, 16, 128))
    v = _Tensor((1, 128, 48, 128))
    assert gd.gated_delta_update(q, q, v, _Tensor((1, 128, 48)), None, None, None) == (
        "blocked_y",
        "blocked_state",
    )
    assert calls == ["blocked"]


def test_prefill_patch_passthrough_for_decode_mask_and_unsupported_shape(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, _ = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    for name in ("gated_delta_blocked_seq", "gated_delta_pipelined"):
        monkeypatch.setattr(
            kernels,
            name,
            lambda *args: pytest.fail("prefill kernel should not be routed"),
        )

    assert patch.apply_qwen35_gdn_prefill_patch() is True

    k = _Tensor((1, 1, 16, 128))
    a = _Tensor((1, 1, 48))
    assert (
        gd.gated_delta_update(k, k, _Tensor((1, 1, 48, 128)), a, None, None, None)[0]
        == "original"
    )

    q = _Tensor((1, 128, 16, 128))
    v = _Tensor((1, 128, 48, 128))
    assert (
        gd.gated_delta_update(
            q, q, v, _Tensor((1, 128, 48)), None, None, None, mask=object()
        )[0]
        == "original"
    )

    bad_v = _Tensor((1, 128, 48, 96 + 16))
    assert (
        gd.gated_delta_update(
            q, q, bad_v, _Tensor((1, 128, 48)), None, None, None
        )[0]
        == "original"
    )


def test_prefill_patch_rejects_non_128_head_dims_satisfying_old_modulus(
    monkeypatch,
):
    """E2: the route gate used to admit any Dk % 16 == 0 / Dv % 32 == 0, but
    both the chunked kernel (A) and the default blocked_seq kernel (S)
    hard-assume Dk=128/Dv=128 internally -- a shape satisfying the old
    modulus without being exactly 128 would silently misbehave rather than
    error. Dk=144 (16*9) and Dv=160 (32*5) both satisfy the old modulus but
    must now be rejected.
    See docs/qwen35-hardening-and-optimization.md E2."""
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, _ = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        kernels,
        "gated_delta_blocked_seq",
        lambda *args: pytest.fail("blocked kernel should not be routed"),
    )

    assert patch.apply_qwen35_gdn_prefill_patch() is True

    a = _Tensor((1, 128, 48))

    # Dk=144: divisible by 16 (old gate), not 128 (new gate).
    q_bad_dk = _Tensor((1, 128, 16, 144))
    v_ok = _Tensor((1, 128, 48, 128))
    assert (
        gd.gated_delta_update(q_bad_dk, q_bad_dk, v_ok, a, None, None, None)[0]
        == "original"
    )

    # Dv=160: divisible by 32 (old gate), not 128 (new gate).
    q_ok = _Tensor((1, 128, 16, 128))
    v_bad_dv = _Tensor((1, 128, 48, 160))
    assert (
        gd.gated_delta_update(q_ok, q_ok, v_bad_dv, a, None, None, None)[0]
        == "original"
    )


def test_prefill_patch_chunked_impl_opt_in(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, _ = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_IMPL", "chunked")

    calls = []
    monkeypatch.setattr(
        kernels,
        "gated_delta_chunked_metal",
        lambda *args: calls.append("chunked") or ("chunked_y", "chunked_state"),
    )

    assert patch.apply_qwen35_gdn_prefill_patch() is True
    q = _Tensor((1, 128, 16, 128))
    v = _Tensor((1, 128, 48, 128))
    assert gd.gated_delta_update(q, q, v, _Tensor((1, 128, 48)), None, None, None) == (
        "chunked_y",
        "chunked_state",
    )
    assert calls == ["chunked"]


def test_blocked_seq_default_block_size_depends_on_input_dtype():
    from omlx.custom_kernels.qwen35_prefill.gdn import _normalize_block_t

    assert _normalize_block_t(None, mx.float32) == 16
    assert _normalize_block_t(None, mx.bfloat16) == 32
    assert _normalize_block_t(None, mx.float16) == 32
    assert _normalize_block_t(32, mx.float32) == 32


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_blocked_seq_matches_stock_kernel_small():
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq

    B, T, Hk, Hv, Dk, Dv = 1, 128, 16, 48, 128, 128
    keys = [mx.random.key(i) for i in range(6)]
    q = (mx.random.normal((B, T, Hk, Dk), key=keys[0]) * Dk**-1.0).astype(mx.bfloat16)
    k = (mx.random.normal((B, T, Hk, Dk), key=keys[1]) * Dk**-0.5).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv), key=keys[2]).astype(mx.bfloat16)
    g = mx.exp(-mx.random.uniform(0.01, 3.0, (B, T, Hv), key=keys[3])).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv), key=keys[4])).astype(mx.float32)
    state = (mx.random.normal((B, Hv, Dv, Dk), key=keys[5]) * 0.1).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)

    y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, state)
    y_fast, s_fast = gated_delta_blocked_seq(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_fast, s_fast)

    y_err = mx.max(mx.abs(y_fast.astype(mx.float32) - y_ref.astype(mx.float32))).item()
    s_rel = (
        mx.max(mx.abs(s_fast - s_ref)) / (mx.max(mx.abs(s_ref)) + 1e-9)
    ).item()
    assert y_err < 2e-2
    assert s_rel < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_blocked_seq_float32_default_fits_threadgroup_memory():
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from omlx.custom_kernels.qwen35_prefill import gated_delta_blocked_seq

    # Exact GDN layout from issue #2162. With float32 inputs, TB=32 requires
    # 40,192 bytes of threadgroup memory and cannot load on a 32 KiB device.
    B, T, Hk, Hv, Dk, Dv = 1, 64, 16, 32, 128, 128
    keys = [mx.random.key(i) for i in range(6)]
    q = (mx.random.normal((B, T, Hk, Dk), key=keys[0]) * Dk**-1.0).astype(
        mx.float32
    )
    k = (mx.random.normal((B, T, Hk, Dk), key=keys[1]) * Dk**-0.5).astype(
        mx.float32
    )
    v = mx.random.normal((B, T, Hv, Dv), key=keys[2]).astype(mx.float32)
    g = mx.exp(-mx.random.uniform(0.01, 3.0, (B, T, Hv), key=keys[3])).astype(
        mx.float32
    )
    beta = mx.sigmoid(mx.random.normal((B, T, Hv), key=keys[4])).astype(
        mx.float32
    )
    state = (mx.random.normal((B, Hv, Dv, Dk), key=keys[5]) * 0.1).astype(
        mx.float32
    )
    mx.eval(q, k, v, g, beta, state)

    y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, state)
    y_fast, s_fast = gated_delta_blocked_seq(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_fast, s_fast)

    y_err = mx.max(mx.abs(y_fast - y_ref)).item()
    s_rel = (
        mx.max(mx.abs(s_fast - s_ref)) / (mx.max(mx.abs(s_ref)) + 1e-9)
    ).item()
    assert y_err < 1e-6
    assert s_rel < 1e-6


def test_prefill_patch_preserves_cache_owned_kernel_dispatch(monkeypatch):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5 import gated_delta, language

    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    monkeypatch.setattr(gated_delta, "gated_delta_update", gated_delta.gated_delta_update)
    monkeypatch.setattr(language, "gated_delta_update", language.gated_delta_update)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    initial = mx.zeros((1, 1, 128, 128))
    final = mx.ones_like(initial)
    calls = []

    def pipelined(q, k, v, g, beta, state):
        calls.append(state)
        return v, final

    monkeypatch.setattr(kernels, "gated_delta_pipelined", pipelined)
    assert patch.apply_qwen35_gdn_prefill_patch()
    cache = ArraysCache(2)
    cache[1] = initial
    q = mx.zeros((1, 64, 1, 128))
    a = mx.zeros((1, 64, 1))
    output, state = gated_delta.gated_delta_update(
        q, q, q, a, a, mx.zeros((1,)), mx.zeros((1,)), cache=cache
    )
    assert len(calls) == 1 and calls[0] is initial
    assert output is q
    assert state is final and cache[1] is final


def _gdn_inputs(B, T, Hk, Hv, dtype, seed=0, Dk=128, Dv=128):
    keys = [mx.random.key(seed * 10 + i) for i in range(6)]
    q = mx.random.normal((B, T, Hk, Dk), key=keys[0])
    k = mx.random.normal((B, T, Hk, Dk), key=keys[1])
    # Qwen normalizes q/k (L2) and scales q by Dk^-0.5 before the recurrence.
    q = q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * Dk**-0.5
    k = k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6)
    v = mx.random.normal((B, T, Hv, Dv), key=keys[2])
    g = mx.exp(-mx.random.uniform(0.01, 3.0, (B, T, Hv), key=keys[3]))
    beta = mx.sigmoid(mx.random.normal((B, T, Hv), key=keys[4]))
    state = mx.random.normal((B, Hv, Dv, Dk), key=keys[5]) * 0.1
    out = (
        q.astype(dtype),
        k.astype(dtype),
        v.astype(dtype),
        g.astype(mx.float32),
        beta.astype(mx.float32),
        state.astype(mx.float32),
    )
    mx.eval(*out)
    return out


def _gdn_reference_fp64(q, k, v, g, beta, state):
    """Sequential recurrence in float64 (mlx_lm gated_delta_ops math)."""
    import numpy as np

    qn = np.array(q.astype(mx.float32), dtype=np.float64)
    kn = np.array(k.astype(mx.float32), dtype=np.float64)
    vn = np.array(v.astype(mx.float32), dtype=np.float64)
    gn = np.array(g, dtype=np.float64)
    bn = np.array(beta, dtype=np.float64)
    S = np.array(state, dtype=np.float64)
    B, T, Hk, _ = qn.shape
    Hv = vn.shape[2]
    hk = np.arange(Hv) // (Hv // Hk)
    ys = np.zeros(vn.shape)
    for t in range(T):
        kt, qt = kn[:, t][:, hk], qn[:, t][:, hk]  # [B, Hv, Dk]
        S = S * gn[:, t][:, :, None, None]
        p = np.einsum("bhvd,bhd->bhv", S, kt)
        delta = (vn[:, t] - p) * bn[:, t][:, :, None]
        S = S + delta[..., None] * kt[:, :, None, :]
        ys[:, t] = np.einsum("bhvd,bhd->bhv", S, qt)
    return ys, S


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
@pytest.mark.parametrize("T", [1, 11, 12, 13, 64, 100])
def test_pipelined_matches_stock_kernel(T):
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from omlx.custom_kernels.qwen35_prefill import gated_delta_pipelined

    q, k, v, g, beta, state = _gdn_inputs(1, T, 16, 48, mx.bfloat16, seed=T)
    y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, state)
    y, s = gated_delta_pipelined(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y, s)

    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32
    y_err = mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item()
    s_rel = (mx.max(mx.abs(s - s_ref)) / (mx.max(mx.abs(s_ref)) + 1e-9)).item()
    assert y_err < 2e-2
    assert s_rel < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
@pytest.mark.parametrize("Hk,Hv", [(16, 48), (16, 32), (2, 2)])
def test_pipelined_float32_matches_fp64_reference(Hk, Hv):
    import numpy as np

    from omlx.custom_kernels.qwen35_prefill import gated_delta_pipelined

    q, k, v, g, beta, state = _gdn_inputs(2, 29, Hk, Hv, mx.float32, seed=Hv)
    y, s = gated_delta_pipelined(q, k, v, g, beta, state)
    mx.eval(y, s)
    y_ref, s_ref = _gdn_reference_fp64(q, k, v, g, beta, state)

    y_rel = np.abs(np.array(y) - y_ref).max() / np.abs(y_ref).max()
    s_rel = np.abs(np.array(s) - s_ref).max() / np.abs(s_ref).max()
    assert y_rel < 1e-5
    assert s_rel < 1e-5


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_pipelined_split_prefill_equals_one_shot():
    from omlx.custom_kernels.qwen35_prefill import gated_delta_pipelined

    q, k, v, g, beta, _ = _gdn_inputs(1, 37, 16, 48, mx.bfloat16, seed=7)
    y_all, s_all = gated_delta_pipelined(q, k, v, g, beta, None)
    y_a, s_a = gated_delta_pipelined(
        q[:, :20], k[:, :20], v[:, :20], g[:, :20], beta[:, :20], None
    )
    y_b, s_b = gated_delta_pipelined(
        q[:, 20:], k[:, 20:], v[:, 20:], g[:, 20:], beta[:, 20:], s_a
    )
    mx.eval(y_all, s_all, y_a, y_b, s_b)
    # Every step runs the same arithmetic in the same order, however the
    # prompt is chunked, so the split run is bit-identical.
    assert mx.array_equal(mx.concatenate([y_a, y_b], axis=1), y_all).item()
    assert mx.array_equal(s_b, s_all).item()


def test_pipelined_falls_back_for_unsupported_layouts(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill.gdn as gdn

    calls = []

    def blocked(q, k, v, g, beta, state=None):
        calls.append((q.shape, v.shape))
        return "blocked_y", "blocked_state"

    monkeypatch.setattr(gdn, "gated_delta_blocked_seq", blocked)
    g = mx.zeros((1, 8, 4))
    for dk, dv in ((64, 128), (128, 40)):
        q = mx.zeros((1, 8, 2, dk), dtype=mx.bfloat16)
        v = mx.zeros((1, 8, 4, dv), dtype=mx.bfloat16)
        assert gdn.gated_delta_pipelined(q, q, v, g, g) == (
            "blocked_y",
            "blocked_state",
        )
    assert calls == [((1, 8, 2, 64), (1, 8, 4, 128)), ((1, 8, 2, 128), (1, 8, 4, 40))]
    q = mx.zeros((1, 8, 2, 128), dtype=mx.bfloat16)
    v = mx.zeros((1, 8, 4, 128), dtype=mx.bfloat16)
    assert gdn.gated_delta_pipelined_supported(q, q, v)
    assert not gdn.gated_delta_pipelined_supported(q, q, v.astype(mx.float32))
