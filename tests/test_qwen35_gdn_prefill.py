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
    monkeypatch.delenv("OMLX_GDN_KERNEL", raising=False)
    monkeypatch.delenv("OMLX_GDN_IMPL", raising=False)
    monkeypatch.delenv("OMLX_GDN_BLOCK_T", raising=False)
    monkeypatch.delenv("OMLX_GDN_MIN_T", raising=False)
    monkeypatch.delenv("OMLX_GDN_STUB", raising=False)
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)


def test_prefill_patch_routes_default_blocked_seq(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, lang = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    calls = []

    def blocked(q, k, v, g, beta, state):
        calls.append(("blocked", g, beta, state))
        return "blocked_y", "blocked_state"

    monkeypatch.setattr(kernels, "gated_delta_blocked_seq", blocked)

    assert patch.apply_qwen35_gdn_prefill_patch() is True
    assert lang.gated_delta_update is gd.gated_delta_update

    q = _Tensor((1, 128, 16, 128))
    k = _Tensor((1, 128, 16, 128))
    v = _Tensor((1, 128, 48, 128))
    a = _Tensor((1, 128, 48))
    assert gd.gated_delta_update(q, k, v, a, object(), object(), object()) == (
        "blocked_y",
        "blocked_state",
    )
    assert calls == [("blocked", "g", "beta", None)]


def test_prefill_patch_passthrough_for_decode_mask_and_unsupported_shape(monkeypatch):
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


def test_prefill_patch_chunked_impl_opt_in(monkeypatch):
    import omlx.custom_kernels.qwen35_prefill as kernels
    import omlx.patches.qwen35_gdn_chunked as patch

    gd, _ = _install_fake_qwen35(monkeypatch)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("OMLX_GDN_IMPL", "chunked")

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


def test_blocked_seq_default_block_size_depends_on_input_dtype(monkeypatch):
    from omlx.custom_kernels.qwen35_prefill.gdn import _normalize_block_t

    assert _normalize_block_t(None, mx.float32) == 16
    assert _normalize_block_t(None, mx.bfloat16) == 32
    assert _normalize_block_t(None, mx.float16) == 32

    monkeypatch.setenv("OMLX_GDN_BLOCK_T", "48")
    assert _normalize_block_t(None, mx.float32) == 48
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
