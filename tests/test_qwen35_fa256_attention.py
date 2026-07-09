# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5/3.6 FA-256 steel attention patch."""

from __future__ import annotations

import math
import sys
import types

import mlx.core as mx
import pytest


def _qkv(q_len=128, kv_len=None, dtype=mx.bfloat16):
    kv_len = q_len if kv_len is None else kv_len
    mx.random.seed(3)
    q = mx.random.normal((1, 24, q_len, 256)).astype(dtype)
    k = mx.random.normal((1, 4, kv_len, 256)).astype(dtype)
    v = mx.random.normal((1, 4, kv_len, 256)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _install_fake_vlm_base(monkeypatch):
    root = types.ModuleType("mlx_vlm")
    models = types.ModuleType("mlx_vlm.models")
    base = types.ModuleType("mlx_vlm.models.base")
    language = types.ModuleType("mlx_vlm.models.qwen3_5.language")

    def original(q, k, v, cache, scale, mask=None, sinks=None):
        return "original"

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
    return base, language


@pytest.fixture(autouse=True)
def _fresh_fa256_patch(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    monkeypatch.delenv("OMLX_FA256_STEEL", raising=False)
    monkeypatch.delenv("OMLX_FA256_MIN_KV_LEN", raising=False)
    monkeypatch.delenv("OMLX_FA256_Q_BLOCK", raising=False)
    monkeypatch.delenv("OMLX_FA256_K_BLOCK", raising=False)
    monkeypatch.delenv("OMLX_FA256_DEBUG", raising=False)
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)


def test_route_gate_is_qwen_fa256_only():
    import omlx.patches.qwen35_fa256_attention as patch

    q, k, _ = _qkv(128, 2048)
    assert patch._should_route(q, k, None, "causal", None, min_kv_len=2048)
    assert patch._should_route(q, k, None, None, None, min_kv_len=2048)
    assert not patch._should_route(q[:, :12], k, None, "causal", None, 2048)
    assert not patch._should_route(q, k[:, :2], None, "causal", None, 2048)
    assert not patch._should_route(q[:, :, :1], k, None, "causal", None, 2048)
    # decode-shaped multi-row (MTP verify, qL = 1 + depth <= 9) -> stock path;
    # the steel prefill kernel is 3-16x slower at tiny q_len (issue #2127)
    for q_len in (2, 4, 9, 15):
        qv, kv, _ = _qkv(q_len, 16384)
        assert not patch._should_route(qv, kv, None, "causal", None, 2048)
    qv, kv, _ = _qkv(16, 16384)
    assert patch._should_route(qv, kv, None, "causal", None, 2048)
    assert not patch._should_route(q, k, None, mx.zeros((128, 2048)), None, 2048)
    assert not patch._should_route(q, k, None, "causal", mx.zeros((4,)), 2048)

    class _QuantCache:
        bits = 4

    assert not patch._should_route(q, k, _QuantCache(), "causal", None, 2048)


def test_vlm_patch_routes_and_passes_through(monkeypatch):
    import omlx.patches.qwen35_fa256_attention as patch

    base, language = _install_fake_vlm_base(monkeypatch)
    calls = []

    def fake_kernel(q, k, v, scale, causal=True, q_block=32, k_block=8):
        calls.append((q.shape, k.shape, scale, causal, q_block, k_block))
        return "steel"

    monkeypatch.setattr(patch, "_native_kernel", lambda: fake_kernel)
    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)

    assert patch.apply_qwen35_fa256_attention_patch(min_kv_len=16)
    q, k, v = _qkv(32, 32)
    scale = 1.0 / math.sqrt(256)
    assert base.scaled_dot_product_attention(q, k, v, None, scale, "causal") == "steel"
    assert language.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert calls == [((1, 24, 32, 256), (1, 4, 32, 256), scale, True, 32, 8)]

    q_decode, _, _ = _qkv(1, 32)
    assert (
        base.scaled_dot_product_attention(q_decode, k, v, None, scale, "causal")
        == "original"
    )


def test_qwen_native_symbols_are_not_registered_on_glm_extension():
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    assert not glm_fast.has_symbol("qwen35_fa256_attention")
    assert not glm_fast.has_symbol("qwen35_q4_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q5_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q6_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_q8_affine_qmm_t")
    assert not glm_fast.has_symbol("qwen35_moe_weighted_sum")


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_native_fa256_matches_mlx_reference_small():
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_fa256_attention"):
        pytest.skip("native qwen35_fa256_attention is unavailable")

    q, k, v = _qkv(128)
    scale = 1.0 / math.sqrt(256)
    out = fast.qwen35_fa256_attention(q, k, v, scale, causal=True)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out, ref)

    err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    rel = (
        mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
        / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-9)
    ).item()
    assert err < 2e-2
    assert rel < 1e-2
