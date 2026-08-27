"""Physical-array exactness gates for DS4 M=6 HC -> RMSNorm continuation."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.cluster import deployment
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
hc = sys.modules["mlx_lm.models.hyper_connection"]


def _config():
    return SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        hidden_size=4096,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_verify_hc_prenorm_is_bit_exact(monkeypatch):
    monkeypatch.setattr(hc, "_VERIFY_HC_PRENORM", True)
    module = hc.HyperConnection(_config())
    module.eval()
    norm = nn.RMSNorm(4096, eps=1e-6)
    mx.random.seed(2026082401)
    module.fn = mx.random.normal((24, 16384)).astype(mx.float32) * 0.01
    module.base = mx.random.normal((24,)).astype(mx.float32) * 0.01
    module.scale = mx.array((0.9, 1.1, 0.8), dtype=mx.float32)
    norm.weight = mx.random.normal((4096,)).astype(mx.bfloat16)
    value = mx.random.normal((1, 6, 4, 4096)).astype(mx.bfloat16)

    collapsed, post, comb = module(value)
    normalized = norm(collapsed)
    fused = module.call_with_norm(value, norm)
    assert fused is not None
    fused_collapsed, fused_normalized, fused_post, fused_comb = fused
    mx.eval(
        collapsed,
        normalized,
        post,
        comb,
        fused_collapsed,
        fused_normalized,
        fused_post,
        fused_comb,
    )

    assert mx.array_equal(fused_collapsed, collapsed).item()
    assert mx.array_equal(fused_normalized, normalized).item()
    assert mx.array_equal(fused_post, post).item()
    assert mx.array_equal(fused_comb, comb).item()


def test_verify_hc_prenorm_fails_closed_outside_exact_shape(monkeypatch):
    monkeypatch.setattr(hc, "_VERIFY_HC_PRENORM", True)
    module = hc.HyperConnection(_config())
    module.eval()
    norm = nn.RMSNorm(4096, eps=1e-6)
    value = mx.zeros((1, 5, 4, 4096), dtype=mx.bfloat16)
    assert module.call_with_norm(value, norm) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_prefill_hc_prenorm_is_bit_exact(monkeypatch):
    monkeypatch.setattr(hc, "_VERIFY_HC_PRENORM", False)
    monkeypatch.setattr(hc, "_PREFILL_HC_PRENORM", True)
    module = hc.HyperConnection(_config())
    module.eval()
    norm = nn.RMSNorm(4096, eps=1e-6)
    mx.random.seed(2026082402)
    module.fn = mx.random.normal((24, 16384)).astype(mx.float32) * 0.01
    module.base = mx.random.normal((24,)).astype(mx.float32) * 0.01
    module.scale = mx.array((0.9, 1.1, 0.8), dtype=mx.float32)
    norm.weight = mx.random.normal((4096,)).astype(mx.bfloat16)
    value = mx.random.normal((1, 1024, 4, 4096)).astype(mx.bfloat16)

    collapsed, post, comb = module(value)
    normalized = norm(collapsed)
    fused = module.call_with_norm(value, norm)
    assert fused is not None
    fused_collapsed, fused_normalized, fused_post, fused_comb = fused
    mx.eval(
        collapsed,
        normalized,
        post,
        comb,
        fused_collapsed,
        fused_normalized,
        fused_post,
        fused_comb,
    )
    assert mx.array_equal(fused_collapsed, collapsed).item()
    assert mx.array_equal(fused_normalized, normalized).item()
    assert mx.array_equal(fused_post, post).item()
    assert mx.array_equal(fused_comb, comb).item()


def test_prefill_hc_prenorm_is_forwarded_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_PREFILL_HC_PRENORM", raising=False)
    assert "OMLX_DSV4_PREFILL_HC_PRENORM=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_PREFILL_HC_PRENORM", "1")
    assert "OMLX_DSV4_PREFILL_HC_PRENORM=1" in deployment._hostfile_envs()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_decode_hc_prenorm_is_bit_exact(monkeypatch):
    monkeypatch.setattr(hc, "_DECODE_HC_PRENORM", True)
    module = hc.HyperConnection(_config())
    module.eval()
    norm = nn.RMSNorm(4096, eps=1e-6)
    mx.random.seed(2026082403)
    module.fn = mx.random.normal((24, 16384)).astype(mx.float32) * 0.01
    module.base = mx.random.normal((24,)).astype(mx.float32) * 0.01
    module.scale = mx.array((0.9, 1.1, 0.8), dtype=mx.float32)
    norm.weight = mx.random.normal((4096,)).astype(mx.bfloat16)
    value = mx.random.normal((1, 1, 4, 4096)).astype(mx.bfloat16)

    collapsed, post, comb = module(value)
    normalized = norm(collapsed)
    fused = module.call_with_norm(value, norm)
    assert fused is not None
    fused_collapsed, fused_normalized, fused_post, fused_comb = fused
    mx.eval(
        collapsed,
        normalized,
        post,
        comb,
        fused_collapsed,
        fused_normalized,
        fused_post,
        fused_comb,
    )
    assert mx.array_equal(fused_collapsed, collapsed).item()
    assert mx.array_equal(fused_normalized, normalized).item()
    assert mx.array_equal(fused_post, post).item()
    assert mx.array_equal(fused_comb, comb).item()


def test_decode_hc_prenorm_is_forwarded_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_DECODE_HC_PRENORM", raising=False)
    assert "OMLX_DSV4_DECODE_HC_PRENORM=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_DECODE_HC_PRENORM", "1")
    assert "OMLX_DSV4_DECODE_HC_PRENORM=1" in deployment._hostfile_envs()
