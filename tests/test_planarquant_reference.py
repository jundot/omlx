# SPDX-License-Identifier: Apache-2.0
"""Pure-MLX reference PlanarQuant3 correctness tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.cache.planarquant.constants import PLANAR_D
from omlx.cache.planarquant.reference import (
    dequantize_block,
    quantize_block,
    roundtrip,
)


def test_roundtrip_mse_near_paper_numbers():
    mx.random.seed(42)
    x = mx.random.normal((2, 4, 8, PLANAR_D)) * 0.1
    x_hat = roundtrip(x)
    diff = x.astype(mx.float32) - x_hat
    mse = float(mx.mean(diff * diff).item())
    # MSE for PlanarQuant3 at d=128 should be ~O(1/d²) ~ 1e-5 range
    assert mse < 1e-3, f"Roundtrip MSE too high: {mse}"


def test_norm_preservation_under_corrected_formula():
    mx.random.seed(7)
    x = mx.random.normal((1, 2, 4, PLANAR_D)) * 0.1
    x_hat = roundtrip(x)
    nx = mx.sqrt(mx.sum(x.astype(mx.float32) * x.astype(mx.float32), axis=-1))
    nxh = mx.sqrt(mx.sum(x_hat * x_hat, axis=-1))
    rel = float(mx.mean(mx.abs(nx - nxh) / (nx + 1e-10)).item())
    assert rel < 0.05, f"Norm preservation broken: rel={rel}"


def test_quantize_returns_packed_layout():
    mx.random.seed(1)
    x = mx.random.normal((1, 4, 8, PLANAR_D)) * 0.1
    packed, norms = quantize_block(x)
    # packed should be (1, 4, 8, qs_size + signs_size) = (1, 4, 8, 48)
    assert packed.shape[-1] == PLANAR_D // 4 + PLANAR_D // 8  # 32 + 16 = 48
    assert packed.dtype == mx.uint8
    assert norms.shape == (1, 4, 8, 1)
    assert norms.dtype == mx.float16


def test_last_dim_not_divisible_by_8_raises():
    x = mx.zeros((1, 1, 1, 10))
    with pytest.raises(ValueError, match="divisible by 8"):
        quantize_block(x)


def test_multi_block_heads():
    mx.random.seed(3)
    x = mx.random.normal((2, 4, 5, PLANAR_D)) * 0.1
    packed, norms = quantize_block(x)
    x_hat = dequantize_block(packed, norms)
    assert x_hat.shape == (2, 4, 5, PLANAR_D)


def test_dequant_zero_input():
    x = mx.zeros((1, 1, 1, PLANAR_D))
    packed, norms = quantize_block(x)
    # Zero input → zero norm → dequant should produce zero
    x_hat = dequantize_block(packed, norms)
    max_val = float(mx.max(mx.abs(x_hat)).item())
    assert max_val < 1e-6, f"Zero input not preserved: {max_val}"


def test_roundtrip_preserves_direction():
    """3-bit quantization preserves vector direction (cosine sim > 0.98)."""
    mx.random.seed(99)
    x = mx.random.normal((1, 1, 1, PLANAR_D)) * 1.0
    x_hat = roundtrip(x)
    x32 = x.astype(mx.float32)
    xh32 = x_hat.astype(mx.float32)
    dot = float(mx.sum(x32 * xh32).item())
    n1 = float(mx.sqrt(mx.sum(x32 * x32)).item())
    n2 = float(mx.sqrt(mx.sum(xh32 * xh32)).item())
    cos_sim = dot / (n1 * n2 + 1e-10)
    assert cos_sim > 0.98, f"Direction not preserved: cos_sim={cos_sim}"
