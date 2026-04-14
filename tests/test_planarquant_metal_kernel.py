# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused PlanarQuant3 Metal kernel (packed layout)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.cache.planarquant.metal_kernels import dequantize_fused
from omlx.cache.planarquant.reference import dequantize_block, quantize_block


def _has_metal() -> bool:
    try:
        return mx.metal.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="Metal unavailable")


@pytest.mark.parametrize("head_dim", [128, 64])
def test_kernel_matches_reference_fp32(head_dim):
    mx.random.seed(42)
    x = mx.random.normal((2, 4, 8, head_dim)) * 0.1
    packed, norms = quantize_block(x)

    ref = dequantize_block(packed, norms)  # fp32
    fused = dequantize_fused(packed, norms, out_dtype=mx.float32)

    max_diff = float(mx.max(mx.abs(ref - fused)).item())
    assert max_diff < 1e-4, f"kernel diverged at D={head_dim}: {max_diff}"


@pytest.mark.parametrize("head_dim", [128])
def test_kernel_fp16_within_fp16_epsilon(head_dim):
    mx.random.seed(7)
    x = mx.random.normal((1, 2, 4, head_dim)) * 0.1
    packed, norms = quantize_block(x)

    ref32 = dequantize_block(packed, norms)
    fused16 = dequantize_fused(packed, norms, out_dtype=mx.float16)

    max_diff = float(mx.max(mx.abs(ref32 - fused16.astype(mx.float32))).item())
    assert max_diff < 5e-4, f"fp16 kernel diverged: {max_diff}"
    assert fused16.dtype == mx.float16


def test_kernel_preserves_batch_shape():
    mx.random.seed(11)
    x = mx.random.normal((3, 7, 11, 128)) * 0.1
    packed, norms = quantize_block(x)
    out = dequantize_fused(packed, norms, out_dtype=mx.float16)
    assert out.shape == (3, 7, 11, 128)


def test_kernel_roundtrip_preserves_norm():
    mx.random.seed(3)
    x = mx.random.normal((2, 4, 5, 128)) * 0.1
    packed, norms = quantize_block(x)
    x_hat = dequantize_fused(packed, norms, out_dtype=mx.float32)

    nx = mx.sqrt(mx.sum(x.astype(mx.float32) * x.astype(mx.float32), axis=-1))
    nxh = mx.sqrt(mx.sum(x_hat * x_hat, axis=-1))
    rel = float(mx.mean(mx.abs(nx - nxh) / (nx + 1e-10)).item())
    assert rel < 0.05, f"norm preservation broken: {rel}"
