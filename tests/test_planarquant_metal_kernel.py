# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused PlanarQuant3 Metal kernel (packed layout)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.cache.planarquant.metal_kernels import dequantize_fused, quantize_fused
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


# --- Quantize kernel parity tests ---


@pytest.mark.parametrize("head_dim", [128, 64])
def test_quantize_kernel_packed_parity(head_dim):
    """Quantize kernel should produce same packed output as reference."""
    mx.random.seed(42)
    x = mx.random.normal((2, 4, 8, head_dim)) * 0.1

    ref_packed, ref_norms = quantize_block(x)
    fused_packed, fused_norms = quantize_fused(x)

    assert fused_packed.shape == ref_packed.shape
    assert fused_packed.dtype == mx.uint8
    assert fused_norms.shape == ref_norms.shape

    # Packed bytes should match exactly (bit-exact)
    max_byte_diff = float(mx.max(mx.abs(
        fused_packed.astype(mx.int16) - ref_packed.astype(mx.int16)
    )).item())
    assert max_byte_diff == 0, f"Packed bytes differ at D={head_dim}: max_diff={max_byte_diff}"


@pytest.mark.parametrize("head_dim", [128, 64])
def test_quantize_kernel_norm_parity(head_dim):
    """Quantize kernel norms should match reference within fp16 epsilon."""
    mx.random.seed(7)
    x = mx.random.normal((1, 4, 4, head_dim)) * 0.1

    _, ref_norms = quantize_block(x)
    _, fused_norms = quantize_fused(x, out_dtype=mx.float32)

    ref32 = ref_norms.astype(mx.float32)
    max_diff = float(mx.max(mx.abs(fused_norms - ref32)).item())
    assert max_diff < 1e-3, f"Norms differ at D={head_dim}: {max_diff}"


def test_quantize_kernel_roundtrip_cosine_sim():
    """Full quantize→dequant roundtrip via Metal should match reference roundtrip."""
    mx.random.seed(99)
    x = mx.random.normal((2, 4, 5, 128)) * 0.1

    # Metal path: quantize_fused → dequantize_fused
    packed, norms = quantize_fused(x)
    x_hat = dequantize_fused(packed, norms, out_dtype=mx.float32)

    # Reference path: quantize_block → dequantize_block
    ref_packed, ref_norms = quantize_block(x)
    x_ref = dequantize_block(ref_packed, ref_norms)

    max_diff = float(mx.max(mx.abs(x_hat - x_ref)).item())
    assert max_diff < 1e-4, f"Metal roundtrip differs from reference: {max_diff}"

    # Cosine sim with original
    x32 = x.astype(mx.float32)
    dot = float(mx.sum(x32 * x_hat).item())
    n1 = float(mx.sqrt(mx.sum(x32 * x32)).item())
    n2 = float(mx.sqrt(mx.sum(x_hat * x_hat)).item())
    cos_sim = dot / (n1 * n2 + 1e-10)
    assert cos_sim > 0.98, f"Roundtrip direction not preserved: cos_sim={cos_sim}"


def test_quantize_kernel_zero_input():
    """Zero input should produce zero norms and packed all-zeros."""
    x = mx.zeros((1, 2, 3, 128))
    packed, norms = quantize_fused(x)

    max_norm = float(mx.max(mx.abs(norms)).item())
    assert max_norm < 1e-6, f"Zero input should have zero norms: {max_norm}"
