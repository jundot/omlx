# SPDX-License-Identifier: Apache-2.0
"""Fused residual+RMS norm (decode_fast) matches the composed fallback."""

import pytest
import mlx.core as mx

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "rows,axis", [(1, 2048), (7, 3072), (3, 4096), (2, 7168), (16, 8192)]
)
def test_matches_composed(dtype, rows, axis):
    mx.random.seed(0)
    x = mx.random.normal((rows, axis)).astype(dtype)
    r = mx.random.normal((rows, axis)).astype(dtype)
    w = mx.random.normal((axis,)).astype(dtype)
    out, summed = fast.rms_norm_residual(x, w, r, 1e-6)
    ref_out, ref_sum = fast._composed_rms_norm_residual(x, w, r, 1e-6)
    mx.eval(out, summed, ref_out, ref_sum)
    tol = 1e-5 if dtype == mx.float32 else 2e-3
    assert mx.allclose(out, ref_out, atol=tol, rtol=tol).item()
    assert mx.allclose(summed, ref_sum, atol=tol, rtol=tol).item()


def test_wrapper_composed_for_odd_strides():
    x = mx.random.normal((4, 4096))[:, ::2]  # non-contiguous last axis
    r = mx.random.normal((4, 2048))
    w = mx.random.normal((2048,))
    out, summed = fast.rms_norm_residual(x, w, r, 1e-6, force_composed=True)
    ref_out, ref_sum = fast._composed_rms_norm_residual(x, w, r, 1e-6)
    mx.eval(out, summed, ref_out, ref_sum)
    assert mx.allclose(out, ref_out, atol=1e-5, rtol=1e-5).item()
    assert mx.allclose(summed, ref_sum, atol=1e-5, rtol=1e-5).item()
