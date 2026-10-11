# SPDX-License-Identifier: Apache-2.0
"""The fused verifier norm must use the served RMS/FP32 SiLU arithmetic."""

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen3_5.language import Qwen3_5RMSNormGated

from omlx.patches import qwen35_gdn_verify_fused as patch
from omlx.patches.qwen35_gdn_verify_fused import _norm_gate_kernel

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _fused(norm, x, gate):
    rows = x.size // 128
    return _norm_gate_kernel(norm.eps)(
        inputs=[x, gate, norm.weight],
        template=[("InT", x.dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape, (rows * 2,)],
        output_dtypes=[x.dtype, mx.float32],
    )[0]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4])
def test_fused_norm_gate_matches_served_arithmetic(dtype, eps):
    norm = Qwen3_5RMSNormGated(128, eps=eps)
    for seed in range(40):
        mx.random.seed(seed)
        norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
        x = (mx.random.normal((1, 1, 32, 128)) * 10 ** (seed % 8 - 4)).astype(dtype)
        gate = (mx.random.normal(x.shape) * (seed / 2 + 0.125)).astype(dtype)
        expected, actual = norm(x, gate), _fused(norm, x, gate)
        mx.eval(expected, actual)
        assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_norm_gate_matches_every_gate_encoding(dtype):
    mx.random.seed(42)
    norm = Qwen3_5RMSNormGated(128)
    norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
    x = mx.ones((1, 1, 512, 128), dtype)
    gate = (
        mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(dtype).reshape(x.shape)
    )
    expected, actual = norm(x, gate), _fused(norm, x, gate)
    mx.eval(expected, actual)
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


def test_unsupported_sigmoid_declines_fusion(monkeypatch):
    from types import SimpleNamespace

    class Cache:
        _speculation = {"length": 2, "records": {}}

        def __getitem__(self, index):
            return None

    norm = Qwen3_5RMSNormGated(128)
    norm.weight = mx.ones((128,), mx.float16)
    layer = SimpleNamespace(
        norm=norm,
        head_k_dim=128,
        head_v_dim=128,
        num_v_heads=8,
        num_k_heads=4,
        A_log=mx.ones((8,), mx.float16),
        dt_bias=mx.ones((8,), mx.float16),
    )
    monkeypatch.setattr(patch, "_PATCHED", True)
    monkeypatch.setattr(patch, "_NORM_CLASS", Qwen3_5RMSNormGated)
    monkeypatch.setattr(patch, "_sigmoid_exp", lambda: None)
    assert not patch.fused_eligible(layer, mx.ones((1,), mx.float16), Cache(), 2)


def test_probe_declines_unknown_served_arithmetic(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    monkeypatch.setattr(
        language, "_precise_swiglu", lambda h, gate, x: mx.ones_like(gate)
    )
    patch._sigmoid_exp.cache_clear()
    try:
        assert patch._sigmoid_exp() is None
    finally:
        patch._sigmoid_exp.cache_clear()
