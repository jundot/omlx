# SPDX-License-Identifier: Apache-2.0
"""Fused GDN operands must not retain a second copy of source weights."""

import gc

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.ops import linear as ops

from omlx.patches import qwen35_gdn_prework as prework
from omlx.patches.qwen35_gdn_prework import (
    _one_row_fused_projections,
    apply_qwen35_gdn_prework_patch,
)


@pytest.fixture(autouse=True)
def install_production_patch(monkeypatch):
    # This entry point also installs GDN kernels. Preserve those class hooks
    # and the installation flag so later tests do not wrap a patched class.
    from mlx_vlm.models.qwen3_5 import language
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    cls = language.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(cls, "_normalize_qk", cls._normalize_qk)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_gated_delta", verifier._gated_delta)
    monkeypatch.setattr(
        verifier,
        "_normalize_gated_delta_qk",
        verifier.__dict__["_normalize_gated_delta_qk"],
    )
    monkeypatch.setattr(prework, "_PATCHED", prework._PATCHED)
    monkeypatch.setattr(
        ops, "_decode_quantized_linears_fused", ops._decode_quantized_linears_fused
    )
    apply_qwen35_gdn_prework_patch()


def projections(input_size=256, rows=(128, 64, 16, 16), bits=4, dtype=mx.bfloat16):
    result = [
        nn.QuantizedLinear(input_size, count, bias=False, group_size=64, bits=bits)
        for count in rows
    ]
    for projection in result:
        projection.set_dtype(dtype)
    mx.eval([projection.parameters() for projection in result])
    return result


@pytest.mark.parametrize("path", ["omlx", "mlx-vlm"])
def test_fused_cache_does_not_duplicate_27b_projection_storage(path):
    linears = projections(5120, (10240, 6144, 48, 48))
    x = mx.zeros((1, 1, 5120), dtype=mx.bfloat16)
    mx.eval(x)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    if path == "omlx":
        _one_row_fused_projections(linears)
    else:
        mx.eval(ops._decode_quantized_linears_fused(linears, x))
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    # Existing code retains another45.3MiB at this geometry. Allow only
    # small output/alignment allocations, never a second projection bank.
    assert mx.get_active_memory() - before < 1024**2
    cache = linears[0]._fused_decode_linears
    assert cache[0] == tuple(
        (id(item.weight), id(item.scales), id(item.biases)) for item in linears
    )
    mx.eval(ops._decode_quantized_linears_fused(linears, x))
    assert linears[0]._fused_decode_linears is cache


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_shared_storage_preserves_decode_prefill_and_cache_reuse(bits, dtype):
    mx.random.seed(51)
    linears = projections(bits=bits, dtype=dtype)
    original_operands = tuple(
        (item.weight, item.scales, item.biases) for item in linears
    )
    x = mx.random.normal((1, 1, 256)).astype(dtype)
    prefill = mx.random.normal((1, 64, 256)).astype(dtype)
    reference_prefill = tuple(item(prefill) for item in linears)
    original = getattr(
        ops._decode_quantized_linears_fused,
        "__wrapped__",
        ops._decode_quantized_linears_fused,
    )
    reference_decode = original(linears, x)
    mx.eval(reference_decode, reference_prefill)
    actual = ops._decode_quantized_linears_fused(linears, x)
    actual_prefill = tuple(item(prefill) for item in linears)
    mx.eval(actual, actual_prefill)
    for item, operands in zip(linears, original_operands):
        for value, original_value in zip(
            (item.weight, item.scales, item.biases), operands
        ):
            assert value.dtype == original_value.dtype
            assert mx.array_equal(value, original_value).item()
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, reference_decode))
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(actual_prefill, reference_prefill)
    )
    cache = linears[0]._fused_decode_linears
    if dtype == mx.bfloat16:
        assert _one_row_fused_projections(linears) is not None
        assert linears[0]._fused_decode_linears is cache
    mx.eval(ops._decode_quantized_linears_fused(linears, x))
    assert linears[0]._fused_decode_linears is cache


def test_replaced_weight_rebuilds_shared_cache_and_changes_output():
    linears = projections()
    x = mx.ones((1, 1, 256), dtype=mx.bfloat16)
    original = ops._decode_quantized_linears_fused(linears, x)
    mx.eval(original)
    cache = linears[0]._fused_decode_linears
    linears[1].weight = mx.zeros_like(linears[1].weight)
    expected = linears[1](x)
    actual = ops._decode_quantized_linears_fused(linears, x)
    mx.eval(expected, actual)
    assert linears[0]._fused_decode_linears is not cache
    assert mx.array_equal(actual[1], expected).item()
    assert not mx.array_equal(actual[1], original[1]).item()
    rebuilt = linears[0]._fused_decode_linears
    mx.eval(ops._decode_quantized_linears_fused(linears, x))
    assert linears[0]._fused_decode_linears is rebuilt


def test_unsupported_projection_group_keeps_original_weights():
    linears = projections()[:3]
    weights = tuple(item.weight for item in linears)
    assert ops._decode_quantized_linears_fused(linears, mx.zeros((1, 1, 256))) is None
    assert all(item.weight is weight for item, weight in zip(linears, weights))


def test_batched_decode_and_target_verify_keep_exact_outputs_after_sharing():
    linears = projections()
    x = mx.random.normal((2, 1, 256)).astype(mx.bfloat16)
    prefill = mx.random.normal((2, 64, 256)).astype(mx.bfloat16)
    original = getattr(
        ops._decode_quantized_linears_fused,
        "__wrapped__",
        ops._decode_quantized_linears_fused,
    )
    expected_decode = original(linears, x)
    expected_verify = ops._target_verify_linears(linears, prefill)
    mx.eval(expected_decode, expected_verify)
    actual_decode = ops._decode_quantized_linears_fused(linears, x)
    actual_verify = ops._target_verify_linears(linears, prefill)
    mx.eval(actual_decode, actual_verify)
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(actual_decode, expected_decode)
    )
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(actual_verify, expected_verify)
    )
