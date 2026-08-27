"""ABI and isolation guards for the fixed DS4 ratio-4 B1 bundle."""

from pathlib import Path

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast

SYMBOL = "deepseek_v4_qkv_compressor_bundle_b1"
PAIR_SYMBOL = "deepseek_v4_qkv_pair_b1"
RATIO128_SYMBOL = "deepseek_v4_qkv_compressor128_bundle_b1"


def test_native_registry_and_wrapper_expose_bundle_only_explicitly():
    assert SYMBOL in fast.NATIVE_SYMBOLS
    assert callable(fast.deepseek_v4_qkv_compressor_bundle_b1)
    assert PAIR_SYMBOL in fast.NATIVE_SYMBOLS
    assert RATIO128_SYMBOL in fast.NATIVE_SYMBOLS


def test_native_source_reports_one_heterogeneous_dispatch_and_fixed_slices():
    source = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_qkv_bundle.cpp"
    ).read_text()
    metal = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_qkv_bundle.metal"
    ).read_text()
    decode_metal = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_qkv_bundle_decode.metal"
    ).read_text()
    decode128_metal = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_qkv_bundle128_decode.metal"
    ).read_text()
    assert "kDispatches = 1" in source
    assert "kPackedRows = 4096" in source
    # Three schedule-specific primitives, each with one dispatch.
    assert source.count("dispatch_threadgroups") == 3
    assert "ds4_qkv_bundle_all_b1" not in metal
    assert "ds4_qkv_bundle_all_b1" in decode_metal
    assert "virtual_group = group_id * 2 + cohort" in decode_metal
    assert "ds4_qkv_bundle128_all_b1" in decode128_metal
    assert "packed_dense_offset = 1536" in metal
    assert "group_row < 1024" in metal
    assert "group_row < 2048" in metal
    assert "group_row < 2304" in metal


@pytest.mark.skipif(
    not fast.has_symbol(SYMBOL),
    reason="glm_moe_dsa extension predates isolated DS4 B1 bundle",
)
def test_native_bundle_rejects_non_contract_shapes_before_gpu_work():
    x = mx.zeros((1, 32), dtype=mx.bfloat16)
    qweight = mx.zeros((1, 8), dtype=mx.uint32)
    qscale = mx.zeros((1, 1), dtype=mx.uint8)
    dense = mx.zeros((1, 32), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="exact ratio-4 B1 contract"):
        fast.deepseek_v4_qkv_compressor_bundle_b1(
            x,
            qweight,
            qscale,
            qweight,
            qscale,
            dense,
            dense,
            dense,
            dense,
        )
