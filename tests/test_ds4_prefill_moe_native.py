"""Focused ABI/shape guards for the isolated DS4 phase-A native symbol."""

from pathlib import Path

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast


SYMBOL = "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks"


def test_fast_wrapper_and_native_symbol_registry_include_phase_a():
    assert callable(getattr(fast, SYMBOL))
    assert SYMBOL in fast.NATIVE_SYMBOLS


def test_cpp_contract_is_fixed_to_measured_shape():
    source = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.cpp"
    ).read_text()
    for contract in (
        "kRoutes = 6144",
        "kExperts = 256",
        "kHidden = 4096",
        "kIntermediate = 1024",
        "kBM = 32",
        "kBN = 32",
        "kBK = 32",
        "kVariant = 2",
        "kActivationLimit = 10.0f",
    ):
        assert contract in source


@pytest.mark.skipif(
    not fast.has_symbol(SYMBOL),
    reason="glm_moe_dsa extension predates isolated DS4 phase-A symbol",
)
def test_native_symbol_rejects_non_contract_shape_before_gpu_work():
    x = mx.zeros((1, 1, 32), dtype=mx.float16)
    weight = mx.zeros((1, 1, 4), dtype=mx.uint32)
    scales = mx.zeros((1, 1, 1), dtype=mx.uint8)
    block_meta = mx.zeros((1, 3), dtype=mx.int32)
    block_count = mx.zeros((1,), dtype=mx.int32)
    with pytest.raises(ValueError, match="isolated symbol requires"):
        fast.deepseek_mxfp4_gather_qmm_pair_swiglu_blocks(
            x,
            weight,
            scales,
            weight,
            scales,
            block_meta,
            block_count,
            10.0,
            2,
        )
