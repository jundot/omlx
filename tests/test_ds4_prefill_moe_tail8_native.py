"""ABI guards for isolated DS4 M=1024 BM8 route-tail symbols."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast


PAIR = "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8"
DOWN = "deepseek_mxfp4_gather_qmm_blocks_tail8"


def test_fast_registry_and_wrappers_expose_tail8_symbols():
    assert PAIR in fast.NATIVE_SYMBOLS
    assert DOWN in fast.NATIVE_SYMBOLS
    assert callable(getattr(fast, PAIR))
    assert callable(getattr(fast, DOWN))


def test_tail8_native_source_is_fixed_shape_and_production_default_off():
    root = Path(__file__).parents[1]
    cpp = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.cpp"
    ).read_text()
    metal = (
        root
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe_tail8.metal"
    ).read_text()
    switch = (root / "omlx/patches/deepseek_v4/switch_layers.py").read_text()

    for contract in (
        "kRoutes = 6144",
        "kExperts = 256",
        "kHidden = 4096",
        "kIntermediate = 1024",
        "kVariant = 2",
    ):
        assert contract in cpp
    assert "MICRO_BM = 8" in metal
    assert "if (rows > 24)" in metal
    assert '"OMLX_DSV4_MOE_TAIL8", "0"' in switch
    assert '"OMLX_DSV4_COMBINED_MOE_PREFILL", "0"' in switch
    assert PAIR in switch
    assert DOWN in switch


@pytest.mark.skipif(
    not fast.has_symbol(PAIR) or not fast.has_symbol(DOWN),
    reason="glm_moe_dsa extension predates isolated tail8 symbols",
)
def test_native_tail8_symbols_reject_non_contract_shapes_before_gpu_work():
    x = mx.zeros((1, 1, 32), dtype=mx.float16)
    weight = mx.zeros((1, 1, 4), dtype=mx.uint32)
    scales = mx.zeros((1, 1, 1), dtype=mx.uint8)
    block_meta = mx.zeros((1, 3), dtype=mx.int32)
    block_count = mx.zeros((1,), dtype=mx.int32)

    with pytest.raises(ValueError, match="fixed M=1024"):
        fast.deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
            x,
            weight,
            scales,
            weight,
            scales,
            block_meta,
            block_count,
        )
    with pytest.raises(ValueError, match="isolated symbol requires"):
        fast.deepseek_mxfp4_gather_qmm_blocks_tail8(
            x,
            weight,
            scales,
            block_meta,
            block_count,
        )


def test_recorded_m3_gate_is_exact_and_clears_composed_speed_threshold():
    results = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experimental/ds4_tp_prefill_moe_tail8_results_2026-08-22.json"
        ).read_text()
    )
    assert results["gate"]["all_boundaries_array_equal"] is True
    assert results["gate"]["both_runs_passed"] is True
    assert all(run["composed"]["speedup"] >= 1.05 for run in results["runs"])
    assert results["decision"]["production_dispatch_default"] == "off"
    assert results["decision"]["m5_binary_synced"] is False
