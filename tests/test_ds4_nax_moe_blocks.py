"""Contract tests for the isolated expert-blocked M5 NAX MoE primitive."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from benchmarks.bench_ds4_nax_moe_blocks import (
    MIN_COMPOSED_SPEEDUP,
    PAIR_SYMBOL,
    SYMBOL,
    _shard_weights,
    route_fixture,
    structural_work_report,
)
from omlx.custom_kernels.glm_moe_dsa import fast


def test_uniform_fixture_exposes_stock_global_tile_recomputation():
    report = structural_work_report()
    assert report["routes"] == 6144
    assert report["active_experts"] == 256
    assert report["min_routes_per_active_expert"] == 24
    assert report["max_routes_per_expert"] == 24
    assert report["stock_global_tiles"] == 96
    assert report["stock_expert_segments"] == 320
    assert report["stock_row_equivalents"] == 20480
    assert report["candidate_expert_blocks"] == 256
    assert report["candidate_row_equivalents"] == 8192
    assert report["row_work_ratio"] == 2.5
    assert report["weight_segment_ratio"] == 1.25


@pytest.mark.parametrize(
    ("name", "active", "minimum", "maximum", "blocks"),
    (
        ("ragged", 255, 1, 33, 256),
        ("skewed", 256, 22, 512, 271),
        ("max-blocks", 256, 1, 5889, 440),
    ),
)
def test_boundary_and_skew_route_fixtures_stay_inside_fixed_block_abi(
    name, active, minimum, maximum, blocks
):
    routes = route_fixture(name)
    report = structural_work_report(routes)
    assert len(routes) == 6144
    assert report["active_experts"] == active
    assert report["min_routes_per_active_expert"] == minimum
    assert report["max_routes_per_expert"] == maximum
    assert report["candidate_expert_blocks"] == blocks
    assert blocks <= 448


def test_gate_requires_lossless_1_45x_composed_speedup():
    assert MIN_COMPOSED_SPEEDUP == 1.45


def test_benchmark_accepts_explicit_equal_or_asymmetric_tp_vectors():
    assert _shard_weights("3,5") == (3, 5)
    assert _shard_weights("4,4") == (4, 4)
    with pytest.raises(ValueError, match="two positive"):
        _shard_weights("8")


def test_native_source_keeps_stock_nax_simd_geometry_and_bf16_boundary():
    root = Path(__file__).parents[1]
    metal = (
        root
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe_nax.metal"
    ).read_text()
    cpp = (
        root
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe_nax.cpp"
    ).read_text()
    cmake = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/CMakeLists.txt"
    ).read_text()
    switch = (
        root / "omlx/patches/deepseek_v4/switch_layers.py"
    ).read_text()

    assert "QuantizedBlockLoader<" in metal
    assert "tile_matmad_nax(" in metal
    assert "NAXTile<float, TM, TN>" in metal
    assert "bfloat16_t, 32, 64, 64, 1, 2" in metal
    assert "kRoutes = 6144" in cpp
    assert "kExperts = 256" in cpp
    assert "kMaxBlocks = 448" in cpp
    assert "N == 1024 || N == 1280" in cpp
    assert "x.dtype() != bfloat16" in cpp
    assert "ds4_prefill_moe_nax.metal" in cmake
    assert '"OMLX_DSV4_NAX_MOE_BLOCKS", "0"' in switch
    assert "_can_use_mxfp4_nax_blocks_prefill" in switch
    assert SYMBOL in switch


def test_fast_registry_exposes_isolated_symbol():
    assert SYMBOL in fast.NATIVE_SYMBOLS
    assert callable(getattr(fast, SYMBOL))
    assert PAIR_SYMBOL in fast.NATIVE_SYMBOLS
    assert callable(getattr(fast, PAIR_SYMBOL))


@pytest.mark.skipif(
    not fast.has_symbol(SYMBOL),
    reason="glm_moe_dsa extension predates the isolated NAX MoE symbol",
)
def test_native_symbol_rejects_non_contract_shape_before_gpu_work():
    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)
    weight = mx.zeros((1, 1, 4), dtype=mx.uint32)
    scales = mx.zeros((1, 1, 1), dtype=mx.uint8)
    block_meta = mx.zeros((1, 3), dtype=mx.int32)
    block_count = mx.zeros((1,), dtype=mx.int32)
    with pytest.raises(ValueError, match="isolated symbol requires BF16"):
        fast.deepseek_mxfp4_gather_qmm_blocks_nax(
            x, weight, scales, block_meta, block_count
        )


@pytest.mark.skipif(
    not fast.has_symbol(PAIR_SYMBOL),
    reason="glm_moe_dsa extension predates the paired NAX MoE symbol",
)
def test_paired_native_symbol_rejects_non_contract_shape_before_gpu_work():
    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)
    weight = mx.zeros((1, 1, 4), dtype=mx.uint32)
    scales = mx.zeros((1, 1, 1), dtype=mx.uint8)
    block_meta = mx.zeros((1, 3), dtype=mx.int32)
    block_count = mx.zeros((1,), dtype=mx.int32)
    with pytest.raises(ValueError, match="requires BF16"):
        fast.deepseek_mxfp4_gather_qmm_pair_blocks_nax(
            x,
            weight,
            scales,
            weight,
            scales,
            block_meta,
            block_count,
        )


def test_recorded_physical_m5_gate_is_exact_and_projects_over_ten_percent():
    root = Path(__file__).parents[1]
    result = json.loads(
        (
            root
            / "docs/experimental/ds4_nax_moe_blocks_m5_2026-08-22.json"
        ).read_text()
    )
    profile = json.loads(
        (
            root
            / "docs/experimental/"
            "ds4_tp_stage_profile_3x5_m5_physical_2026-08-22.json"
        ).read_text()
    )
    assert result["device"]["device_name"] == "Apple M5 Max"
    assert result["all_boundaries_array_equal"] is True
    assert result["gate"]["passed"] is True
    assert result["gate"]["composed_speedup"] >= MIN_COMPOSED_SPEEDUP
    assert all(item["mismatches"] == 0 for item in result["parity"].values())

    fractions = profile["observed_attribution"]["fractions"]
    pair_fraction = fractions["routed_moe_pair"]
    down_fraction = fractions["routed_moe_down"]
    pair_speedup = result["timings"]["pair_plus_limited_swiglu"]["speedup"]
    down_speedup = result["timings"]["down_fixed_input"]["speedup"]
    projected_speedup = 1 / (
        1
        - pair_fraction
        - down_fraction
        + pair_fraction / pair_speedup
        + down_fraction / down_speedup
    )
    assert profile["device"]["device_name"] == "Apple M5 Max"
    assert pair_fraction + down_fraction > 0.38
    assert projected_speedup > 1.10
