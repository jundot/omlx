from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.bench_ds4_tp_prefill_moe_tail8 import (
    MEASURED_M3_PAIR_CONCAT_MS,
    MEASURED_M3_SHARED_X_MS,
    Shape,
    analysis_report,
    current_mma_rows,
    persistent_fusion_blocker,
    poisson_padding,
    projected_time_ms,
    required_mma_fraction,
    tail8_mma_rows,
)


def test_tail8_skips_padding_without_adding_expert_blocks_or_weight_reads():
    assert current_mma_rows(24) == 32
    assert tail8_mma_rows(24) == 24
    assert current_mma_rows(40) == 64
    assert tail8_mma_rows(40) == 40

    report = analysis_report(Shape())
    fixture = report["uniform_24_route_fixture"]
    assert fixture["current_mma_rows"] == 8192
    assert fixture["tail8_mma_rows"] == 6144
    assert fixture["mma_row_reduction"] == 0.25
    assert fixture["block_count_change"] == 0
    assert fixture["weight_read_amplification"] == 1.0


def test_poisson_mean24_still_has_material_tail_headroom():
    model = poisson_padding(24.0)
    assert model["expected_routes"] == pytest.approx(24.0)
    assert model["mma_row_reduction"] == pytest.approx(0.1790211, rel=1e-5)
    assert model["mma_only_speedup"] == pytest.approx(1.2180581, rel=1e-5)


def test_measured_phase_a_break_even_is_quantified_not_assumed():
    uniform_reduction = 0.25
    break_even = required_mma_fraction(
        MEASURED_M3_SHARED_X_MS,
        MEASURED_M3_PAIR_CONCAT_MS,
        uniform_reduction,
    )
    promotion = required_mma_fraction(
        MEASURED_M3_SHARED_X_MS,
        MEASURED_M3_PAIR_CONCAT_MS / 1.05,
        uniform_reduction,
    )
    assert break_even == pytest.approx(0.093574, rel=1e-4)
    assert promotion == pytest.approx(0.279594, rel=1e-4)
    assert projected_time_ms(MEASURED_M3_SHARED_X_MS, promotion, 0.25) == (
        pytest.approx(MEASURED_M3_PAIR_CONCAT_MS / 1.05)
    )


def test_single_fused_no_mid_path_is_blocked_by_real_m3_threadgroup_limit():
    blocker = persistent_fusion_blocker(Shape())
    assert blocker["m3_max_threadgroup_bytes"] == 32768
    by_bm = {row["bm"]: row for row in blocker["candidates"]}
    assert by_bm[16]["mid_bytes"] == 32768
    assert by_bm[16]["fits_32k"] is False
    assert by_bm[24]["fits_32k"] is False
    assert by_bm[32]["fits_32k"] is False
    assert by_bm[8]["fits_32k"] is True
    assert by_bm[8]["weight_read_amplification_at_24_routes"] == 3


def test_prototype_is_microtiled_exact_order_and_not_build_wired():
    root = Path(__file__).parents[1]
    source_path = root / "benchmarks/prototypes/ds4_tp_prefill_moe_tail8.metal"
    source = source_path.read_text()
    cmake = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/CMakeLists.txt"
    ).read_text()

    assert "MICRO_BM = 8" in source
    assert "if (rows > 16)" in source
    assert "if (rows > 24)" in source
    assert "loader_up.load_unsafe();" in source
    assert source.count("loader_up.load_unsafe();") == 1
    assert "Sigmoid{}(gate)" in source
    assert "prototype_ds4_mxfp4_down_tail8" in source
    assert str(source_path.relative_to(root)) not in cmake


def test_analysis_refuses_to_claim_unmeasured_speed():
    report = analysis_report(Shape())
    assert report["live_gpu_benchmark_safe"] is False
    assert report["speed_claimed"] is False
