from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.bench_ds4_tp_prefill_moe_phase_b import (
    PHASE_B_ABI,
    PHASE_B_SYMBOL,
    SUPERTILE_CANDIDATES,
    Shape,
    analysis_report,
    byte_model,
    output_partitions,
    reference_reduce,
    tile_plan,
    tiled_reduce,
    validate_output_partitions,
)


def test_supertile_sweep_freezes_scratch_and_dispatch_tradeoff():
    shape = Shape()
    expected = {
        128: (1.5, 64),
        256: (3.0, 32),
        512: (6.0, 16),
        1024: (12.0, 8),
        2048: (24.0, 4),
        4096: (48.0, 2),
    }

    for supertile, (scratch_mib, dispatches) in expected.items():
        plan = tile_plan(shape, supertile)
        assert plan.scratch_bytes / 2**20 == scratch_mib
        assert plan.total_dispatches == dispatches
        assert plan.down_weight_read_amplification == 1.0
        assert plan.down_workgroups == 256 * (4096 // 32)
        validate_output_partitions(shape, supertile)


def test_output_partitions_cover_every_weight_row_once_without_overlap():
    shape = Shape()
    for supertile in SUPERTILE_CANDIDATES:
        partitions = output_partitions(shape, supertile)
        columns = [column for part in partitions for column in part]
        assert columns == list(range(shape.hidden))
        assert len(columns) == len(set(columns))


def test_tiled_reduction_is_bit_exact_for_every_candidate_width():
    tokens = 2
    hidden = 12
    # Original route ids are deliberately shuffled into expert-major order.
    sorted_route_ids = (7, 1, 10, 3, 8, 0, 11, 4, 5, 9, 2, 6)
    inverse = [0] * (tokens * 6)
    for sorted_row, route_id in enumerate(sorted_route_ids):
        inverse[route_id] = sorted_row
    down = tuple(
        tuple(
            ((row + 1) * (column - 5.25)) / 17.0
            for column in range(hidden)
        )
        for row in range(tokens * 6)
    )
    scores = (
        (0.51, 0.125, 0.0625, 0.03125, 0.2, 0.07125),
        (0.02, 0.18, 0.3, 0.07, 0.41, 0.02),
    )
    expected = reference_reduce(
        down, inverse, scores, tokens=tokens, hidden=hidden
    )

    for supertile in (1, 2, 3, 4, 6, 12):
        actual = tiled_reduce(
            down,
            inverse,
            scores,
            tokens=tokens,
            hidden=hidden,
            supertile=supertile,
        )
        assert actual == expected


def test_invalid_supertile_is_rejected_before_native_dispatch():
    shape = Shape()
    for value in (0, 96, 768, 8192):
        with pytest.raises(ValueError):
            tile_plan(shape, value)


def test_byte_model_is_a_ceiling_not_a_speed_claim():
    model = byte_model(Shape(), moe_runtime_fraction=0.5)
    assert model["full_route_tensor_mib"] == 48.0
    assert model["current_tail_payload_mib"] == 392.0
    assert model["candidate_tail_payload_mib"] == 104.0
    assert model["tail_payload_speedup_ceiling"] == pytest.approx(392 / 104)
    assert 1.0 < model["phase_b_shape_e2e_ceiling"] < 1.1
    assert model["phase_b_shape_e2e_ceiling"] < model["phase_ab_shape_e2e_ceiling"]

    report = analysis_report(Shape(), moe_runtime_fraction=0.5)
    assert report["candidate_symbol"] == PHASE_B_SYMBOL
    assert report["abi"] == PHASE_B_ABI
    assert PHASE_B_ABI["inputs"][-2:] == ("scores_f32", "supertile_i32")
    assert PHASE_B_ABI["output"] == "local_output_bf16"
    assert report["claims"]["down_weight_read_amplification"] == 1.0
    assert report["claims"]["speed_claimed_without_gpu_gate"] is False
    assert report["exactness"]["sum_order"] == [0, 1, 2, 3, 4, 5]
    assert report["exactness"]["atomics_allowed"] is False


def test_isolated_source_has_no_atomic_or_per_slot_down_dispatch():
    source = (
        Path(__file__).parents[1]
        / "benchmarks/prototypes/ds4_tp_prefill_moe_phase_b.metal"
    ).read_text()

    assert "prototype_ds4_mxfp4_down_tile_f16_bm32_bn32" in source
    assert "prototype_ds4_moe_top6_reduce_tile_bf16" in source
    assert "for (int slot = 0; slot < 6; ++slot)" in source
    assert "route_value * route_score" in source
    assert "total = bfloat16_t(total + weighted)" in source
    down_source = source.split("// Exact current tail", 1)[0]
    assert "scores" not in down_source
    assert "atomic_" not in source
    assert "down_slot" not in source
