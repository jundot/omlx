from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest

from benchmarks.bench_ds4_tp_prefill_moe_campaign import (
    ExpertBlock,
    Shape,
    analysis_report,
    bf16,
    bf16_bits,
    deterministic_expert_map,
    materialization_ledger,
    reference_top6_scalar,
    roofline,
    sorted_route_top6_scalar,
    synthetic_routes,
    validate_expert_map,
)


def test_expert_map_is_stable_prefix_addressed_and_complete():
    routes = (2, 0, 2, 1, 0, 2, 1, 0)
    order, blocks = deterministic_expert_map(routes, experts=4, bm=2)

    assert order == (1, 4, 7, 3, 6, 0, 2, 5)
    assert blocks == (
        ExpertBlock(0, 0, 2),
        ExpertBlock(2, 0, 1),
        ExpertBlock(3, 1, 2),
        ExpertBlock(5, 2, 2),
        ExpertBlock(7, 2, 1),
    )
    validate_expert_map(routes, order, blocks, experts=4, bm=2)


def test_expert_map_rejects_atomic_completion_order():
    routes = (0, 1, 0, 1)
    order, blocks = deterministic_expert_map(routes, experts=2, bm=2)

    with pytest.raises(ValueError, match="stable"):
        validate_expert_map(
            routes,
            (2, 0, 1, 3),
            blocks,
            experts=2,
            bm=2,
        )


def test_m1024_contract_has_one_full_bm32_block_per_expert():
    shape = Shape()
    routes = synthetic_routes(shape)
    order, blocks = deterministic_expert_map(routes, shape.experts, shape.bm)

    assert len(order) == 6144
    assert len(blocks) == 256
    assert {block.rows for block in blocks} == {24}
    assert tuple(block.expert for block in blocks) == tuple(range(256))


def test_sorted_down_reduction_restores_original_top6_slot_order():
    values = (0.101, -3.75, 12.125, 0.007, 2.5, -0.333)
    scores = (0.51, 0.125, 0.0625, 0.03125, 0.2, 0.07125)
    expected = reference_top6_scalar(values, scores)

    for order in itertools.permutations(range(6)):
        assert sorted_route_top6_scalar(order, values, scores) == expected


def test_bfloat16_slot_order_is_not_replaceable_by_unordered_atomics():
    # A deliberately cancellation-heavy row. BF16 additions in a different
    # completion order do not preserve the current MLX slot-order result.
    weighted = (256.0, 1.0, -256.0, 0.5, 0.5, 0.5)
    slot_order = 0.0
    for value in weighted:
        slot_order = bf16(slot_order + bf16(value))
    completion_order = 0.0
    for index in (0, 2, 1, 3, 4, 5):
        completion_order = bf16(completion_order + bf16(weighted[index]))

    assert bf16_bits(slot_order) != bf16_bits(completion_order)


def test_materialization_and_roofline_contract_for_tp4_4_m1024():
    ledger = materialization_ledger(Shape())
    assert ledger["current_mib"]["sorted_route_input"] == 48.0
    assert ledger["current_mib"]["gate_up_pair"] == 24.0
    assert ledger["current_mib"]["activated_mid"] == 12.0
    assert ledger["current_mib"]["sorted_down_routes"] == 48.0
    assert ledger["phase_ab_target_mib"]["bounded_down_tile_scratch"] == 0.375
    assert ledger["phase_ab_removed_core_persistent_mib"] == 84.0

    model = roofline(Shape(), moe_runtime_fraction=0.5)
    assert model["phase_a_infinite_e2e_ceiling"] == pytest.approx(1.5)
    assert model["phase_ab_infinite_e2e_ceiling"] == pytest.approx(2.0)
    assert 1.0 < model["phase_a_shape_e2e_ceiling"] < 1.5
    assert model["phase_a_shape_e2e_ceiling"] < model["phase_ab_shape_e2e_ceiling"]
    assert model["phase_ab_shape_e2e_ceiling"] < 2.0

    report = analysis_report(Shape(), moe_runtime_fraction=0.5)
    assert report["phase_b_contract"]["route_score_stage"] == (
        "after_fp16_down_store_and_bf16_cast"
    )
    assert report["phase_b_contract"]["active_experts_per_slot"] == [256] * 6
    assert report["phase_b_contract"][
        "per_slot_down_weight_read_amplification"
    ] == 6.0


def test_isolated_metal_prototype_keeps_shared_x_and_ordered_non_atomic_tail():
    prototype = (
        Path(__file__).parents[1]
        / "benchmarks/prototypes/ds4_tp_prefill_moe_phase_a.metal"
    ).read_text()
    native = (
        Path(__file__).parents[1]
        / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.metal"
    ).read_text()

    assert "prototype_ds4_mxfp4_pair_swiglu_f16_bm32_bn32" in prototype
    assert "prototype_ds4_moe_top6_post_down_bf16" in prototype
    assert "for (int slot = 0; slot < 6; ++slot)" in prototype
    assert "route_value * route_score" in prototype
    assert "atomic_" not in prototype

    assert "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks" in native
    assert native.count("loader_x.load_safe(x_dims);") == 1
    assert "mma_up.mma(Xs, Wup);" in native
    assert "mma_gate.mma(Xs, Wgate);" in native
    assert "Sigmoid{}(gate)" in native
    assert "T(T(gate * sigmoid_gate) * up)" in native
    assert "atomic_" not in native


def _python_references_exact_symbol(source: str, symbol: str) -> bool:
    tree = ast.parse(source)
    return any(
        (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == symbol
        )
        or (isinstance(node, ast.Name) and node.id == symbol)
        or (isinstance(node, ast.Attribute) and node.attr == symbol)
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("fast.has_symbol('rejected_kernel')", True),
        ("fast.rejected_kernel(x)", True),
        ('fast.has_symbol("rejected_kernel_tail8")', False),
    ),
)
def test_exact_native_symbol_reference_ignores_longer_qualified_names(
    source,
    expected,
):
    assert _python_references_exact_symbol(source, "rejected_kernel") is expected


def test_phase_a_native_symbol_is_isolated_from_production_dispatch():
    root = Path(__file__).parents[1]
    # Match the exact Phase-A symbol, independent of quote style or whether a
    # caller accesses the binding as a Python attribute. Do not confuse it
    # with the separately qualified ``...blocks_tail8`` production symbol.
    symbol = "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks"
    allowed = {
        root / "omlx/custom_kernels/glm_moe_dsa/fast.py",
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/bindings.cpp",
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.cpp",
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.h",
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.metal",
    }
    production_hits = []
    for path in (root / "omlx").rglob("*.py"):
        if path in allowed:
            continue
        if _python_references_exact_symbol(path.read_text(), symbol):
            production_hits.append(path)
    assert production_hits == []


def test_recorded_two_host_phase_a_gate_is_rejected():
    path = (
        Path(__file__).parents[1]
        / "docs/experimental/ds4_tp_prefill_moe_phase_a_results_2026-08-22.json"
    )
    report = json.loads(path.read_text())
    m3 = report["hosts"]["m3_ultra_rank0"]
    m5 = report["hosts"]["m5_max_rank1"]

    assert m3["candidate_vs_pair_concat_exact"] is True
    assert m5["candidate_vs_pair_concat_exact"] is True
    assert m3["speedup_vs_faster_baseline"] < 1.05
    assert m5["speedup_vs_faster_baseline"] < 1.05
    assert m5["pair_concat_vs_stock_exact"] is False
    assert report["outcome"]["promoted"] is False
    assert report["outcome"]["production_dispatch"] is False
