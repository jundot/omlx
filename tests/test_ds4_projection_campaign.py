from __future__ import annotations

from pathlib import Path

from benchmarks.bench_ds4_projection_campaign import (
    analysis_report,
    projection_schema,
    rank_shape,
)


def test_asymmetric_projection_shapes_match_signed_3x5_plan():
    rank0 = rank_shape(0)
    rank1 = rank_shape(1)
    assert (rank0.local_heads, rank0.q_b_rows, rank0.o_a_input) == (
        24,
        12288,
        1536,
    )
    assert (rank1.local_heads, rank1.q_b_rows, rank1.o_a_input) == (
        40,
        20480,
        2560,
    )


def test_ratio4_schema_includes_row_local_indexer_projections():
    schema = projection_schema(4, rank_shape(0))
    assert schema["q_b"]["n"] == 12288
    assert schema["o_a"]["batches"] == 8
    assert schema["index_q_b"]["m"] == 512
    assert schema["index_q_b"]["n"] == 8192
    assert schema["index_weights"]["n"] == 64


def test_gate_requires_exact_1_30x_and_no_production_dispatch():
    report = analysis_report()
    assert report["gates"] == {
        "projection_bucket_min_speedup": 1.30,
        "all_projection_boundaries": "mx.array_equal",
        "production_dispatch": False,
    }
    assert report["m5_gate_up_concat"]["duplicate_steady_state_bytes"] == 0


def test_classic_and_nax_sweeps_include_stock_and_nine_alternatives():
    root = Path(__file__).parents[1]
    classic = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_projection_qmm.metal"
    ).read_text()
    nax = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_projection_qmm_nax.metal"
    ).read_text()
    cpp = (
        root / "omlx/custom_kernels/glm_moe_dsa/csrc/ds4_projection_qmm.cpp"
    ).read_text()

    assert classic.count("  instantiate_ds4_projection_mxfp8(type,") == 10
    assert nax.count("instantiate_ds4_projection_mxfp8_nax(bfloat16_t,") == 10
    assert "return {32, 32, 32};" in cpp
    assert "return {64, 64, 64, 2, 2};" in cpp


def test_symbol_is_built_and_only_reached_through_the_narrow_default_off_gate():
    root = Path(__file__).parents[1]
    cmake = (root / "omlx/custom_kernels/glm_moe_dsa/csrc/CMakeLists.txt").read_text()
    model = (root / "omlx/patches/deepseek_v4/deepseek_v4_model.py").read_text()
    switch = (root / "omlx/patches/deepseek_v4/switch_layers.py").read_text()

    assert "ds4_projection_qmm.metal" in cmake
    assert "ds4_projection_qmm_nax.metal" in cmake
    assert '"OMLX_DSV4_NAX_OA_PREFILL", "0"' in model
    assert "_can_use_nax_oa_prefill" in model
    assert "ds4_projection_mxfp8_qmm" in model
    assert "ds4_projection_mxfp8_qmm" not in switch
