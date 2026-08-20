# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 planner accounting: pooled MQA cache — not MLA, not dense.

V4 has ``q_lora_rank`` but no ``kv_lora_rank``, one KV head over a shared
latent, and K-only pools that hold one entry per ``compress_ratio`` tokens.
Before these fixes the planner (a) refused every ``tensor_parallel_size``,
because the replicated single KV head was kept as a divisor constraint and
``1 % N != 0`` for any N > 1, and (b) priced the cache with the dense
1-head K+V formula — ~6-13x over — while also dividing the reservation by
the TP degree even though ``shard()`` splits only query heads and every
rank keeps the pools whole.
"""

import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    _deepseek_v4_kv_bytes_by_layer,
    _deepseek_v4_pooled_kv,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_per_layer,
    _kv_cache_replicated_across_tp,
    _tensor_parallel_divisors,
    plan_hybrid,
)

GIB = 1024**3

V4_PRO = {
    "model_type": "deepseek_v4",
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "index_head_dim": 128,
    "num_hidden_layers": 61,
    "compress_ratios": [0] + [4] * 30 + [128] * 30,
}

DSV32_MLA = {
    "model_type": "deepseek_v32",
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1536,
    "head_dim": 128,
}

DENSE_GQA = {
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


# ── tensor-parallel divisors ─────────────────────────────────────────────────


def test_v4_single_kv_head_is_not_a_tp_divisor():
    assert _tensor_parallel_divisors(V4_PRO) == (128,)


def test_v4_gqa_variant_keeps_kv_constraint():
    config = dict(V4_PRO, num_key_value_heads=8)
    assert _tensor_parallel_divisors(config) == (128, 8)


def test_non_v4_mqa_stays_constrained():
    config = {
        "model_type": "custom_mqa",
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
    }
    assert _tensor_parallel_divisors(config) == (32, 1)


def test_plan_hybrid_accepts_tp2_with_v4_divisors():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(1 * GIB,) * 61,
        tensor_parallel_heads=128,
        tensor_parallel_kv_heads=1,
        tensor_parallel_divisors=_tensor_parallel_divisors(V4_PRO),
        supports_tensor_parallel=True,
    )
    nodes = [
        NodeBudget(node_id=f"n{i}", capacity_bytes=64 * GIB,
                   reserve_bytes=2 * GIB, rank=i)
        for i in range(2)
    ]
    plan = plan_hybrid(model, nodes, tensor_parallel_size=2)
    assert plan.tensor_parallel_size == 2


def test_plan_hybrid_legacy_kv_divisor_rejected_tp2():
    """The pre-fix behavior this exemption removes: (128, 1) refuses TP=2."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(1 * GIB,) * 61,
        tensor_parallel_heads=128,
        tensor_parallel_kv_heads=1,
        supports_tensor_parallel=True,
    )
    assert model.tensor_parallel_divisors == (128, 1)
    nodes = [
        NodeBudget(node_id=f"n{i}", capacity_bytes=64 * GIB,
                   reserve_bytes=2 * GIB, rank=i)
        for i in range(2)
    ]
    with pytest.raises(PlanningError, match="not divisible"):
        plan_hybrid(model, nodes, tensor_parallel_size=2)


# ── pooled-cache detection ───────────────────────────────────────────────────


def test_v4_pooled_shape_detected():
    assert _deepseek_v4_pooled_kv(V4_PRO)


def test_mla_and_dense_shapes_not_pooled():
    assert not _deepseek_v4_pooled_kv(DSV32_MLA)
    assert not _deepseek_v4_pooled_kv(DENSE_GQA)


def test_v4_cache_replicated_across_tp():
    assert _kv_cache_replicated_across_tp(V4_PRO)
    assert _kv_cache_replicated_across_tp(DSV32_MLA)
    assert not _kv_cache_replicated_across_tp(DENSE_GQA)


# ── per-token pricing ────────────────────────────────────────────────────────


def test_v4_kv_bytes_averages_pool_ratios():
    # pooled entry = (512 + 128) * 2 bytes, x2 for PoolingCache's geometric
    # backing capacity = 2560; 30 ratio-4 layers at 640 + 30 ratio-128
    # layers at 20, ratio-0 free: 19800 // 61 = 324 — vs dense 2048.
    assert _kv_bytes_per_token_per_layer(V4_PRO) == 324


def test_v4_kv_bytes_fallback_without_ratios():
    config = {k: v for k, v in V4_PRO.items() if k != "compress_ratios"}
    assert _kv_bytes_per_token_per_layer(config) == 640


def test_v4_kv_bytes_malformed_ratios_fall_back():
    assert (
        _kv_bytes_per_token_per_layer(dict(V4_PRO, compress_ratios=[7] * 61))
        == 640
    )
    assert (
        _kv_bytes_per_token_per_layer(dict(V4_PRO, compress_ratios=[4] * 10))
        == 640
    )


def test_v4_by_layer_table_prices_each_ratio():
    # pooled entry (incl. capacity x2) = 2560 B: ratio-0 layers store only
    # the constant 128-token window (no per-token growth), ratio-4 layers
    # cost 640 B/tok, ratio-128 layers 20 B/tok.
    table = _deepseek_v4_kv_bytes_by_layer(V4_PRO)
    assert len(table) == 61
    assert table[0] == 0
    assert set(table[1:31]) == {640}
    assert set(table[31:]) == {20}


def test_uneven_stage_uses_exact_table_slice_not_the_average():
    """A stage holding only the expensive ratio-4 layers must be priced by
    its own layers: the scalar average (324 B/tok/layer) under-reserves it
    by ~2x, which is exactly the failure the per-layer table exists for."""
    table = _deepseek_v4_kv_bytes_by_layer(V4_PRO)
    model = _v4_layout(
        _kv_bytes_per_token_per_layer(V4_PRO),
        _kv_cache_replicated_across_tp(V4_PRO),
    )
    model = ModelLayout.from_dict(model.to_dict())  # survives the wire format
    assert model.kv_bytes_per_token_by_layer == ()
    import dataclasses

    model = dataclasses.replace(model, kv_bytes_per_token_by_layer=table)
    assert (
        ModelLayout.from_dict(model.to_dict()).kv_bytes_per_token_by_layer
        == table
    )

    tokens = 1000
    ratio4_stage = _kv_bytes_for_stage(model, 30, tokens, start_layer=1)
    ratio128_stage = _kv_bytes_for_stage(model, 30, tokens, start_layer=31)
    assert ratio4_stage == 30 * 640 * tokens
    assert ratio128_stage == 30 * 20 * tokens
    # Without a start_layer the scalar average remains the (whole-model
    # correct, per-stage blunt) fallback.
    assert _kv_bytes_for_stage(model, 30, tokens) == 324 * 30 * tokens
    # Replicated pooled cache: TP must not divide the exact slice either.
    assert (
        _kv_bytes_for_stage(
            model, 30, tokens, tensor_parallel_size=2, start_layer=1
        )
        == ratio4_stage
    )


def test_v4_kv_bytes_requires_head_dim():
    config = {k: v for k, v in V4_PRO.items() if k != "head_dim"}
    assert _kv_bytes_per_token_per_layer(config) == 0


def test_mla_and_dense_pricing_unchanged():
    assert _kv_bytes_per_token_per_layer(DSV32_MLA) == (512 + 64) * 2
    assert _kv_bytes_per_token_per_layer(DENSE_GQA) == 8 * 128 * 2 * 2


# ── end to end: the reservation the mis-price breaks ─────────────────────────


def _v4_sized_nodes():
    # Budget chosen so ~31.5 GiB of sharded weights plus the true pooled
    # reservation (~1.6 GiB replicated) fits, while the dense mis-price
    # (~19 GiB, then halved by the TP split it should not get) does not.
    return [
        NodeBudget(node_id=f"n{i}", capacity_bytes=37 * GIB,
                   reserve_bytes=2 * GIB, rank=i)
        for i in range(2)
    ]


def _v4_layout(kv_per_token_per_layer, kv_replicated):
    return ModelLayout(
        source="test",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(1 * GIB,) * 61,
        tensor_parallel_heads=128,
        tensor_parallel_kv_heads=1,
        tensor_parallel_divisors=_tensor_parallel_divisors(V4_PRO),
        supports_tensor_parallel=True,
        kv_bytes_per_token_per_layer=kv_per_token_per_layer,
        kv_replicated_across_tp=kv_replicated,
    )


def test_v4_pooled_reservation_fits_at_full_context():
    model = _v4_layout(
        _kv_bytes_per_token_per_layer(V4_PRO),
        _kv_cache_replicated_across_tp(V4_PRO),
    )
    plan = plan_hybrid(
        model, _v4_sized_nodes(), tensor_parallel_size=2,
        context_tokens=163840,
    )
    assert plan.tensor_parallel_size == 2


def test_v4_dense_mispricing_rejected_same_budget():
    model = _v4_layout(1 * 512 * 2 * 2, False)  # the pre-fix accounting
    with pytest.raises(PlanningError):
        plan_hybrid(
            model, _v4_sized_nodes(), tensor_parallel_size=2,
            context_tokens=163840,
        )
