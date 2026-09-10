# SPDX-License-Identifier: Apache-2.0
"""A plan must fit once the KV cache is full, not just once weights are loaded.

Reserving only weight bytes is how a stage that "fits" dies on the first long
prompt — the failure that took a 128 GiB MacBook down mid-session.
"""

import pytest

from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    _kv_bytes_per_token_per_layer,
    _kv_cache_replicated_across_tp,
    plan_hybrid,
    plan_unequal_pipeline,
)

GIB = 1024**3


def _model(layers=32, layer_gib=2, kv_per_token_per_layer=0, heads=48,
           kv_replicated=False):
    return ModelLayout(
        source="test",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(layer_gib * GIB,) * layers,
        tensor_parallel_heads=heads,
        supports_tensor_parallel=True,
        kv_bytes_per_token_per_layer=kv_per_token_per_layer,
        kv_replicated_across_tp=kv_replicated,
    )


def _nodes(count, capacity_gib=64):
    return [
        NodeBudget(node_id=f"n{i}", capacity_bytes=capacity_gib * GIB,
                   reserve_bytes=2 * GIB, rank=i)
        for i in range(count)
    ]


def test_standard_attention_kv_is_two_tensors_per_head():
    """num_kv_heads * head_dim * 2 (K and V) * 2 bytes."""

    config = {"num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256}
    assert _kv_bytes_per_token_per_layer(config) == 4 * 256 * 2 * 2


def test_head_dim_is_derived_when_absent():
    config = {"num_attention_heads": 8, "hidden_size": 4096}
    assert _kv_bytes_per_token_per_layer(config) == 8 * 512 * 2 * 2


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "hidden_size": 8192,
            },
            8 * 128 * 2 * 2,
        ),
        (
            {
                "num_attention_heads": 40,
                "hidden_size": 5120,
            },
            40 * 128 * 2 * 2,
        ),
    ],
    ids=("qwen2.5-72b", "llama-13b"),
)
def test_large_hidden_sizes_without_head_dim_still_reserve_kv(config, expected):
    """Real model widths above the count-field ceiling must not become zero KV."""

    assert _kv_bytes_per_token_per_layer(config) == expected


def test_non_divisible_hidden_size_is_not_rounded_down():
    config = {"num_attention_heads": 8, "hidden_size": 4097}
    assert _kv_bytes_per_token_per_layer(config) == 0


def test_mla_models_are_not_over_counted():
    """GLM/DeepSeek store a latent key plus RoPE under one head.

    The uniform formula over-counts these by more than an order of magnitude,
    which would refuse plans that fit comfortably.
    """

    mla = {"kv_lora_rank": 512, "qk_rope_head_dim": 64,
           "num_attention_heads": 64, "num_key_value_heads": 64, "head_dim": 128}
    uniform = {"num_attention_heads": 64, "num_key_value_heads": 64, "head_dim": 128}

    assert _kv_bytes_per_token_per_layer(mla) == (512 + 64) * 2
    assert _kv_bytes_per_token_per_layer(mla) < _kv_bytes_per_token_per_layer(uniform) / 10
    assert _kv_cache_replicated_across_tp(mla) is True
    assert _kv_cache_replicated_across_tp(uniform) is False


def test_glm5_next_nope_hybrid_uses_sparse_layer_average():
    text_config = {
        "num_hidden_layers": 4,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 0,
        "index_head_dim": 128,
        "index_kpool": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
    }
    config = {"model_type": "glm5_next", "text_config": text_config}

    # One of four layers grows with context: (512 + 128 / 4) * fp16 / 4.
    assert _kv_bytes_per_token_per_layer(config) == 272
    assert _kv_cache_replicated_across_tp(config) is True


def test_an_unreadable_config_reserves_nothing_rather_than_guessing():
    assert _kv_bytes_per_token_per_layer({}) == 0
    assert _kv_bytes_per_token_per_layer({"num_attention_heads": 8}) == 0


def test_kv_is_counted_as_resident_memory():
    model = _model(layers=16, layer_gib=1, kv_per_token_per_layer=128 * 1024)
    plan = plan_unequal_pipeline(model, _nodes(2), context_tokens=8192)

    for a in plan.assignments:
        assert a.kv_cache_bytes > 0
        assert a.planned_weight_bytes == (
            a.fixed_weight_bytes + a.layer_weight_bytes + a.kv_cache_bytes
        )


def test_a_plan_that_fits_weights_but_not_kv_is_refused():
    """The exact failure mode: weights fit, cache does not."""

    model = _model(layers=16, layer_gib=3, kv_per_token_per_layer=256 * 1024)

    # No context: weights alone fit.
    assert plan_unequal_pipeline(model, _nodes(2, capacity_gib=32), context_tokens=0)

    # Same plan with a real context no longer fits, and says why.
    with pytest.raises(PlanningError, match="KV cache"):
        plan_unequal_pipeline(model, _nodes(2, capacity_gib=32), context_tokens=32768)


def test_longer_context_reserves_proportionally_more():
    model = _model(layers=16, layer_gib=1, kv_per_token_per_layer=64 * 1024)
    short = plan_unequal_pipeline(model, _nodes(2), context_tokens=4096)
    long = plan_unequal_pipeline(model, _nodes(2), context_tokens=16384)

    assert sum(a.kv_cache_bytes for a in long.assignments) == 4 * sum(
        a.kv_cache_bytes for a in short.assignments
    )
    assert short.target_context_tokens == 4096
    assert long.target_context_tokens == 16384
    assert long.to_dict()["cluster"]["target_context_tokens"] == 16384
    assert short.plan_hash != long.plan_hash


def test_a_node_reserves_only_for_the_layers_it_holds():
    """KV is per layer, so an unequal split reserves unequally."""

    model = _model(layers=32, layer_gib=1, kv_per_token_per_layer=64 * 1024)
    nodes = [
        NodeBudget(node_id="small", capacity_bytes=20 * GIB, reserve_bytes=2 * GIB, rank=0),
        NodeBudget(node_id="big", capacity_bytes=90 * GIB, reserve_bytes=2 * GIB, rank=1),
    ]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=8192)
    by_id = {a.node_id: a for a in plan.assignments}

    assert by_id["small"].layer_count < by_id["big"].layer_count
    assert by_id["small"].kv_cache_bytes < by_id["big"].kv_cache_bytes


def test_performance_rebalancing_never_moves_kv_beyond_a_nodes_memory():
    """A faster rank may receive more layers only while their cache still fits.

    This is the MiniMax 256k regression: the safe preview put 11 layers on the
    MacBook, then measured performance moved 19 there using weights alone and
    activation failed even though a valid split existed.
    """

    def profile(node_id, rank, rate):
        return NodePerformanceProfile(
            node_id=node_id,
            rank=rank,
            decode_weight_bytes_per_second=rate,
            prefill_weight_bytes_per_second=rate,
            collective_latency_seconds=0.001,
            collective_bandwidth_bytes_per_second=10_000,
            backend="ring",
            measured_at="2026-07-30T00:00:00+00:00",
            samples=5,
        )

    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
        kv_bytes_per_token_per_layer=10,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "slow",
                60,
                rank=0,
                performance=profile("slow", 0, 10),
            ),
            NodeBudget(
                "fast",
                100,
                rank=1,
                performance=profile("fast", 1, 40),
            ),
        ],
        context_tokens=1,
    )

    by_id = {assignment.node_id: assignment for assignment in plan.assignments}
    assert by_id["fast"].layer_count == 5
    assert by_id["slow"].layer_count == 3
    assert all(assignment.headroom_bytes >= 0 for assignment in plan.assignments)


def test_tensor_and_pipeline_reserve_the_same_kv_per_node():
    """Neither split saves KV — they divide the same cache along different axes.

    Under pipeline the node holds half the layers at full head width; under
    tensor parallelism it holds every layer at half the heads. Same bytes. Worth
    pinning, because it means "switch to tensor parallelism" is never a fix for
    a KV-bound plan — only more Macs or less context is.
    """

    model = _model(layers=32, layer_gib=1, kv_per_token_per_layer=64 * 1024, heads=48)
    pipelined = plan_hybrid(model, _nodes(2, 90), tensor_parallel_size=1, context_tokens=8192)
    tensored = plan_hybrid(model, _nodes(2, 90), tensor_parallel_size=2, context_tokens=8192)

    assert max(a.kv_cache_bytes for a in tensored.assignments) == max(
        a.kv_cache_bytes for a in pipelined.assignments
    )


def test_an_mla_cache_is_reserved_whole_on_every_tp_member():
    """The latent cache is not per-head: sharding divides heads, not it.

    Under pipeline each node holds half the layers' caches. Under TP each
    member holds every layer's cache whole — twice the pipeline reservation,
    where standard attention reserves the same bytes either way.
    """

    model = _model(layers=32, layer_gib=1, kv_per_token_per_layer=1152,
                   heads=64, kv_replicated=True)
    pipelined = plan_hybrid(model, _nodes(2, 90), tensor_parallel_size=1,
                            context_tokens=8192)
    tensored = plan_hybrid(model, _nodes(2, 90), tensor_parallel_size=2,
                           context_tokens=8192)

    assert max(a.kv_cache_bytes for a in tensored.assignments) == 2 * max(
        a.kv_cache_bytes for a in pipelined.assignments
    )


def test_a_replicated_cache_that_only_fits_divided_is_refused():
    """The under-reservation this pins: budgets sized for 1/N of the cache.

    With the flag off the same budgets plan cleanly, which is exactly the plan
    that used to be produced for MLA models and then died loading.
    """

    def small_layers(replicated):
        return ModelLayout(
            source="test",
            fixed_weight_bytes=1 * GIB,
            layer_weight_bytes=(64 * 1024**2,) * 32,
            tensor_parallel_heads=64,
            supports_tensor_parallel=True,
            # 16 GiB whole-model cache at 8192 tokens.
            kv_bytes_per_token_per_layer=64 * 1024,
            kv_replicated_across_tp=replicated,
        )

    fits_divided = small_layers(False)
    replicated = small_layers(True)

    assert plan_hybrid(fits_divided, _nodes(2, 14), tensor_parallel_size=2,
                       context_tokens=8192)
    with pytest.raises(PlanningError):
        plan_hybrid(replicated, _nodes(2, 14), tensor_parallel_size=2,
                    context_tokens=8192)


def test_kv_replication_survives_the_wire():
    """Peers exchange layouts as JSON; the flag must not be lost in transit."""

    model = _model(kv_per_token_per_layer=1152, kv_replicated=True)
    assert ModelLayout.from_dict(model.to_dict()).kv_replicated_across_tp is True
