# SPDX-License-Identifier: Apache-2.0
"""Moving the pipeline split by hand, and seeing what it costs in context.

The motivating case: MiniMax-M3-4bit is 225 GiB over 60 layers. It loads on a
256 GiB Studio alone but leaves so little room that only ~1k tokens of context
fit. Split across a second Mac, the same model reaches hundreds of thousands of
tokens — so the split point is not a tuning detail, it is the difference
between a usable model and an unusable one.
"""

from __future__ import annotations

import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    plan_unequal_pipeline,
)

GiB = 1024**3


def _model(total_gib=225, layers=60, kv_per_layer=2048):
    """MiniMax-M3 shaped: 60 layers, 4 KV heads x 128 dims x 2 bytes x K/V."""

    total = int(total_gib * GiB)
    base, remainder = divmod(total, layers)
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=0,
        layer_weight_bytes=tuple(
            base + (1 if index < remainder else 0) for index in range(layers)
        ),
        kv_bytes_per_token_per_layer=kv_per_layer,
        supports_pipeline=True,
    )


def _mbp(cap_gib=0):
    return NodeBudget(
        node_id="mbp", capacity_bytes=128 * GiB, reserve_bytes=12 * GiB,
        rank=0, max_weight_bytes=int(cap_gib * GiB),
    )


def _studio(cap_gib=0):
    return NodeBudget(
        node_id="studio", capacity_bytes=243 * GiB, reserve_bytes=16 * GiB,
        rank=1, max_weight_bytes=int(cap_gib * GiB),
    )


def _by_node(plan):
    return {item.node_id: item for item in plan.assignments}


# --- The split control ------------------------------------------------------


def test_a_pinned_node_is_not_given_more_than_its_cap():
    plan = plan_unequal_pipeline(_model(), [_mbp(60), _studio()], context_tokens=8192)
    mbp = _by_node(plan)["mbp"]
    assert mbp.layer_weight_bytes + mbp.fixed_weight_bytes <= 60 * GiB


def test_pinning_one_node_lower_moves_work_to_the_other():
    loose = _by_node(plan_unequal_pipeline(_model(), [_mbp(), _studio()], context_tokens=8192))
    tight = _by_node(plan_unequal_pipeline(_model(), [_mbp(48), _studio()], context_tokens=8192))
    assert tight["mbp"].layer_count < loose["mbp"].layer_count
    assert tight["studio"].layer_count > loose["studio"].layer_count


def test_no_cap_means_the_planner_balances_as_before():
    """The control is opt-in; leaving it alone must change nothing."""

    unset = plan_unequal_pipeline(_model(), [_mbp(), _studio()], context_tokens=8192)
    generous = plan_unequal_pipeline(
        _model(), [_mbp(120), _studio(240)], context_tokens=8192
    )
    assert unset.plan_hash == generous.plan_hash


def test_a_cap_above_the_machine_is_clamped_not_believed():
    node = NodeBudget(
        node_id="mbp", capacity_bytes=128 * GiB, reserve_bytes=12 * GiB,
        max_weight_bytes=900 * GiB,
    )
    assert node.weight_ceiling_bytes == node.usable_bytes


def test_pinning_everything_too_low_fails_with_the_shortfall():
    with pytest.raises(PlanningError, match="does not fit"):
        plan_unequal_pipeline(
            _model(), [_mbp(40), _studio(60)], context_tokens=8192
        )


def test_a_negative_cap_is_rejected():
    with pytest.raises(ValueError, match="max_weight_bytes"):
        NodeBudget(node_id="a", capacity_bytes=GiB, max_weight_bytes=-1)


# --- What the split costs, which is the point of showing it -----------------


def test_each_node_reports_the_context_it_could_hold():
    plan = plan_unequal_pipeline(_model(), [_mbp(), _studio()], context_tokens=8192)
    for item in plan.assignments:
        assert item.max_context_tokens > 0
        assert item.kv_bytes_per_token > 0


def test_a_node_holding_fewer_layers_holds_more_context():
    """Fewer layers is less KV per token and more memory left for it."""

    plan = plan_unequal_pipeline(_model(), [_mbp(48), _studio()], context_tokens=8192)
    nodes = _by_node(plan)
    assert nodes["mbp"].layer_count < nodes["studio"].layer_count
    assert nodes["mbp"].max_context_tokens > nodes["studio"].max_context_tokens


def test_the_cluster_limit_is_the_weakest_stage_not_the_average():
    """Every request passes through every stage; the shortest one decides."""

    plan = plan_unequal_pipeline(_model(), [_mbp(48), _studio()], context_tokens=8192)
    assert plan.max_context_tokens == min(
        item.max_context_tokens for item in plan.assignments
    )


def test_moving_the_split_away_from_balance_costs_context():
    """The measured result on the real pairing: 971k balanced, 644k at 60 GiB."""

    balanced = plan_unequal_pipeline(_model(), [_mbp(), _studio()], context_tokens=8192)
    pinned = plan_unequal_pipeline(_model(), [_mbp(60), _studio()], context_tokens=8192)
    assert pinned.max_context_tokens < balanced.max_context_tokens


def test_the_capped_node_still_gets_its_whole_machine_for_cache():
    """Capping weights frees memory for KV — it must not also cap the cache."""

    pinned = _by_node(
        plan_unequal_pipeline(_model(), [_mbp(48), _studio()], context_tokens=8192)
    )["mbp"]
    spare = pinned.capacity_bytes - pinned.reserve_bytes - pinned.layer_weight_bytes
    assert pinned.max_context_tokens == spare // pinned.kv_bytes_per_token


def test_a_model_with_no_kv_shape_reports_unknown_not_unlimited():
    layout = _model(kv_per_layer=0)
    plan = plan_unequal_pipeline(layout, [_mbp(), _studio()], context_tokens=8192)
    assert plan.max_context_tokens == 0
    assert all(item.max_context_tokens == 0 for item in plan.assignments)


def test_the_plan_reports_kv_and_context_for_the_interface():
    plan = plan_unequal_pipeline(_model(), [_mbp(), _studio()], context_tokens=131072)
    cluster = plan.to_dict()["cluster"]
    assert cluster["kv_cache_bytes"] > 0
    assert cluster["max_context_tokens"] > 0
    assert plan.to_dict()["assignments"][0]["max_context_tokens"] > 0


def test_the_studio_alone_cannot_hold_a_long_context_but_the_pair_can():
    """The motivating case, stated as a test."""

    alone = NodeBudget(
        node_id="studio", capacity_bytes=243 * GiB, reserve_bytes=16 * GiB, rank=0
    )
    with pytest.raises(PlanningError, match="KV cache"):
        plan_unequal_pipeline(_model(), [alone], context_tokens=131072)

    paired = plan_unequal_pipeline(
        _model(), [_mbp(), _studio()], context_tokens=131072
    )
    assert paired.max_context_tokens > 131072
