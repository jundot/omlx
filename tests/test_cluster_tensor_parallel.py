# SPDX-License-Identifier: Apache-2.0

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    plan_hybrid,
)


def test_plan_hybrid_4_nodes_tp2():
    """Test hybrid planning: 4 nodes, tp=2, 2 pipeline stages."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=32,
    )
    nodes = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(4)
    ]

    plan = plan_hybrid(model, nodes, tensor_parallel_size=2)

    assert len(plan.assignments) == 4
    assert plan.tensor_parallel_size == 2
    assert plan.pipeline_stages == 2

    # Verify rank mapping: rank = stage * tp_size + tp_rank
    # stage 0 = ranks 0,1 (tp_rank 0,1); stage 1 = ranks 2,3 (tp_rank 0,1).
    # Stage 0 holds the *late* layers: MLX-LM sends activations from the highest
    # rank down to rank zero, so rank 0 is the tail of the pipeline.
    for assignment in plan.assignments:
        expected_tp_rank = assignment.rank % 2
        assert assignment.tensor_parallel_rank == expected_tp_rank
        assert assignment.tensor_parallel_size == 2
        assert assignment.sharded_weight_bytes > 0


def test_plan_hybrid_rank_mapping():
    """Verify the exact rank -> (stage, tp_rank) mapping for 4 nodes tp=2."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=32,
    )
    nodes = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(4)
    ]

    plan = plan_hybrid(model, nodes, tensor_parallel_size=2)

    # rank 0 -> stage 0, tp_rank 0
    # rank 1 -> stage 0, tp_rank 1
    # rank 2 -> stage 1, tp_rank 0
    # rank 3 -> stage 1, tp_rank 1
    by_rank = {a.rank: a for a in plan.assignments}
    assert by_rank[0].tensor_parallel_rank == 0
    assert by_rank[1].tensor_parallel_rank == 1
    assert by_rank[2].tensor_parallel_rank == 0
    assert by_rank[3].tensor_parallel_rank == 1


def test_plan_hybrid_not_divisible():
    """World size not divisible by TP size should raise PlanningError."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=32,
    )
    nodes = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(3)
    ]

    try:
        plan_hybrid(model, nodes, tensor_parallel_size=2)
        raise AssertionError("should have raised PlanningError")
    except PlanningError as e:
        assert "not divisible" in str(e)


def test_plan_hybrid_heads_not_divisible():
    """TP size that does not divide heads should raise PlanningError."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=33,  # Not divisible by 2
    )
    nodes = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(4)
    ]

    try:
        plan_hybrid(model, nodes, tensor_parallel_size=2)
        raise AssertionError("should have raised PlanningError")
    except PlanningError as e:
        assert "not divisible" in str(e)


def test_plan_hybrid_single_node():
    """Hybrid with tp=1 and 1 node should work (pipeline only)."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 4,
        tensor_parallel_heads=32,
    )
    nodes = [
        NodeBudget(
            node_id="single",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=0,
        ),
    ]

    plan = plan_hybrid(model, nodes, tensor_parallel_size=1)

    assert len(plan.assignments) == 1
    assert plan.tensor_parallel_size == 1
    assert plan.pipeline_stages == 1
    assert plan.assignments[0].tensor_parallel_rank == 0


def test_plan_hybrid_assignment_to_dict():
    """Test that hybrid assignments serialize with TP fields."""
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=32,
    )
    nodes = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=32 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(4)
    ]

    plan = plan_hybrid(model, nodes, tensor_parallel_size=2)

    for assignment in plan.assignments:
        d = assignment.to_dict()
        assert "tensor_parallel_rank" in d
        assert "tensor_parallel_size" in d
        assert "sharded_weight_bytes" in d
        assert d["tensor_parallel_size"] == 2


# ---------------------------------------------------------------------------
# Topology invariant (B1)
#
# The assertion that actually matters: every rank in a tensor-parallel group
# must hold the SAME layer range. They split each of those layers between them
# via shard_linear and all-reduce per layer, which is only meaningful if they
# are working on the same layers. An earlier plan_hybrid gave every rank its own
# range while still reporting pipeline_stages=2, so ranks 0 and 1 would have
# all-reduced across different layers.
# ---------------------------------------------------------------------------


def _grid_model(layers=32, layer_gib=2, fixed_gib=1, heads=48):
    # 48 heads so tp=2, 3 and 4 all divide evenly.
    return ModelLayout(
        source="test",
        fixed_weight_bytes=fixed_gib * 1024**3,
        layer_weight_bytes=(layer_gib * 1024**3,) * layers,
        tensor_parallel_heads=heads,
    )


def _grid_nodes(count, capacity_gib=32):
    return [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=capacity_gib * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(count)
    ]


def test_tp_group_members_share_one_layer_range():
    """Every TP group holds exactly one layer range, and there are `stages` of them."""

    for nodes_count, tp_size in ((4, 2), (6, 3), (6, 2), (4, 4), (3, 1)):
        plan = plan_hybrid(
            _grid_model(), _grid_nodes(nodes_count), tensor_parallel_size=tp_size
        )
        expected_stages = nodes_count // tp_size
        assert plan.pipeline_stages == expected_stages

        by_group: dict[int, set[tuple[int, int]]] = {}
        for assignment in plan.assignments:
            group = assignment.rank // tp_size
            by_group.setdefault(group, set()).add(
                (assignment.start_layer, assignment.end_layer)
            )

        for group, ranges in by_group.items():
            assert len(ranges) == 1, (
                f"{nodes_count} nodes tp={tp_size}: TP group {group} spans "
                f"{len(ranges)} different layer ranges {sorted(ranges)} — its "
                f"members must hold identical layers"
            )

        distinct = {(a.start_layer, a.end_layer) for a in plan.assignments}
        assert len(distinct) == expected_stages


def test_tp_group_members_cover_every_layer_exactly_once():
    """Stages tile the model: contiguous, no gaps, no overlap."""

    plan = plan_hybrid(_grid_model(), _grid_nodes(4), tensor_parallel_size=2)
    ranges = sorted({(a.start_layer, a.end_layer) for a in plan.assignments})
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 32
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert prev_end == next_start


def test_tp_divides_the_layer_bytes_a_node_holds():
    """TP shards the layers themselves, so a node holds 1/N of its stage."""

    # The no-TP baseline needs room for a whole 32 GiB stage on one node.
    solo = plan_hybrid(
        _grid_model(), _grid_nodes(2, capacity_gib=40), tensor_parallel_size=1
    )
    paired = plan_hybrid(_grid_model(), _grid_nodes(4), tensor_parallel_size=2)

    # 32 layers x 2 GiB = 64 GiB. Two stages either way, so each stage is 32 GiB.
    # Without TP one node carries all 32; with tp=2 each member carries 16.
    assert {a.layer_weight_bytes for a in solo.assignments} == {32 * 1024**3}
    assert {a.layer_weight_bytes for a in paired.assignments} == {16 * 1024**3}

    # And the parts sum back to the whole stage, with no double counting.
    for group in (0, 1):
        members = [a for a in paired.assignments if a.rank // 2 == group]
        assert sum(a.layer_weight_bytes for a in members) == 32 * 1024**3
        for member in members:
            assert member.planned_weight_bytes == (
                member.fixed_weight_bytes + member.layer_weight_bytes
            )


def test_tp_lets_a_model_fit_that_one_node_cannot_hold():
    """The point of TP: halving per-node layer bytes fits a model that otherwise won't."""

    model = _grid_model(layers=32, layer_gib=2, fixed_gib=1)
    # 20 GiB usable each: a 32 GiB stage does not fit one node, but 16 GiB does.
    tight = [
        NodeBudget(
            node_id=f"node-{i}",
            capacity_bytes=22 * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=i,
        )
        for i in range(4)
    ]

    try:
        plan_hybrid(model, tight[:2], tensor_parallel_size=1)
        raise AssertionError("2 nodes without TP should not fit this model")
    except PlanningError:
        pass

    plan = plan_hybrid(model, tight, tensor_parallel_size=2)
    assert plan.pipeline_stages == 2
    for assignment in plan.assignments:
        assert assignment.planned_weight_bytes <= (
            assignment.capacity_bytes - assignment.reserve_bytes
        )
