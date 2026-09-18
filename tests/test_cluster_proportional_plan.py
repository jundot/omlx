# SPDX-License-Identifier: Apache-2.0
"""Tests for the RAM-proportional largest-remainder N-node allocator."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    allocate_layers_proportional,
    plan_proportional_pipeline,
    plan_unequal_pipeline,
)

GIB = 1024**3


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# allocate_layers_proportional
# ---------------------------------------------------------------------------


def test_proportional_split_matches_ram_ratio():
    # exo's reference case: 256 GB + 128 GB splits ≈⅔–⅓.
    counts = allocate_layers_proportional(80, [256 * GIB, 128 * GIB])

    assert counts == (53, 27)
    assert sum(counts) == 80


def test_proportional_split_three_equal_nodes():
    counts = allocate_layers_proportional(80, [100, 100, 100])

    # 26.67 each: two nodes round up, ties break toward earlier position.
    assert counts == (27, 27, 26)
    assert sum(counts) == 80


def test_proportional_split_guarantees_one_layer_per_node():
    counts = allocate_layers_proportional(10, [1000, 1])

    assert counts == (9, 1)


def test_proportional_split_remainder_goes_to_largest_fraction():
    counts = allocate_layers_proportional(4, [1, 1, 1])

    assert counts == (2, 1, 1)


def test_proportional_split_is_deterministic():
    shares = [37, 91, 55, 12]

    assert allocate_layers_proportional(61, shares) == allocate_layers_proportional(
        61, shares
    )
    assert sum(allocate_layers_proportional(61, shares)) == 61


def test_proportional_split_rejects_more_nodes_than_layers():
    with pytest.raises(PlanningError, match="cannot each receive a layer"):
        allocate_layers_proportional(2, [1, 1, 1])


@pytest.mark.parametrize("layer_count, shares", [(0, [1]), (-3, [1])])
def test_proportional_split_rejects_nonpositive_layer_count(layer_count, shares):
    with pytest.raises(ValueError):
        allocate_layers_proportional(layer_count, shares)


@pytest.mark.parametrize("shares", [[0], [10, -1], [10, 0]])
def test_proportional_split_rejects_nonpositive_shares(shares):
    with pytest.raises(ValueError, match="positive"):
        allocate_layers_proportional(8, shares)


# ---------------------------------------------------------------------------
# plan_proportional_pipeline
# ---------------------------------------------------------------------------


def _model(layers=(10, 10, 10, 10), **overrides):
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=1,
        layer_weight_bytes=tuple(layers),
        supports_pipeline=True,
        **overrides,
    )


def _nodes(*specs):
    return [
        NodeBudget(
            node_id=node_id,
            capacity_bytes=capacity,
            reserve_bytes=reserve,
            rank=rank,
            **extra,
        )
        for rank, (node_id, capacity, reserve, extra) in enumerate(specs)
    ]


def test_proportional_plan_two_unequal_nodes():
    # usable 90 vs 50 → 4 layers split 2.57/1.43 → 3/1 after largest remainder.
    plan = plan_proportional_pipeline(
        _model(),
        _nodes(("large", 100, 10, {}), ("small", 60, 10, {})),
    )

    assert plan.optimization == "ram-proportional"
    by_rank = {item.rank: item for item in plan.assignments}
    # Highest rank owns the earliest layers (MLX pipeline order).
    assert (by_rank[1].start_layer, by_rank[1].end_layer) == (0, 1)
    assert by_rank[1].node_id == "small"
    assert (by_rank[0].start_layer, by_rank[0].end_layer) == (1, 4)
    assert by_rank[0].node_id == "large"
    # Full, contiguous coverage.
    covered = sorted((item.start_layer, item.end_layer) for item in plan.assignments)
    assert covered == [(0, 1), (1, 4)]


def test_proportional_plan_three_nodes_covers_every_layer_once():
    plan = plan_proportional_pipeline(
        _model(layers=(10,) * 12),
        _nodes(
            ("a", 200, 10, {}),
            ("b", 100, 10, {}),
            ("c", 60, 10, {}),
        ),
    )

    assert len(plan.assignments) == 3
    covered = sorted((item.start_layer, item.end_layer) for item in plan.assignments)
    assert covered[0][0] == 0
    assert covered[-1][1] == 12
    for previous, following in zip(covered, covered[1:]):
        assert previous[1] == following[0]
    # Shares 190:90:50 → counts ∝ usable RAM, descending by node size.
    counts = {
        item.node_id: item.end_layer - item.start_layer for item in plan.assignments
    }
    assert counts["a"] > counts["b"] > counts["c"]
    assert sum(counts.values()) == 12


def test_proportional_plan_respects_weight_ceiling():
    with pytest.raises(PlanningError, match="small.*split cap|weight ceiling"):
        plan_proportional_pipeline(
            _model(),
            _nodes(
                ("large", 100, 10, {}),
                ("small", 60, 10, {"max_weight_bytes": 5}),
            ),
        )


def test_proportional_plan_validates_kv_fit():
    model = _model(kv_bytes_per_token_per_layer=1)
    with pytest.raises(PlanningError, match="KV cache"):
        plan_proportional_pipeline(
            model,
            _nodes(("large", 100, 10, {}), ("small", 60, 10, {})),
            context_tokens=1_000_000,
        )


def test_proportional_plan_hash_is_stable_and_distinct():
    nodes = _nodes(("large", 100, 10, {}), ("small", 60, 10, {}))

    first = plan_proportional_pipeline(_model(), nodes)
    second = plan_proportional_pipeline(_model(), nodes)
    balanced = plan_unequal_pipeline(_model(), nodes)

    assert first.plan_hash == second.plan_hash
    # The allocator is part of plan identity even when the layer ranges agree.
    assert first.plan_hash != balanced.plan_hash


def test_proportional_plan_single_node_holds_everything():
    plan = plan_proportional_pipeline(
        _model(),
        _nodes(("only", 100, 10, {})),
    )

    assert len(plan.assignments) == 1
    assert (plan.assignments[0].start_layer, plan.assignments[0].end_layer) == (0, 4)


# ---------------------------------------------------------------------------
# Route-level allocation selection
# ---------------------------------------------------------------------------


def test_plan_route_runs_proportional_allocator(monkeypatch):
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: _model(layers=(10,) * 12),
    )

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/example",
            "allocation": "proportional",
            "nodes": [
                {"node_id": "a", "capacity_bytes": 200, "reserve_bytes": 10},
                {"node_id": "b", "capacity_bytes": 100, "reserve_bytes": 10},
                {"node_id": "c", "capacity_bytes": 60, "reserve_bytes": 10},
            ],
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["optimization"] == "ram-proportional"
    assert payload["placement_signature"]
    counts = {item["node_id"]: item["layer_count"] for item in payload["assignments"]}
    assert counts["a"] > counts["b"] > counts["c"]


def test_plan_route_defaults_to_balanced_allocator(monkeypatch):
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: _model(layers=(10,) * 12),
    )

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/example",
            "nodes": [
                {"node_id": "a", "capacity_bytes": 200, "reserve_bytes": 10},
                {"node_id": "b", "capacity_bytes": 100, "reserve_bytes": 10},
            ],
        },
    )

    assert response.status_code == 200, response.json()
    assert response.json()["optimization"] == "memory"


def test_plan_route_rejects_proportional_tensor_parallel(monkeypatch):
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: _model(layers=(10,) * 12),
    )

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/example",
            "allocation": "proportional",
            "tensor_parallel_size": 2,
            "nodes": [
                {"node_id": "a", "capacity_bytes": 200, "reserve_bytes": 10},
                {"node_id": "b", "capacity_bytes": 100, "reserve_bytes": 10},
            ],
        },
    )

    assert response.status_code == 400
    assert "pipeline-only" in response.json()["detail"]
