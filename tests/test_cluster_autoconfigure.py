# SPDX-License-Identifier: Apache-2.0
"""One-click activation picks the split and backend so the user does not have to."""

import asyncio
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest

from omlx.cluster.autoconfigure import (
    candidate_tensor_parallel_sizes,
    choose_backend,
    choose_parallelism,
    describe_transports,
    transports_are_fast_enough,
)
from omlx.cluster.planner import ModelLayout, NodeBudget, PlanningError


def _model(layers=32, layer_gib=2, fixed_gib=1, heads=48, shardable=True):
    return ModelLayout(
        source="test",
        fixed_weight_bytes=fixed_gib * 1024**3,
        layer_weight_bytes=(layer_gib * 1024**3,) * layers,
        tensor_parallel_heads=heads,
        supports_tensor_parallel=shardable,
        supports_pipeline=True,
    )


def _nodes(count, capacity_gib=64):
    return [
        NodeBudget(
            node_id=f"node-{index}",
            capacity_bytes=capacity_gib * 1024**3,
            reserve_bytes=2 * 1024**3,
            rank=index,
        )
        for index in range(count)
    ]


def _link(kind, speed=120, tb_version="TB5"):
    return SimpleNamespace(kind=kind, link_speed_gbps=speed, tb_version=tb_version)


def test_candidates_must_divide_both_nodes_and_heads():
    # Hybrid TP+pipeline is not a runtime-supported choice: use every node or
    # pipeline only.
    assert candidate_tensor_parallel_sizes(_model(heads=48), 4) == (4, 1)
    # 6 heads with 4 nodes: four-way TP does not divide, so only pipeline.
    assert candidate_tensor_parallel_sizes(_model(heads=6), 4) == (1,)
    # A head count that divides nothing leaves only the no-TP option.
    assert candidate_tensor_parallel_sizes(_model(heads=7), 4) == (1,)


def test_two_fast_linked_macs_get_tensor_parallelism():
    """The headline case: plug two Macs together, get one model split across both."""

    choice = choose_parallelism(
        _model(), _nodes(2), transports=[_link("thunderbolt")]
    )

    assert choice.tensor_parallel_size == 2
    assert choice.pipeline_stages == 1
    assert "tensor parallelism" in choice.reason
    # Both ranks hold the same layers, split between them.
    ranges = {(a.start_layer, a.end_layer) for a in choice.plan.assignments}
    assert len(ranges) == 1


def test_a_slow_link_falls_back_to_pipeline_stages():
    """TP all-reduces every layer; over Ethernet that is the wrong trade."""

    choice = choose_parallelism(
        _model(), _nodes(2), transports=[_link("ethernet", speed=10, tb_version=None)]
    )

    assert choice.tensor_parallel_size == 1
    assert choice.pipeline_stages == 2
    assert "too slow" in choice.reason


def test_explicit_tensor_can_override_a_slow_link_with_a_warning():
    choice = choose_parallelism(
        _model(),
        _nodes(2),
        transports=[_link("ethernet", speed=10, tb_version=None)],
        strategy="tensor",
    )

    assert choice.tensor_parallel_size == 2
    assert any("explicitly selected" in warning for warning in choice.warnings)


def test_capacity_preference_spends_nodes_on_pipeline_stages():
    speed = choose_parallelism(
        _model(), _nodes(4), transports=[_link("thunderbolt")], prefer="speed"
    )
    capacity = choose_parallelism(
        _model(), _nodes(4), transports=[_link("thunderbolt")], prefer="capacity"
    )

    assert speed.tensor_parallel_size > capacity.tensor_parallel_size
    assert capacity.pipeline_stages > speed.pipeline_stages


def test_a_model_too_big_for_the_cluster_is_refused_with_a_reason():
    tiny = [
        NodeBudget(
            node_id=f"node-{index}",
            capacity_bytes=6 * 1024**3,
            reserve_bytes=1 * 1024**3,
            rank=index,
        )
        for index in range(2)
    ]
    with pytest.raises(PlanningError, match="no workable split"):
        choose_parallelism(_model(), tiny, transports=[_link("thunderbolt")])


def test_an_impossible_head_count_is_refused_before_planning():
    # 4 nodes but 7 heads: only tp=1 is structurally possible, and it must fit.
    choice = choose_parallelism(_model(heads=7), _nodes(4))
    assert choice.tensor_parallel_size == 1


def test_choosing_tp_without_transport_data_warns_rather_than_pretending():
    choice = choose_parallelism(_model(), _nodes(2))
    assert choice.tensor_parallel_size == 2
    assert any("not probed" in warning for warning in choice.warnings)


def test_backend_follows_the_fabric():
    assert choose_backend([_link("rdma")])[0] == "jaccl"
    assert choose_backend([_link("thunderbolt")])[0] == "jaccl-ring"
    assert choose_backend([_link("ethernet")])[0] == "ring"
    assert choose_backend([])[0] == "ring"
    # Every branch explains itself; the dashboard shows this to the user.
    assert all(choose_backend(t)[1] for t in ([_link("rdma")], [_link("ethernet")], []))


def test_transport_summary_reports_the_slowest_link():
    summary = describe_transports([_link("thunderbolt", 120), _link("thunderbolt", 40)])
    assert "40 Gb/s" in summary
    assert describe_transports([]) == "Link speed unknown"


def test_mixed_fabric_is_not_treated_as_fast():
    """One Ethernet hop in the set means TP should not span the whole cluster."""

    assert transports_are_fast_enough([_link("thunderbolt"), _link("ethernet")], 2) is False
    assert transports_are_fast_enough([_link("thunderbolt"), _link("rdma")], 2) is True
    assert transports_are_fast_enough([_link("ethernet")], 1) is True


# ---------------------------------------------------------------------------
# The endpoint the one-click button calls.
# ---------------------------------------------------------------------------


def _app():
    from fastapi import FastAPI

    from omlx.cluster import routes

    app = FastAPI()
    app.include_router(routes.router)
    return app


def _autoconfigure_payload(node_count=2, capacity_gib=64):
    return {
        "model_size_bytes": 40 * 1024**3,
        "layer_count": 32,
        "nodes": [
            {
                "node_id": f"node-{index}",
                "capacity_bytes": capacity_gib * 1024**3,
                "reserve_bytes": 4 * 1024**3,
            }
            for index in range(node_count)
        ],
        "hosts": [],
        "detect_transports": False,
    }


def test_autoconfigure_returns_an_activatable_proposal():
    from fastapi.testclient import TestClient

    payload = _autoconfigure_payload()
    payload.update(
        {
            "auto_tune": False,
            "sampling_rank_only": True,
            "async_overlap": False,
            "cache_affinity": False,
            "max_kv_size": 4096,
            "target_context_tokens": 4096,
            "ring_connections_per_ip": 7,
        }
    )
    with TestClient(_app()) as client:
        response = client.post(
            "/admin/api/cluster/autoconfigure", json=payload
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["backend"] in {"ring", "jaccl", "jaccl-ring"}
    assert body["backend_reason"]
    assert body["summary"]
    assert body["tensor_parallel_size"] >= 1
    assert body["pipeline_stages"] >= 1
    assert (
        body["tensor_parallel_size"] * body["pipeline_stages"]
        == len(body["activation"]["nodes"])
    )

    # The activation block must be shaped for POST /deployments as-is.
    activation = body["activation"]
    assert activation["preflight"] is True
    assert activation["backend"] == body["backend"]
    assert activation["tensor_parallel_size"] == body["tensor_parallel_size"]
    assert activation["auto_tune"] is False
    assert activation["sampling_rank_only"] is True
    assert activation["async_overlap"] is False
    assert activation["cache_affinity"] is False
    assert activation["max_kv_size"] == 4096
    assert activation["target_context_tokens"] == 4096
    assert body["plan"]["cluster"]["target_context_tokens"] == 4096
    assert activation["ring_connections_per_ip"] == (
        7 if activation["backend"] == "ring" else None
    )


def test_autoconfigure_refuses_a_model_that_cannot_fit():
    from fastapi.testclient import TestClient

    payload = _autoconfigure_payload(capacity_gib=6)
    with TestClient(_app()) as client:
        response = client.post("/admin/api/cluster/autoconfigure", json=payload)

    assert response.status_code == 400
    assert "split" in response.json()["detail"]


def test_autoconfigure_does_not_activate_anything():
    """Proposing must never start processes on a peer Mac."""

    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        before = client.get("/admin/api/cluster/deployments").json()
        client.post("/admin/api/cluster/autoconfigure", json=_autoconfigure_payload())
        after = client.get("/admin/api/cluster/deployments").json()

    assert before == after


def test_autoconfigure_never_marks_incomplete_staging_ready(monkeypatch):
    """Missing shard files are a launch blocker, not merely a warning."""

    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    monkeypatch.setattr(
        routes,
        "_staging_for",
        lambda _request, _choice: {
            "ready": False,
            "error": "model shards are missing on studio",
            "nodes": [],
        },
    )
    with TestClient(_app()) as client:
        response = client.post(
            "/admin/api/cluster/autoconfigure", json=_autoconfigure_payload()
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ready_to_activate"] is False
    assert body["staging"]["error"] == "model shards are missing on studio"


def test_autoconfigure_never_marks_an_unverified_fabric_ready(monkeypatch):
    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    monkeypatch.setattr(
        routes,
        "detect_cluster_transports",
        lambda _hosts: SimpleNamespace(transports=()),
    )
    monkeypatch.setattr(
        routes,
        "_resolve_fabric",
        lambda _hosts: {
            "ok": False,
            "backend": "ring",
            "backend_reason": "no verified cluster route",
            "blocker": "studio.local did not answer",
            "link": {"reason": "studio.local did not answer"},
            "hosts": [],
        },
    )
    monkeypatch.setattr(
        routes,
        "_execution_for_request",
        lambda *_args, **_kwargs: pytest.fail(
            "an unverified fabric must not start the distributed probe"
        ),
    )
    monkeypatch.setattr(
        routes,
        "_model_and_nodes",
        lambda _request: (_model(), _nodes(2)),
    )
    monkeypatch.setattr(routes, "_staging_for", lambda *_args: None)
    payload = _autoconfigure_payload()
    payload.pop("model_size_bytes")
    payload.update(
        {
            "model_path": "/models/test",
            "hosts": [
                {
                    "node_id": "node-0",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.1.1"],
                },
                {
                    "node_id": "node-1",
                    "ssh": "studio.local",
                    "ips": ["10.0.1.2"],
                },
            ],
            "detect_transports": True,
            "measure_performance": True,
            "preflight": False,
        }
    )

    with TestClient(_app()) as client:
        response = client.post("/admin/api/cluster/autoconfigure", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fabric_ready"] is False
    assert body["ready_to_activate"] is False
    assert body["fabric_blocker"] == "studio.local did not answer"
    assert body["preflight"].startswith("Cluster link is not ready:")


def test_performance_is_measured_and_applied_before_staging(tmp_path, monkeypatch):
    """The first copied shard map must already be the measured fast one."""

    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    events = []
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10 * 1024**3,) * 8,
        supports_pipeline=True,
    )
    nodes = _nodes(2, capacity_gib=100)

    monkeypatch.setattr(
        routes,
        "_model_and_nodes",
        lambda _request: (model, nodes),
    )

    def probe(deployment):
        events.append(("probe", [item.layer_count for item in deployment.assignments]))
        return {
            "ok": True,
            "backend": "ring",
            "profiles": [
                {
                    "node_id": "node-0",
                    "rank": 0,
                    "decode_weight_bytes_per_second": 10.0,
                    "prefill_weight_bytes_per_second": 10.0,
                    "collective_latency_seconds": 0.001,
                    "collective_bandwidth_bytes_per_second": 10_000.0,
                    "backend": "ring",
                    "measured_at": "2026-07-29T12:00:00+00:00",
                    "samples": 5,
                },
                {
                    "node_id": "node-1",
                    "rank": 1,
                    "decode_weight_bytes_per_second": 40.0,
                    "prefill_weight_bytes_per_second": 40.0,
                    "collective_latency_seconds": 0.001,
                    "collective_bandwidth_bytes_per_second": 10_000.0,
                    "backend": "ring",
                    "measured_at": "2026-07-29T12:00:00+00:00",
                    "samples": 5,
                },
            ],
        }

    def staging(_request, choice):
        events.append(("staging", [item.layer_count for item in choice.plan.assignments]))
        return {"ready": True, "nodes": []}

    monkeypatch.setattr(routes, "run_cluster_performance_probe", probe)
    monkeypatch.setattr(routes, "_staging_for", staging)
    payload = {
        "model_path": str(tmp_path),
        "nodes": [
            {
                "node_id": f"node-{rank}",
                "capacity_bytes": 100 * 1024**3,
                "reserve_bytes": 2 * 1024**3,
            }
            for rank in range(2)
        ],
        "hosts": [
            {"node_id": "node-0", "ssh": "127.0.0.1", "ips": ["10.0.0.1"]},
            {"node_id": "node-1", "ssh": "studio", "ips": ["10.0.0.2"]},
        ],
        "strategy": "pipeline",
        "detect_transports": False,
        "preflight": False,
        "measure_performance": True,
    }

    with TestClient(_app()) as client:
        response = client.post("/admin/api/cluster/autoconfigure", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert [event[0] for event in events] == ["probe", "staging"]
    assert events[0][1] == [4, 4]
    assert events[1][1][1] > events[1][1][0]
    assert body["performance_probe"]["status"] == "applied_before_staging"
    assert all(node["performance"] for node in body["activation"]["nodes"])
    assert body["activation"]["approved_placement"] == body["plan"][
        "placement_signature"
    ]


def test_rejected_performance_split_is_not_replayed_during_activation(
    tmp_path, monkeypatch
):
    """Fallback nodes must not retain profiles from a plan that was rejected."""

    from fastapi.testclient import TestClient

    from omlx.cluster import routes

    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10 * 1024**3,) * 8,
        supports_pipeline=True,
    )
    nodes = _nodes(2, capacity_gib=100)
    monkeypatch.setattr(routes, "_model_and_nodes", lambda _request: (model, nodes))
    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda _deployment: {
            "ok": True,
            "backend": "ring",
            "profiles": [
                {
                    "node_id": f"node-{rank}",
                    "rank": rank,
                    "decode_weight_bytes_per_second": 10.0 + rank,
                    "prefill_weight_bytes_per_second": 10.0 + rank,
                    "collective_latency_seconds": 0.001,
                    "collective_bandwidth_bytes_per_second": 10_000.0,
                    "backend": "ring",
                    "measured_at": "2026-07-30T00:00:00+00:00",
                    "samples": 5,
                }
                for rank in range(2)
            ],
        },
    )
    monkeypatch.setattr(
        routes,
        "_build_performance_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PlanningError("measured split exceeds the context budget")
        ),
    )
    monkeypatch.setattr(
        routes,
        "_staging_for",
        lambda _request, _choice: {"ready": True, "nodes": []},
    )
    payload = {
        "model_path": str(tmp_path),
        "nodes": [
            {
                "node_id": f"node-{rank}",
                "capacity_bytes": 100 * 1024**3,
                "reserve_bytes": 2 * 1024**3,
            }
            for rank in range(2)
        ],
        "hosts": [
            {"node_id": "node-0", "ssh": "127.0.0.1", "ips": ["10.0.0.1"]},
            {"node_id": "node-1", "ssh": "studio", "ips": ["10.0.0.2"]},
        ],
        "strategy": "pipeline",
        "detect_transports": False,
        "preflight": False,
        "measure_performance": True,
    }

    with TestClient(_app()) as client:
        response = client.post("/admin/api/cluster/autoconfigure", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["performance_probe"]["status"] == "memory_fallback"
    assert all(node["performance"] is None for node in body["activation"]["nodes"])


@pytest.mark.asyncio
async def test_autoconfigure_keeps_event_loop_responsive_during_staging(
    monkeypatch,
):
    from omlx.cluster import routes

    started = threading.Event()
    release = threading.Event()

    def blocking_staging(_request, _choice):
        started.set()
        release.wait(timeout=2)
        return None

    monkeypatch.setattr(routes, "_staging_for", blocking_staging)
    app = _app()

    @app.get("/event-loop-probe")
    async def event_loop_probe():
        return {"ok": True}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        proposal = asyncio.create_task(
            client.post(
                "/admin/api/cluster/autoconfigure",
                json=_autoconfigure_payload(),
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            status = await asyncio.wait_for(
                client.get("/event-loop-probe"),
                timeout=0.5,
            )
            assert status.status_code == 200
        finally:
            release.set()
        response = await proposal

    assert response.status_code == 200


def test_staging_timeout_is_a_fail_soft_result(tmp_path, monkeypatch):
    from omlx.cluster import routes

    request = routes.ClusterAutoconfigureRequest(
        model_path=str(tmp_path),
        nodes=[
            {
                "node_id": "local",
                "capacity_bytes": 64 * 1024**3,
                "reserve_bytes": 4 * 1024**3,
            }
        ],
    )
    choice = SimpleNamespace(plan=SimpleNamespace(assignments=()))

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh", "peer"], 120)

    monkeypatch.setattr(routes, "stage_manifest", timeout)

    result = routes._staging_for(request, choice)

    assert result["ready"] is False
    assert "timed out" in result["error"]


def test_head_count_is_read_from_the_model_config(tmp_path):
    """Without this the planner refuses every TP degree and one-click gives PP.

    tensor_parallel_heads defaulted to 1 for every real model, so `1 % 2 != 0`
    rejected tensor parallelism no matter the hardware.
    """

    import json
    import struct

    from omlx.cluster.planner import inspect_safetensors_layout

    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "llama",  # a real mlx-lm module that implements shard()
            "hidden_size": 4096,
            "num_attention_heads": 32,
        })
    )
    header = {
        "model.embed_tokens.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16", "shape": [1], "data_offsets": [2, 6],
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "dtype": "F16", "shape": [1], "data_offsets": [6, 10],
        },
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * 10
    )

    layout = inspect_safetensors_layout(str(tmp_path))
    assert layout.tensor_parallel_heads == 32
    assert candidate_tensor_parallel_sizes(layout, 2) == (2, 1)


def test_a_model_without_a_config_gets_no_tensor_parallelism(tmp_path):
    """Unknown head count must mean 'no TP', never a guessed number."""

    import json
    import struct

    from omlx.cluster.planner import inspect_safetensors_layout

    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16", "shape": [1], "data_offsets": [0, 4],
        },
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * 4
    )

    layout = inspect_safetensors_layout(str(tmp_path))
    assert layout.tensor_parallel_heads == 1
    assert candidate_tensor_parallel_sizes(layout, 2) == (1,)


def test_an_architecture_without_shard_never_gets_tensor_parallelism():
    """Most mlx-lm models do not implement shard(); refusing at plan time beats
    failing after every rank has loaded its weights over the network."""

    unsupported = _model(shardable=False)
    assert candidate_tensor_parallel_sizes(unsupported, 4) == (1,)

    choice = choose_parallelism(
        unsupported, _nodes(2), transports=[_link("thunderbolt")]
    )
    assert choice.tensor_parallel_size == 1
    assert "does not implement" in choice.reason


def test_explicit_pipeline_refuses_an_architecture_without_pipeline_support():
    unsupported = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 8,
        tensor_parallel_heads=8,
        supports_tensor_parallel=True,
        supports_pipeline=False,
    )

    with pytest.raises(PlanningError, match="does not implement"):
        choose_parallelism(
            unsupported,
            _nodes(2),
            transports=[_link("ethernet", speed=10, tb_version=None)],
            strategy="pipeline",
        )


def test_automatic_never_falls_back_to_an_unsupported_pipeline():
    unsupported = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 8,
        tensor_parallel_heads=8,
        tensor_parallel_kv_heads=8,
        supports_tensor_parallel=True,
        supports_pipeline=False,
    )

    with pytest.raises(PlanningError, match="no tensor-parallel degree"):
        choose_parallelism(
            unsupported,
            _nodes(2),
            transports=[_link("ethernet", speed=10, tb_version=None)],
        )


def test_grouped_query_attention_bounds_the_split():
    """n_kv_heads is usually far smaller than n_heads and binds first."""

    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=32,
        tensor_parallel_kv_heads=2,
        supports_tensor_parallel=True,
    )
    # 32 heads would allow 4, but only 2 KV heads exist to divide.
    assert candidate_tensor_parallel_sizes(model, 4) == (1,)


def test_architecture_specific_linear_heads_bound_the_split():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * 1024**3,
        layer_weight_bytes=(2 * 1024**3,) * 32,
        tensor_parallel_heads=24,
        tensor_parallel_kv_heads=24,
        tensor_parallel_divisors=(24, 8),
        supports_tensor_parallel=True,
    )

    # Regular attention permits three-way TP, but the linear-attention head
    # group does not. Refuse before downloading or loading any rank.
    assert candidate_tensor_parallel_sizes(model, 3) == (1,)
    assert ModelLayout.from_dict(model.to_dict()).tensor_parallel_divisors == (24, 8)


def test_kv_heads_default_to_attention_heads():
    """Models without grouped-query attention omit num_key_value_heads."""

    assert _model(heads=48).tensor_parallel_kv_heads == 48


# ---------------------------------------------------------------------------
# C3: transport-aware rank placement. TP all-reduces per layer per token;
# pipeline sends one tensor per stage. Fast links belong inside TP groups.
# ---------------------------------------------------------------------------


def _pair(source, peer, kind="thunderbolt"):
    return SimpleNamespace(
        kind=kind, source_node_id=source, peer_node_id=peer,
        link_speed_gbps=120, tb_version="TB5",
    )


def test_fast_linked_pairs_are_grouped_not_split():
    """A,B on TB5 and C,D on TB5, with A-C slow: groups must be {A,B},{C,D}."""

    from omlx.cluster.autoconfigure import order_hosts_for_topology

    hosts = ["A", "C", "B", "D"]  # deliberately interleaved input order
    fabric = [
        _pair("A", "B"), _pair("B", "A"),
        _pair("C", "D"), _pair("D", "C"),
        _pair("A", "C", kind="ethernet"),
    ]

    placement = order_hosts_for_topology(hosts, fabric, 2)

    groups = [set(placement.hosts[i:i + 2]) for i in (0, 2)]
    assert {"A", "B"} in groups
    assert {"C", "D"} in groups
    # The placement must explain itself: which groups, and on what evidence.
    assert "A+B" in placement.reason and "C+D" in placement.reason
    assert "link speed" in placement.reason
    assert any("not measured" in w for w in placement.warnings)


def test_autoconfigure_reorders_nodes_and_hosts_as_one_rank_map(monkeypatch):
    """Topology placement must not make the server reject its own proposal."""

    from fastapi.testclient import TestClient

    from omlx.cluster import routes
    from omlx.cluster.autoconfigure import ParallelismChoice
    from omlx.cluster.planner import ModelLayout, plan_hybrid

    hosts = ["A", "C", "B", "D"]
    fabric = [
        _pair("A", "B"),
        _pair("B", "A"),
        _pair("C", "D"),
        _pair("D", "C"),
        _pair("A", "C", kind="ethernet"),
    ]

    def choose(model, nodes, **_kwargs):
        plan = plan_hybrid(model, nodes, tensor_parallel_size=2)
        return ParallelismChoice(2, 2, plan, "test topology")

    monkeypatch.setattr(routes, "choose_parallelism", choose)
    monkeypatch.setattr(
        routes,
        "synthetic_model_layout",
        lambda **_kwargs: ModelLayout(
            source="test",
            fixed_weight_bytes=0,
            layer_weight_bytes=(1 * 1024**3,) * 32,
            tensor_parallel_heads=2,
            tensor_parallel_kv_heads=2,
            supports_tensor_parallel=True,
        ),
    )
    monkeypatch.setattr(
        routes,
        "detect_cluster_transports",
        lambda _hosts: SimpleNamespace(transports=fabric),
    )
    monkeypatch.setattr(
        routes,
        "_resolve_fabric",
        lambda _hosts: (_ for _ in ()).throw(OSError("not needed")),
    )
    monkeypatch.setattr(routes, "_staging_for", lambda *_args: None)
    payload = {
        "model_size_bytes": 32 * 1024**3,
        "layer_count": 32,
        "nodes": [
            {
                "node_id": host,
                "capacity_bytes": 64 * 1024**3,
                "reserve_bytes": 2 * 1024**3,
            }
            for host in hosts
        ],
        "hosts": [
            {"node_id": host, "ssh": host, "ips": [f"10.0.0.{index + 1}"]}
            for index, host in enumerate(hosts)
        ],
        "preflight": False,
    }

    with TestClient(_app()) as client:
        response = client.post("/admin/api/cluster/autoconfigure", json=payload)

    assert response.status_code == 200, response.text
    activation = response.json()["activation"]
    assert [item["node_id"] for item in activation["nodes"]] == ["A", "B", "C", "D"]
    assert [item["node_id"] for item in activation["hosts"]] == ["A", "B", "C", "D"]
    assert activation["approved_placement"] == response.json()["plan"][
        "placement_signature"
    ]


def test_placement_falls_back_loudly_when_the_fabric_will_not_split():
    """Better to say so than to silently put an all-reduce on Ethernet."""

    from omlx.cluster.autoconfigure import order_hosts_for_topology

    hosts = ["A", "B", "C", "D"]
    fabric = [_pair("A", "B"), _pair("B", "A")]  # C and D linked to nothing

    placement = order_hosts_for_topology(hosts, fabric, 2)

    assert placement.hosts == ("A", "B", "C", "D")
    assert placement.warnings
    assert "slow link" in placement.warnings[0]


def test_no_transport_data_leaves_order_alone_with_a_warning():
    from omlx.cluster.autoconfigure import order_hosts_for_topology

    placement = order_hosts_for_topology(["A", "B", "C", "D"], [], 2)
    assert placement.hosts == ("A", "B", "C", "D")
    assert placement.warnings


def test_tp1_needs_no_placement():
    from omlx.cluster.autoconfigure import order_hosts_for_topology

    placement = order_hosts_for_topology(["A", "B"], [], 1)
    assert placement.hosts == ("A", "B")
    assert not placement.warnings


def test_slow_spanning_groups_are_reported():
    from omlx.cluster.autoconfigure import tp_groups_spanning_slow_links

    fabric = [_pair("A", "B"), _pair("B", "A")]
    # A+B is fast; C+D has no fast link between them.
    offenders = tp_groups_spanning_slow_links(["A", "B", "C", "D"], fabric, 2)
    assert offenders == (("C", "D"),)

    # Everything fast -> nothing to report.
    fabric += [_pair("C", "D"), _pair("D", "C")]
    assert tp_groups_spanning_slow_links(["A", "B", "C", "D"], fabric, 2) == ()


# ---------------------------------------------------------------------------
# E: pre-flight, so activation fails before it starts rather than after a
# full distributed load.
# ---------------------------------------------------------------------------


def test_preflight_flags_an_unreachable_peer():
    from omlx.cluster.autoconfigure import describe_preflight, preflight_issues

    issues = preflight_issues({"mac-1": None}, model_path="/models/llama")
    assert [i.kind for i in issues] == ["unreachable"]
    assert "unreachable" in describe_preflight(issues)


def test_preflight_flags_a_missing_model_and_a_version_skew():
    from omlx.cluster.autoconfigure import preflight_issues

    statuses = {
        "mac-1": {
            "runtime": {"mlx_version": "0.31.0"},
            "models": ["/models/other"],
        }
    }
    issues = preflight_issues(
        statuses,
        model_path="/models/llama",
        local_versions={"mlx_version": "0.32.0"},
    )
    kinds = {issue.kind for issue in issues}
    assert kinds == {"runtime_mismatch", "model_missing"}


def test_preflight_is_quiet_when_everything_matches():
    from omlx.cluster.autoconfigure import describe_preflight, preflight_issues

    statuses = {
        "mac-1": {
            "runtime": {"mlx_version": "0.32.0"},
            "models": ["/models/llama"],
        }
    }
    issues = preflight_issues(
        statuses,
        model_path="/models/llama",
        local_versions={"mlx_version": "0.32.0"},
    )
    assert issues == ()
    assert describe_preflight(issues) == "All peers are reachable and ready."


def test_preflight_does_not_invent_problems_from_missing_data():
    """A peer that does not report its models must not be called broken."""

    from omlx.cluster.autoconfigure import preflight_issues

    statuses = {"mac-1": {"runtime": {}}}
    assert preflight_issues(statuses, model_path="/models/llama") == ()


def test_preflight_unwraps_the_live_peer_probe_response():
    from omlx.cluster.autoconfigure import preflight_issues

    issues = preflight_issues(
        {
            "spark-a": {
                "runtime_compatible": False,
                "runtime_mismatches": ["mlx local=0.32.0 remote=0.31.0"],
                "status": {
                    "runtime": {"mlx_version": "0.31.0"},
                    "node": {"accelerator": "cuda", "memory_kind": "unified"},
                },
            }
        },
        model_path="/models/llama",
        local_versions={"mlx_version": "0.32.0"},
    )

    assert [(issue.kind, issue.detail) for issue in issues] == [
        ("runtime_mismatch", "mlx local=0.32.0 remote=0.31.0")
    ]


def test_preflight_refuses_discrete_cuda_without_a_live_vram_guard():
    from omlx.cluster.autoconfigure import preflight_issues

    issues = preflight_issues(
        {
            "cuda-box": {
                "runtime_compatible": True,
                "status": {
                    "runtime": {"mlx_version": "0.32.0"},
                    "node": {"accelerator": "cuda", "memory_kind": "vram"},
                },
            }
        },
        model_path=None,
    )

    assert [issue.kind for issue in issues] == ["cuda_memory_guard_unavailable"]


def test_rdma_detection_does_not_ssh_to_itself(monkeypatch):
    """MLX's check_rdma shells out to `ssh 127.0.0.1 ibv_devices`.

    On a Mac without SSH-to-self that fails and reports no RDMA on hardware
    that has it — observed on a machine with rdma_en1/en2/en6 present. The
    local host must be queried directly.
    """

    import subprocess as sp

    from omlx.cluster import transport

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh":
            raise AssertionError(f"must not ssh for a local host: {command}")

        class Result:
            stdout = "device node_GUID\n---- ----\nrdma_en6 abc123\n"

        return Result()

    monkeypatch.setattr(sp, "run", fake_run)
    monkeypatch.setattr(transport, "subprocess", sp)

    assert transport._rdma_devices("127.0.0.1") == ["rdma_en6"]
    assert calls == [["ibv_devices"]]


def test_rdma_requires_every_host_to_have_a_device(monkeypatch):
    from omlx.cluster import transport

    monkeypatch.setattr(
        transport,
        "_rdma_devices",
        lambda host: ["rdma_en6"] if host == "a" else [],
    )
    assert transport._rdma_available(["a"]) is True
    assert transport._rdma_available(["a", "b"]) is False
    assert transport._rdma_available([]) is False


# ---------------------------------------------------------------------------
# Explicit strategy choice: "make this faster" vs "run something bigger".
# ---------------------------------------------------------------------------


def test_pipeline_strategy_never_uses_tensor_parallelism():
    """Two M4 Mac minis running a 70B want layers split, not heads."""

    choice = choose_parallelism(
        _model(), _nodes(2), transports=[_link("thunderbolt")], strategy="pipeline"
    )
    assert choice.tensor_parallel_size == 1
    assert choice.pipeline_stages == 2
    # Each Mac holds a different half of the layers.
    ranges = sorted({(a.start_layer, a.end_layer) for a in choice.plan.assignments})
    assert len(ranges) == 2


def test_tensor_strategy_forces_tensor_parallelism():
    """Two M4 Mac minis speeding up a 27B want both working on every token."""

    choice = choose_parallelism(
        _model(), _nodes(2), transports=[_link("thunderbolt")], strategy="tensor"
    )
    assert choice.tensor_parallel_size == 2
    assert choice.pipeline_stages == 1
    ranges = {(a.start_layer, a.end_layer) for a in choice.plan.assignments}
    assert len(ranges) == 1, "TP peers share one layer range"


def test_asking_for_tensor_when_impossible_explains_why():
    """Silently falling back to pipeline would hide the reason from the user."""

    with pytest.raises(PlanningError, match="does not implement sharding"):
        choose_parallelism(
            _model(shardable=False), _nodes(2),
            transports=[_link("thunderbolt")], strategy="tensor",
        )

    with pytest.raises(PlanningError, match="head group"):
        choose_parallelism(
            _model(heads=7), _nodes(2),
            transports=[_link("thunderbolt")], strategy="tensor",
        )


def test_auto_still_picks_for_you():
    auto = choose_parallelism(
        _model(), _nodes(2), transports=[_link("thunderbolt")], strategy="auto"
    )
    assert auto.tensor_parallel_size == 2


def test_auto_uses_complete_end_to_end_measurements_instead_of_cable_label():
    pipeline = SimpleNamespace(
        prompt_tokens_per_second=140.0,
        decode_tokens_per_second=24.0,
        time_to_first_token_seconds=1.0,
    )
    tensor = SimpleNamespace(
        prompt_tokens_per_second=80.0,
        decode_tokens_per_second=10.0,
        time_to_first_token_seconds=2.0,
    )

    choice = choose_parallelism(
        _model(),
        _nodes(2),
        transports=[_link("thunderbolt")],
        strategy="auto",
        measurements={1: pipeline, 2: tensor},
        workload_profile="balanced",
    )

    assert choice.tensor_parallel_size == 1
    assert "measured end-to-end results" in choice.reason
    assert "140.0 prompt tok/s" in choice.reason


def test_complete_measurements_can_override_a_slow_link_heuristic():
    pipeline = SimpleNamespace(
        prompt_tokens_per_second=20.0,
        decode_tokens_per_second=4.0,
        time_to_first_token_seconds=10.0,
    )
    tensor = SimpleNamespace(
        prompt_tokens_per_second=40.0,
        decode_tokens_per_second=8.0,
        time_to_first_token_seconds=5.0,
    )

    choice = choose_parallelism(
        _model(),
        _nodes(2),
        transports=[_link("ethernet", speed=10, tb_version=None)],
        strategy="auto",
        measurements={1: pipeline, 2: tensor},
        workload_profile="balanced",
    )

    assert choice.tensor_parallel_size == 2
    assert "measured end-to-end results" in choice.reason


def test_auto_does_not_compare_one_measured_strategy_with_one_unknown():
    pipeline = SimpleNamespace(
        prompt_tokens_per_second=140.0,
        decode_tokens_per_second=24.0,
        time_to_first_token_seconds=1.0,
    )

    choice = choose_parallelism(
        _model(),
        _nodes(2),
        transports=[_link("thunderbolt")],
        strategy="auto",
        measurements={1: pipeline},
    )

    assert choice.tensor_parallel_size == 2
    assert "measured end-to-end results" not in choice.reason


def test_every_strategy_is_described_for_the_ui():
    from omlx.cluster.autoconfigure import STRATEGIES

    assert set(STRATEGIES) == {"auto", "tensor", "pipeline"}
    for key, meta in STRATEGIES.items():
        assert meta["label"] and meta["summary"] and meta["detail"], key
        # The description must say what it is FOR, not just what it does.
        assert len(meta["detail"]) > 80, key


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy must be"):
        choose_parallelism(_model(), _nodes(2), strategy="magic")


# ---------------------------------------------------------------------------
# Measured bandwidth beats the cable's label.
# ---------------------------------------------------------------------------


def _profile(node_id, gb_per_s, rank=0):
    return SimpleNamespace(
        node_id=node_id,
        rank=rank,
        collective_bandwidth_bytes_per_second=gb_per_s * 1000**3,
    )


def test_the_faster_of_two_thunderbolt_links_wins_the_group():
    """Two links, same kind, 4x apart — placement must prefer the fast one."""

    from omlx.cluster.autoconfigure import order_hosts_for_topology

    hosts = ["A", "B", "C", "D"]
    fabric = [
        _pair("A", "B"), _pair("A", "C"), _pair("A", "D"),
        _pair("B", "C"), _pair("B", "D"), _pair("C", "D"),
    ]
    # A and C were measured fast; B and D slow but still above the floor.
    profiles = [
        _profile("A", 6.6), _profile("C", 6.6),
        _profile("B", 1.6), _profile("D", 1.6),
    ]

    placement = order_hosts_for_topology(hosts, fabric, 2, profiles)

    groups = [set(placement.hosts[i : i + 2]) for i in (0, 2)]
    assert {"A", "C"} in groups, "the measured-fast pair must be grouped"
    assert "measured" in placement.reason
    assert not placement.warnings


def test_a_measured_slow_link_is_not_used_for_tensor_parallelism():
    """A Thunderbolt cable that only delivers 0.5 GB/s is not a fast link."""

    from omlx.cluster.autoconfigure import order_hosts_for_topology

    hosts = ["A", "B", "C", "D"]
    fabric = [_pair("A", "B"), _pair("C", "D"), _pair("A", "C")]
    profiles = [_profile(node, 0.5) for node in ("A", "B", "C", "D")]

    placement = order_hosts_for_topology(hosts, fabric, 2, profiles)

    assert placement.hosts == ("A", "B", "C", "D"), "order left unchanged"
    assert "no link fast enough" in placement.reason
    assert placement.warnings


def test_the_reported_group_speed_is_its_slowest_link():
    """An all-reduce runs at the weakest member, not the average."""

    from omlx.cluster.autoconfigure import order_hosts_for_topology

    hosts = ["A", "B"]
    fabric = [_pair("A", "B")]
    placement = order_hosts_for_topology(
        hosts, fabric, 2, [_profile("A", 6.6), _profile("B", 2.0)]
    )
    # tp_size == len(hosts): a single group, no placement choice to make.
    assert placement.hosts == ("A", "B")


def test_a_slow_group_is_reported_by_measured_bandwidth():
    from omlx.cluster.autoconfigure import tp_groups_spanning_slow_links

    hosts = ["A", "B"]
    fabric = [_pair("A", "B")]
    assert not tp_groups_spanning_slow_links(hosts, fabric, 2, [_profile("A", 6.6), _profile("B", 6.6)])
    slow = tp_groups_spanning_slow_links(
        hosts, fabric, 2, [_profile("A", 0.4), _profile("B", 0.4)]
    )
    assert slow == (("A", "B"),), "a measured-slow TB link must still be flagged"


# ---------------------------------------------------------------------------
# RDMA matrix
#
# choose_backend() returns "jaccl" as soon as RDMA is detected, but a
# ClusterDeployment with backend="jaccl" raised "JACCL requires a full RDMA
# connectivity matrix" because nothing assembled ClusterHost.rdma. These tests
# pin both the matrix and the reason given when there cannot be one.
# ---------------------------------------------------------------------------

GIB = 1024**3


def _interfaces(name, addresses, *, rdma=(), thunderbolt=None):
    from omlx.cluster.transport import HostInterfaces, InterfaceAddress

    interfaces = tuple(InterfaceAddress(*entry) for entry in addresses)
    return HostInterfaces(
        host=name,
        addresses=interfaces,
        rdma_interfaces=frozenset(rdma),
        thunderbolt_interfaces=frozenset(
            rdma if thunderbolt is None else thunderbolt
        ),
    )


def _rdma_pair():
    return [
        _interfaces("127.0.0.1", [("en4", "10.0.1.1", 24)], rdma={"en4"}),
        _interfaces("studio.local", [("en5", "10.0.1.2", 24)], rdma={"en5"}),
    ]


def test_the_matrix_names_the_device_each_rank_uses_for_each_peer():
    from omlx.cluster.autoconfigure import build_rdma_matrix

    matrix = build_rdma_matrix(_rdma_pair())

    assert matrix.ok
    assert matrix.rows == ((None, "rdma_en4"), ("rdma_en5", None))


def test_a_jaccl_deployment_built_from_the_matrix_is_accepted():
    """The failure this closes: backend and hosts agreed for the first time."""

    from omlx.cluster.autoconfigure import build_rdma_matrix
    from omlx.cluster.deployment import ClusterDeployment, ClusterHost
    from omlx.cluster.planner import PipelineAssignment

    matrix = build_rdma_matrix(_rdma_pair())
    assignments = tuple(
        PipelineAssignment(
            node_id=node_id,
            rank=rank,
            start_layer=rank * 4,
            end_layer=rank * 4 + 4,
            layer_weight_bytes=8 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=96 * GIB,
        )
        for rank, node_id in enumerate(("large", "small"))
    )

    deployment = ClusterDeployment(
        deployment_id="pair",
        model="mlx-community/model-4bit",
        backend="jaccl",
        hosts=(
            ClusterHost("large", "127.0.0.1", ("10.0.1.1",), matrix.rows[0]),
            ClusterHost("small", "studio.local", ("10.0.1.2",), matrix.rows[1]),
        ),
        assignments=assignments,
        plan_hash="a" * 64,
    )

    assert deployment.hostfile_dict()["hosts"][0]["rdma"] == [None, "rdma_en4"]


def test_a_mesh_gives_each_pair_its_own_port_not_one_per_host():
    from omlx.cluster.autoconfigure import build_rdma_matrix

    mesh = [
        _interfaces(
            "a",
            [("en1", "10.0.1.1", 30), ("en2", "10.0.2.1", 30)],
            rdma={"en1", "en2"},
        ),
        _interfaces(
            "b",
            [("en1", "10.0.1.2", 30), ("en2", "10.0.3.1", 30)],
            rdma={"en1", "en2"},
        ),
        _interfaces(
            "c",
            [("en1", "10.0.2.2", 30), ("en2", "10.0.3.2", 30)],
            rdma={"en1", "en2"},
        ),
    ]

    matrix = build_rdma_matrix(mesh)

    assert matrix.rows == (
        (None, "rdma_en1", "rdma_en2"),
        ("rdma_en1", None, "rdma_en2"),
        ("rdma_en1", "rdma_en2", None),
    )


def test_a_mac_without_rdma_is_named_so_the_fallback_to_ring_is_explainable():
    from omlx.cluster.autoconfigure import build_rdma_matrix

    interfaces = _rdma_pair()
    interfaces[1] = _interfaces("studio.local", [("en5", "10.0.1.2", 24)])

    matrix = build_rdma_matrix(interfaces)

    assert not matrix.ok
    assert matrix.rows == ()
    assert "studio.local" in matrix.reason
    assert "Recovery" in matrix.reason


def test_hosts_that_only_share_the_lan_do_not_get_a_silent_rdma_matrix():
    """Both Macs have RDMA devices, but the only routable pair is the office."""

    from omlx.cluster.autoconfigure import build_rdma_matrix

    matrix = build_rdma_matrix(
        [
            _interfaces("127.0.0.1", [("en0", "192.168.4.21", 24)], rdma={"en4"}),
            _interfaces("studio.local", [("en0", "192.168.4.22", 24)], rdma={"en5"}),
        ]
    )

    assert not matrix.ok
    assert "en0" in matrix.reason
    assert "not an RDMA device" in matrix.reason


def test_hosts_that_share_no_subnet_report_that_rather_than_a_partial_matrix():
    from omlx.cluster.autoconfigure import build_rdma_matrix

    matrix = build_rdma_matrix(
        [
            _interfaces("127.0.0.1", [("en4", "10.0.1.1", 24)], rdma={"en4"}),
            _interfaces("studio.local", [("en5", "192.168.9.4", 24)], rdma={"en5"}),
        ]
    )

    assert not matrix.ok
    assert "share no subnet" in matrix.reason


def test_a_single_host_is_not_a_cluster():
    from omlx.cluster.autoconfigure import build_rdma_matrix

    matrix = build_rdma_matrix(_rdma_pair()[:1])

    assert not matrix.ok
    assert "two hosts" in matrix.reason


def test_the_backend_choice_and_the_matrix_disagree_visibly_not_silently():
    """choose_backend saying jaccl is a claim the matrix must be able to confirm."""

    from omlx.cluster.autoconfigure import build_rdma_matrix, choose_backend

    backend, _ = choose_backend([_link("rdma")])
    matrix = build_rdma_matrix(
        [
            _interfaces("127.0.0.1", [("en0", "192.168.4.21", 24)], rdma={"en4"}),
            _interfaces("studio.local", [("en0", "192.168.4.22", 24)], rdma={"en5"}),
        ]
    )

    assert backend == "jaccl"
    assert not matrix.ok, "the caller must be able to see the claim was wrong"
    assert matrix.reason


# ---------------------------------------------------------------------------
# F: peer imports.
#
# Rank 1 died mid-load with "No module named 'mlx_vlm'" because the Studio's
# worker venv lacked it and nothing had asked. Which imports a rank needs is a
# property of the model, so it is read off the model's config and oMLX's own
# patch dispatch rather than listed here.
# ---------------------------------------------------------------------------

import json  # noqa: E402


def _model_dir(directory, **config):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return directory


@pytest.fixture
def derivation_tree(tmp_path, monkeypatch):
    """Point the import derivation at a throwaway dispatcher and patch tree."""

    from omlx.cluster import autoconfigure

    def install(dispatcher_source, patches):
        for name, source in patches.items():
            package = tmp_path / "patches" / name
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(source)
        dispatcher = tmp_path / "model_loading.py"
        dispatcher.write_text(dispatcher_source)
        monkeypatch.setattr(autoconfigure, "_OMLX_ROOT", tmp_path)
        monkeypatch.setattr(autoconfigure, "_DISPATCHER", dispatcher)
        autoconfigure._dispatch_rules.cache_clear()
        autoconfigure._third_party_imports.cache_clear()
        return autoconfigure

    yield install

    autoconfigure._dispatch_rules.cache_clear()
    autoconfigure._third_party_imports.cache_clear()


_ONE_BRANCH_DISPATCHER = (
    "def maybe_apply_pre_load_patches(model_name, model_settings=None,"
    " for_vlm=False):\n"
    "    if model_type == 'invented_arch':\n"
    "        from ..patches.invented import apply_invented_patch\n"
)


def _peer(found=(), missing=(), *, pip=None, uv=None, python="/peer/.venv/bin/python"):
    """A runner that answers the import probe the way one peer would."""

    def runner(argv, **kwargs):
        report = {
            "python": python,
            "found": list(found),
            "missing": list(missing),
            "pip": pip,
            "uv": uv,
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(report), "")

    return runner


def test_a_minimax_rank_needs_mlx_vlm_even_though_a_rank_is_an_mlx_lm_server(tmp_path):
    """The exact shape of the failure: a text-path rank with a hidden mlx-vlm dep.

    ``omlx.patches.minimax_m3_mlx_lm`` exists precisely because a rank is an
    ``mlx_lm.server``, and it registers a model built on vendored mlx-vlm — so
    "this is not a VLM load" is not a reason to skip asking for mlx-vlm.
    """

    from omlx.cluster.autoconfigure import required_imports

    modules = {
        requirement.module
        for requirement in required_imports(
            _model_dir(tmp_path, model_type="minimax_m3_vl")
        )
    }

    assert "mlx_vlm" in modules


def test_a_plain_llama_is_not_asked_to_bring_mlx_vlm(tmp_path):
    """Over-asking blocks a cluster that would have worked."""

    from omlx.cluster.autoconfigure import required_imports

    modules = {
        requirement.module
        for requirement in required_imports(_model_dir(tmp_path, model_type="llama"))
    }

    assert "mlx_vlm" not in modules
    assert {"mlx", "mlx_lm"} <= modules


def test_a_model_type_only_the_dispatcher_knows_about_is_still_covered(
    tmp_path, derivation_tree
):
    """The requirement is read out of the dispatcher, not restated here.

    A new architecture lands as a branch in ``maybe_apply_pre_load_patches``
    plus a patch package. Neither the model type nor the package that branch
    needs may have to be repeated here for the peer check to cover it.
    """

    autoconfigure = derivation_tree(
        _ONE_BRANCH_DISPATCHER, {"invented": "import newthing\n"}
    )

    modules = {
        requirement.module
        for requirement in autoconfigure.required_imports(
            _model_dir(tmp_path / "model", model_type="invented_arch")
        )
    }

    assert "newthing" in modules


def test_a_model_the_dispatcher_has_no_branch_for_is_not_charged_for_one(
    tmp_path, derivation_tree
):
    autoconfigure = derivation_tree(
        _ONE_BRANCH_DISPATCHER, {"invented": "import newthing\n"}
    )

    modules = {
        requirement.module
        for requirement in autoconfigure.required_imports(
            _model_dir(tmp_path / "model", model_type="something_else")
        )
    }

    assert modules == set()


def test_a_patch_that_tolerates_a_missing_import_is_not_demanded_of_a_peer(
    tmp_path, derivation_tree
):
    """oMLX logs and carries on, so the peer does not have to have it."""

    autoconfigure = derivation_tree(
        _ONE_BRANCH_DISPATCHER,
        {
            "invented": (
                "import strict\n"
                "try:\n"
                "    import fussy\n"
                "except ImportError:\n"
                "    fussy = None\n"
            )
        },
    )

    modules = {
        requirement.module
        for requirement in autoconfigure.required_imports(
            _model_dir(tmp_path / "model", model_type="invented_arch")
        )
    }

    assert modules == {"strict"}


def test_an_import_a_patch_defers_into_a_function_is_still_required(
    tmp_path, derivation_tree
):
    """Deferring the import is how the failure lands mid-load instead of up front."""

    autoconfigure = derivation_tree(
        _ONE_BRANCH_DISPATCHER,
        {"invented": "def apply_invented_patch():\n    import deferred\n"},
    )

    modules = {
        requirement.module
        for requirement in autoconfigure.required_imports(
            _model_dir(tmp_path / "model", model_type="invented_arch")
        )
    }

    assert modules == {"deferred"}


def test_a_vlm_load_and_a_rank_of_the_same_model_need_different_things(tmp_path):
    """A rank takes the mlx-lm branches; only an mlx-vlm load takes the others."""

    from omlx.cluster.autoconfigure import required_imports

    model = _model_dir(tmp_path, model_type="qwen3_5_moe", vision_config={})

    rank = {requirement.module for requirement in required_imports(model)}
    vlm = {
        requirement.module for requirement in required_imports(model, for_vlm=True)
    }

    assert "mlx_vlm" not in rank
    assert "mlx_vlm" in vlm


def test_a_model_directory_that_cannot_be_read_still_reports_the_common_imports():
    """An unreadable config is not a reason to claim a rank needs nothing."""

    from omlx.cluster.autoconfigure import required_imports

    modules = {
        requirement.module
        for requirement in required_imports("/no/such/model/anywhere")
    }

    assert {"mlx", "mlx_lm"} <= modules


def test_a_missing_import_names_the_module_and_what_asked_for_it(tmp_path):
    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], uv="/Users/omlx/.local/bin/uv"),
    )

    assert [issue.kind for issue in issues] == ["import_missing"]
    assert "mlx_vlm" in issues[0].detail
    assert "omlx.patches.minimax_m3_mlx_lm" in issues[0].detail


def test_a_missing_import_carries_a_command_that_installs_it(tmp_path):
    """A preflight that says "install mlx-vlm" sends the user to a search engine."""

    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], uv="/Users/omlx/.local/bin/uv"),
    )

    command = issues[0].remediation
    assert command.startswith("ssh studio ")
    assert "mlx-vlm" in command


def test_a_worker_venv_without_pip_is_told_to_use_the_installer_it_has(tmp_path):
    """uv builds venvs with no pip, so `python -m pip install` cannot work there."""

    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], pip=None, uv="/Users/omlx/.local/bin/uv"),
    )

    command = issues[0].remediation
    assert "/Users/omlx/.local/bin/uv pip install" in command
    assert "-m pip install" not in command


def test_a_peer_with_pip_is_told_to_use_pip(tmp_path):
    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], pip="/peer/.venv/bin/pip"),
    )

    assert "/peer/.venv/bin/python -m pip install" in issues[0].remediation


def test_a_peer_with_neither_installer_gets_a_command_that_still_works(tmp_path):
    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], pip=None, uv=None),
    )

    assert "-m ensurepip" in issues[0].remediation


def test_the_command_installs_the_version_omlx_pins(tmp_path):
    """A peer on a different mlx-vlm fails differently, not less."""

    from omlx.cluster.autoconfigure import _declared_requirements, peer_import_issues

    pinned = _declared_requirements().get("mlx-vlm", "")
    if "git+" not in pinned:
        pytest.skip("oMLX metadata is unavailable in this environment")

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], uv="/Users/omlx/.local/bin/uv"),
    )

    assert pinned in issues[0].remediation


def test_the_command_names_the_interpreter_the_peer_actually_ran(tmp_path):
    """The launcher's own path is a guess about someone else's Mac."""

    from omlx.cluster.autoconfigure import peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        python_executable="/here/.venv/bin/python",
        runner=_peer(
            missing=["mlx_vlm"],
            uv="/Users/omlx/.local/bin/uv",
            python="/over/there/.venv/bin/python",
        ),
    )

    assert "/over/there/.venv/bin/python" in issues[0].remediation
    assert "/here/.venv/bin/python" not in issues[0].remediation


def test_a_peer_that_can_import_everything_raises_nothing(tmp_path):
    from omlx.cluster.autoconfigure import peer_import_issues

    assert (
        peer_import_issues(
            {"studio": "studio"},
            model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
            runner=_peer(found=["mlx", "mlx_lm", "mlx_vlm"]),
        )
        == ()
    )


def test_the_local_rank_is_not_asked_over_ssh(tmp_path):
    """Rank zero is the process asking the question."""

    from omlx.cluster.autoconfigure import peer_import_issues

    def runner(argv, **kwargs):
        raise AssertionError("the local rank must not be probed over SSH")

    assert (
        peer_import_issues(
            {"here": "127.0.0.1", "also-here": "localhost"},
            model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
            runner=runner,
        )
        == ()
    )


def test_a_peer_that_cannot_be_reached_is_reported_rather_than_launched_into(tmp_path):
    from omlx.cluster.autoconfigure import peer_import_issues

    def runner(argv, **kwargs):
        raise OSError("ssh: connect to host studio port 22: Host is down")

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="llama"),
        runner=runner,
    )

    assert [issue.kind for issue in issues] == ["unreachable"]
    assert issues[0].remediation


def test_a_peer_that_answers_with_junk_is_not_taken_as_a_pass(tmp_path):
    """Silence and noise both have to fail closed, or the check is decoration."""

    from omlx.cluster.autoconfigure import peer_import_issues

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "Last login: Mon\n", "")

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="llama"),
        runner=runner,
    )

    assert [issue.kind for issue in issues] == ["unreachable"]


def test_a_login_banner_ahead_of_the_report_does_not_hide_a_missing_import(tmp_path):
    """Plenty of Macs print an MOTD on every non-interactive login."""

    from omlx.cluster.autoconfigure import peer_import_issues

    def runner(argv, **kwargs):
        report = {"python": "/peer/.venv/bin/python", "missing": ["mlx_vlm"]}
        return subprocess.CompletedProcess(
            argv, 0, "Welcome to the Studio\n" + json.dumps(report) + "\n", ""
        )

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=runner,
    )

    assert [issue.kind for issue in issues] == ["import_missing"]


def test_a_peer_missing_the_interpreter_says_so_rather_than_reporting_no_imports(
    tmp_path,
):
    from omlx.cluster.autoconfigure import peer_import_issues

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 127, "", "zsh: no such file or directory: /peer/.venv/bin/python"
        )

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="llama"),
        python_executable="/peer/.venv/bin/python",
        runner=runner,
    )

    assert [issue.kind for issue in issues] == ["unreachable"]
    assert "/peer/.venv/bin/python" in issues[0].detail


def test_the_peer_uses_the_same_prompt_free_ssh_as_the_launch(tmp_path):
    """Preflight and launch must share one changed-key refusal policy."""

    from omlx.cluster.autoconfigure import peer_import_issues

    seen = {}

    def runner(argv, **kwargs):
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="llama"),
        runner=runner,
    )

    assert "BatchMode=yes" in seen["argv"]
    assert "StrictHostKeyChecking=accept-new" in seen["argv"]
    assert "CheckHostIP=no" in seen["argv"]


def test_the_summary_shows_the_fix_and_not_only_the_failure(tmp_path):
    from omlx.cluster.autoconfigure import describe_preflight, peer_import_issues

    issues = peer_import_issues(
        {"studio": "studio"},
        model_path=_model_dir(tmp_path, model_type="minimax_m3_vl"),
        runner=_peer(missing=["mlx_vlm"], uv="/Users/omlx/.local/bin/uv"),
    )

    summary = describe_preflight(issues)

    assert "studio" in summary
    assert issues[0].remediation in summary


def test_the_probe_runs_standalone_under_a_bare_interpreter():
    """It has to work on a peer whose venv has almost nothing in it."""

    import sys

    from omlx.cluster.autoconfigure import _IMPORT_PROBE

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, "json,module_that_does_not_exist"],
        capture_output=True,
        text=True,
        check=False,
    )

    report = json.loads(completed.stdout)
    assert report["found"] == ["json"]
    assert report["missing"] == ["module_that_does_not_exist"]
