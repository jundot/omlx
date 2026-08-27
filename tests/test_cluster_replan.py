# SPDX-License-Identifier: Apache-2.0
"""Tests for the one-action cluster replan endpoint and its derivation helpers.

Everything is offline: the engine pool, peer liveness, preflight, and model
layout inspection are doubled, and the registry lives in a tmp_path.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.planner import ModelLayout
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.registry import configure_cluster_registry
from omlx.cluster.replan import (
    hosts_from_deployment,
    nodes_from_deployment,
    placement_view,
    summarize_deployment,
)

GIB = 1024**3


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _layout(path):
    return ModelLayout(
        source=path,
        fixed_weight_bytes=1,
        layer_weight_bytes=(10, 10, 10, 10),
        supports_pipeline=True,
    )


def _install_layout(monkeypatch):
    monkeypatch.setattr(routes, "inspect_safetensors_layout", _layout)
    monkeypatch.setattr(
        routes,
        "preflight_remote_hosts",
        lambda deployment: [
            {"rank": rank, "node_id": host.node_id}
            for rank, host in enumerate(deployment.hosts)
        ],
    )
    # Route tests establish peer reachability out of band; never let the
    # synthetic hosts reach the real SSH liveness probe.
    monkeypatch.setattr(routes, "check_peers", lambda *args, **kwargs: ())


def _deployment_payload(model_path, *, capacity_large=100, capacity_small=60):
    return {
        "model_path": str(model_path),
        "backend": "ring",
        # The synthetic probe is the real network thing; replan coverage uses
        # the memory-only path.
        "auto_tune": False,
        "nodes": [
            {"node_id": "large", "capacity_bytes": capacity_large, "reserve_bytes": 10},
            {"node_id": "small", "capacity_bytes": capacity_small, "reserve_bytes": 10},
        ],
        "hosts": [
            {"node_id": "large", "ssh": "127.0.0.1", "ips": ["192.168.5.1"]},
            {"node_id": "small", "ssh": "studio.local", "ips": ["192.168.5.2"]},
        ],
    }


def _approval_for(payload):
    plan = routes._create_cluster_plan(
        routes.ClusterPlanRequest(
            model_path=payload["model_path"],
            nodes=payload["nodes"],
            execution_profile=payload.get("execution_profile", "balanced"),
            tensor_parallel_size=payload.get("tensor_parallel_size", 1),
            serving_mode=payload.get("serving_mode", "sharded"),
            prefill_rank=payload.get("prefill_rank"),
            decode_rank=payload.get("decode_rank"),
            target_context_tokens=payload.get("target_context_tokens", 8192),
            prompt_cache_ssd=payload.get("prompt_cache_ssd"),
        )
    )
    return routes._placement_signature(plan.to_dict())


class _ReadyEngine:
    def __init__(self, deployment):
        self.deployment = deployment

    async def generate(self, *_args, **_kwargs):
        return SimpleNamespace(completion_tokens=1)

    def cluster_status(self):
        return {"phase": "ready", "ranks": []}


class _RecordingPool:
    """Engine-pool double that records the reload dance."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.model_id = "public-model"
        self.entry = SimpleNamespace(engine=None)
        self.reloads = 0

    def resolve_cluster_model_id(self, model_path):
        assert model_path == self.model_path
        return self.model_id

    def register_cluster_model(self, model_path, *, estimated_size):
        assert model_path == self.model_path
        assert estimated_size > 0
        return self.model_id, True

    def unregister_cluster_model(self, model_id):
        assert model_id == self.model_id
        return True

    def get_entry(self, model_id):
        assert model_id == self.model_id
        return self.entry

    async def prepare_cluster_reload(self, model_id):
        assert model_id == self.model_id
        self.reloads += 1
        self.entry.engine = None

    async def get_engine(self, model_id):
        assert model_id == self.model_id
        deployment = routes.get_cluster_registry().get_for_model(self.model_path)
        self.entry.engine = _ReadyEngine(deployment)
        return self.entry.engine


class _BusyPool(_RecordingPool):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.busy = False

    async def prepare_cluster_reload(self, model_id):
        if self.busy:
            from omlx.exceptions import ModelBusyError

            raise ModelBusyError(self.model_id, "replan the distributed cluster")
        await super().prepare_cluster_reload(model_id)


@pytest.fixture()
def active_deployment(tmp_path, monkeypatch):
    """Activate a two-node ring deployment and return its context."""

    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "nemotron"
    model_path.mkdir(parents=True)
    _install_layout(monkeypatch)
    pool = _RecordingPool(model_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    body = _deployment_payload(model_path)
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)
    assert response.status_code == 200, response.json()
    return SimpleNamespace(
        model_path=model_path,
        pool=pool,
        deployment=response.json()["deployment"],
    )


def test_nodes_from_deployment_round_trip(active_deployment):
    deployment = ClusterDeployment.from_dict(active_deployment.deployment)

    nodes = nodes_from_deployment(deployment)

    assert [node["node_id"] for node in nodes] == ["large", "small"]
    assert nodes[0]["capacity_bytes"] == 100
    assert nodes[0]["reserve_bytes"] == 10
    assert nodes[0]["role"] == "headless"
    # Split-control preferences are not recoverable from a signed plan.
    assert "max_weight_bytes" not in nodes[0]
    assert "target_weight_bytes" not in nodes[0]


def test_nodes_from_deployment_preserves_signed_performance(active_deployment):
    deployment = ClusterDeployment.from_dict(active_deployment.deployment)
    profiles = tuple(
        NodePerformanceProfile(
            node_id=assignment.node_id,
            rank=assignment.rank,
            decode_weight_bytes_per_second=100.0 - 40.0 * assignment.rank,
            prefill_weight_bytes_per_second=90.0 - 30.0 * assignment.rank,
            collective_latency_seconds=0.00003,
            collective_bandwidth_bytes_per_second=6.2e9,
            backend="ring",
            measured_at="2026-08-22T00:00:00+00:00",
            samples=5,
        )
        for assignment in deployment.assignments
    )
    deployment = replace(deployment, performance_profiles=profiles)

    nodes = nodes_from_deployment(deployment)

    assert nodes[0]["performance"] == profiles[0].to_dict()
    assert nodes[1]["performance"] == profiles[1].to_dict()


def test_hosts_from_deployment_round_trip(active_deployment):
    deployment = ClusterDeployment.from_dict(active_deployment.deployment)

    hosts = hosts_from_deployment(deployment)

    assert hosts[0]["ssh"] == "127.0.0.1"
    assert hosts[1]["ssh"] == "studio.local"
    assert hosts[1]["ips"] == ["192.168.5.2"]


def test_summarize_and_placement_view(active_deployment):
    deployment = ClusterDeployment.from_dict(active_deployment.deployment)

    summary = summarize_deployment(deployment)
    assert summary["deployment_id"] == deployment.deployment_id
    assert summary["world_size"] == 2
    assert summary["backend"] == "ring"
    assert len(summary["assignments"]) == 2

    # The placement view must feed the same diff machinery a plan dict does.
    signature = routes._placement_signature(placement_view(deployment))
    assert len(signature) == 16


def test_replan_preview_derives_current_cluster(active_deployment):
    response = _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": active_deployment.deployment["deployment_id"]},
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["mode"] == "preview"
    assert payload["derived"] == {
        "nodes": True,
        "hosts": True,
        "backend": True,
        "path_map": False,
    }
    assert payload["current"]["world_size"] == 2
    assert payload["current"]["mtp_enabled"] is False
    assert payload["changes"]["changed"] is False
    assert len(payload["steps"]) == 3
    assert payload["plan"]["placement_signature"]
    # A preview must not touch the pool at all.
    assert active_deployment.pool.reloads == 1  # only the initial activation


def test_replan_inherits_tensor_parallelism_and_context(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "nemotron"
    model_path.mkdir(parents=True)
    _install_layout(monkeypatch)
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: replace(
            _layout(path),
            tensor_parallel_heads=2,
            tensor_parallel_kv_heads=2,
            tensor_parallel_divisors=(2,),
            supports_tensor_parallel=True,
        ),
    )
    pool = _RecordingPool(model_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    body = _deployment_payload(model_path)
    body.update(tensor_parallel_size=2, target_context_tokens=32768)
    body["approved_placement"] = _approval_for(body)
    activation = _client().post("/admin/api/cluster/deployments", json=body)
    assert activation.status_code == 200, activation.json()

    deployment_id = activation.json()["deployment"]["deployment_id"]
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": deployment_id, "mtp_enabled": True},
    )

    assert preview.status_code == 200, preview.json()
    assert preview.json()["plan"]["tensor_parallel_size"] == 2
    assert preview.json()["plan"]["cluster"]["target_context_tokens"] == 32768


def test_replan_snapshot_toggle_is_signed_and_forces_reload(active_deployment):
    deployment_id = active_deployment.deployment["deployment_id"]
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": deployment_id, "prompt_cache_ssd": True},
    )

    assert preview.status_code == 200, preview.json()
    payload = preview.json()
    assert payload["plan"]["prompt_cache_ssd"] is True
    assert payload["changes"]["settings"]["prompt_cache_ssd"] == {
        "before": False,
        "after": True,
    }

    applied = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "prompt_cache_ssd": True,
            "approved_placement": payload["plan"]["placement_signature"],
        },
    )

    assert applied.status_code == 200, applied.json()
    assert active_deployment.pool.reloads == 2
    assert active_deployment.pool.entry.engine.deployment.execution.prompt_cache_ssd


def test_replan_preserves_disaggregated_phase_ownership(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "phase-model"
    model_path.mkdir(parents=True)
    _install_layout(monkeypatch)
    pool = _RecordingPool(model_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    body = _deployment_payload(model_path, capacity_large=100, capacity_small=100)
    body.update(
        serving_mode="disaggregated",
        prefill_rank=1,
        decode_rank=0,
    )
    body["approved_placement"] = _approval_for(body)
    activation = _client().post("/admin/api/cluster/deployments", json=body)
    assert activation.status_code == 200, activation.json()

    deployment_id = activation.json()["deployment"]["deployment_id"]
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": deployment_id, "execution_profile": "throughput"},
    )

    assert preview.status_code == 200, preview.json()
    plan = preview.json()["plan"]
    assert plan["serving_mode"] == "disaggregated"
    assert plan["prefill_rank"] == 1
    assert plan["decode_rank"] == 0
    assert all(item["start_layer"] == 0 for item in plan["assignments"])


def test_replan_carries_path_map_forward(tmp_path, monkeypatch):
    """A replan of a per-node-path deployment must not revert to same-path."""

    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "nemotron"
    model_path.mkdir(parents=True)
    _install_layout(monkeypatch)
    pool = _RecordingPool(model_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    path_map = {
        "large": str(model_path),
        "small": "/Volumes/shared/nemotron",
    }
    body = _deployment_payload(model_path)
    body["path_map"] = path_map
    plan = routes._create_cluster_plan(
        routes.ClusterPlanRequest(
            model_path=body["model_path"],
            nodes=body["nodes"],
            path_map=path_map,
        )
    )
    body["approved_placement"] = routes._placement_signature(plan.to_dict())
    response = _client().post("/admin/api/cluster/deployments", json=body)
    assert response.status_code == 200, response.json()
    deployment_id = response.json()["deployment"]["deployment_id"]
    assert response.json()["deployment"]["path_map"] == path_map

    preview = _client().post(
        "/admin/api/cluster/replan", json={"deployment_id": deployment_id}
    )
    assert preview.status_code == 200, preview.json()
    payload = preview.json()
    assert payload["derived"]["path_map"] is True
    assert payload["plan"]["path_map"] == path_map

    # An explicit override wins over the carried-forward map.
    override = _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": deployment_id, "path_map": {}},
    )
    assert override.status_code == 200, override.json()
    assert override.json()["derived"]["path_map"] is False


def test_replan_preview_detects_changed_split(active_deployment):
    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "nodes": [
                {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
                # A bigger reserve on the small node moves layers to the large one.
                {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 40},
            ],
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["derived"]["nodes"] is False
    assert payload["derived"]["hosts"] is True
    assert payload["changes"]["changed"] is True


def test_replan_applied_runs_full_dance(active_deployment):
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "target_context_tokens": 16384,
        },
    ).json()

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "target_context_tokens": 16384,
            "approved_placement": preview["plan"]["placement_signature"],
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["mode"] == "applied"
    assert payload["readiness"]["canary_passed"] is True
    assert payload["replan"]["previous"]["deployment_id"] == (
        active_deployment.deployment["deployment_id"]
    )
    # One reload for the initial activation, one for the replan.
    assert active_deployment.pool.reloads == 2
    engine = active_deployment.pool.entry.engine
    assert engine.deployment.target_context_tokens == 16384


def test_replan_mtp_toggle_is_signed_and_forces_reload(active_deployment):
    deployment_id = active_deployment.deployment["deployment_id"]
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "mtp_enabled": True,
            "mtp_num_draft_tokens": 3,
        },
    )

    assert preview.status_code == 200, preview.json()
    signed_plan = preview.json()["plan"]
    assert signed_plan["mtp_enabled"] is True
    assert signed_plan["mtp_num_draft_tokens"] == 3
    assert preview.json()["changes"]["changed"] is True
    assert preview.json()["changes"]["settings"] == {
        "mtp_enabled": {"before": False, "after": True},
        "mtp_num_draft_tokens": {"before": None, "after": 3},
    }

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "mtp_enabled": True,
            "mtp_num_draft_tokens": 3,
            "approved_placement": signed_plan["placement_signature"],
        },
    )

    assert response.status_code == 200, response.json()
    assert active_deployment.pool.reloads == 2
    engine = active_deployment.pool.entry.engine
    assert engine.deployment.mtp_enabled is True
    assert engine.deployment.mtp_num_draft_tokens == 3

    disabled = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "mtp_enabled": False,
            "mtp_num_draft_tokens": None,
        },
    )
    assert disabled.status_code == 200, disabled.json()
    assert disabled.json()["plan"].get("mtp_enabled") is None
    assert disabled.json()["plan"].get("mtp_num_draft_tokens") is None
    assert disabled.json()["changes"]["settings"] == {
        "mtp_enabled": {"before": True, "after": False},
        "mtp_num_draft_tokens": {"before": 3, "after": None},
    }


def test_replan_rejects_a_stale_approval(active_deployment):
    _client().post(
        "/admin/api/cluster/replan",
        json={"deployment_id": active_deployment.deployment["deployment_id"]},
    )

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "target_context_tokens": 4096,
            "approved_placement": "f" * 16,
        },
    )

    assert response.status_code == 409
    assert "not the plan" in response.json()["detail"]
    assert active_deployment.pool.reloads == 1


def test_replan_refuses_to_interrupt_active_requests(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "nemotron"
    model_path.mkdir(parents=True)
    _install_layout(monkeypatch)
    pool = _BusyPool(model_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    body = _deployment_payload(model_path)
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)
    assert response.status_code == 200, response.json()
    deployment_id = response.json()["deployment"]["deployment_id"]
    pool.busy = True

    preview = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "target_context_tokens": 16384,
        },
    ).json()
    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": deployment_id,
            "target_context_tokens": 16384,
            "approved_placement": preview["plan"]["placement_signature"],
        },
    )

    assert response.status_code == 409
    assert "serving a request" in response.json()["detail"]
    # The old deployment must survive a refused replan.
    listed = _client().get("/admin/api/cluster/deployments")
    assert listed.json()["deployments"][0]["deployment_id"] == deployment_id


def test_replan_unknown_deployment_is_404(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    _install_layout(monkeypatch)
    monkeypatch.setattr(
        routes, "_get_engine_pool", lambda: _RecordingPool(tmp_path / "m")
    )

    response = _client().post(
        "/admin/api/cluster/replan", json={"deployment_id": "nope"}
    )

    assert response.status_code == 404


def test_replan_without_context_requires_explicit_cluster(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    _install_layout(monkeypatch)
    monkeypatch.setattr(
        routes, "_get_engine_pool", lambda: _RecordingPool(tmp_path / "m")
    )

    response = _client().post(
        "/admin/api/cluster/replan",
        json={"model_path": "/models/never-deployed"},
    )

    assert response.status_code == 400
    assert "nodes and hosts" in response.json()["detail"]


def test_replan_membership_change_adds_a_node(active_deployment):
    """N-node replan: explicit nodes/hosts express the membership change."""

    model_path = active_deployment.model_path
    nodes = [
        {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
        {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
        {"node_id": "mini", "capacity_bytes": 50, "reserve_bytes": 10},
    ]
    hosts = [
        {"node_id": "large", "ssh": "127.0.0.1", "ips": ["192.168.5.1"]},
        {"node_id": "small", "ssh": "studio.local", "ips": ["192.168.5.2"]},
        {"node_id": "mini", "ssh": "mini.local", "ips": ["192.168.5.3"]},
    ]
    preview = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "nodes": nodes,
            "hosts": hosts,
        },
    )
    assert preview.status_code == 200, preview.json()
    assert preview.json()["plan"]["assignments"][2]["node_id"] == "mini"

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "deployment_id": active_deployment.deployment["deployment_id"],
            "nodes": nodes,
            "hosts": hosts,
            "approved_placement": preview.json()["plan"]["placement_signature"],
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["deployment"]["world_size"] if "world_size" in payload["deployment"] else True
    assert len(payload["deployment"]["assignments"]) == 3
    assert payload["replan"]["previous"]["world_size"] == 2
    ranks = sorted(
        (item["rank"], item["node_id"])
        for item in payload["deployment"]["assignments"]
    )
    assert ranks == [(0, "large"), (1, "small"), (2, "mini")]
