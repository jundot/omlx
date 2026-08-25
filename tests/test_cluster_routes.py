# SPDX-License-Identifier: Apache-2.0
"""Tests for authenticated cluster admin route handlers."""

import base64
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.enrollment import ClusterEnrollmentStore
from omlx.cluster.models import (
    ClusterStatus,
    RDMACapability,
    RuntimeCapability,
    TransportState,
)


def _status() -> ClusterStatus:
    return ClusterStatus(
        collected_at="2026-07-26T12:00:00+00:00",
        hostname="MacBook-Pro",
        platform="macOS-26.5.2-arm64",
        chip_name="Apple M5 Max",
        physical_memory_bytes=128 * 1024**3,
        recommended_working_set_bytes=115_448_725_504,
        runtime=RuntimeCapability(
            omlx_version="0.5.3",
            mlx_version="0.32.0",
            mlx_lm_version="0.31.3",
            python_version="3.11.14",
            macos_version="26.5.2",
        ),
        transport_state=TransportState.ENABLED_NO_PEER,
        rdma=RDMACapability(
            control_status="enabled",
            devices=("rdma_en1",),
        ),
        thunderbolt_ports=(),
        warnings=("No peer",),
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _enrollment_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(routes.join_router)
    return TestClient(app)


def test_ssh_key_generation_requires_explicit_overwrite(monkeypatch, tmp_path):
    from omlx.cluster import ssh_keys

    calls = []
    key_pair = SimpleNamespace(
        key_type="ed25519",
        fingerprint="SHA256:test",
        public_key="ssh-ed25519 AAAA test",
        private_key_path=tmp_path / "omlx_cluster",
        public_key_path=tmp_path / "omlx_cluster.pub",
        created_at=123.0,
    )

    def generate(*, overwrite=False):
        calls.append(overwrite)
        return key_pair

    monkeypatch.setattr(ssh_keys, "generate_ssh_key_pair", generate)

    created = _client().post("/admin/api/cluster/ssh-key/generate")
    rotated = _client().post("/admin/api/cluster/ssh-key/generate?overwrite=true")

    assert created.status_code == 200
    assert rotated.status_code == 200
    assert created.json()["available"] is True
    assert rotated.json()["created_at"] == 123.0
    assert calls == [False, True]


def _worker_claim() -> dict:
    return {
        "node_id": "cuda-worker-1-machine",
        "hostname": "cuda-worker-1",
        "ssh_user": "omlxworker",
        "ssh_port": 22,
        "addresses": ["10.42.0.21"],
        "accelerator": "cuda",
        "platform": "Linux-aarch64",
    }


def _approval_for(payload: dict) -> str:
    """Placement identity for a direct activation request."""

    plan = routes._create_cluster_plan(
        routes.ClusterPlanRequest(
            model_path=payload["model_path"],
            nodes=payload["nodes"],
            execution_profile=payload.get("execution_profile", "balanced"),
            tensor_parallel_size=payload.get("tensor_parallel_size", 1),
            target_context_tokens=payload.get("target_context_tokens", 8192),
        )
    )
    return routes._placement_signature(plan.to_dict())


class _ReadyClusterEngine:
    def __init__(self, deployment, *, fail_canary: bool = False):
        self.deployment = deployment
        self.fail_canary = fail_canary

    async def generate(self, *_args, **_kwargs):
        if self.fail_canary:
            raise RuntimeError("synthetic canary failure")
        return SimpleNamespace(completion_tokens=1)

    def cluster_status(self):
        return {
            "phase": "ready",
            "ranks": [
                {
                    "rank": item.rank,
                    "node_id": item.node_id,
                    "measured_weight_bytes": item.planned_weight_bytes,
                }
                for item in self.deployment.assignments
            ],
        }


class _ReadyClusterPool:
    def __init__(
        self,
        model_path: str,
        *,
        fail_canary: bool = False,
        remote_only: bool = False,
    ):
        self.model_path = model_path
        self.model_id = "public-model"
        self.fail_canary = fail_canary
        self.remote_only = remote_only
        self.cluster_registered = False
        self.entry = SimpleNamespace(engine=None)
        self.reloads = 0

    def resolve_cluster_model_id(self, model_path):
        assert model_path == self.model_path
        if self.remote_only and not self.cluster_registered:
            from omlx.exceptions import ModelNotFoundError

            raise ModelNotFoundError(model_path, [])
        return self.model_id

    def register_cluster_model(self, model_path, *, estimated_size):
        assert model_path == self.model_path
        assert estimated_size > 0
        self.cluster_registered = True
        return self.model_id, True

    def unregister_cluster_model(self, model_id):
        assert model_id == self.model_id
        self.cluster_registered = False
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
        self.entry.engine = _ReadyClusterEngine(
            deployment,
            fail_canary=self.fail_canary,
        )
        return self.entry.engine

    def get_loaded_model_ids(self):
        return [self.model_id] if self.entry.engine is not None else []


def _install_ready_pool(
    monkeypatch,
    model_path,
    *,
    fail_canary=False,
    remote_only=False,
):
    pool = _ReadyClusterPool(
        str(model_path),
        fail_canary=fail_canary,
        remote_only=remote_only,
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    # These route tests exercise activation after peer reachability has been
    # established. Do not let their synthetic ``studio.local`` host escape to
    # the real SSH liveness probe; peer-loss behavior has dedicated coverage.
    monkeypatch.setattr(routes, "check_peers", lambda *args, **kwargs: ())
    return pool


def test_cluster_status_route(monkeypatch):
    monkeypatch.setattr(routes, "collect_cluster_status", lambda **_: _status())
    monkeypatch.setattr(
        routes,
        "read_runtime_markers",
        lambda: {"jobs": [{"rank": 1, "live": True}], "warnings": []},
    )

    response = _client().get(
        "/admin/api/cluster/status",
        params={"route_to": "198.51.100.197"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["node"]["chip_name"] == "Apple M5 Max"
    assert payload["transport"]["state"] == "enabled_no_peer"
    assert payload["runtime_jobs"]["jobs"][0]["rank"] == 1


def test_cluster_status_route_rejects_invalid_route_target():
    response = _client().get(
        "/admin/api/cluster/status",
        params={"route_to": "studio.local"},
    )
    assert response.status_code == 400
    assert "IPv4 or IPv6" in response.json()["detail"]


def test_link_setup_route_exposes_only_the_fixed_gui_operation(monkeypatch):
    from omlx.cluster.transport import LinkStatus

    seen = []
    monkeypatch.setattr(
        routes,
        "configure_link",
        lambda hosts: (
            seen.append(hosts)
            or LinkStatus(
                state="rdma_ready",
                title="ready",
                detail="ready",
                backend="jaccl",
                ready=True,
            )
        ),
    )

    response = _client().post(
        "/admin/api/cluster/link-setup",
        json={"hosts": ["127.0.0.1", "Studio.local"]},
    )

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert seen == [["127.0.0.1", "Studio.local"]]

    injected = _client().post(
        "/admin/api/cluster/link-setup",
        json={
            "hosts": ["127.0.0.1", "Studio.local"],
            "command": "arbitrary shell text",
        },
    )
    assert injected.status_code == 422


def test_link_setup_rejects_ssh_options_before_configuring(monkeypatch):
    called = []
    monkeypatch.setattr(
        routes,
        "configure_link",
        lambda hosts: called.append(hosts),
    )

    response = _client().post(
        "/admin/api/cluster/link-setup",
        json={"hosts": ["127.0.0.1", "-oProxyCommand=touch /tmp/pwned"]},
    )

    assert response.status_code == 400
    assert "invalid SSH target" in response.json()["detail"]
    assert called == []


def test_link_setup_route_reports_cancelled_native_authorization(monkeypatch):
    from omlx.cluster.transport import LinkAuthorizationCancelledError

    def cancelled(hosts):
        raise LinkAuthorizationCancelledError("Administrator approval was cancelled.")

    monkeypatch.setattr(routes, "configure_link", cancelled)

    response = _client().post(
        "/admin/api/cluster/link-setup",
        json={"hosts": ["127.0.0.1", "Studio.local"]},
    )

    assert response.status_code == 409
    assert "cancelled" in response.json()["detail"]


def test_cluster_runtime_route_is_lightweight(monkeypatch):
    monkeypatch.setattr(
        routes,
        "read_runtime_markers",
        lambda: {"jobs": [{"rank": 0, "phase": "ready"}], "warnings": []},
    )

    response = _client().get("/admin/api/cluster/runtime")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["phase"] == "ready"


def test_cluster_diagnostics_bundles_and_redacts_local_evidence(monkeypatch):
    monkeypatch.setattr(routes, "collect_cluster_status", lambda **_: _status())
    monkeypatch.setattr(
        routes,
        "read_runtime_markers",
        lambda: {
            "jobs": [
                {
                    "rank": 0,
                    "phase": "failed",
                    "model": f"{Path.home()}/.omlx/models/example",
                    "detail": "user@Studio.local stopped",
                    "join_key": "must-not-leak-either",
                    "pairing_token": "must-not-leak",
                }
            ],
            "warnings": [],
        },
    )
    monkeypatch.setattr(routes, "_get_engine_pool", None)
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(
            list=lambda: (),
            to_dict=lambda: {"schema_version": 1, "deployments": []},
        ),
    )

    response = _client().get("/admin/api/cluster/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["scope"] == "local-only cluster support bundle"
    assert payload["status"]["node"]["chip_name"] == "Apple M5 Max"
    marker = payload["runtime"]["jobs"][0]
    assert marker["model"].startswith("~/.omlx/")
    assert marker["detail"] == "<user>@Studio.local stopped"
    assert marker["join_key"] == "<redacted>"
    assert marker["pairing_token"] == "<redacted>"


def test_cluster_discovery_never_implies_trust(monkeypatch):
    monkeypatch.setattr(
        routes,
        "discover_all_peers",
        lambda: {
            "peers": [
                {
                    "name": "Studio",
                    "ssh": "studio.local",
                    "service": "oMLX Distributed",
                    "transport": "unknown",
                }
            ],
            "trusted": False,
            "warning": None,
            "pairing_token": "test-token",
        },
    )

    response = _client().get("/admin/api/cluster/discover")

    assert response.status_code == 200
    assert response.json()["peers"][0]["ssh"] == "studio.local"
    assert response.json()["trusted"] is False


def test_cluster_worker_smoke_route(monkeypatch):
    monkeypatch.setattr(
        routes,
        "run_worker_smoke",
        lambda **_: {
            "ok": True,
            "protocol_version": 1,
            "worker_pid": 42,
        },
    )

    response = _client().post(
        "/admin/api/cluster/worker-smoke",
        params={"timeout": 1.0},
    )

    assert response.status_code == 200
    assert response.json()["worker_pid"] == 42


def test_cluster_collective_smoke_route(monkeypatch):
    monkeypatch.setattr(
        routes,
        "run_local_collective_smoke",
        lambda **_: {
            "ok": True,
            "backend": "ring",
            "rank_count": 2,
        },
    )

    response = _client().post(
        "/admin/api/cluster/collective-smoke",
        params={"timeout": 2.0},
    )

    assert response.status_code == 200
    assert response.json()["rank_count"] == 2


def test_cluster_pipeline_smoke_route(monkeypatch):
    monkeypatch.setattr(
        routes,
        "run_local_pipeline_smoke",
        lambda **_: {
            "ok": True,
            "model_type": "nemotron_h",
            "rank_count": 2,
        },
    )

    response = _client().post(
        "/admin/api/cluster/pipeline-smoke",
        params={"timeout": 4.0},
    )

    assert response.status_code == 200
    assert response.json()["model_type"] == "nemotron_h"


def test_cluster_peer_probe_route(monkeypatch):
    monkeypatch.setattr(
        routes,
        "probe_remote_host",
        lambda ssh, route_to=None: {
            "ok": True,
            "ssh": ssh,
            "route_to": route_to,
        },
    )

    response = _client().post(
        "/admin/api/cluster/peer-probe",
        json={"ssh": "studio.local", "route_to": "192.168.5.1"},
    )

    assert response.status_code == 200
    assert response.json()["ssh"] == "studio.local"


def test_cluster_node_budgets_use_each_hosts_live_admission_ceiling(monkeypatch):
    gib = 1024**3
    asked = {}
    monkeypatch.setattr(
        "omlx.cluster.node_role._enforcer_ceiling_bytes",
        lambda: 100 * gib,
    )
    monkeypatch.setattr(
        routes,
        "probe_remote_admission_ceiling",
        lambda ssh, *, python_executable: (
            asked.update(ssh=ssh, python=python_executable) or 213 * gib + 123
        ),
    )

    response = _client().post(
        "/admin/api/cluster/node-budgets",
        json={
            "hosts": [
                {
                    "node_id": "local",
                    "ssh": "127.0.0.1",
                },
                {
                    "node_id": "studio",
                    "ssh": "studio.local",
                    "python_executable": "/opt/omlx/bin/python",
                },
            ],
            "roles": {"local": "workstation", "studio": "headless"},
        },
    )
    assert asked == {
        "ssh": "studio.local",
        "python": "/opt/omlx/bin/python",
    }

    assert response.status_code == 200
    local, studio = response.json()["nodes"]
    assert local["capacity_bytes"] == 100 * gib
    assert studio["capacity_bytes"] == 213 * gib + 123
    assert studio["capacity_source"] == "admission_ceiling"
    assert studio["capacity_bytes"] < 223 * gib


def test_cluster_node_budgets_let_the_probe_discover_an_unknown_interpreter(monkeypatch):
    """#2680: sys.executable is the coordinator's bundled binary, not the peer's."""

    gib = 1024**3
    asked = {}
    monkeypatch.setattr(
        "omlx.cluster.node_role._enforcer_ceiling_bytes",
        lambda: 100 * gib,
    )
    monkeypatch.setattr(
        routes,
        "probe_remote_admission_ceiling",
        lambda ssh, *, python_executable: (
            asked.update(ssh=ssh, python=python_executable) or 64 * gib
        ),
    )

    response = _client().post(
        "/admin/api/cluster/node-budgets",
        json={
            "hosts": [{"node_id": "studio", "ssh": "studio.local"}],
            "roles": {"studio": "headless"},
        },
    )

    assert response.status_code == 200
    assert asked == {"ssh": "studio.local", "python": None}


def test_cluster_node_budgets_reject_ssh_options_before_probing(monkeypatch):
    called = []
    monkeypatch.setattr(
        routes,
        "probe_remote_admission_ceiling",
        lambda ssh: called.append(ssh),
    )

    response = _client().post(
        "/admin/api/cluster/node-budgets",
        json={
            "hosts": [
                {
                    "node_id": "peer",
                    "ssh": "-oProxyCommand=touch /tmp/pwned",
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "invalid SSH target" in response.json()["detail"]
    assert called == []


def test_cluster_plan_route_builds_unequal_pipeline():
    gib = 1024**3

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_size_bytes": 300 * gib,
            "layer_count": 80,
            "nodes": [
                {
                    "node_id": "studio",
                    "capacity_bytes": 256 * gib,
                    "reserve_bytes": 8 * gib,
                    "memory_guard_tier": "aggressive",
                },
                {
                    "node_id": "mobile",
                    "capacity_bytes": 128 * gib,
                    "reserve_bytes": 8 * gib,
                    "memory_guard_tier": "safe",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "unequal_contiguous_pipeline"
    assert payload["assignments"][0]["node_id"] == "studio"
    assert payload["assignments"][0]["layer_count"] == 54
    assert payload["assignments"][1]["node_id"] == "mobile"
    assert payload["assignments"][1]["layer_count"] == 26
    assert [
        item["memory_guard_tier"] for item in payload["assignments"]
    ] == ["aggressive", "safe"]


def test_cluster_plan_route_requires_exactly_one_model_source():
    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "nodes": [
                {
                    "node_id": "local",
                    "capacity_bytes": 128 * 1024**3,
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_cluster_plan_route_uses_downloaded_model_headers(monkeypatch):
    from omlx.cluster.planner import ModelLayout

    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1 * 1024**3,
            layer_weight_bytes=(2 * 1024**3,) * 4,
        ),
    )

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/example",
            "nodes": [
                {
                    "node_id": "local",
                    "capacity_bytes": 16 * 1024**3,
                    "reserve_bytes": 2 * 1024**3,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"]["source"] == "/models/example"


def test_cluster_plan_context_changes_kv_memory_and_signed_placement(monkeypatch):
    from omlx.cluster.planner import ModelLayout

    gib = 1024**3
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=str(path),
            fixed_weight_bytes=gib,
            layer_weight_bytes=(gib,) * 8,
            kv_bytes_per_token_per_layer=64 * 1024,
            supports_pipeline=True,
        ),
    )
    base = {
        "model_path": "/models/context",
        "nodes": [
            {
                "node_id": "local",
                "capacity_bytes": 128 * gib,
                "reserve_bytes": 2 * gib,
            }
        ],
    }

    short = _client().post(
        "/admin/api/cluster/plan",
        json=base | {"target_context_tokens": 8192},
    )
    long = _client().post(
        "/admin/api/cluster/plan",
        json=base | {"target_context_tokens": 32768},
    )

    assert short.status_code == long.status_code == 200
    short_plan, long_plan = short.json(), long.json()
    assert short_plan["cluster"]["target_context_tokens"] == 8192
    assert long_plan["cluster"]["target_context_tokens"] == 32768
    assert long_plan["cluster"]["kv_cache_bytes"] == (
        4 * short_plan["cluster"]["kv_cache_bytes"]
    )
    assert long_plan["placement_signature"] != short_plan["placement_signature"]


def test_selected_context_is_the_runtime_kv_ceiling():
    request = SimpleNamespace(
        execution_profile="balanced",
        auto_tune=False,
        sampling_rank_only=True,
        async_overlap=True,
        cache_affinity=True,
        max_kv_size=None,
        target_context_tokens=262144,
        ring_connections_per_ip=None,
    )

    execution = routes._execution_for_request(request, (), backend="ring")

    assert execution.max_kv_size == 262144


def test_cluster_plan_route_refuses_hybrid_tp_the_worker_cannot_run(monkeypatch):
    gib = 1024**3

    from omlx.cluster.planner import ModelLayout

    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1 * gib,
            layer_weight_bytes=(2 * gib,) * 80,
            tensor_parallel_heads=32,
        ),
    )

    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/example",
            "nodes": [
                {
                    "node_id": "studio",
                    "capacity_bytes": 256 * gib,
                    "reserve_bytes": 8 * gib,
                },
                {
                    "node_id": "mobile",
                    "capacity_bytes": 256 * gib,
                    "reserve_bytes": 8 * gib,
                },
                {
                    "node_id": "ultra",
                    "capacity_bytes": 256 * gib,
                    "reserve_bytes": 8 * gib,
                },
                {
                    "node_id": "mini",
                    "capacity_bytes": 256 * gib,
                    "reserve_bytes": 8 * gib,
                },
            ],
            "tensor_parallel_size": 2,
        },
    )

    assert response.status_code == 400
    assert "must use every detected node" in response.json()["detail"]


def test_cluster_deployment_recomputes_plan_and_preflights(tmp_path, monkeypatch):
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.registry import configure_cluster_registry

    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "nemotron"
    model_path.mkdir(parents=True)
    pool = _install_ready_pool(monkeypatch, model_path, remote_only=True)
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1,
            layer_weight_bytes=(10, 10, 10, 10),
            supports_pipeline=True,
        ),
    )
    monkeypatch.setattr(
        routes,
        "preflight_remote_hosts",
        lambda deployment: [
            {"rank": rank, "node_id": host.node_id, "runtime": {"mlx": "0.32.0"}}
            for rank, host in enumerate(deployment.hosts)
        ],
    )
    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda deployment: {
            "ok": True,
            "profiles": [
                {
                    "node_id": host.node_id,
                    "rank": rank,
                    "decode_weight_bytes_per_second": 100 + rank,
                    "prefill_weight_bytes_per_second": 200 + rank,
                    "collective_latency_seconds": 0.001,
                    "collective_bandwidth_bytes_per_second": 10_000,
                    "backend": "jaccl",
                    "measured_at": "2026-07-26T12:00:00+00:00",
                    "samples": 5,
                    "source": "synthetic_mlx_probe",
                }
                for rank, host in enumerate(deployment.hosts)
            ],
        },
    )

    body = {
            "deployment_id": "nemotron-pool",
            "model_path": str(model_path),
            "backend": "jaccl",
            "nodes": [
                {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
                {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
            ],
            "hosts": [
                {
                    "node_id": "large",
                    "ssh": "127.0.0.1",
                    "ips": ["192.168.5.1"],
                    "rdma": [None, "rdma_en5"],
                },
                {
                    "node_id": "small",
                    "ssh": "studio.local",
                    "ips": ["192.168.5.2"],
                    "rdma": ["rdma_en5", None],
                },
            ],
        }
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["deployment"]["plan_hash"] == payload["plan"]["plan_hash"]
    assert payload["load_behavior"] == "eager"
    assert payload["readiness"]["state"] == "ready"
    assert payload["readiness"]["canary_passed"] is True
    assert len(payload["readiness"]["ranks"]) == 2
    assert payload["api"] == {
        "base_url": "/v1",
        "model": "public-model",
        "chat_completions": "/v1/chat/completions",
        "responses": "/v1/responses",
    }
    assert pool.cluster_registered is True
    assert pool.reloads == 1
    assert len(payload["preflight"]) == 2
    assert payload["performance_probe"]["status"] in {"applied", "placement_locked"}
    assert (
        payload["plan"]["placement_signature"]
        == body["approved_placement"]
    )
    assert payload["deployment"]["execution"]["profile"] == "balanced"

    listed = _client().get("/admin/api/cluster/deployments")
    assert listed.status_code == 200
    assert listed.json()["deployments"][0]["deployment_id"] == "nemotron-pool"

    removed = _client().delete("/admin/api/cluster/deployments/nemotron-pool")
    assert removed.status_code == 200


def test_cluster_deployment_keeps_memory_plan_when_benchmark_is_unavailable(
    tmp_path,
    monkeypatch,
):
    from omlx.cluster.launch import DistributedLaunchError
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.registry import configure_cluster_registry

    configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "fallback"
    model_path.mkdir(parents=True)
    _install_ready_pool(monkeypatch, model_path)
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1,
            layer_weight_bytes=(10, 10, 10, 10),
            supports_pipeline=True,
        ),
    )
    monkeypatch.setattr(
        routes,
        "preflight_remote_hosts",
        lambda deployment: [{"rank": 0}, {"rank": 1}],
    )
    monkeypatch.setattr(
        routes,
        "run_cluster_performance_probe",
        lambda deployment: (_ for _ in ()).throw(
            DistributedLaunchError("benchmark link unavailable")
        ),
    )

    body = {
            "deployment_id": "fallback-pool",
            "model_path": str(model_path),
            "backend": "ring",
            "nodes": [
                {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
                {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
            ],
            "hosts": [
                {
                    "node_id": "large",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.0.1"],
                },
                {
                    "node_id": "small",
                    "ssh": "studio.local",
                    "ips": ["10.0.0.2"],
                },
            ],
        }
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["performance_probe"]["status"] == "memory_fallback"
    assert payload["plan"]["optimization"] == "memory"
    assert payload["deployment"]["performance_profiles"] == []


def test_cluster_activation_rolls_back_when_canary_fails(tmp_path, monkeypatch):
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.registry import configure_cluster_registry

    registry = configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "canary-failure"
    model_path.mkdir(parents=True)
    pool = _install_ready_pool(
        monkeypatch,
        model_path,
        fail_canary=True,
    )
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1,
            layer_weight_bytes=(10, 10, 10, 10),
            supports_pipeline=True,
        ),
    )
    monkeypatch.setattr(
        routes,
        "preflight_remote_hosts",
        lambda deployment: [{"rank": 0}, {"rank": 1}],
    )

    body = {
            "deployment_id": "canary-failure",
            "model_path": str(model_path),
            "backend": "ring",
            "auto_tune": False,
            "nodes": [
                {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
                {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
            ],
            "hosts": [
                {
                    "node_id": "large",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.0.1"],
                },
                {
                    "node_id": "small",
                    "ssh": "studio.local",
                    "ips": ["10.0.0.2"],
                },
            ],
        }
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)

    assert response.status_code == 503
    assert "canary failure" in response.json()["detail"]
    assert registry.list() == ()
    assert pool.entry.engine is None
    assert pool.reloads == 2


def test_cluster_deployment_rejects_unsafe_ssh_target(tmp_path, monkeypatch):
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.registry import configure_cluster_registry

    configure_cluster_registry(tmp_path)
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=0,
            layer_weight_bytes=(1, 1),
            supports_pipeline=True,
        ),
    )

    body = {
            "model_path": str(tmp_path / "model"),
            "backend": "ring",
            "nodes": [
                {"node_id": "local", "capacity_bytes": 10},
                {"node_id": "peer", "capacity_bytes": 10},
            ],
            "hosts": [
                {
                    "node_id": "local",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.0.1"],
                },
                {
                    "node_id": "peer",
                    "ssh": "peer.local; bad",
                    "ips": ["10.0.0.2"],
                },
            ],
        }
    body["approved_placement"] = _approval_for(body)
    response = _client().post("/admin/api/cluster/deployments", json=body)

    assert response.status_code == 400
    assert "invalid SSH target" in response.json()["detail"]


def test_cluster_deployment_cannot_disable_preflight():
    response = _client().post(
        "/admin/api/cluster/deployments",
        json={
            "model_path": "/models/nemotron",
            "backend": "ring",
            "preflight": False,
            "nodes": [
                {"node_id": "local", "capacity_bytes": 10},
                {"node_id": "peer", "capacity_bytes": 10},
            ],
            "hosts": [
                {
                    "node_id": "local",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.0.1"],
                },
                {
                    "node_id": "peer",
                    "ssh": "peer.local",
                    "ips": ["10.0.0.2"],
                },
            ],
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# The fabric: addresses and backend, read rather than typed.
#
# A rank was told to bind an address that had been right the week before, and
# the launch died with EADDRNOTAVAIL. Nothing on the page had asked either Mac
# where it actually answered.
# ---------------------------------------------------------------------------


def _interfaces(name, addresses, *, rdma=(), thunderbolt=None):
    from omlx.cluster.transport import HostInterfaces, InterfaceAddress

    return HostInterfaces(
        host=name,
        addresses=tuple(InterfaceAddress(*entry) for entry in addresses),
        rdma_interfaces=frozenset(rdma),
        thunderbolt_interfaces=frozenset(rdma if thunderbolt is None else thunderbolt),
    )


def _readings(monkeypatch, hosts):
    monkeypatch.setattr(routes, "probe_host_interfaces", lambda host: hosts[host])
    monkeypatch.setattr(
        routes,
        "verify_link_reachability",
        lambda link: (True, link.reason),
    )


def _no_links():
    from omlx.cluster.transport import TransportMatrix

    return TransportMatrix(transports=(), backend="ring")


def test_transport_diagnostics_reject_ssh_options_before_running_probes(
    monkeypatch,
):
    called = []

    monkeypatch.setattr(
        routes,
        "detect_cluster_transports",
        lambda hosts: called.append(("transports", hosts)),
    )
    monkeypatch.setattr(
        routes,
        "check_peers",
        lambda hosts, **kwargs: called.append(("health", hosts)),
    )
    monkeypatch.setattr(
        routes,
        "assess_link",
        lambda hosts: called.append(("link", hosts)),
    )

    malicious = "-oProxyCommand=touch /tmp/pwned"
    endpoints = (
        "/admin/api/cluster/transports",
        "/admin/api/cluster/peer-health",
        "/admin/api/cluster/link-status",
    )
    for endpoint in endpoints:
        response = _client().get(endpoint, params={"hosts": malicious})
        assert response.status_code == 400
        assert "invalid SSH target" in response.json()["detail"]

    assert called == []


def test_transport_diagnostics_accept_user_at_host(monkeypatch):
    seen = {}

    def detect(hosts):
        seen["transports"] = hosts
        return _no_links()

    monkeypatch.setattr(
        routes,
        "detect_cluster_transports",
        detect,
    )
    monkeypatch.setattr(routes, "record_peer_transports", lambda transports: None)

    response = _client().get(
        "/admin/api/cluster/transports",
        params={"hosts": "user@studio.local"},
    )

    assert response.status_code == 200
    assert seen["transports"] == ["user@studio.local"]


def test_pairing_token_route_authenticates_with_the_shared_secret():
    secret = "correct-horse-battery-staple"
    generated = _client().post(
        "/admin/api/cluster/pairing-token",
        json={"shared_secret": secret},
    )

    assert generated.status_code == 200
    token = generated.json()["pairing_token"]
    accepted = _client().post(
        "/admin/api/cluster/verify-pairing-token",
        json={"token": token, "shared_secret": secret},
    )
    rejected = _client().post(
        "/admin/api/cluster/verify-pairing-token",
        json={"token": token, "shared_secret": "a-different-shared-secret"},
    )

    assert accepted.json() == {"valid": True}
    assert rejected.json() == {"valid": False}


def test_cuda_join_command_is_single_use_pinned_and_not_cached(
    monkeypatch, tmp_path
):
    from omlx.cluster import ssh_keys

    store = ClusterEnrollmentStore(tmp_path)
    public_key = "ssh-ed25519 " + base64.b64encode(b"controller-key-material").decode()
    fingerprint = ssh_keys.ssh_public_key_fingerprint(public_key)
    monkeypatch.setattr(routes, "get_cluster_enrollment", lambda: store)
    monkeypatch.setattr(routes, "worker_source_digest", lambda: "b" * 64)
    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(public_key=public_key, fingerprint=fingerprint),
    )

    response = _enrollment_client().post(
        "/admin/api/cluster/join-keys",
        json={
            "controller_ip": "10.42.0.10",
            "controller_port": 8000,
            "scheme": "http",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert "join_key" not in payload
    assert "--source-digest " + "b" * 64 in payload["command"]
    assert "/cluster/join/bootstrap.py" in payload["command"]
    assert payload["controller_key_fingerprint"] == fingerprint
    assert payload["single_use"] is True
    assert payload["expires_at"] - payload["created_at"] == 30 * 60
    assert '"join_key":' not in _enrollment_client().get(
        "/admin/api/cluster/join-status"
    ).text


def test_cuda_join_command_only_accepts_a_private_ipv4_controller():
    client = _enrollment_client()
    for controller_ip in ("127.0.0.1", "8.8.8.8", "fd00::10"):
        response = client.post(
            "/admin/api/cluster/join-keys",
            json={"controller_ip": controller_ip, "controller_port": 8000},
        )
        assert response.status_code == 400
        assert "controller" in response.json()["detail"]


def test_cuda_worker_claim_replay_and_completion_fail_closed(
    monkeypatch, tmp_path
):
    from omlx.cluster import ssh_keys

    store = ClusterEnrollmentStore(tmp_path)
    public_key = "ssh-ed25519 " + base64.b64encode(b"controller-key-material").decode()
    fingerprint = ssh_keys.ssh_public_key_fingerprint(public_key)
    worker_key = "ssh-ed25519 " + base64.b64encode(b"worker-key-material").decode()
    worker_fingerprint = ssh_keys.ssh_public_key_fingerprint(worker_key)
    monkeypatch.setattr(routes, "get_cluster_enrollment", lambda: store)
    monkeypatch.setattr(routes, "worker_source_bundle", lambda: b"source-bundle")
    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(public_key=public_key, fingerprint=fingerprint),
    )
    pinned = []
    monkeypatch.setattr(
        ssh_keys,
        "pin_enrolled_host_key",
        lambda **kwargs: pinned.append(kwargs),
    )
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="b" * 64,
    )
    client = _enrollment_client()

    missing = client.post("/cluster/join/claim", json=_worker_claim())
    claimed = client.post(
        "/cluster/join/claim",
        json=_worker_claim(),
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    replayed = client.post(
        "/cluster/join/claim",
        json=_worker_claim(),
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert missing.status_code == 401
    assert claimed.status_code == 200
    assert claimed.headers["cache-control"] == "no-store"
    assert replayed.status_code == 401
    assert replayed.json() == {"detail": "invalid enrollment credential"}
    session = claimed.json()["session_token"]
    source = client.get(
        "/cluster/join/source",
        headers={"Authorization": f"Bearer {session}"},
    )
    assert source.content == b"source-bundle"
    assert source.headers["cache-control"] == "no-store"

    completion = _worker_claim() | {
        "python_executable": "/opt/omlx-cluster-worker/venv/bin/python",
        "source_digest": "b" * 64,
        "ssh_host_public_key": worker_key,
        "ssh_host_fingerprint": worker_fingerprint,
        "runtime": {"mlx": "0.32.0"},
    }
    changed = client.post(
        "/cluster/join/complete",
        json=completion | {"addresses": ["10.42.0.22"]},
        headers={"Authorization": f"Bearer {session}"},
    )
    assert changed.status_code == 400
    assert "identity changed" in changed.json()["detail"]
    assert pinned == []

    completed = client.post(
        "/cluster/join/complete",
        json=completion,
        headers={"Authorization": f"Bearer {session}"},
    )
    assert completed.status_code == 200
    assert completed.json()["ssh"] == "omlxworker@10.42.0.21"
    assert pinned == [{"hostname": "10.42.0.21", "public_key": worker_key}]
    status = client.get("/admin/api/cluster/join-status")
    assert status.json()["nodes"][0]["node_id"] == "cuda-worker-1-machine"
    assert raw_key not in status.text
    assert session not in status.text


def test_cuda_worker_completion_refuses_a_forged_host_fingerprint(
    monkeypatch, tmp_path
):
    from omlx.cluster import ssh_keys

    store = ClusterEnrollmentStore(tmp_path)
    controller_key = "ssh-ed25519 " + base64.b64encode(b"controller").decode()
    monkeypatch.setattr(routes, "get_cluster_enrollment", lambda: store)
    monkeypatch.setattr(
        ssh_keys,
        "get_or_create_ssh_key",
        lambda: SimpleNamespace(
            public_key=controller_key,
            fingerprint=ssh_keys.ssh_public_key_fingerprint(controller_key),
        ),
    )
    worker_key = "ssh-ed25519 " + base64.b64encode(
        b"worker-host-key-material-long-enough"
    ).decode()
    raw_key, _ = store.issue_join_key(
        controller_url="http://10.42.0.10:8000",
        source_digest="b" * 64,
    )
    client = _enrollment_client()
    claimed = client.post(
        "/cluster/join/claim",
        json=_worker_claim(),
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    session = claimed.json()["session_token"]
    response = client.post(
        "/cluster/join/complete",
        json=_worker_claim()
        | {
            "python_executable": "/opt/omlx-cluster-worker/venv/bin/python",
            "source_digest": "b" * 64,
            "ssh_host_public_key": worker_key,
            "ssh_host_fingerprint": "SHA256:" + "A" * 43,
        },
        headers={"Authorization": f"Bearer {session}"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "worker SSH fingerprint mismatch"


def test_key_exchange_routes_keep_the_shared_secret_out_of_the_url(monkeypatch):
    from omlx.cluster import ssh_keys

    calls = []

    def generate(**kwargs):
        calls.append(("generate", kwargs))
        return "authenticated-exchange-token"

    def exchange(**kwargs):
        calls.append(("exchange", kwargs))
        return {"success": True}

    monkeypatch.setattr(ssh_keys, "generate_key_exchange_for_peer", generate)
    monkeypatch.setattr(ssh_keys, "exchange_keys_with_peer", exchange)
    secret = "correct-horse-battery-staple"

    generated = _client().post(
        "/admin/api/cluster/ssh-key/exchange-token",
        json={"node_id": "studio.local", "shared_secret": secret},
    )
    exchanged = _client().post(
        "/admin/api/cluster/ssh-key/exchange",
        json={
            "exchange_token": "authenticated-exchange-token",
            "shared_secret": secret,
        },
    )

    assert generated.json() == {"exchange_token": "authenticated-exchange-token"}
    assert exchanged.json() == {"success": True}
    assert calls == [
        (
            "generate",
            {"node_id": "studio.local", "shared_secret": secret},
        ),
        (
            "exchange",
            {
                "peer_token": "authenticated-exchange-token",
                "shared_secret": secret,
            },
        ),
    ]


def test_the_fabric_reports_the_address_each_mac_answers_on(monkeypatch):
    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1", [("en4", "10.0.1.1", 30)], rdma={"en4"}
            ),
            "studio.local": _interfaces(
                "studio.local", [("en5", "10.0.1.2", 30)], rdma={"en5"}
            ),
        },
    )

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"})
        .json()
    )

    assert body["ok"]
    assert [host["ips"] for host in body["hosts"]] == [["10.0.1.1"], ["10.0.1.2"]]
    assert body["backend"] == "jaccl"
    assert body["fell_back"] is False
    # Entry [i][j] is the device rank i uses to reach rank j: what a jaccl
    # deployment demands and what a pair of device names cannot express.
    assert [host["rdma"] for host in body["hosts"]] == [
        [None, "rdma_en4"],
        ["rdma_en5", None],
    ]


def test_the_fabric_rejects_a_shared_subnet_that_does_not_answer(monkeypatch):
    """Matching netmasks are a candidate, not proof of a working cable."""

    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1", [("en4", "10.0.1.1", 30)], rdma={"en4"}
            ),
            "studio.local": _interfaces(
                "studio.local", [("en5", "10.0.1.2", 30)], rdma={"en5"}
            ),
        },
    )
    monkeypatch.setattr(
        routes,
        "verify_link_reachability",
        lambda _link: (
            False,
            "127.0.0.1 routes the address, but the peer did not answer.",
        ),
    )

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"})
        .json()
    )

    assert body["ok"] is False
    assert body["backend"] == "ring"
    assert all(host["ips"] == [] for host in body["hosts"])
    assert "did not answer" in body["backend_reason"]


def test_the_fabric_reports_an_unpaired_mac_instead_of_a_scrubbed_500(monkeypatch):
    """An SSH failure is the peer's state, not a server fault (#2423 tester A1).

    The unhandled raise became a generic 500 whose body FastAPI scrubs, which
    swallowed the permission-denied stderr the dashboard maps to pairing
    guidance.
    """

    def _refuse(_hosts):
        raise RuntimeError(
            "reading interfaces on studio.local failed: "
            "user@studio.local: Permission denied (publickey,password)."
        )

    monkeypatch.setattr(routes, "_resolve_fabric", _refuse)

    response = _client().get(
        "/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"}
    )

    assert response.status_code == 503
    assert "Permission denied" in response.json()["detail"]


def test_link_status_redacts_the_username_but_keeps_the_evidence(monkeypatch):
    """Remote stderr leaves the admin API without the local account name."""

    from omlx.cluster.transport import LinkStatus

    monkeypatch.setattr(
        routes,
        "assess_link",
        lambda _hosts: LinkStatus(
            state="unreachable",
            title="The other Mac rejected the login",
            detail="user@studio.local: Permission denied (publickey).",
            backend="ring",
            ready=False,
            commands=("ssh-copy-id -i ~/.ssh/omlx_cluster.pub user@studio.local",),
        ),
    )

    body = (
        _client()
        .get(
            "/admin/api/cluster/link-status",
            params={"hosts": "127.0.0.1,studio.local"},
        )
        .json()
    )

    assert "<user>@studio.local" in body["detail"]
    assert "Permission denied" in body["detail"]
    # Pasteable fixes keep their real target untouched.
    assert body["commands"] == [
        "ssh-copy-id -i ~/.ssh/omlx_cluster.pub user@studio.local"
    ]


def test_the_fabric_falls_back_from_unreachable_rdma_to_verified_ethernet(
    monkeypatch,
):
    """Dual-network Macs use the fastest path that actually answers."""

    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1",
                [("en4", "10.0.1.1", 30), ("en0", "192.168.4.21", 24)],
                rdma={"en4"},
            ),
            "studio.local": _interfaces(
                "studio.local",
                [("en5", "10.0.1.2", 30), ("en0", "192.168.4.22", 24)],
                rdma={"en5"},
            ),
        },
    )
    checked = []

    def verify(link):
        checked.append(link.kind)
        return (link.kind == "ethernet", f"{link.kind} did not answer")

    monkeypatch.setattr(routes, "verify_link_reachability", verify)

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"})
        .json()
    )

    assert checked == ["rdma", "ethernet"]
    assert body["ok"] is True
    assert body["backend"] == "ring"
    assert body["rdma"]["ok"] is False
    assert "verified" in body["rdma"]["reason"]
    assert [host["ips"] for host in body["hosts"]] == [
        ["192.168.4.21"],
        ["192.168.4.22"],
    ]


def test_the_fabric_verifies_every_pair_not_only_the_coordinator_links(monkeypatch):
    hosts = ("127.0.0.1", "studio.local", "mini.local")
    _readings(
        monkeypatch,
        {
            host: _interfaces(
                host,
                [(f"en{index + 4}", f"10.0.1.{index + 1}", 24)],
                rdma={f"en{index + 4}"},
            )
            for index, host in enumerate(hosts)
        },
    )
    checked = []

    def verify(link):
        pair = frozenset((link.source.host, link.peer.host))
        checked.append(pair)
        if pair == frozenset(("studio.local", "mini.local")):
            return False, "studio.local cannot reach mini.local"
        return True, link.reason

    monkeypatch.setattr(routes, "verify_link_reachability", verify)

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": ",".join(hosts)})
        .json()
    )

    assert set(checked) == {
        frozenset(("127.0.0.1", "studio.local")),
        frozenset(("127.0.0.1", "mini.local")),
        frozenset(("studio.local", "mini.local")),
    }
    assert body["ok"] is False
    assert body["backend"] == "ring"
    assert body["rdma"]["ok"] is False
    assert "studio.local cannot reach mini.local" in body["blocker"]


def test_a_thunderbolt_pair_without_rdma_falls_back_to_the_ring_out_loud(monkeypatch):
    """A Thunderbolt link makes choose_backend say jaccl-ring, which needs a matrix.

    Posting that backend with no matrix dies inside ClusterDeployment's
    constructor, where there is nothing the user can act on.
    """

    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1", [("en4", "10.0.1.1", 30)], rdma={"en4"}
            ),
            "studio.local": _interfaces(
                "studio.local", [("en5", "10.0.1.2", 30)], thunderbolt={"en5"}
            ),
        },
    )

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"})
        .json()
    )

    assert body["backend"] == "ring"
    assert body["fell_back"] is True
    assert "studio.local" in body["backend_reason"]
    assert "ring" in body["backend_reason"]
    assert all(host["rdma"] == [] for host in body["hosts"])


def test_macs_that_share_no_subnet_get_no_address_and_a_reason(monkeypatch):
    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces("127.0.0.1", [("en0", "192.168.4.21", 24)]),
            "studio.local": _interfaces("studio.local", [("en0", "10.9.9.9", 24)]),
        },
    )

    body = (
        _client()
        .get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1,studio.local"})
        .json()
    )

    assert not body["ok"]
    assert all(host["ips"] == [] for host in body["hosts"])
    assert "share no subnet" in body["backend_reason"]


def test_one_host_is_not_a_fabric():
    response = _client().get("/admin/api/cluster/fabric", params={"hosts": "127.0.0.1"})

    assert response.status_code == 400


def test_fabric_rejects_an_ssh_option_before_running_a_probe(monkeypatch):
    def unexpected_probe(_host):
        raise AssertionError("an invalid SSH target reached the probe")

    monkeypatch.setattr(
        routes,
        "probe_host_interfaces",
        unexpected_probe,
    )

    response = _client().get(
        "/admin/api/cluster/fabric",
        params={"hosts": "127.0.0.1,-oProxyCommand=bad"},
    )

    assert response.status_code == 400
    assert "invalid SSH target" in response.json()["detail"]


def test_autoconfigure_rejects_an_invalid_manual_ip_before_planning():
    response = _client().post(
        "/admin/api/cluster/autoconfigure",
        json={
            "model_size_bytes": 40 * 1024**3,
            "layer_count": 32,
            "nodes": [
                {"node_id": f"node-{index}", "capacity_bytes": 64 * 1024**3}
                for index in range(2)
            ],
            "hosts": [
                {
                    "node_id": "node-0",
                    "ssh": "127.0.0.1",
                    "ips": ["not-an-ip"],
                },
                {
                    "node_id": "node-1",
                    "ssh": "studio.local",
                    "ips": ["10.77.0.2"],
                },
            ],
            "detect_transports": False,
        },
    )

    assert response.status_code == 400
    assert "invalid communication IP" in response.text


def test_autoconfigure_activates_on_discovered_addresses_not_supplied_ones(
    monkeypatch,
):
    """The proposal must carry what was read, not what the page happened to send."""

    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1", [("en4", "10.0.1.1", 30)], rdma={"en4"}
            ),
            "studio.local": _interfaces(
                "studio.local", [("en5", "10.0.1.2", 30)], rdma={"en5"}
            ),
        },
    )
    monkeypatch.setattr(routes, "detect_cluster_transports", lambda hosts: _no_links())
    monkeypatch.setattr(routes, "probe_remote_host", lambda ssh, **_: None)

    body = (
        _client()
        .post(
            "/admin/api/cluster/autoconfigure",
            json={
                "model_size_bytes": 40 * 1024**3,
                "layer_count": 32,
                "nodes": [
                    {"node_id": f"node-{index}", "capacity_bytes": 64 * 1024**3}
                    for index in range(2)
                ],
                "hosts": [
                    {"node_id": "node-0", "ssh": "127.0.0.1", "ips": ["192.168.4.21"]},
                    {
                        "node_id": "node-1",
                        "ssh": "studio.local",
                        "ips": ["192.168.4.22"],
                    },
                ],
            },
        )
        .json()
    )

    assert [host["ips"] for host in body["activation"]["hosts"]] == [
        ["10.0.1.1"],
        ["10.0.1.2"],
    ]
    assert body["activation"]["backend"] == body["backend"] == "jaccl"
    assert [host["rdma"] for host in body["activation"]["hosts"]] == [
        [None, "rdma_en4"],
        ["rdma_en5", None],
    ]


def test_autoconfigure_uses_verified_ethernet_when_rdma_is_unreachable(
    monkeypatch,
):
    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces(
                "127.0.0.1",
                [("en4", "10.0.1.1", 30), ("en0", "192.168.4.21", 24)],
                rdma={"en4"},
            ),
            "studio.local": _interfaces(
                "studio.local",
                [("en5", "10.0.1.2", 30), ("en0", "192.168.4.22", 24)],
                rdma={"en5"},
            ),
        },
    )
    monkeypatch.setattr(
        routes,
        "verify_link_reachability",
        lambda link: (link.kind == "ethernet", f"{link.kind} did not answer"),
    )
    monkeypatch.setattr(routes, "detect_cluster_transports", lambda _hosts: _no_links())
    monkeypatch.setattr(routes, "probe_remote_host", lambda _ssh, **_: None)

    body = (
        _client()
        .post(
            "/admin/api/cluster/autoconfigure",
            json={
                "model_size_bytes": 40 * 1024**3,
                "layer_count": 32,
                "nodes": [
                    {"node_id": f"node-{index}", "capacity_bytes": 64 * 1024**3}
                    for index in range(2)
                ],
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
                "preflight": False,
                "auto_tune": False,
            },
        )
        .json()
    )

    assert body["backend"] == "ring"
    assert body["activation"]["backend"] == "ring"
    assert [host["ips"] for host in body["activation"]["hosts"]] == [
        ["192.168.4.21"],
        ["192.168.4.22"],
    ]
    assert [host["rdma"] for host in body["activation"]["hosts"]] == [[], []]


def test_autoconfigure_preserves_manual_addresses_when_detection_is_disabled(
    monkeypatch,
):
    """An operator-selected route must survive all the way to activation."""

    def unexpected_discovery(*_args, **_kwargs):
        raise AssertionError("manual addressing must not trigger rediscovery")

    monkeypatch.setattr(routes, "detect_cluster_transports", unexpected_discovery)
    monkeypatch.setattr(routes, "_resolve_fabric", unexpected_discovery)

    body = (
        _client()
        .post(
            "/admin/api/cluster/autoconfigure",
            json={
                "model_size_bytes": 40 * 1024**3,
                "layer_count": 32,
                "nodes": [
                    {"node_id": f"node-{index}", "capacity_bytes": 64 * 1024**3}
                    for index in range(2)
                ],
                "hosts": [
                    {
                        "node_id": "node-0",
                        "ssh": "127.0.0.1",
                        "ips": ["10.77.0.1"],
                    },
                    {
                        "node_id": "node-1",
                        "ssh": "studio.local",
                        "ips": ["10.77.0.2"],
                    },
                ],
                "detect_transports": False,
                "preflight": False,
                "auto_tune": False,
            },
        )
        .json()
    )

    assert body["fabric"] is None
    assert body["backend"] == "ring"
    assert [host["ips"] for host in body["activation"]["hosts"]] == [
        ["10.77.0.1"],
        ["10.77.0.2"],
    ]
    assert [host["rdma"] for host in body["activation"]["hosts"]] == [[], []]


def test_a_peer_that_cannot_import_blocks_activation_with_its_fix(
    monkeypatch, tmp_path
):
    """Rank 1 died on "No module named 'mlx_vlm'" after every rank paid for weights."""

    from omlx.cluster.autoconfigure import PreflightIssue
    from omlx.cluster.planner import ModelLayout

    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=str(path),
            fixed_weight_bytes=1024**3,
            layer_weight_bytes=(1024**3,) * 8,
            supports_pipeline=True,
        ),
    )
    monkeypatch.setattr(routes, "detect_cluster_transports", lambda hosts: _no_links())
    monkeypatch.setattr(routes, "probe_remote_host", lambda ssh, **_: None)
    monkeypatch.setattr(
        routes,
        "peer_import_issues",
        lambda hosts, **_: (
            PreflightIssue(
                node_id="node-1",
                kind="import_missing",
                detail="/opt/venv/bin/python cannot import mlx_vlm",
                remediation='ssh studio.local "uv pip install mlx-vlm"',
            ),
        ),
    )
    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces("127.0.0.1", [("en4", "10.0.1.1", 30)]),
            "studio.local": _interfaces("studio.local", [("en5", "10.0.1.2", 30)]),
        },
    )

    body = (
        _client()
        .post(
            "/admin/api/cluster/autoconfigure",
            json={
                "model_path": str(model),
                "model_size_bytes": None,
                "layer_count": 32,
                "nodes": [
                    {"node_id": f"node-{index}", "capacity_bytes": 64 * 1024**3}
                    for index in range(2)
                ],
                "hosts": [
                    {"node_id": "node-0", "ssh": "127.0.0.1", "ips": ["10.0.1.1"]},
                    {"node_id": "node-1", "ssh": "studio.local", "ips": ["10.0.1.2"]},
                ],
            },
        )
        .json()
    )

    assert body["ready_to_activate"] is False
    issue = next(
        item for item in body["preflight_issues"] if item["kind"] == "import_missing"
    )
    # The command has to survive to the page verbatim: a summary that names the
    # problem and hides the one-line fix sends the user back to the logs.
    assert issue["remediation"] == 'ssh studio.local "uv pip install mlx-vlm"'


def test_cuda_fabric_verification_is_a_gui_api_and_uses_live_peer_facts(monkeypatch):
    def peer(ssh, **_kwargs):
        suffix = "1" if ssh == "spark-a.local" else "2"
        return {
            "runtime_compatible": True,
            "status": {
                "node": {
                    "accelerator": "cuda",
                    "distributed_backends": ["ring", "nccl"],
                    "fabric_kind": "connectx-7",
                },
                "transport": {
                    "rdma": {
                        "devices": ["mlx5_0", "mlx5_1"],
                        "addresses": {
                            "mlx5_0": f"192.168.100.{suffix}",
                            "mlx5_1": f"192.168.101.{suffix}",
                        },
                        "network_interfaces": {
                            "mlx5_0": "enp1s0f0np0",
                            "mlx5_1": "enp2s0f0np0",
                        },
                    }
                },
            },
        }

    captured = {}

    def verify(hosts):
        captured["hosts"] = hosts
        return {
            "ok": True,
            "verified": True,
            "group_id": "connectx-verified",
            "transport": "nccl",
            "payload_bytes_per_second": 3 * 1024**3,
        }

    monkeypatch.setattr(routes, "probe_remote_host", peer)
    monkeypatch.setattr(routes, "run_cuda_fabric_probe", verify)

    response = _client().post(
        "/admin/api/cluster/cuda-fabric/verify",
        json={
            "hosts": [
                {"node_id": "spark-a", "ssh": "spark-a.local"},
                {"node_id": "spark-b", "ssh": "spark-b.local"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["verified"] is True
    assert [host.ips for host in captured["hosts"]] == [
        ("192.168.100.1", "192.168.101.1"),
        ("192.168.100.2", "192.168.101.2"),
    ]
    assert all(
        host.interfaces == ("enp1s0f0np0", "enp2s0f0np0")
        for host in captured["hosts"]
    )


def test_cuda_fabric_verification_refuses_unmeasured_connectx(monkeypatch):
    monkeypatch.setattr(
        routes,
        "probe_remote_host",
        lambda *_args, **_kwargs: {
            "status": {
                "node": {
                    "accelerator": "cuda",
                    "distributed_backends": ["nccl"],
                    "fabric_kind": "connectx-7",
                },
                "transport": {
                    "rdma": {
                        "devices": ["mlx5_0"],
                        "addresses": {},
                        "network_interfaces": {"mlx5_0": "enp1s0f0np0"},
                    }
                },
            }
        },
    )

    response = _client().post(
        "/admin/api/cluster/cuda-fabric/verify",
        json={
            "hosts": [
                {"node_id": "spark-a", "ssh": "spark-a.local"},
                {"node_id": "spark-b", "ssh": "spark-b.local"},
            ]
        },
    )

    assert response.status_code == 409
    assert "no direct-link IP" in response.json()["detail"]


def test_verified_cuda_pair_is_kept_adjacent_in_outer_ring():
    hosts = [
        routes.ClusterHostRequest(node_id="mac", ssh="127.0.0.1", ips=["10.0.0.1"]),
        routes.ClusterHostRequest(node_id="spark-a", ssh="spark-a", ips=["10.0.0.2"]),
        routes.ClusterHostRequest(node_id="other", ssh="other", ips=["10.0.0.3"]),
        routes.ClusterHostRequest(node_id="spark-b", ssh="spark-b", ips=["10.0.0.4"]),
    ]
    nodes = [
        routes.ClusterPlanNodeRequest(node_id="mac", capacity_bytes=64 * 1024**3),
        routes.ClusterPlanNodeRequest(
            node_id="spark-a",
            capacity_bytes=115 * 1024**3,
            accelerator="cuda",
            fabric_group_id="connectx-proof",
            fabric_verified=True,
        ),
        routes.ClusterPlanNodeRequest(node_id="other", capacity_bytes=64 * 1024**3),
        routes.ClusterPlanNodeRequest(
            node_id="spark-b",
            capacity_bytes=115 * 1024**3,
            accelerator="cuda",
            fabric_group_id="connectx-proof",
            fabric_verified=True,
        ),
    ]

    ordered = routes._coalesce_verified_cuda_groups(
        ["127.0.0.1", "spark-a", "other", "spark-b"],
        hosts,
        nodes,
    )

    assert ordered == ["127.0.0.1", "spark-a", "spark-b", "other"]


def _incident_store(tmp_path, monkeypatch):
    """Configure a fresh incident store without leaking into other tests."""

    from omlx.cluster import incidents as incidents_module
    from omlx.cluster.incidents import IncidentStore

    store = IncidentStore(tmp_path)
    monkeypatch.setattr(incidents_module, "_configured_incidents", store)
    return store


def test_activation_failure_records_matching_incident(tmp_path, monkeypatch):
    from omlx.cluster.planner import PlanningError

    store = _incident_store(tmp_path, monkeypatch)

    def raise_planning(_request):
        raise PlanningError("the model does not fit the approved budgets")

    monkeypatch.setattr(routes, "_create_deployment", raise_planning)
    body = {
        "model_path": "~/.omlx/models/example",
        "backend": "ring",
        "nodes": [
            {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
            {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
        ],
        "hosts": [
            {"node_id": "large", "ssh": "127.0.0.1", "ips": ["192.168.5.1"]},
            {"node_id": "small", "ssh": "studio.local", "ips": ["192.168.5.2"]},
        ],
        "approved_placement": "0" * 32,
    }

    response = _client().post("/admin/api/cluster/deployments", json=body)

    assert response.status_code == 400
    detail = response.json()["detail"]
    recorded = store.list()
    assert len(recorded) == 1
    incident = recorded[0]
    assert incident.message == detail
    assert incident.severity == "error"
    assert incident.state_code == "activation_rejected"
    assert incident.source == "coordinator"
    assert incident.guidance_code


def test_cluster_incidents_cursor_only_returns_unseen(tmp_path, monkeypatch):
    from omlx.cluster.incidents import Severity

    store = _incident_store(tmp_path, monkeypatch)
    first = store.record(
        Severity.ERROR, "coordinator", "activation_launch_failed", "rank 1 died"
    )
    store.record(
        Severity.WARN, "peer:studio", "peer_unhealthy", "studio stopped answering"
    )
    client = _client()

    everything = client.get("/admin/api/cluster/incidents?since=0")
    assert everything.status_code == 200
    payload = everything.json()
    assert [item["message"] for item in payload["incidents"]] == [
        "rank 1 died",
        "studio stopped answering",
    ]
    latest = payload["latest_seq"]
    assert latest > first.seq

    unseen = client.get(f"/admin/api/cluster/incidents?since={first.seq}")
    assert [item["message"] for item in unseen.json()["incidents"]] == [
        "studio stopped answering"
    ]

    caught_up = client.get(f"/admin/api/cluster/incidents?since={latest}")
    caught_up_payload = caught_up.json()
    assert caught_up_payload["incidents"] == []
    assert caught_up_payload["latest_seq"] == latest
    # The epoch names the seq numbering so a client can detect a log reset.
    assert isinstance(caught_up_payload["epoch"], str) and caught_up_payload["epoch"]


def test_dismissed_incident_stays_in_diagnostics_bundle(tmp_path, monkeypatch):
    from omlx.cluster.incidents import Severity

    store = _incident_store(tmp_path, monkeypatch)
    incident = store.record(
        Severity.ERROR,
        "coordinator",
        "staging_failed",
        "Model staging failed on small",
    )
    monkeypatch.setattr(routes, "collect_cluster_status", lambda **_: _status())
    monkeypatch.setattr(
        routes, "read_runtime_markers", lambda: {"jobs": [], "warnings": []}
    )
    monkeypatch.setattr(routes, "_get_engine_pool", None)
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(
            list=lambda: (),
            to_dict=lambda: {"schema_version": 1, "deployments": []},
        ),
    )
    client = _client()

    dismissed = client.post(f"/admin/api/cluster/incidents/{incident.id}/dismiss")
    assert dismissed.status_code == 200
    assert dismissed.json() == {"ok": True}
    missing = client.post("/admin/api/cluster/incidents/nope/dismiss")
    assert missing.status_code == 404

    bundle = client.get("/admin/api/cluster/diagnostics")
    assert bundle.status_code == 200
    bundled = bundle.json()["incidents"]
    assert len(bundled) == 1
    assert bundled[0]["id"] == incident.id
    assert bundled[0]["dismissed_at"] is not None

    # Dismissal is invisible to the cursor: the record keeps its seq, so a
    # caller that already saw it is not re-sent a deletion it could act on.
    feed = client.get("/admin/api/cluster/incidents?since=0").json()
    assert feed["incidents"][0]["dismissed_at"] is not None


def test_peer_health_transition_records_one_incident(tmp_path, monkeypatch):
    store = _incident_store(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "_PEER_HEALTH_LAST_HEALTHY", {})
    monkeypatch.setattr(
        routes, "describe_failure", lambda health: "studio stopped answering"
    )

    def fake_peer(healthy):
        return SimpleNamespace(
            healthy=healthy,
            node_id="studio",
            to_dict=lambda: {"node_id": "studio", "healthy": healthy},
        )

    state = {"healthy": True}
    monkeypatch.setattr(
        routes,
        "check_peers",
        lambda hosts_by_rank, **_: (fake_peer(state["healthy"]),),
    )
    client = _client()
    url = "/admin/api/cluster/peer-health?hosts=studio.local&deployment_id=d1"

    assert client.get(url).json()["healthy"] is True
    assert store.list() == ()

    # The first unhealthy poll after a healthy one narrates the death — once
    # per transition, not once per poll.
    state["healthy"] = False
    assert client.get(url).json()["healthy"] is False
    assert client.get(url).json()["healthy"] is False
    recorded = store.list()
    assert len(recorded) == 1
    incident = recorded[0]
    assert incident.state_code == "peer_unhealthy"
    assert incident.source == "peer:studio"
    assert incident.deployment_id == "d1"
    assert incident.message == "studio stopped answering"

    # A never-healthy deployment (fresh server) does not emit on sight.
    fresh = "/admin/api/cluster/peer-health?hosts=studio.local&deployment_id=d2"
    assert client.get(fresh).json()["healthy"] is False
    assert len(store.list()) == 1
