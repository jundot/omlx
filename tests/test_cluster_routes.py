# SPDX-License-Identifier: Apache-2.0
"""Tests for authenticated cluster admin route handlers."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.exceptions import ModelBusyError


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _plan(body: dict) -> dict:
    """Plan and sign the way activation does."""

    plan = routes._create_cluster_plan(routes.ClusterPlanRequest(**body))
    return routes._plan_with_signature(plan.to_dict())


def _approval_for(payload: dict) -> str:
    """Placement identity for a direct activation request."""

    plan = routes._create_cluster_plan(
        routes.ClusterPlanRequest(
            model_path=payload["model_path"],
            nodes=payload["nodes"],
            execution_profile=payload.get("execution_profile", "balanced"),
            tensor_parallel_size=payload.get("tensor_parallel_size", 1),
            target_context_tokens=payload.get("target_context_tokens", 8192),
            mtp=payload.get("mtp", False),
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
        reload_busy: bool = False,
    ):
        self.model_path = model_path
        self.model_id = "public-model"
        self.fail_canary = fail_canary
        self.remote_only = remote_only
        self.reload_busy = reload_busy
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
        if self.entry.engine is not None:
            raise RuntimeError(
                f"cluster model '{model_id}' is still loaded and cannot be removed"
            )
        self.cluster_registered = False
        return True

    def get_entry(self, model_id):
        assert model_id == self.model_id
        return self.entry

    async def prepare_cluster_reload(self, model_id):
        assert model_id == self.model_id
        if self.reload_busy and self.entry.engine is not None:
            raise ModelBusyError(model_id, "activate distributed cluster")
        self.reloads += 1
        self.entry.engine = None
        self.entry.pending_unload_reason = None

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
    reload_busy=False,
):
    pool = _ReadyClusterPool(
        str(model_path),
        fail_canary=fail_canary,
        remote_only=remote_only,
        reload_busy=reload_busy,
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    # These route tests exercise activation after peer reachability has been
    # established. Do not let their synthetic ``studio.local`` host escape to
    # the real SSH liveness probe; peer-loss behavior has dedicated coverage.
    monkeypatch.setattr(routes, "check_peers", lambda *args, **kwargs: ())
    pool.verified_teardowns = []
    monkeypatch.setattr(
        routes,
        "stop_deployment_processes",
        lambda deployment: (
            pool.verified_teardowns.append(deployment.deployment_id)
            or {
                "verified": True,
                "deployment_id": deployment.deployment_id,
                "ranks_checked": deployment.world_size,
                "manifest_retired": False,
            }
        ),
    )
    return pool


def test_cluster_runtime_route_is_lightweight(monkeypatch):
    monkeypatch.setattr(
        routes,
        "read_runtime_markers",
        lambda: {"jobs": [{"rank": 0, "phase": "ready"}], "warnings": []},
    )

    response = _client().get("/admin/api/cluster/runtime")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["phase"] == "ready"


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


def test_cluster_node_budgets_let_the_probe_discover_an_unknown_interpreter(
    monkeypatch,
):
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


def test_cluster_plan_builds_unequal_pipeline():
    gib = 1024**3

    payload = _plan(
        {
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
        }
    )

    assert payload["strategy"] == "unequal_contiguous_pipeline"
    assert payload["assignments"][0]["node_id"] == "studio"
    assert payload["assignments"][0]["layer_count"] == 54
    assert payload["assignments"][1]["node_id"] == "mobile"
    assert payload["assignments"][1]["layer_count"] == 26
    assert [item["memory_guard_tier"] for item in payload["assignments"]] == [
        "aggressive",
        "safe",
    ]


def test_cluster_plan_requires_exactly_one_model_source():
    from omlx.cluster.planner import PlanningError

    with pytest.raises(PlanningError, match="exactly one"):
        _plan({"nodes": [{"node_id": "local", "capacity_bytes": 128 * 1024**3}]})


def test_cluster_plan_uses_downloaded_model_headers(monkeypatch):
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

    payload = _plan(
        {
            "model_path": "/models/example",
            "nodes": [
                {
                    "node_id": "local",
                    "capacity_bytes": 16 * 1024**3,
                    "reserve_bytes": 2 * 1024**3,
                }
            ],
        }
    )

    assert payload["model"]["source"] == "/models/example"


def test_new_cluster_plans_follow_the_saved_lightning_mtp_setting(monkeypatch):
    from omlx.cluster.planner import ModelLayout, PlanningError

    head = ModelLayout("/models/draft", 0, (1,), mtp_weight_bytes=1)
    plain = ModelLayout("/models/plain", 0, (1,))
    saved = {}
    pool = SimpleNamespace(
        get_status=lambda: {
            "models": [
                {"id": "draft", "model_path": "/models/draft", "model_type": "llm"}
            ]
        }
    )
    manager = SimpleNamespace(
        has_settings=lambda model_id: model_id in saved,
        get_settings=lambda model_id: SimpleNamespace(mtp_enabled=saved[model_id]),
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)

    # Like a single Mac, a checkpoint with a head starts with MTP on.
    assert routes._cluster_mtp(None, head, "/models/draft") is True
    saved["draft"] = False
    assert routes._cluster_mtp(None, head, "/models/draft") is False
    assert routes._cluster_mtp(True, head, "/models/draft") is True
    assert routes._cluster_mtp(None, plain, "/models/plain") is False
    with pytest.raises(PlanningError, match="no Lightning MTP head"):
        routes._cluster_mtp(True, plain, None)


def test_cluster_plan_follows_the_saved_engram_ssd_offload_setting(monkeypatch):
    from omlx.cluster.planner import ModelLayout

    gib = 1024**3
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=str(path),
            fixed_weight_bytes=gib,
            layer_weight_bytes=(4 * gib,) * 4,
            tensor_parallel_ssd_layer_bytes=(0, 3 * gib, 0, 0),
            supports_pipeline=True,
        ),
    )
    saved = {"v41": True}
    pool = SimpleNamespace(
        get_status=lambda: {
            "models": [{"id": "v41", "model_path": "/models/v41", "model_type": "llm"}]
        }
    )
    manager = SimpleNamespace(
        has_settings=lambda model_id: model_id in saved,
        get_settings=lambda model_id: SimpleNamespace(
            deepseek_v41_engram_ssd_offload=saved[model_id]
        ),
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    body = {
        "model_path": "/models/v41",
        "nodes": [
            {"node_id": "local", "capacity_bytes": 128 * gib, "reserve_bytes": 2 * gib}
        ],
    }

    offloaded = _plan(body)
    saved["v41"] = False
    resident = _plan(body)

    assert offloaded["assignments"][0]["ssd_weight_bytes"] == 3 * gib
    assert resident["assignments"][0]["ssd_weight_bytes"] == 0
    assert offloaded["placement_signature"] != resident["placement_signature"]


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

    short_plan = _plan(base | {"target_context_tokens": 8192})
    long_plan = _plan(base | {"target_context_tokens": 32768})

    assert short_plan["cluster"]["target_context_tokens"] == 8192
    assert long_plan["cluster"]["target_context_tokens"] == 32768
    assert long_plan["cluster"]["kv_cache_bytes"] == (
        4 * short_plan["cluster"]["kv_cache_bytes"]
    )
    assert long_plan["placement_signature"] != short_plan["placement_signature"]


def test_cluster_plan_validates_and_signs_a_tensor_split(monkeypatch):
    from omlx.cluster.planner import ModelLayout, PlanningError

    gib = 1024**3
    layout = ModelLayout(
        source="v41",
        fixed_weight_bytes=gib,
        layer_weight_bytes=(8 * gib,) * 8,
        tensor_parallel_divisors=(64, 8, 36),
        supports_tensor_parallel=True,
        tensor_parallel_split_units=(8, 36),
        tensor_parallel_split_bytes=(4 * gib, 56 * gib),
    )
    monkeypatch.setattr(routes, "inspect_safetensors_layout", lambda path: layout)
    nodes = [
        {"node_id": n, "capacity_bytes": c * gib} for n, c in (("a", 256), ("b", 128))
    ]

    def plan(split):
        payload = {"model_path": "/m", "tensor_parallel_size": 2, "nodes": nodes}
        return _plan(payload | {"tensor_split": split})

    first = plan({"attention": [5, 3], "experts": [25, 11]})
    second = plan({"attention": [5, 3], "experts": [24, 12]})
    # Approving one split must not approve another.
    assert first["placement_signature"] != second["placement_signature"]
    with pytest.raises(PlanningError):
        plan({"attention": [5]})


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


def test_cluster_plan_refuses_hybrid_tp_the_worker_cannot_run(monkeypatch):
    gib = 1024**3

    from omlx.cluster.planner import ModelLayout, PlanningError

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

    with pytest.raises(PlanningError, match="must use every detected node"):
        _plan(
            {
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
            }
        )


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
    assert payload["plan"]["placement_signature"] == body["approved_placement"]
    assert payload["deployment"]["execution"]["profile"] == "balanced"

    listed = _client().get("/admin/api/cluster/deployments")
    assert listed.status_code == 200
    assert listed.json()["deployments"][0]["deployment_id"] == "nemotron-pool"

    unloaded = _client().post("/admin/api/cluster/deployments/nemotron-pool/unload")
    assert unloaded.status_code == 200, unloaded.json()
    assert unloaded.json()["configured"] is True
    assert unloaded.json()["stopped"] is True
    assert unloaded.json()["teardown"]["verified"] is True
    assert pool.verified_teardowns == ["nemotron-pool"]
    assert pool.entry.engine is None
    assert routes.get_cluster_registry().get("nemotron-pool") is not None

    loaded = _client().post("/admin/api/cluster/deployments/nemotron-pool/load")
    assert loaded.status_code == 200, loaded.json()
    assert loaded.json()["loaded"] is True
    assert loaded.json()["canary_completion_tokens"] == 1
    assert len(loaded.json()["ranks"]) == 2
    assert pool.entry.engine is not None

    # Verify reload succeeds even if entry has a stuck pending_unload_reason
    pool.entry.pending_unload_reason = "drain timeout"
    pool.reloads = 0
    loaded_again = _client().post("/admin/api/cluster/deployments/nemotron-pool/load")
    assert loaded_again.status_code == 200, loaded_again.json()
    assert pool.reloads == 1
    assert pool.entry.pending_unload_reason is None

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


# reload_busy: rank telemetry still counts the failed canary, so the rollback
# cannot unload the engine. The canary error must still reach the caller.
@pytest.mark.parametrize("reload_busy", [False, True])
def test_cluster_activation_rolls_back_when_canary_fails(
    tmp_path, monkeypatch, reload_busy
):
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.registry import configure_cluster_registry

    registry = configure_cluster_registry(tmp_path)
    model_path = tmp_path / "models" / "canary-failure"
    model_path.mkdir(parents=True)
    pool = _install_ready_pool(
        monkeypatch,
        model_path,
        fail_canary=True,
        reload_busy=reload_busy,
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
    if not reload_busy:
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


def test_a_thunderbolt_cable_the_cluster_cannot_use_is_named_before_activation(
    monkeypatch,
):
    from dataclasses import replace

    def mac(name, address):
        return replace(
            _interfaces(name, [("en1", address, 24)], thunderbolt={"bridge0"}),
            wifi_interfaces=frozenset({"en1"}),
            self_assigned_interfaces=frozenset({"bridge0"}),
        )

    _readings(
        monkeypatch,
        {
            "127.0.0.1": mac("127.0.0.1", "192.168.4.21"),
            "studio.local": mac("studio.local", "192.168.4.22"),
        },
    )
    monkeypatch.setattr(routes, "detect_cluster_transports", lambda hosts: _no_links())
    monkeypatch.setattr(routes, "probe_remote_host", lambda ssh, **_: None)

    proposal = (
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
                    {"node_id": "node-0", "ssh": "127.0.0.1", "ips": ["10.0.0.1"]},
                    {"node_id": "node-1", "ssh": "studio.local", "ips": ["10.0.0.2"]},
                ],
            },
        )
        .json()
    )

    # The cluster still runs, over Wi-Fi, and says what to fix first.
    assert proposal["backend"] == "ring"
    (warning,) = [w for w in proposal["warnings"] if "169.254" in w]
    assert "self-assigned 169.254" in warning and "Wi-Fi" in warning


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


def test_autoconfigure_lightning_mtp_plan_is_the_one_activation_recomputes(
    monkeypatch,
):
    from omlx.cluster.planner import ModelLayout

    gib = 1024**3
    monkeypatch.setattr(
        routes,
        "inspect_safetensors_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=gib,
            layer_weight_bytes=(2 * gib,) * 8,
            supports_pipeline=True,
            mtp_weight_bytes=gib,
        ),
    )
    _readings(
        monkeypatch,
        {
            "127.0.0.1": _interfaces("127.0.0.1", [("en4", "10.0.1.1", 30)]),
            "studio.local": _interfaces("studio.local", [("en5", "10.0.1.2", 30)]),
        },
    )
    monkeypatch.setattr(routes, "detect_cluster_transports", lambda hosts: _no_links())
    monkeypatch.setattr(routes, "probe_remote_host", lambda ssh, **_: None)
    monkeypatch.setattr(routes, "peer_import_issues", lambda *_, **__: ())
    monkeypatch.setattr(routes, "_staging_for", lambda *_: None)

    body = (
        _client()
        .post(
            "/admin/api/cluster/autoconfigure",
            json={
                "model_path": "/models/draft",
                "strategy": "pipeline",
                "mtp": True,
                "nodes": [
                    {"node_id": f"node-{index}", "capacity_bytes": 64 * gib}
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

    activation = body["activation"]
    assert activation["mtp"] is True
    # Placing ranks rebuilds the plan; it must still carry the head.
    assert activation["approved_placement"] == _approval_for(activation)


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


@pytest.mark.parametrize("local_only", [False, True])
def test_forget_cluster_with_unreachable_peer(monkeypatch, local_only):
    from omlx.cluster.launch import DistributedTeardownError

    deployment = SimpleNamespace(model="org/model", deployment_id="offline")
    removed = []
    prepared = []

    class Pool:
        def resolve_cluster_model_id(self, model):
            return "model"

        async def prepare_cluster_reload(self, model_id, **kwargs):
            prepared.append(kwargs)

        def unregister_cluster_model(self, model_id):
            assert removed == ["offline"]

    def stop(deployment, *, local_only=False):
        if not local_only:
            raise DistributedTeardownError("peer is unreachable")

    monkeypatch.setattr(routes, "_engine_pool", lambda: Pool())
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(
            get=lambda identifier: deployment,
            remove=lambda identifier: removed.append(identifier) or True,
        ),
    )
    monkeypatch.setattr(routes, "stop_deployment_processes", stop)
    suffix = "?local_only=true" if local_only else ""
    response = _client().delete("/admin/api/cluster/deployments/offline" + suffix)
    assert response.status_code == (200 if local_only else 503)
    assert removed == (["offline"] if local_only else [])
    assert prepared == ([{"local_only": True}] if local_only else [{}])
    if local_only:
        assert response.json()["stopped"] is False
        assert response.json()["local_only"] is True


def test_forget_cluster_preserves_setup_when_local_stop_fails(monkeypatch):
    from omlx.cluster.launch import DistributedTeardownError

    deployment = SimpleNamespace(model="org/model", deployment_id="offline")
    removed = []

    class Pool:
        def resolve_cluster_model_id(self, model):
            return "model"

        async def prepare_cluster_reload(self, model_id, **kwargs):
            raise DistributedTeardownError("local worker survived")

    monkeypatch.setattr(routes, "_engine_pool", lambda: Pool())
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(
            get=lambda identifier: deployment,
            remove=lambda identifier: removed.append(identifier),
        ),
    )
    response = _client().delete(
        "/admin/api/cluster/deployments/offline?local_only=true"
    )
    assert response.status_code == 503
    assert removed == []


@pytest.mark.parametrize("node_id", ["peer-a", None, "self"])
def test_forget_member_preserves_other_pairings_or_leaves_cluster(monkeypatch, node_id):
    from omlx.cluster import pairing_routes

    peers = {"peer-a", "peer-b"}
    placements = [
        SimpleNamespace(
            deployment_id="uses-a", hosts=[SimpleNamespace(node_id="peer-a")]
        ),
        SimpleNamespace(
            deployment_id="uses-b", hosts=[SimpleNamespace(node_id="peer-b")]
        ),
    ]
    stopped = []

    async def stop(deployment_id, *, local_only):
        assert local_only
        stopped.append(deployment_id)

    manager = SimpleNamespace(
        node_id="self",
        paired_devices=lambda: [{"node_id": node} for node in sorted(peers)],
        unpair=lambda node: peers.remove(node),
    )
    monkeypatch.setattr(pairing_routes, "_manager", lambda: manager)
    monkeypatch.setattr(
        routes, "get_cluster_registry", lambda: SimpleNamespace(list=lambda: placements)
    )
    monkeypatch.setattr(routes, "deactivate_cluster_deployment", stop)
    app = FastAPI()
    app.include_router(routes.router)
    suffix = "" if node_id is None else f"?node_id={node_id}"
    response = TestClient(app).delete("/admin/api/cluster/forget" + suffix)
    assert response.status_code == 200
    assert response.json()["stopped"] is False
    if node_id == "peer-a":
        assert peers == {"peer-b"}
        assert stopped == ["uses-a"]
    else:
        assert peers == set()
        assert stopped == ["uses-a", "uses-b"]


@pytest.mark.parametrize("status", [409, 503])
def test_forget_member_does_not_unpair_when_local_teardown_fails(monkeypatch, status):
    from fastapi import HTTPException
    from omlx.cluster import pairing_routes

    removed = []
    manager = SimpleNamespace(
        node_id="self",
        paired_devices=lambda: [{"node_id": "offline"}],
        unpair=removed.append,
    )
    deployment = SimpleNamespace(
        deployment_id="model", hosts=[SimpleNamespace(node_id="offline")]
    )

    async def stop(*args, **kwargs):
        raise HTTPException(status_code=status, detail="local teardown blocked")

    monkeypatch.setattr(pairing_routes, "_manager", lambda: manager)
    monkeypatch.setattr(
        routes,
        "get_cluster_registry",
        lambda: SimpleNamespace(list=lambda: [deployment]),
    )
    monkeypatch.setattr(routes, "deactivate_cluster_deployment", stop)
    app = FastAPI()
    app.include_router(routes.router)
    assert (
        TestClient(app).delete("/admin/api/cluster/forget?node_id=offline").status_code
        == status
    )
    assert not removed


def test_forget_member_maps_pairing_errors(monkeypatch):
    from omlx.cluster import pairing_routes
    from omlx.cluster.pairing import PairingStateError

    def unpair(node):
        raise PairingStateError("approval is already in progress")

    manager = SimpleNamespace(
        node_id="self", paired_devices=lambda: [{"node_id": "busy"}], unpair=unpair
    )
    monkeypatch.setattr(pairing_routes, "_manager", lambda: manager)
    monkeypatch.setattr(
        routes, "get_cluster_registry", lambda: SimpleNamespace(list=lambda: [])
    )
    app = FastAPI()
    app.include_router(routes.router)
    response = TestClient(app).delete("/admin/api/cluster/forget?node_id=busy")
    assert response.status_code == 404
    assert response.json()["detail"] == "approval is already in progress"
