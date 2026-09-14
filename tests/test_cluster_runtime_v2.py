# SPDX-License-Identifier: Apache-2.0
"""Runtime-membership v2 tests: backend selection, N-node launch manifest,
and ring-backend verified-teardown parity.

All offline: process groups, killpg, SSH sweeps, pools, and probes are mocked.
"""

import signal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import launch, routes
from omlx.cluster.backends import (
    MemberFabric,
    members_from_host_records,
    select_cluster_backend,
)
from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    decode_worker_contract,
)
from omlx.cluster.planner import ModelLayout, PipelineAssignment
from omlx.cluster.registry import configure_cluster_registry


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_backend_selection_picks_jaccl_when_all_members_ready():
    selection = select_cluster_backend(
        [
            MemberFabric("a", True, ("rdma_en5",)),
            MemberFabric("b", True, ("rdma_en1", "rdma_en2")),
            MemberFabric("c", True, ("rdma_en0",)),
        ]
    )

    assert selection.backend == "jaccl"
    assert selection.blockers == ()
    assert len(selection.members) == 3


def test_backend_selection_falls_back_to_ring_and_names_blockers():
    selection = select_cluster_backend(
        [
            MemberFabric("a", True, ("rdma_en5",)),
            MemberFabric("b", False, ()),
            MemberFabric("c", True, ()),  # enabled but deviceless
        ]
    )

    assert selection.backend == "ring"
    assert selection.blockers == ("b", "c")
    assert "b has rdma_ctl disabled" in selection.reason
    assert "c reports no RDMA device" in selection.reason


def test_backend_selection_requires_two_members_and_unique_ids():
    with pytest.raises(ValueError, match="at least two"):
        select_cluster_backend([MemberFabric("a", True, ("rdma_en0",))])
    with pytest.raises(ValueError, match="unique"):
        select_cluster_backend(
            [MemberFabric("a", True, ("rdma_en0",)), MemberFabric("a")]
        )


def test_members_from_host_records_reads_complete_matrices():
    hosts = [
        SimpleNamespace(node_id="a", rdma=[None, "rdma_en5"]),
        SimpleNamespace(node_id="b", rdma=["rdma_en5", None]),
    ]
    members = members_from_host_records(hosts)
    assert all(member.rdma_ctl_enabled for member in members)
    assert select_cluster_backend(members).backend == "jaccl"

    incomplete = [
        SimpleNamespace(node_id="a", rdma=[]),
        SimpleNamespace(node_id="b", rdma=["rdma_en5", None]),
    ]
    members = members_from_host_records(incomplete)
    assert [member.rdma_ctl_enabled for member in members] == [False, True]
    selection = select_cluster_backend(members)
    assert selection.backend == "ring"
    assert selection.blockers == ("a",)


def test_backend_selection_endpoint():
    response = _client().post(
        "/admin/api/cluster/backend-selection",
        json={
            "members": [
                {
                    "node_id": "studio",
                    "rdma_ctl_enabled": True,
                    "rdma_devices": ["rdma_en5"],
                },
                {"node_id": "macbook", "rdma_ctl_enabled": False},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "ring"
    assert payload["blockers"] == ["macbook"]
    assert {m["node_id"] for m in payload["members"]} == {"studio", "macbook"}


def test_backend_selection_endpoint_all_rdma_picks_jaccl():
    response = _client().post(
        "/admin/api/cluster/backend-selection",
        json={
            "members": [
                {
                    "node_id": "a",
                    "rdma_ctl_enabled": True,
                    "rdma_devices": ["rdma_en1"],
                },
                {
                    "node_id": "b",
                    "rdma_ctl_enabled": True,
                    "rdma_devices": ["rdma_en5"],
                },
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["backend"] == "jaccl"


# ---------------------------------------------------------------------------
# Replan with backend="auto"
# ---------------------------------------------------------------------------


def _install_route_doubles(monkeypatch):
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
    monkeypatch.setattr(routes, "check_peers", lambda *args, **kwargs: ())


def test_replan_auto_selects_jaccl_from_complete_matrix(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    _install_route_doubles(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_engine_pool",
        lambda: SimpleNamespace(),
    )
    nodes = [
        {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
        {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
    ]
    hosts = [
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
    ]

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "model_path": "/models/example",
            "backend": "auto",
            "nodes": nodes,
            "hosts": hosts,
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["mode"] == "preview"
    assert payload["backend"] == "jaccl"
    assert payload["backend_decision"]["backend"] == "jaccl"
    assert payload["deployment_id"]


def test_replan_auto_falls_back_to_ring(tmp_path, monkeypatch):
    configure_cluster_registry(tmp_path)
    _install_route_doubles(monkeypatch)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: SimpleNamespace())

    response = _client().post(
        "/admin/api/cluster/replan",
        json={
            "model_path": "/models/example",
            "backend": "auto",
            "nodes": [
                {"node_id": "large", "capacity_bytes": 100, "reserve_bytes": 10},
                {"node_id": "small", "capacity_bytes": 60, "reserve_bytes": 10},
            ],
            "hosts": [
                {"node_id": "large", "ssh": "127.0.0.1", "ips": ["192.168.5.1"]},
                {"node_id": "small", "ssh": "studio.local", "ips": ["192.168.5.2"]},
            ],
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["backend"] == "ring"
    assert payload["backend_decision"]["blockers"] == ["large", "small"]


# ---------------------------------------------------------------------------
# N-node launch manifest
# ---------------------------------------------------------------------------


def _three_node_ring_deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="cluster-nnode",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "user@studio.local", ("10.0.0.2",)),
            ClusterHost("mini", "user@mini.local", ("10.0.0.3",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 6, 9, 5, 1, 1, 16),
            PipelineAssignment("studio", 1, 3, 6, 3, 1, 1, 8),
            PipelineAssignment("mini", 2, 0, 3, 3, 1, 1, 8),
        ),
        plan_hash="d" * 64,
    )


def test_launch_manifest_records_every_host(tmp_path):
    deployment = _three_node_ring_deployment()

    path = launch._write_launch_manifest(
        tmp_path, deployment, launcher_pid=43210, api_port=32100
    )

    manifest = launch._read_launch_manifest(path)
    assert manifest is not None
    assert manifest["deployment_id"] == "cluster-nnode"
    assert manifest["process_group"] == 43210
    assert manifest["hosts"] == [
        {"rank": 0, "node_id": "local", "ssh": "127.0.0.1"},
        {"rank": 1, "node_id": "studio", "ssh": "user@studio.local"},
        {"rank": 2, "node_id": "mini", "ssh": "user@mini.local"},
    ]


def test_three_node_ring_deployment_validates_and_maps_backend():
    deployment = _three_node_ring_deployment()

    assert deployment.world_size == 3
    assert deployment.distributed_init_backend == "ring"
    hostfile = deployment.hostfile_dict()
    assert hostfile["backend"] == "ring"
    assert len(hostfile["hosts"]) == 3
    # The worker contract carries all three assignments, rank-indexed.
    plan_hash, assignments, _, tp = decode_worker_contract(
        deployment.encode_worker_plan()
    )
    assert plan_hash == deployment.plan_hash
    assert [item.rank for item in assignments] == [0, 1, 2]
    assert tp == 1


# ---------------------------------------------------------------------------
# Ring carries the same verified-teardown semantics as JACCL
# ---------------------------------------------------------------------------


class _Launcher:
    pid = 43210
    stdout = None
    stderr = None

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def _teardown_mocks(monkeypatch, group_exit_results):
    results = iter(group_exit_results)
    monkeypatch.setattr(launch, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        launch,
        "_wait_for_process_group_exit",
        lambda _pgid, _timeout: next(results),
    )
    signals = []
    monkeypatch.setattr(
        launch.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    return signals


def test_ring_teardown_verifies_killpg_and_raises_on_survivors(monkeypatch):
    supervisor = launch.DistributedJobSupervisor(
        _three_node_ring_deployment(),
        preflight=False,
        stop_timeout=0.1,
    )
    supervisor.process = _Launcher()
    signals = _teardown_mocks(monkeypatch, [False, False, False])
    swept = []
    monkeypatch.setattr(
        launch.DistributedJobSupervisor,
        "_sweep_rank_leftovers",
        lambda self, **_kw: swept.append(True) or [],
    )

    with pytest.raises(launch.DistributedTeardownError, match="survived SIGKILL"):
        supervisor.stop()

    # TERM, then KILL twice — identical escalation to the JACCL path, because
    # there is exactly one teardown path and it is backend-agnostic.
    assert signals == [
        (43210, signal.SIGTERM),
        (43210, signal.SIGKILL),
        (43210, signal.SIGKILL),
    ]
    # The rank sweep runs even when the group is the problem, and it covers
    # every host of the N-node ring, not just local ranks.
    assert swept == [True]
    # State is kept so a later stop() retries instead of forgetting the job.
    assert supervisor.process is not None


def test_ring_teardown_sweeps_leftover_ranks_by_marker_pid(monkeypatch):
    deployment = _three_node_ring_deployment()
    supervisor = launch.DistributedJobSupervisor(
        deployment,
        preflight=False,
        stop_timeout=0.1,
    )
    supervisor.process = _Launcher()
    _teardown_mocks(monkeypatch, [True])
    swept_hosts = []

    def fake_sweep(deployment_id, hosts, **kwargs):
        swept_hosts.extend(hosts)
        return []

    monkeypatch.setattr(launch, "_sweep_rank_processes", fake_sweep)

    supervisor.stop()

    assert supervisor.process is None
    assert supervisor.status().phase == "stopped"
    assert swept_hosts == [
        {"rank": 0, "node_id": "local", "ssh": "127.0.0.1"},
        {"rank": 1, "node_id": "studio", "ssh": "user@studio.local"},
        {"rank": 2, "node_id": "mini", "ssh": "user@mini.local"},
    ]


def test_ring_teardown_reports_unkillable_leftover_rank(monkeypatch):
    supervisor = launch.DistributedJobSupervisor(
        _three_node_ring_deployment(),
        preflight=False,
        stop_timeout=0.1,
    )
    supervisor.process = _Launcher()
    _teardown_mocks(monkeypatch, [True])
    monkeypatch.setattr(
        launch,
        "_sweep_rank_processes",
        lambda *args, **kwargs: ["rank 2 (mini): pid 999 survived SIGKILL"],
    )

    with pytest.raises(launch.DistributedTeardownError, match="rank 2 \\(mini\\)"):
        supervisor.stop()

    assert supervisor.process is not None
