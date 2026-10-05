# SPDX-License-Identifier: Apache-2.0
"""The supervisor re-verifies stage links every launch and frees them once ranks are gone."""

from __future__ import annotations

from dataclasses import replace

import pytest

from omlx.cluster import launch
from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.rdma import launch_links
from omlx.cluster.rdma.stage_plan import StageLink

_LINK = StageLink(1, 0, "linka", "/tmp/mcdma-rpcd.linka.sock")
_REPORT = {"active": True, "reason": "rank 1 sends to rank 0 over linka", "edges": []}


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="cluster-rdma",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("mac", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("spark-a", "worker@10.0.0.2", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("mac", 0, 3, 8, 5, 1, 1, 16),
            PipelineAssignment("spark-a", 1, 0, 3, 3, 1, 1, 8),
        ),
        plan_hash="c" * 64,
    )


@pytest.fixture
def released(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(
        launch,
        "reap_orphaned_launches",
        lambda _state_dir: {"reaped": [], "failures": []},
    )
    monkeypatch.setattr(launch, "_set_serve_release", lambda *_a, **_k: None)
    monkeypatch.setattr(launch, "_install_cluster_ssh_wrapper", lambda _path: None)
    monkeypatch.setattr(
        launch, "_available_launch_ports", lambda _deployment: (18080, 18100)
    )
    monkeypatch.setattr(launch, "_available_control_port", lambda _host: 18200)
    monkeypatch.setattr(
        launch.DistributedJobSupervisor, "_reap_remote_ranks", lambda self: None
    )
    monkeypatch.setattr(launch, "_release_rdma_stage_links", calls.append)
    return calls


def _supervisor(tmp_path, attach):
    return launch.DistributedJobSupervisor(
        _deployment(),
        preflight=False,
        state_dir=str(tmp_path),
        attach_stage_links=attach,
    )


def test_each_launch_encodes_the_links_it_just_verified(
    tmp_path, monkeypatch, released
):
    seen = []

    def argv(deployment, **_kwargs):
        seen.append(deployment.stage_links)
        return ["true"]

    def popen(*_args, **_kwargs):
        raise OSError("launcher missing")

    monkeypatch.setattr(launch, "build_mlx_launch_argv", argv)
    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    supervisor = _supervisor(
        tmp_path,
        lambda deployment: (replace(deployment, stage_links=(_LINK,)), _REPORT),
    )
    with pytest.raises(OSError, match="launcher missing"):
        supervisor.start()
    assert seen == [(_LINK,)]
    assert supervisor.status().to_dict()["stage_links"] == _REPORT
    assert released == ["cluster-rdma"]


def test_a_launch_that_fails_before_any_rank_frees_its_links(
    tmp_path, monkeypatch, released
):
    def argv(_deployment, **_kwargs):
        raise ValueError("pipeline plan is too large")

    monkeypatch.setattr(launch, "build_mlx_launch_argv", argv)
    supervisor = _supervisor(
        tmp_path,
        lambda deployment: (replace(deployment, stage_links=(_LINK,)), _REPORT),
    )
    with pytest.raises(ValueError, match="too large"):
        supervisor.start()
    assert released == ["cluster-rdma"]


def test_a_broken_link_check_launches_over_mlx_instead(monkeypatch):
    def explode(_deployment):
        raise RuntimeError("daemon answered nonsense")

    monkeypatch.setattr(launch_links, "attach_stage_links", explode)
    launch_links.claim_link("linka", "cluster-rdma")
    deployment, report = launch._attach_rdma_stage_links(
        replace(_deployment(), stage_links=(_LINK,))
    )
    assert deployment.stage_links == ()
    assert report == {
        "active": False,
        "reason": "RDMA stage-link check failed: daemon answered nonsense",
        "edges": [],
    }
    assert launch_links.claimed_links() == {}


def test_status_reports_what_the_ranks_decided_not_only_the_pre_launch_check(tmp_path):
    supervisor = _supervisor(tmp_path, lambda deployment: (deployment, _REPORT))
    supervisor.stage_link_report = _REPORT
    edge = {
        **_LINK.to_dict(),
        "active": False,
        "reason": "mcdma-rpcd reports the link down",
    }
    supervisor.rank_ready_events = {
        0: {"rank": 0, "stage_links": {"active": False, "edges": [edge]}},
        1: {"rank": 1, "stage_links": {"active": False, "edges": [edge]}},
    }
    report = supervisor.status().to_dict()["stage_links"]
    assert report["active"] is False
    assert report["reason"] == "mcdma-rpcd reports the link down"
