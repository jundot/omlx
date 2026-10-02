# SPDX-License-Identifier: Apache-2.0
"""Tests for the isolated cluster worker process boundary."""

from pathlib import Path

from omlx.cluster.models import WORKER_PROTOCOL_VERSION
from omlx.cluster.supervisor import (
    JacclLaunchConfig,
    WorkerSupervisor,
    run_worker_smoke,
)


def test_jaccl_launch_config_matches_mlx_environment_contract():
    config = JacclLaunchConfig(
        rank=1,
        coordinator="169.254.42.1:5000",
        ibv_devices_file=Path("/tmp/omlx-ibv-devices.json"),
        ring=True,
    )

    assert config.environment() == {
        "MLX_RANK": "1",
        "MLX_JACCL_COORDINATOR": "169.254.42.1:5000",
        "MLX_IBV_DEVICES": "/tmp/omlx-ibv-devices.json",
        "MLX_JACCL_RING": "1",
    }


def test_worker_supervisor_ready_ping_shutdown_round_trip():
    supervisor = WorkerSupervisor(rank=0, plan_hash="test-plan", timeout=5.0)
    ready = supervisor.start()
    child_pid = ready["pid"]

    assert ready["type"] == "ready"
    assert ready["protocol_version"] == WORKER_PROTOCOL_VERSION
    assert ready["rank"] == 0
    assert ready["plan_hash"] == "test-plan"
    assert child_pid > 0

    pong = supervisor.ping(nonce="test-nonce")
    assert pong["type"] == "pong"
    assert pong["nonce"] == "test-nonce"

    stopped = supervisor.stop()
    assert stopped is not None
    assert stopped["type"] == "stopped"
    assert supervisor.process is None


def test_run_worker_smoke_returns_complete_receipt():
    result = run_worker_smoke(timeout=5.0)

    assert result["ok"] is True
    assert result["protocol_version"] == WORKER_PROTOCOL_VERSION
    assert result["ready"]["type"] == "ready"
    assert result["pong"]["type"] == "pong"
    assert result["stopped"]["type"] == "stopped"
    assert result["elapsed_seconds"] >= 0
