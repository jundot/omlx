# SPDX-License-Identifier: Apache-2.0
"""End-to-end CLI tests for the runnable cluster prototype."""

import json
import subprocess
import sys


def test_cluster_help_is_exposed():
    result = subprocess.run(
        [sys.executable, "-m", "omlx.cli", "cluster", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "status" in result.stdout
    assert "worker-smoke" in result.stdout
    assert "collective-smoke" in result.stdout
    assert "pipeline-smoke" in result.stdout
    assert "plan" in result.stdout


def test_cluster_status_json_is_runnable():
    result = subprocess.run(
        [sys.executable, "-m", "omlx.cli", "cluster", "status", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["protocol_version"] == "1.0"
    assert "recommended_working_set_bytes" in payload["node"]
    assert payload["transport"]["state"]


def test_cluster_worker_smoke_json_is_runnable():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omlx.cli",
            "cluster",
            "worker-smoke",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["ready"]["type"] == "ready"
    assert payload["pong"]["type"] == "pong"
    assert payload["stopped"]["type"] == "stopped"


def test_cluster_pipeline_smoke_json_is_runnable():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omlx.cli",
            "cluster",
            "pipeline-smoke",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["model_type"] == "nemotron_h"
    assert payload["rank_count"] == 2


def test_cluster_status_rejects_hostname_route_target():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omlx.cli",
            "cluster",
            "status",
            "--route-to",
            "studio.local",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "IPv4 or IPv6" in result.stderr


def test_cluster_unequal_plan_json_is_runnable():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omlx.cli",
            "cluster",
            "plan",
            "--model-size",
            "300GiB",
            "--layers",
            "80",
            "--node",
            "studio=256GiB",
            "--node",
            "mobile=128GiB",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "unequal_contiguous_pipeline"
    assert payload["model"]["total_weight_bytes"] == 300 * 1024**3
    assert payload["assignments"][0]["node_id"] == "studio"
    assert (
        payload["assignments"][0]["layer_count"]
        > payload["assignments"][1]["layer_count"]
    )
