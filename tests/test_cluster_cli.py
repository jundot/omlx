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


def test_cluster_doctor_help_is_exposed():
    result = subprocess.run(
        [sys.executable, "-m", "omlx.cli", "cluster", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "doctor" in result.stdout


def test_cluster_doctor_runs_with_stubbed_runner(monkeypatch):
    """The CLI action drives run_fabric_doctor without touching real hosts."""

    from types import SimpleNamespace

    from omlx import cli
    from omlx.cluster import doctor as doctor_module
    from omlx.cluster.doctor import DoctorFinding, DoctorReport

    seen = {}

    def stub(hosts, **_):
        seen["hosts"] = tuple(hosts)
        return DoctorReport(
            hosts=tuple(hosts),
            findings=(
                DoctorFinding(
                    check_id="link_presence", state="pass", evidence="en3 up"
                ),
            ),
            verdict="Fabric verified — every check passed.",
        )

    monkeypatch.setattr(doctor_module, "run_fabric_doctor", stub)
    args = SimpleNamespace(
        cluster_action="doctor", host=["peer.local"], json=True
    )
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exit_code = cli.cluster_command(args)
    assert exit_code == 0
    # One --host means "this Mac plus that peer".
    assert seen["hosts"] == ("127.0.0.1", "peer.local")
    payload = json.loads(out.getvalue())
    assert payload["verdict"] == "Fabric verified — every check passed."
    assert payload["ok"] is True
    assert payload["findings"][0]["check_id"] == "link_presence"


def test_cluster_doctor_red_report_exits_nonzero(monkeypatch):
    from types import SimpleNamespace

    from omlx import cli
    from omlx.cluster import doctor as doctor_module
    from omlx.cluster.doctor import DoctorFinding, DoctorReport

    def stub(hosts, **_):
        return DoctorReport(
            hosts=tuple(hosts),
            findings=(
                DoctorFinding(
                    check_id="subnet_collision",
                    state="fail",
                    evidence="WARP routes 10.0.0.0/8 through utun4",
                    remedy="Click Start Cluster again — it will pick a different, collision-free subnet automatically.",
                ),
            ),
            verdict="Fabric Doctor stopped at subnet_collision: collision",
        )

    monkeypatch.setattr(doctor_module, "run_fabric_doctor", stub)
    args = SimpleNamespace(
        cluster_action="doctor",
        host=["a.local", "b.local"],
        json=False,
    )
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exit_code = cli.cluster_command(args)
    captured = out.getvalue()
    assert exit_code == 1
    assert "FAIL" in captured
    assert "subnet_collision" in captured
    assert "Start Cluster" in captured
    assert "Fabric Doctor stopped at subnet_collision" in captured


def test_cluster_doctor_requires_a_host():
    import contextlib
    import io
    from types import SimpleNamespace

    from omlx import cli

    err = io.StringIO()
    args = SimpleNamespace(cluster_action="doctor", host=None, json=False)
    with contextlib.redirect_stderr(err):
        assert cli.cluster_command(args) == 2
    assert "--host" in err.getvalue()
