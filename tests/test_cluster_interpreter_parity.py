# SPDX-License-Identifier: Apache-2.0
"""Report interpreter differences alongside package parity checks (#2695).

The runtime gate compared omlx/mlx/mlx-lm and the cluster protocol, but never
the interpreter underneath them.  ``runtime.python_version`` was collected and
carried all the way into the status payload, and nothing read it.

Policy: missing, malformed, or different-major reports are hard mismatches.
Minor and patch differences launch but remain visible to an operator debugging
unexpected behaviour.
"""

import json
import pathlib
import platform
import subprocess

import pytest
from test_cluster_launch import _deployment

from omlx.cluster import launch
from omlx.cluster.launch import (
    DistributedLaunchError,
    _local_probe_versions,
    _local_runtime_versions,
    preflight_remote_hosts,
    probe_remote_host,
)
from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION

PEER_PYTHON = "/opt/omlx/bin/python"


def _status(python_version: str | None) -> dict:
    # probe.py reports mlx/mlx-lm as module constants, so a fake probe payload
    # must mirror that source and not dist-info (#2726).
    versions = _local_probe_versions()
    runtime = {
        "omlx_version": versions["omlx"],
        "mlx_version": versions["mlx"],
        "mlx_lm_version": versions["mlx-lm"],
        "python_executable": PEER_PYTHON,
    }
    if python_version is not None:
        runtime["python_version"] = python_version
    return {
        "protocol_version": CLUSTER_PROTOCOL_VERSION,
        "node": {"hostname": "studio"},
        "runtime": runtime,
        "transport": {},
    }


def _probe(python_version: str | None) -> dict:
    payload = _status(python_version)

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    return probe_remote_host("studio", python_executable=PEER_PYTHON, runner=runner)


# --- probe_remote_host ------------------------------------------------------


def test_probe_accepts_a_different_python_minor_but_reports_it(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.9")

    result = _probe("3.12.13")

    assert result["runtime_compatible"] is True
    assert result["ok"] is True
    assert result["runtime_mismatches"] == []
    assert result["runtime_warnings"] == [
        "python minor differs: local=3.11.9 remote=3.12.13"
    ]
    assert result["status"]["warnings"] == [
        "python minor differs: local=3.11.9 remote=3.12.13"
    ]


def test_probe_accepts_a_patch_difference_but_still_reports_it(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.13")

    result = _probe("3.12.14")

    assert result["runtime_compatible"] is True
    assert result["runtime_mismatches"] == []
    assert result["runtime_warnings"] == [
        "python patch differs: local=3.12.13 remote=3.12.14"
    ]
    # Visible where the dashboard already renders per-node warnings.
    assert result["status"]["warnings"] == [
        "python patch differs: local=3.12.13 remote=3.12.14"
    ]


def test_probe_is_silent_when_both_ranks_run_the_same_interpreter(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.13")

    result = _probe("3.12.13")

    assert result["runtime_compatible"] is True
    assert result["runtime_mismatches"] == []
    assert result["runtime_warnings"] == []


@pytest.mark.parametrize("reported", [None, "", "   "])
def test_probe_treats_an_unreported_interpreter_as_a_mismatch(monkeypatch, reported):
    """Silence is not agreement — an old worker that omits it is not verified."""

    monkeypatch.setattr(platform, "python_version", lambda: "3.12.13")

    result = _probe(reported)

    assert result["runtime_compatible"] is False
    assert result["runtime_mismatches"] == ["python local=3.12.13 remote=missing"]


def test_probe_still_reports_package_mismatches_alongside_the_interpreter(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.9")
    payload = _status("3.12.13")
    payload["runtime"]["mlx_version"] = "0.0.1-wrong"

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    result = probe_remote_host("studio", python_executable=PEER_PYTHON, runner=runner)

    assert result["runtime_compatible"] is False
    assert any(entry.startswith("mlx local=") for entry in result["runtime_mismatches"])
    assert result["runtime_warnings"] == [
        "python minor differs: local=3.11.9 remote=3.12.13"
    ]


def test_probe_ignores_stale_local_omlx_metadata(monkeypatch):
    real_version = launch.importlib.metadata.version

    def stale_metadata(name):
        if name == "omlx":
            return "0.0.0-stale"
        return real_version(name)

    monkeypatch.setattr(launch.importlib.metadata, "version", stale_metadata)
    payload = _status(platform.python_version())

    from omlx._version import __version__

    payload["runtime"]["omlx_version"] = __version__

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    result = probe_remote_host("studio", python_executable=PEER_PYTHON, runner=runner)

    assert result["runtime_compatible"] is True
    assert result["runtime_mismatches"] == []


def test_probe_still_rejects_a_genuine_omlx_source_version_difference():
    payload = _status(platform.python_version())
    payload["runtime"]["omlx_version"] = "0.0.0-other"

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    result = probe_remote_host("studio", python_executable=PEER_PYTHON, runner=runner)

    assert result["runtime_compatible"] is False
    assert result["runtime_mismatches"] == [
        f"omlx local={_local_probe_versions()['omlx']} remote=0.0.0-other"
    ]


# --- preflight_remote_hosts -------------------------------------------------


@pytest.fixture
def stub_admission(monkeypatch):
    """The memory plan is validated elsewhere; keep these tests to parity."""

    monkeypatch.setattr(
        launch, "_validate_deployment_admission", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        launch, "ceiling_breakdown", lambda *_a, **_k: {"hard_limit": 512 * 1024**3},
        raising=False,
    )
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 512 * 1024**3},
    )


def _preflight_runner(python_version: str | None):
    versions = _local_runtime_versions()

    def runner(argv, **_kwargs):
        payload = {
            **versions,
            "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
            "admission-ceiling-bytes": 512 * 1024**3,
            "model-exists": True,
        }
        if python_version is not None:
            payload["python"] = python_version
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    return runner


def test_preflight_allows_a_different_python_minor_and_records_it(
    monkeypatch, stub_admission
):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.9")

    results = preflight_remote_hosts(
        _deployment(),
        python_executable=PEER_PYTHON,
        runner=_preflight_runner("3.12.13"),
    )

    assert results[0]["runtime_warnings"] == []
    assert results[1]["runtime_warnings"] == [
        "python minor differs: local=3.11.9 remote=3.12.13"
    ]


def test_preflight_allows_a_patch_difference_and_records_it(
    monkeypatch, stub_admission
):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.13")

    results = preflight_remote_hosts(
        _deployment(),
        python_executable=PEER_PYTHON,
        runner=_preflight_runner("3.12.14"),
    )

    assert results[0]["runtime_warnings"] == []
    assert results[1]["runtime_warnings"] == [
        "python patch differs: local=3.12.13 remote=3.12.14"
    ]


def test_preflight_refuses_a_rank_that_reports_no_interpreter(
    monkeypatch, stub_admission
):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.13")

    with pytest.raises(
        DistributedLaunchError, match="python local=3.12.13 remote=missing"
    ):
        preflight_remote_hosts(
            _deployment(),
            python_executable=PEER_PYTHON,
            runner=_preflight_runner(None),
        )


def test_preflight_asks_the_rank_for_its_interpreter_version():
    """The remote script must actually measure it rather than assume."""

    assert "platform" in launch._PREFLIGHT_SCRIPT
    assert "python_version()" in launch._PREFLIGHT_SCRIPT
    assert "v['python']" in launch._PREFLIGHT_SCRIPT


def test_preflight_reads_omlx_source_version_before_installed_metadata():
    script = launch._PREFLIGHT_SCRIPT

    assert script.index("if name == 'omlx':") < script.index("return m.version(name)")


# --- the parity rule itself -------------------------------------------------


@pytest.mark.parametrize(
    ("local", "remote", "blocking", "warning"),
    [
        ("3.12.13", "3.12.13", None, None),
        (
            "3.12.13",
            "3.12.14",
            None,
            "python patch differs: local=3.12.13 remote=3.12.14",
        ),
        (
            "3.11.9",
            "3.12.13",
            None,
            "python minor differs: local=3.11.9 remote=3.12.13",
        ),
        (
            "3.12.13",
            "3.11.9",
            None,
            "python minor differs: local=3.12.13 remote=3.11.9",
        ),
        ("3.12.13", "4.0.0", "python local=3.12.13 remote=4.0.0", None),
        ("3.12.13", "", "python local=3.12.13 remote=missing", None),
        # A free-threaded or otherwise suffixed build is still 3.13.
        (
            "3.13.1",
            "3.13.1+freethreaded",
            None,
            "python patch differs: local=3.13.1 remote=3.13.1+freethreaded",
        ),
        # Not a version at all: refuse rather than guess.
        ("3.12.13", "banana", "python local=3.12.13 remote=banana", None),
    ],
)
def test_interpreter_parity_rule(local, remote, blocking, warning):
    assert launch._interpreter_parity(local, remote) == (blocking, warning)


def test_every_probe_branch_returns_the_same_result_keys():
    """A caller must not have to know which branch produced the result.

    Observed live: the bootstrap fallback returned no runtime_warnings at all,
    so the field read as None on one path and [] on the other.
    """

    def bootstrap_runner(argv, **_kwargs):
        command = argv[-1]
        if "import sys; print(sys.executable)" in command:
            return subprocess.CompletedProcess(argv, 0, "/usr/bin/python3\n", "")
        if "worker_runtime_evidence" in command:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(
                    {
                        "node": {"worker_runtime_evidence": ["/Applications/oMLX.app"]},
                        "runtime": {},
                        "transport": {},
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(argv, 1, "", "No module named 'omlx'")

    bootstrap = launch.probe_remote_system_host(
        "studio", preferred_python=PEER_PYTHON, runner=bootstrap_runner
    )

    for key in ("runtime_compatible", "runtime_mismatches", "runtime_warnings"):
        assert key in bootstrap, key
    assert bootstrap["runtime_warnings"] == []


def test_dashboard_does_not_render_a_warned_peer_as_an_unqualified_match():
    cluster = pathlib.Path(
        "omlx/admin/templates/dashboard/_cluster.html"
    ).read_text(encoding="utf-8")

    assert "clusterPeerProbe.runtime_warnings" in cluster
    assert "text-amber-700" in cluster
    # 'Runtime match' must be the else-branch, not the whole truthy branch.
    assert "runtime_warnings || []).length ? 'text-amber-700' : 'text-green-700'" in cluster


def test_interpreter_parity_ignores_a_non_string_report():
    assert launch._interpreter_parity("3.12.13", None) == (
        "python local=3.12.13 remote=missing",
        None,
    )
    assert launch._interpreter_parity("3.12.13", 312) == (
        "python local=3.12.13 remote=missing",
        None,
    )
