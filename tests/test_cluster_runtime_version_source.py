# SPDX-License-Identifier: Apache-2.0
"""Cluster version comparisons must use matching local and remote sources.

The coordinator used to prefer ``importlib.metadata`` while the peer reported
``omlx._version.__version__``. On an editable install whose ``dist-info`` has
drifted from the source tree, a node then disagreed with itself and the gate
blocked two machines running byte-identical code. The same source distinction
also applies to the MLX and mlx-lm probe versions.
"""

from __future__ import annotations

import ast
import builtins
import importlib.metadata
import json
import subprocess

import pytest

from omlx._version import __version__ as source_version
from omlx.cluster import launch
from omlx.cluster.launch import (
    _local_probe_versions,
    _local_runtime_versions,
    probe_remote_host,
)
from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION
from omlx.cluster.probe import __version__ as probe_version
from omlx.utils import hardware

PEER_PYTHON = "/opt/omlx/bin/python"


@pytest.fixture
def stale_metadata(monkeypatch):
    """Pretend the installed dist-info was built at a different version."""

    stale = "0.0.1.dev999"
    assert stale != source_version

    def fake_version(name: str) -> str:
        if name == "omlx":
            return stale
        return {"mlx": "9.9.9", "mlx-lm": "8.8.8"}[name]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    return stale


def test_the_peer_reports_the_source_version():
    # probe.py is the peer side; it has always read the source tree.
    assert probe_version == source_version


def test_coordinator_reads_omlx_from_the_source_not_stale_metadata(stale_metadata):
    assert launch._package_version("omlx") == source_version
    assert launch._package_version("omlx") != stale_metadata


def test_coordinator_and_peer_agree_when_dist_info_has_drifted(stale_metadata):
    # The whole point: identical nodes must not read as a version mismatch.
    assert launch._local_runtime_versions()["omlx"] == probe_version


def test_third_party_packages_still_come_from_metadata(stale_metadata):
    # mlx and mlx-lm have no source constant to read; metadata stays correct.
    assert launch._package_version("mlx") == "9.9.9"
    assert launch._package_version("mlx-lm") == "8.8.8"


def test_unknown_package_without_metadata_is_reported_as_unknown(monkeypatch):
    def raise_missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_missing)
    assert launch._package_version("mlx") == "unknown"


def test_omlx_version_survives_a_source_checkout_with_no_dist_info(monkeypatch):
    def raise_missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_missing)
    assert launch._package_version("omlx") == source_version


def _preflight_package_version():
    """Execute the ``package_version`` the remote preflight script ships.

    The script is a string sent over SSH, so the only way to test the code that
    actually runs on the peer is to execute that string's definition.
    """

    script = launch._PREFLIGHT_SCRIPT
    start = script.index("def package_version(name):")
    end = script.index("x=pathlib.Path(")

    class _StaleMetadata:
        PackageNotFoundError = importlib.metadata.PackageNotFoundError

        @staticmethod
        def version(name: str) -> str:
            if name == "omlx":
                return "0.0.1.dev999"
            return {"mlx": "9.9.9", "mlx-lm": "8.8.8"}[name]

    namespace: dict = {"m": _StaleMetadata}
    exec(script[start:end], namespace)  # noqa: S102 - the shipped script itself
    return namespace["package_version"]


def test_remote_preflight_script_reads_omlx_from_the_source_too():
    package_version = _preflight_package_version()

    assert package_version("omlx") == source_version
    assert package_version("mlx") == "9.9.9"


def test_remote_preflight_agrees_with_the_coordinator(stale_metadata):
    # preflight_remote_hosts and probe_remote_host must not disagree either.
    # Asserting the shared value too: agreeing on the stale number would
    # satisfy an equality-only check while leaving the bug in place.
    agreed = _preflight_package_version()("omlx")
    assert agreed == launch._package_version("omlx") == source_version


def test_the_whole_preflight_script_is_valid_python():
    # The script is hand-assembled from string literals and only ever executed
    # on a remote host, so a stray indent would surface as an SSH preflight
    # traceback on someone else's Mac. Parse the whole thing here instead.
    ast.parse(launch._PREFLIGHT_SCRIPT)


def _without_omlx_version(monkeypatch):
    """Simulate a peer older than omlx._version (added in 0.1.2)."""

    real_import = builtins.__import__

    def refuse_version(name, *args, **kwargs):
        if name == "omlx._version":
            raise ImportError("No module named 'omlx._version'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_version)


def test_coordinator_degrades_to_metadata_when_the_source_constant_is_absent(
    stale_metadata, monkeypatch
):
    _without_omlx_version(monkeypatch)

    # Not a crash: an old peer must still produce a legible version mismatch.
    assert launch._package_version("omlx") == stale_metadata


def test_remote_preflight_degrades_to_metadata_too(monkeypatch):
    package_version = _preflight_package_version()
    _without_omlx_version(monkeypatch)

    assert package_version("omlx") == "0.0.1.dev999"


@pytest.fixture
def drifted_metadata(monkeypatch):
    """dist-info that disagrees with the loaded MLX modules."""
    stale = {"mlx": "0.0.1-stale", "mlx-lm": "0.0.2-stale"}

    def fake_version(name: str) -> str:
        if name in stale:
            return stale[name]
        raise launch.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(launch.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(hardware, "get_mlx_version", lambda: "9.9.9")
    monkeypatch.setattr(hardware, "get_mlx_lm_version", lambda: "8.8.8")
    return stale


def _probe_with(peer_versions: dict[str, str]) -> dict:
    payload = {
        "protocol_version": CLUSTER_PROTOCOL_VERSION,
        "node": {"hostname": "studio"},
        "runtime": {
            "omlx_version": peer_versions["omlx"],
            "mlx_version": peer_versions["mlx"],
            "mlx_lm_version": peer_versions["mlx-lm"],
            "python_version": launch.platform.python_version(),
            "python_executable": PEER_PYTHON,
        },
        "transport": {},
    }

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    return probe_remote_host("studio", python_executable=PEER_PYTHON, runner=runner)


def test_probe_reads_mlx_from_the_module_like_the_peer_does(drifted_metadata):
    versions = _local_probe_versions()

    assert versions["mlx"] == "9.9.9"
    assert versions["mlx-lm"] == "8.8.8"
    assert versions["mlx"] != drifted_metadata["mlx"]


def test_preflight_still_reads_metadata_like_its_own_script(drifted_metadata):
    versions = _local_runtime_versions()

    assert versions["mlx"] == drifted_metadata["mlx"]
    assert versions["mlx-lm"] == drifted_metadata["mlx-lm"]


def test_identical_nodes_pass_the_probe_gate_when_dist_info_has_drifted(
    drifted_metadata,
):
    result = _probe_with(_local_probe_versions())

    assert result["runtime_compatible"] is True, result["runtime_mismatches"]
    assert result["runtime_mismatches"] == []


def test_a_genuine_mlx_difference_is_still_blocking(drifted_metadata):
    peer = _local_probe_versions() | {"mlx": "1.2.3"}

    result = _probe_with(peer)

    assert result["runtime_compatible"] is False
    assert any(
        "mlx local=9.9.9 remote=1.2.3" in message
        for message in result["runtime_mismatches"]
    )


def test_mlx_missing_on_both_ends_is_not_a_mismatch(monkeypatch):
    monkeypatch.setattr(hardware, "get_mlx_version", lambda: "Unknown")
    monkeypatch.setattr(hardware, "get_mlx_lm_version", lambda: "Unknown")

    result = _probe_with(_local_probe_versions())

    assert result["runtime_compatible"] is True, result["runtime_mismatches"]


def test_both_local_sources_still_agree_about_omlx(drifted_metadata):
    assert _local_probe_versions()["omlx"] == _local_runtime_versions()["omlx"]
