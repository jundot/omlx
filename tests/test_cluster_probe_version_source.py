# SPDX-License-Identifier: Apache-2.0
"""Each comparison reads the local side the way its own remote does (#2726).

#2705 fixed this for ``omlx``. The same skew remained for ``mlx`` and
``mlx-lm`` on the probe path: the coordinator read their ``dist-info`` while
``probe.py`` reports their module constants. The two remote paths also
disagreed with each other, so a pair of Macs could pass preflight and fail the
probe gate, or the reverse.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from omlx.cluster import launch
from omlx.cluster.launch import (
    _local_probe_versions,
    _local_runtime_versions,
    probe_remote_host,
)
from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION
from omlx.utils import hardware

PEER_PYTHON = "/opt/omlx/bin/python"


@pytest.fixture
def drifted_metadata(monkeypatch):
    """dist-info that disagrees with the loaded module, as an editable install."""

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
    # _PREFLIGHT_SCRIPT calls importlib.metadata on the peer, so the local side
    # of that comparison must keep doing the same.
    versions = _local_runtime_versions()

    assert versions["mlx"] == drifted_metadata["mlx"]
    assert versions["mlx-lm"] == drifted_metadata["mlx-lm"]


def test_identical_nodes_pass_the_probe_gate_when_dist_info_has_drifted(
    drifted_metadata,
):
    # The peer runs the same code, so it reports the same module constants.
    result = _probe_with(_local_probe_versions())

    assert result["runtime_compatible"] is True, result["runtime_mismatches"]
    assert result["runtime_mismatches"] == []


def test_a_genuine_mlx_difference_is_still_blocking(drifted_metadata):
    peer = _local_probe_versions() | {"mlx": "1.2.3"}

    result = _probe_with(peer)

    assert result["runtime_compatible"] is False
    assert any("mlx local=9.9.9 remote=1.2.3" in m for m in result["runtime_mismatches"])


def test_mlx_missing_on_both_ends_is_not_a_mismatch(monkeypatch):
    # hardware.* returns "Unknown"; _package_version returns "unknown". Reading
    # one against the other reported a mismatch between two ranks in the same
    # state, purely on capitalisation.
    monkeypatch.setattr(hardware, "get_mlx_version", lambda: "Unknown")
    monkeypatch.setattr(hardware, "get_mlx_lm_version", lambda: "Unknown")

    result = _probe_with(_local_probe_versions())

    assert result["runtime_compatible"] is True, result["runtime_mismatches"]


def test_both_local_sources_still_agree_about_omlx(drifted_metadata):
    # #2705's fix must not be undone: omlx comes from the source tree on both.
    assert _local_probe_versions()["omlx"] == _local_runtime_versions()["omlx"]
