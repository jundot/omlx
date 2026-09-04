# SPDX-License-Identifier: Apache-2.0
"""Both ranks must read the oMLX version from the same place (#2705).

The coordinator used to prefer ``importlib.metadata`` while the peer reported
``omlx._version.__version__``. On an editable install whose ``dist-info`` has
drifted from the source tree — any ``git pull`` across a version bump without a
reinstall — a node then disagreed with itself and the gate blocked two machines
running byte-identical code.
"""

from __future__ import annotations

import ast
import builtins
import importlib.metadata

import pytest

from omlx._version import __version__ as source_version
from omlx.cluster import launch
from omlx.cluster.probe import __version__ as probe_version


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
