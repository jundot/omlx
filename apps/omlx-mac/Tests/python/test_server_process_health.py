# SPDX-License-Identifier: Apache-2.0
"""Run ServerProcess restart health regressions compiled with swiftc."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2]
CASES = ("restart_after_stale_status", "probe_after_exit", "live_status_fallback")


@pytest.fixture(scope="module")
def health_harness(tmp_path_factory):
    if sys.platform != "darwin":
        pytest.skip("ServerProcess uses macOS frameworks")
    compiler = shutil.which("swiftc")
    if compiler is None:
        pytest.skip("Install the Xcode Command Line Tools to run Swift regressions")
    directory = tmp_path_factory.mktemp("server-process-health")
    source = directory / "ServerProcessHealth.swift"
    source.write_text(
        (APP / "Sources/Server/ServerProcess.swift").read_text()
        + "\n"
        + (APP / "Tests/Fixtures/ServerProcessHealthHarness.swift").read_text()
    )
    executable = directory / "health-tests"
    result = subprocess.run(
        [
            compiler,
            "-swift-version",
            "6",
            "-parse-as-library",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("scenario", CASES)
def test_server_process_health(health_harness, tmp_path, scenario):
    # Stands in for Python and ignores the serve arguments.
    child = tmp_path / "child"
    child.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
    child.chmod(0o755)
    result = subprocess.run(
        [str(health_harness), str(tmp_path), str(child), scenario],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {scenario}" in result.stdout
