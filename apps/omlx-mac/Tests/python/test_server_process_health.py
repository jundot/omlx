# SPDX-License-Identifier: Apache-2.0
"""Execute ServerProcess health/restart regressions with the macOS Swift compiler.

The harness is appended to the unmodified production source so a same-file
extension can control private lifecycle state and probe ordering. Only external
dependencies are stubbed; no lifecycle method is copied or rewritten. This runs
in the Python CI suite without an Xcode test host or the bundled Python runtime.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2]
CASES = (
    "restart_after_stale_status",
    "stop_during_backoff",
    "exit_cancels_health_loop",
    "no_child_health",
    "dead_child_status",
    "live_health",
    "live_status_fallback",
    "live_failure_and_recovery",
    "probe_after_exit",
    "probe_after_replacement",
    "cancelled_probe",
)


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
    # Accept the real spawn arguments, but never start Python or load a model.
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
