# SPDX-License-Identifier: Apache-2.0
"""Run ServerProcess restart health regressions compiled with swiftc."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2]
CASES = (
    "restart_after_stale_status",
    "probe_after_exit",
    "live_status_fallback",
    "reuse_running_sandboxed_server",
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
    # ServerProcess references SecureServer/SecureServerLaunch, so the shared
    # definitions have to be in the same compilation unit. Use the real file
    # rather than a stub: a stub would let the harness pass while the shipped
    # spawn path drifted away from it.
    source.write_text(
        (APP / "Sources/SecureServer/SecureServerShared.swift").read_text()
        + "\n"
        + (APP / "Sources/Server/ServerProcess.swift")
        .read_text()
        .replace("import AppKit\n", "")
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
            # DEBUG so SecureServer honours OMLX_SECURE_MODE_STATE_PATH and the
            # scenarios do not depend on this machine's real sandbox record.
            "-D",
            "DEBUG",
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
    env = dict(os.environ)
    # Keep the harness off this machine's real Secure Server record: where the
    # mode is enabled, ServerProcess rightly refuses to spawn locally and every
    # scenario would fail for an unrelated reason. A missing file reads as off,
    # which is the state a fresh CI runner has.
    env["OMLX_SECURE_MODE_STATE_PATH"] = str(tmp_path / "secure-server-off.json")
    result = subprocess.run(
        [str(health_harness), str(tmp_path), str(child), scenario],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {scenario}" in result.stdout
