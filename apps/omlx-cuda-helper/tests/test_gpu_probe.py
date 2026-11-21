# SPDX-License-Identifier: Apache-2.0
"""The GPU probe must say when it failed, not only what it found.

On 2026-11-20 nvidia-smi failed inside the helper container; the probe
answered with no GPUs, the host planned 0 layers, and prefill ran locally
with nothing naming why. These tests pin the distinction: an empty list from
a probe that answered, an empty list from a probe that failed, and the error
text the second one now carries.
"""

from __future__ import annotations

import logging
import sys

from omlx_cuda_helper import gpus
from omlx_cuda_helper.control import ControlServer
from omlx_cuda_helper.files import ModelStore
from omlx_cuda_helper.state import State
from omlx_cuda_helper.supervisor import Supervisor

NVML = "Failed to initialize NVML: Driver/library version mismatch"


def test_a_failed_command_keeps_its_exit_code_and_stderr():
    failure = gpus._run(
        sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"
    )
    assert failure.returncode == 3
    assert failure.stdout == ""
    assert failure.stderr.strip() == "boom"
    assert failure.failed is True


def test_a_command_that_cannot_run_is_not_a_silent_empty_answer():
    missing = gpus._run("/nonexistent/nvidia-smi")
    # No process ran, so there is no exit code — but there is a reason.
    assert missing.returncode is None and missing.stdout == ""
    assert missing.stderr
    assert missing.failed is True


def test_a_failed_probe_still_returns_no_gpus_and_says_why(monkeypatch, caplog):
    monkeypatch.setattr(gpus, "_run", lambda *command: gpus.RunResult("", 255, NVML))
    with caplog.at_level(logging.WARNING, logger="omlx_cuda_helper.gpus"):
        # Existing callers ask for a list and still get one.
        assert gpus.gpus() == []
        found, status = gpus.probe()
    assert found == []
    assert status == {"ok": False, "error": "nvidia-smi exited 255: " + NVML}
    # And the container log carries it, which is half the point.
    assert NVML in caplog.text


def test_a_probe_that_answered_with_no_gpus_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(gpus, "_run", lambda *command: gpus.RunResult("", 0, ""))
    found, status = gpus.probe()
    assert found == []
    assert status == {"ok": True, "error": ""}


def test_the_info_payload_carries_the_probe_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gpus, "probe", lambda only=None: ([], {"ok": False, "error": NVML})
    )
    control = ControlServer(
        State(tmp_path / "state", name="nvidia1"),
        ModelStore(tmp_path / "models"),
        Supervisor(None, tmp_path / "run"),
        {},
    )
    try:
        payload = control._info("127.0.0.1")
    finally:
        control.stop()
    assert payload["gpus"] == []
    assert payload["gpu_probe"] == {"ok": False, "error": NVML}
    # Additive: everything the host read before is still in the answer.
    assert payload["op"] == "ok"
    assert {"version", "ram", "ram_available", "links", "session"} <= set(payload)
