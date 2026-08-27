# SPDX-License-Identifier: Apache-2.0
"""Cluster rank worker memory must count toward local load admission.

Regression coverage for the 2026-08-21 incident: a 59 GB local model was
admitted on a 128 GB node already holding a 75 GB cluster rank worker,
because admission only ever measured its own process.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

from omlx.engine_pool import _cluster_rank_resident_bytes

_GIB = 1024**3


def _write_marker(
    state_dir,
    name,
    *,
    pid,
    age_seconds=0.0,
    load_bytes=None,
    measured_bytes=None,
    deployment="dep-a",
):
    stamp = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    payload = {"deployment_id": deployment, "pid": pid, "updated_at": stamp}
    if load_bytes is not None:
        payload["load_memory_bytes"] = load_bytes
    if measured_bytes is not None:
        payload["measured_weight_bytes"] = measured_bytes
    (state_dir / name).write_text(json.dumps(payload))


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid  # reaped by wait(); not reusable within the test


def test_fresh_live_rank_marker_is_counted(tmp_path):
    _write_marker(
        tmp_path, "model-x-rank-1.json", pid=os.getpid(), load_bytes=75 * _GIB
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 75 * _GIB


def test_stale_marker_is_ignored(tmp_path):
    _write_marker(
        tmp_path,
        "model-x-rank-1.json",
        pid=os.getpid(),
        age_seconds=3600,
        load_bytes=75 * _GIB,
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 0


def test_dead_rank_pid_is_ignored(tmp_path):
    _write_marker(
        tmp_path, "model-x-rank-1.json", pid=_dead_pid(), load_bytes=75 * _GIB
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 0


def test_admitted_deployments_own_markers_are_excluded(tmp_path):
    _write_marker(
        tmp_path, "model-x-rank-0.json", pid=os.getpid(), load_bytes=77 * _GIB
    )
    _write_marker(
        tmp_path, "model-y-rank-1.json", pid=os.getpid(), load_bytes=75 * _GIB,
        deployment="dep-b",
    )
    assert (
        _cluster_rank_resident_bytes(
            state_dir=tmp_path, exclude_deployment_id="dep-a"
        )
        == 75 * _GIB
    )


def test_measured_weight_bytes_is_the_fallback(tmp_path):
    _write_marker(
        tmp_path, "model-x-rank-1.json", pid=os.getpid(), measured_bytes=71 * _GIB
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 71 * _GIB


def test_multiple_live_ranks_sum(tmp_path):
    _write_marker(
        tmp_path, "model-x-rank-0.json", pid=os.getpid(), load_bytes=77 * _GIB
    )
    _write_marker(
        tmp_path, "model-y-rank-0.json", pid=os.getpid(), load_bytes=59 * _GIB,
        deployment="dep-b",
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 136 * _GIB


def test_malformed_and_unrelated_files_are_ignored(tmp_path):
    (tmp_path / "model-x-rank-0.json").write_text("{not json")
    (tmp_path / "launch-model-x.json").write_text(
        json.dumps({"pid": os.getpid(), "load_memory_bytes": 90 * _GIB})
    )
    _write_marker(
        tmp_path,
        "model-x-rank-1.json",
        pid=os.getpid(),
        load_bytes="not-a-number",
        measured_bytes=None,
    )
    assert _cluster_rank_resident_bytes(state_dir=tmp_path) == 0


def test_missing_state_dir_is_zero(tmp_path):
    assert _cluster_rank_resident_bytes(state_dir=tmp_path / "nope") == 0
