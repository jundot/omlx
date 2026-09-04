# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime-marker pruning (design doc §D3).

A marker (`<deployment_id>-rank-<N>.json`) is removed only on a clean exit
(`RuntimeMarker.remove`); crashes, SIGKILL, the OOM reaper and power cuts
all leave it behind on purpose, as crash evidence, and nothing ever pruned
ancient ones. The reaper here must only ever delete a marker that is (a)
outside the newest-N-per-model keep window, (b) older than the age gate,
and (c) owned by a dead pid — never a live rank's marker.
"""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from omlx.cluster.inference_worker import (
    _MARKER_KEEP_NEWEST_PER_MODEL,
    _MARKER_MAX_AGE_DAYS,
    _prune_runtime_markers,
)


def _reaped_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _write_marker(
    state_dir: Path,
    name: str,
    *,
    model: str = "org/model",
    pid: int,
    age_days: float = 0.0,
) -> Path:
    stamp = datetime.now(UTC) - timedelta(days=age_days)
    payload = {"phase": "ready", "updated_at": stamp.isoformat(), "pid": pid, "model": model}
    path = state_dir / f"{name}.json"
    path.write_text(json.dumps(payload))
    mtime = time.time() - age_days * 86400
    os.utime(path, (mtime, mtime))
    return path


class TestOldDeadOwnerMarkersPruned:
    def test_old_dead_owner_marker_beyond_keep_window_deleted(self, tmp_path):
        dead = _reaped_pid()
        # Fill the keep window with 3 newer markers, then one that's old
        # and falls outside it.
        for i in range(_MARKER_KEEP_NEWEST_PER_MODEL):
            _write_marker(tmp_path, f"deploy-{i}-rank-0", pid=dead, age_days=i + 1)
        old = _write_marker(
            tmp_path,
            "deploy-old-rank-0",
            pid=dead,
            age_days=_MARKER_MAX_AGE_DAYS + 5,
        )

        _prune_runtime_markers(tmp_path)

        assert not old.exists()


class TestMarkersProtected:
    def test_within_keep_window_survives_even_when_old(self, tmp_path):
        dead = _reaped_pid()
        # Only one marker for this model: always within the keep-3 window,
        # regardless of age — the doc's rationale is "may be the only
        # diagnostic evidence for a recent-ish crash."
        only = _write_marker(
            tmp_path, "deploy-only-rank-0", pid=dead, age_days=_MARKER_MAX_AGE_DAYS + 30
        )

        _prune_runtime_markers(tmp_path)

        assert only.exists()

    def test_under_age_gate_survives_outside_keep_window(self, tmp_path):
        dead = _reaped_pid()
        for i in range(_MARKER_KEEP_NEWEST_PER_MODEL):
            _write_marker(tmp_path, f"deploy-{i}-rank-0", pid=dead, age_days=i + 1)
        recent = _write_marker(
            tmp_path, "deploy-recent-rank-0", pid=dead, age_days=_MARKER_MAX_AGE_DAYS - 5
        )

        _prune_runtime_markers(tmp_path)

        assert recent.exists()

    def test_live_owner_marker_never_pruned(self, tmp_path):
        for i in range(_MARKER_KEEP_NEWEST_PER_MODEL):
            _write_marker(
                tmp_path, f"deploy-{i}-rank-0", pid=_reaped_pid(), age_days=i + 1
            )
        live = _write_marker(
            tmp_path,
            "deploy-live-rank-0",
            pid=os.getpid(),
            age_days=_MARKER_MAX_AGE_DAYS + 30,
        )

        _prune_runtime_markers(tmp_path)

        assert live.exists()

    def test_keep_window_is_per_model_not_global(self, tmp_path):
        dead = _reaped_pid()
        # Model A has 3 newer markers filling its own keep window; model B's
        # single old marker must not be evicted by model A's occupancy.
        for i in range(_MARKER_KEEP_NEWEST_PER_MODEL):
            _write_marker(
                tmp_path, f"a-deploy-{i}-rank-0", model="org/model-a", pid=dead, age_days=i + 1
            )
        b_marker = _write_marker(
            tmp_path,
            "b-deploy-rank-0",
            model="org/model-b",
            pid=dead,
            age_days=1,
        )

        _prune_runtime_markers(tmp_path)

        assert b_marker.exists()

    def test_symlinked_marker_never_touched(self, tmp_path):
        outside = tmp_path / "outside.json"
        dead = _reaped_pid()
        stamp = datetime.now(UTC) - timedelta(days=_MARKER_MAX_AGE_DAYS + 5)
        outside.write_text(
            json.dumps({"phase": "ready", "updated_at": stamp.isoformat(), "pid": dead})
        )
        for i in range(_MARKER_KEEP_NEWEST_PER_MODEL):
            _write_marker(tmp_path, f"deploy-{i}-rank-0", pid=dead, age_days=i + 1)
        link = tmp_path / "deploy-link-rank-0.json"
        link.symlink_to(outside)

        _prune_runtime_markers(tmp_path)

        assert outside.exists()

    def test_missing_state_dir_is_a_no_op(self, tmp_path):
        _prune_runtime_markers(tmp_path / "does-not-exist")  # must not raise
