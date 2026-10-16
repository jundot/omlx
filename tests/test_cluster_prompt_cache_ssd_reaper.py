# SPDX-License-Identifier: Apache-2.0
"""Tests for the cluster prompt-cache-ssd deployment reaper (design doc §D1/R3).

Every relaunch mints a fresh deployment id, so the "next store on the same
directory reclaims the leftovers" comment in telemetry.py's teardown never
actually holds after a non-clean exit (crash, SIGKILL, power cut) — nothing
else ever enumerates sibling directories. These tests exercise the reaper
that fixes that: each safety condition (registry membership, marker
liveness, age gate, symlink refusal) gets its own negative test, since a
directory-deletion routine's review currency is what it refuses to touch.
"""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.inference_worker import (
    _DEAD_DEPLOYMENT_IDLE_SECONDS,
    _reap_dead_prompt_cache_ssd_dirs,
)
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.registry import ClusterRegistry


def _deployment(deployment_id: str = "cluster-test") -> ClusterDeployment:
    assignments = (
        PipelineAssignment("local", 0, 3, 8, 5, 1, 1, 16),
        PipelineAssignment("studio", 1, 0, 3, 3, 1, 1, 8),
    )
    return ClusterDeployment(
        deployment_id=deployment_id,
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "user@studio.local", ("10.0.0.2",)),
        ),
        assignments=assignments,
        plan_hash="c" * 64,
    )


def _reaped_pid() -> int:
    """A pid that certainly no longer exists: one we started and collected."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _make_env(tmp_path: Path):
    """base_path/cluster/runtime layout matching the real deployment tree."""
    base_path = tmp_path / "omlx-home"
    state_dir = base_path / "cluster" / "runtime"
    root = state_dir / "prompt-cache-ssd"
    root.mkdir(parents=True)
    return base_path, state_dir, root


def _make_dir(root: Path, name: str, *, age_seconds: float = 7200) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "boundary.safetensors").write_bytes(b"snapshot")
    stamp = time.time() - age_seconds
    os.utime(d, (stamp, stamp))
    return d


def _write_marker(state_dir: Path, name: str, *, pid: int, age_seconds: float = 0.0):
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    payload = {"phase": "ready", "updated_at": stamp.isoformat(), "pid": pid}
    (state_dir / f"{name}.json").write_text(json.dumps(payload))


class TestDeadDeploymentReaped:
    def test_dead_deployment_no_registry_no_marker_reaped(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        ClusterRegistry(base_path)  # empty registry, written or not is fine
        dead_dir = _make_dir(root, "dead-deploy-rank-0")

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert not dead_dir.exists()

    def test_dead_deployment_with_stale_marker_reaped(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        dead_dir = _make_dir(root, "dead-deploy-rank-0")
        _write_marker(state_dir, "dead-deploy-rank-0", pid=_reaped_pid())

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert not dead_dir.exists()


class TestLiveOrRecentDeploymentSurvives:
    def test_registry_active_deployment_survives(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        registry = ClusterRegistry(base_path)
        registry.upsert(_deployment("cluster-test"))
        active_dir = _make_dir(root, "cluster-test-rank-0")

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert active_dir.exists()

    def test_live_marker_owner_survives_even_if_not_registered(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        ClusterRegistry(base_path)
        live_dir = _make_dir(root, "orphan-deploy-rank-0")
        _write_marker(state_dir, "orphan-deploy-rank-0", pid=os.getpid())

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert live_dir.exists()

    def test_recently_idle_dead_deployment_survives_the_age_gate(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        ClusterRegistry(base_path)
        fresh_dir = _make_dir(
            root, "dead-deploy-rank-0", age_seconds=_DEAD_DEPLOYMENT_IDLE_SECONDS / 2
        )

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert fresh_dir.exists()

    def test_symlinked_entry_never_touched(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        ClusterRegistry(base_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "boundary.safetensors").write_bytes(b"do not delete me")
        old = time.time() - 7200
        os.utime(outside, (old, old))
        (root / "dead-deploy-rank-0").symlink_to(outside)

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert (outside / "boundary.safetensors").exists()

    def test_malformed_directory_name_untouched(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        ClusterRegistry(base_path)
        odd_dir = _make_dir(root, "not-a-deployment-dir")

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert odd_dir.exists()

    def test_corrupt_registry_skips_reaping_entirely(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        registry_path = base_path / "cluster" / "deployments.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{not valid json")
        dead_dir = _make_dir(root, "dead-deploy-rank-0")

        _reap_dead_prompt_cache_ssd_dirs(root, state_dir)

        assert dead_dir.exists()

    def test_missing_root_is_a_no_op(self, tmp_path):
        base_path, state_dir, root = _make_env(tmp_path)
        missing_root = root / "does-not-exist"

        _reap_dead_prompt_cache_ssd_dirs(missing_root, state_dir)  # must not raise
