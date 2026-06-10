# SPDX-License-Identifier: Apache-2.0
"""Tests for VideoJobManager (omlx/video/manager.py) with a fake worker.

The manager spawns [worker_python, -I, worker_script, --spec, spec.json];
these tests point worker_python at sys.executable and worker_script at a
tiny stdlib-only script written into tmp_path, so no model / mflux / venv
is needed. Spec reference: docs/video-generation-engine-spec.md section 4.2.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

import omlx.video.manager as vm
from omlx.settings import VideoSettings
from omlx.video.manager import QueueFullError, VideoJob, VideoJobManager

GB = 1024**3


# ---------------------------------------------------------------------------
# Fake worker scripts (stdlib only -- they run under python -I)
# ---------------------------------------------------------------------------

_PRELUDE = """\
import json, sys, time

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

spec_path = sys.argv[sys.argv.index("--spec") + 1]
with open(spec_path) as f:
    spec = json.load(f)
"""

_SUCCESS_BODY = """\
emit({"phase": "loading"})
emit({"phase": "loaded"})
emit({"phase": "denoise", "step": 1, "total_steps": 2})
emit({"phase": "denoise", "step": 2, "total_steps": 2})
emit({"phase": "saving"})
with open(spec["output_path"], "wb") as f:
    f.write(b"FAKE-MP4-BYTES")
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "completed", "lifetime_max_phys_gb": 1.5}, f)
sys.exit(0)
"""

_CRASH_BODY = """\
emit({"phase": "loading"})
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "failed", "code": "worker_crashed",
               "message": "boom"}, f)
sys.exit(1)
"""

_NO_OUTPUT_BODY = """\
emit({"phase": "loading"})
emit({"phase": "saving"})
sys.exit(0)
"""

_STALL_BODY = """\
emit({"phase": "loading"})
time.sleep(60)
sys.exit(0)
"""

# Prints a heartbeat every 0.5s "forever" (bounded so a leaked process
# cannot outlive the test session by much)
_CHATTY_BODY = """\
for _ in range(240):
    emit({"phase": "denoise"})
    time.sleep(0.5)
sys.exit(0)
"""


def _write_worker(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(_PRELUDE + body)
    return script


# ---------------------------------------------------------------------------
# Fake enforcer
# ---------------------------------------------------------------------------


class FakeEnforcer:
    """Records lease-related calls so tests can assert order + release."""

    def __init__(self, ceiling_gb: float = 100.0, peak_bytes: int = 0):
        self.is_running = True
        self._ceiling = int(ceiling_gb * GB)
        self.peak = peak_bytes
        self._soft_threshold = 0.85
        self._prefill_transient_margin_bytes = 0
        self.calls: list[tuple] = []

    def get_final_ceiling(self) -> int:
        return self._ceiling

    def recent_peak_bytes(self) -> int:
        return self.peak

    def acquire_video_lease(self, lease_bytes: int) -> None:
        self.calls.append(("acquire", lease_bytes))

    def set_video_worker_pid(self, pid) -> None:
        self.calls.append(("set_pid", pid))

    def release_video_lease(self) -> None:
        self.calls.append(("release",))

    # assertion helpers ----------------------------------------------------

    def call_names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def assert_lease_cycle(self, lease_bytes: int) -> None:
        """One acquire -> set_pid(real) -> set_pid(None) -> release cycle."""
        names = self.call_names()
        assert names.count("acquire") == names.count("release") == 1
        assert ("acquire", lease_bytes) in self.calls
        assert names.index("acquire") < names.index("release")
        pids = [c[1] for c in self.calls if c[0] == "set_pid"]
        assert pids[-1] is None  # cleared before release
        assert isinstance(pids[0], int) and pids[0] > 0
        # acquire happens before the pid is registered
        assert names.index("acquire") < names.index("set_pid")
        # release is the very last lease call
        assert names[-1] == "release"


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> VideoSettings:
    kwargs = dict(
        enabled=True,
        worker_python=sys.executable,
        memory_lease_gb=1.0,
        max_queued_jobs=4,
        job_timeout_seconds=60,
        progress_stall_timeout_seconds=30,
        artifacts_max_count=50,
        artifacts_max_gb=50.0,
    )
    kwargs.update(overrides)
    return VideoSettings(**kwargs)


def _make_job(job_id: str = "video_t1", **param_overrides) -> VideoJob:
    params = dict(prompt="a cat", width=256, height=256, frames=5,
                  steps=2, fps=16, seed=7)
    params.update(param_overrides)
    return VideoJob(id=job_id, model_id="wan-test",
                    model_dir="/nonexistent/model", params=params)


def _make_manager(tmp_path: Path, worker_body: str,
                  settings: VideoSettings | None = None,
                  enforcer: FakeEnforcer | None = None,
                  ) -> tuple[VideoJobManager, FakeEnforcer]:
    enforcer = enforcer or FakeEnforcer()
    script = _write_worker(tmp_path, "fake_worker.py", worker_body)
    manager = VideoJobManager(
        settings=settings or _make_settings(),
        base_path=tmp_path,
        enforcer=enforcer,
        worker_script=script,
    )
    return manager, enforcer


async def _wait_until(cond, timeout: float = 12.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(interval)
    return False


async def _wait_terminal(job: VideoJob, timeout: float = 12.0) -> None:
    ok = await _wait_until(
        lambda: job.status in ("completed", "failed"), timeout=timeout
    )
    assert ok, (
        f"job did not reach a terminal state within {timeout}s "
        f"(status={job.status}, phase={job.phase!r})"
    )


# ---------------------------------------------------------------------------
# (1) success path
# ---------------------------------------------------------------------------


async def test_success_completes_with_artifact_and_lease_cycle(tmp_path):
    manager, enforcer = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        job = await manager.submit(_make_job("video_ok1"))
        await _wait_terminal(job)

        assert job.status == "completed"
        assert job.error is None
        assert job.progress == 100
        assert job.phase == "done"
        assert job.artifact_path is not None
        artifact = Path(job.artifact_path)
        assert artifact.exists() and artifact.stat().st_size > 0
        assert artifact == manager.artifacts_dir / job.id / "output.mp4"
        assert job.peak_memory_gb == 1.5
        assert job.wall_seconds is not None

        # wire shape
        wire = job.to_dict()
        assert wire["object"] == "video"
        assert wire["status"] == "completed"
        assert wire["progress"] == 100
        assert wire["error"] is None
        assert wire["size"] == "256x256"

        # lease acquired AND released, in order, pid registered then cleared
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)

        # persisted record reflects completion
        with open(manager.jobs_dir / f"{job.id}.json") as f:
            persisted = json.load(f)
        assert persisted["status"] == "completed"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (2) crash with failure manifest
# ---------------------------------------------------------------------------


async def test_crash_propagates_manifest_error_and_releases_lease(tmp_path):
    manager, enforcer = _make_manager(tmp_path, _CRASH_BODY)
    try:
        job = await manager.submit(_make_job("video_crash1"))
        await _wait_terminal(job)

        assert job.status == "failed"
        assert job.error == {"code": "worker_crashed", "message": "boom"}
        assert job.artifact_path is None
        # lease released even on failure
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (3) exit 0 but no output file
# ---------------------------------------------------------------------------


async def test_exit_zero_without_output_is_output_invalid(tmp_path):
    manager, enforcer = _make_manager(tmp_path, _NO_OUTPUT_BODY)
    try:
        job = await manager.submit(_make_job("video_noout1"))
        await _wait_terminal(job)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_OUTPUT_INVALID
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (4) stall: silent worker killed by the watchdog
# ---------------------------------------------------------------------------


async def test_stalled_worker_is_killed(tmp_path):
    settings = _make_settings(progress_stall_timeout_seconds=2)
    manager, enforcer = _make_manager(tmp_path, _STALL_BODY,
                                      settings=settings)
    try:
        job = await manager.submit(_make_job("video_stall1"))
        # one heartbeat then 60s of silence; watchdog ticks every 2s so the
        # kill should land well within ~8s
        await _wait_terminal(job, timeout=12.0)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_WORKER_STALLED
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (5) per-run timeout
# ---------------------------------------------------------------------------


async def test_job_timeout_kills_chatty_worker(tmp_path):
    settings = _make_settings(job_timeout_seconds=2)
    manager, enforcer = _make_manager(tmp_path, _CHATTY_BODY,
                                      settings=settings)
    try:
        job = await manager.submit(_make_job("video_timeout1"))
        await _wait_terminal(job, timeout=12.0)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_JOB_TIMEOUT
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (6) queue depth cap
# ---------------------------------------------------------------------------


async def test_queue_full_raises_when_cap_reached(tmp_path):
    settings = _make_settings(max_queued_jobs=1)
    manager, _ = _make_manager(tmp_path, _CHATTY_BODY, settings=settings)
    try:
        job_a = await manager.submit(_make_job("video_qa"))
        # wait until the dispatcher picks A up (queue drains)
        ok = await _wait_until(
            lambda: job_a.status == "in_progress" and manager.queue_depth() == 0
        )
        assert ok, "first job never started"

        await manager.submit(_make_job("video_qb"))  # fills the queue
        assert manager.queue_depth() == 1
        with pytest.raises(QueueFullError):
            await manager.submit(_make_job("video_qc"))
        assert manager.get("video_qc") is None
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (7) DELETE a running job
# ---------------------------------------------------------------------------


async def test_delete_running_job_kills_worker_and_removes_record(tmp_path):
    manager, _ = _make_manager(tmp_path, _CHATTY_BODY)
    try:
        job = await manager.submit(_make_job("video_del1"))
        ok = await _wait_until(
            lambda: job.status == "in_progress"
            and manager._current_proc is not None
        )
        assert ok, "job never started"
        proc = manager._current_proc

        assert await manager.delete(job.id) is True

        assert proc.returncode is not None  # worker terminated
        assert manager.get(job.id) is None
        assert not (manager.jobs_dir / f"{job.id}.json").exists()
        assert not (manager.artifacts_dir / job.id).exists()
        # deleting again reports not found
        assert await manager.delete(job.id) is False
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (8) startup replay marks in-flight jobs as failed
# ---------------------------------------------------------------------------


async def test_restart_replay_fails_inflight_jobs(tmp_path):
    jobs_dir = tmp_path / "video-jobs"
    jobs_dir.mkdir(parents=True)
    inflight = _make_job("video_replay1")
    inflight.status = "in_progress"
    inflight.started_at = time.time()
    with open(jobs_dir / "video_replay1.json", "w") as f:
        json.dump(inflight.to_persist(), f)
    done = _make_job("video_replay2")
    done.status = "completed"
    done.progress = 100
    done.completed_at = time.time()
    with open(jobs_dir / "video_replay2.json", "w") as f:
        json.dump(done.to_persist(), f)

    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        replayed = manager.get("video_replay1")
        assert replayed is not None
        assert replayed.status == "failed"
        assert replayed.error is not None
        assert replayed.error["code"] == vm.ERR_SERVER_RESTARTED
        assert replayed.completed_at is not None
        # the failure is persisted back to disk
        with open(jobs_dir / "video_replay1.json") as f:
            assert json.load(f)["status"] == "failed"
        # terminal jobs replay unchanged
        survivor = manager.get("video_replay2")
        assert survivor is not None and survivor.status == "completed"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (9) retention: LRU purge beyond artifacts_max_count
# ---------------------------------------------------------------------------


async def test_retention_purges_oldest_artifact_but_keeps_record(tmp_path):
    settings = _make_settings(artifacts_max_count=1)
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY, settings=settings)
    try:
        job1 = await manager.submit(_make_job("video_ret1"))
        await _wait_terminal(job1)
        assert job1.status == "completed"
        assert job1.artifact_path is not None

        job2 = await manager.submit(_make_job("video_ret2"))
        await _wait_terminal(job2)
        assert job2.status == "completed"

        ok = await _wait_until(lambda: job1.artifact_path is None)
        assert ok, "retention sweep did not purge the older artifact"
        assert job1.expires_at is not None
        assert job1.status == "completed"  # record kept, status unchanged
        assert manager.get(job1.id) is not None
        assert not (manager.artifacts_dir / job1.id).exists()
        # newest artifact survives
        assert job2.artifact_path is not None
        assert Path(job2.artifact_path).exists()
        assert job2.expires_at is None
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (10) memory admission deferral
# ---------------------------------------------------------------------------


async def test_admission_defers_then_proceeds(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_ADMISSION_RECHECK_S", 0.2)
    enforcer = FakeEnforcer(ceiling_gb=100.0, peak_bytes=200 * GB)
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY, enforcer=enforcer)
    try:
        job = await manager.submit(_make_job("video_adm1"))
        ok = await _wait_until(lambda: "waiting for memory" in job.phase)
        assert ok, f"job never reported memory wait (phase={job.phase!r})"
        assert job.status == "queued"
        assert enforcer.call_names() == []  # no lease while deferred

        enforcer.peak = 0  # pressure clears
        await _wait_terminal(job)
        assert job.status == "completed"
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (11) watchdog: footprint over lease
# ---------------------------------------------------------------------------


async def test_watchdog_kills_worker_over_lease(tmp_path, monkeypatch):
    lease = 1 * GB
    monkeypatch.setattr(vm, "get_phys_footprint", lambda pid=None: lease + GB)
    manager, enforcer = _make_manager(tmp_path, _CHATTY_BODY)
    try:
        job = await manager.submit(_make_job("video_lease1"))
        await _wait_terminal(job, timeout=12.0)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_LEASE_EXCEEDED
        enforcer.assert_lease_cycle(lease_bytes=lease)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (12) watchdog: footprint monitor failure (3x zero reads)
# ---------------------------------------------------------------------------


async def test_watchdog_kills_worker_when_monitor_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "get_phys_footprint", lambda pid=None: 0)
    manager, enforcer = _make_manager(tmp_path, _CHATTY_BODY)
    try:
        job = await manager.submit(_make_job("video_mon1"))
        # 3 zero reads at 2s watchdog cadence -> killed around t=6s
        await _wait_terminal(job, timeout=14.0)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_MONITOR_FAILED
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()
