# SPDX-License-Identifier: Apache-2.0
"""Tests for VideoJobManager (omlx/video/manager.py) with a fake worker.

The manager spawns [worker_python, -I, worker_script, --spec, spec.json];
these tests point worker_python at sys.executable and worker_script at a
tiny stdlib-only script written into tmp_path, so no model / mflux / venv
is needed. Spec reference: docs/video-generation-engine-spec.md section 4.2.
"""

import asyncio
import json
import shutil
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


# ---------------------------------------------------------------------------
# (13) submit with input_reference (I2V conditioning image)
# ---------------------------------------------------------------------------


async def test_submit_writes_input_reference_and_image_path(tmp_path):
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        png = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
        job = await manager.submit(
            _make_job("video_ref1"), input_reference=(png, ".png")
        )

        # The reference lands in the job's blob dir synchronously at submit
        ref = manager.artifacts_dir / job.id / "input_reference.png"
        assert ref.exists()
        assert ref.read_bytes() == png
        assert job.params["image_path"] == str(ref)

        # The reference does not disturb the normal run; the worker spec
        # carries image_path through
        await _wait_terminal(job)
        assert job.status == "completed"
        with open(manager.artifacts_dir / job.id / "spec.json") as f:
            spec = json.load(f)
        assert spec["image_path"] == str(ref)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (14) extend: dispatch resolves the source artifact into the spec
# ---------------------------------------------------------------------------


async def test_extend_job_writes_extend_from_into_spec(tmp_path):
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        src = await manager.submit(_make_job("video_src1"))
        await _wait_terminal(src)
        assert src.status == "completed"
        assert src.artifact_path is not None

        job = await manager.submit(
            _make_job("video_ext1", extend_source_id=src.id)
        )
        await _wait_terminal(job)
        assert job.status == "completed"

        with open(manager.artifacts_dir / job.id / "spec.json") as f:
            spec = json.load(f)
        # No output_raw.mp4 in the source blob dir -> falls back to the
        # artifact itself
        assert spec["extend_from"] == src.artifact_path
        assert spec["extend_source_id"] == src.id
        assert Path(spec["extend_from"]).exists()
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (15) extend: missing source fails before lease/spawn
# ---------------------------------------------------------------------------


async def test_extend_missing_source_fails_without_lease_or_spawn(tmp_path):
    manager, enforcer = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        job = await manager.submit(
            _make_job("video_ext_orphan", extend_source_id="video_gone")
        )
        await _wait_terminal(job)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_EXTEND_SOURCE_MISSING
        # Failed before lease acquisition: no acquire/set_pid/release at all
        assert enforcer.calls == []
        # _run_job bails before writing the spec, so no worker was spawned
        assert not (manager.artifacts_dir / job.id / "spec.json").exists()
        assert not (manager.artifacts_dir / job.id / "output.mp4").exists()
    finally:
        await manager.shutdown()


async def test_extend_source_artifact_deleted_fails_before_dispatch(tmp_path):
    manager, enforcer = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        src = await manager.submit(_make_job("video_src2"))
        await _wait_terminal(src)
        assert src.status == "completed"
        acquires_before = enforcer.call_names().count("acquire")
        # Source record survives but its artifact blob is gone
        shutil.rmtree(manager.artifacts_dir / src.id)

        job = await manager.submit(
            _make_job("video_ext2", extend_source_id=src.id)
        )
        await _wait_terminal(job)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_EXTEND_SOURCE_MISSING
        # No second lease cycle for the failed extend job
        assert enforcer.call_names().count("acquire") == acquires_before
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (16) extend_source_path: output_raw.mp4 preferred over artifact_path
# ---------------------------------------------------------------------------


async def test_extend_source_path_prefers_raw_then_artifact(tmp_path):
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        job = _make_job("video_srcsel")
        blob_dir = manager.artifacts_dir / job.id
        blob_dir.mkdir(parents=True)
        artifact = blob_dir / "output.mp4"
        artifact.write_bytes(b"upscaled")
        job.artifact_path = str(artifact)
        raw = blob_dir / "output_raw.mp4"
        raw.write_bytes(b"raw")

        # The pre-upscale stitchable video wins over the artifact
        assert manager.extend_source_path(job) == raw

        raw.unlink()
        assert manager.extend_source_path(job) == artifact

        artifact.unlink()
        assert manager.extend_source_path(job) is None

        # No artifact_path recorded at all -> None as well
        job.artifact_path = None
        assert manager.extend_source_path(job) is None
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (17) retention: pending extend jobs protect their source artifact
# ---------------------------------------------------------------------------


async def test_retention_protects_pending_extend_source(tmp_path):
    settings = _make_settings(artifacts_max_count=1)
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY, settings=settings)
    try:
        src = await manager.submit(_make_job("video_prot_src"))
        await _wait_terminal(src)
        assert src.status == "completed"

        # A queued job extending src; injected directly (not enqueued) so it
        # stays "queued" while later jobs run their retention sweeps
        pending = _make_job("video_prot_pending", extend_source_id=src.id)
        manager._jobs[pending.id] = pending

        job_b = await manager.submit(_make_job("video_prot_b"))
        await _wait_terminal(job_b)
        job_c = await manager.submit(_make_job("video_prot_c"))
        await _wait_terminal(job_c)

        # cap=1 over three artifacts: the oldest UNPROTECTED one (job_b)
        # purges
        ok = await _wait_until(lambda: job_b.artifact_path is None)
        assert ok, "unprotected artifact was not purged"
        # ...while the protected source survives despite being oldest of all
        assert src.artifact_path is not None
        assert Path(src.artifact_path).exists()
        assert src.expires_at is None
        # newest artifact survives as usual
        assert job_c.artifact_path is not None
        assert Path(job_c.artifact_path).exists()
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (18) progress mapping for post-generation phases
# ---------------------------------------------------------------------------


async def test_apply_progress_upscaling_and_stitching_mapping(tmp_path):
    manager, _ = _make_manager(tmp_path, _SUCCESS_BODY)
    try:
        # Plain job (no upscale): denoise keeps the full 5-95 band
        plain = _make_job("video_prog0")
        manager._apply_progress(
            plain, {"phase": "denoise", "step": 1, "total_steps": 2}
        )
        assert plain.progress == 50
        manager._apply_progress(
            plain, {"phase": "denoise", "step": 2, "total_steps": 2}
        )
        assert plain.progress == 95

        # Upscale job: denoise compresses to 5-65 so the minutes-long
        # SeedVR2 pass owns 65-97 and the bar never looks hung
        job = _make_job("video_prog1", upscale_resolution=1080)
        manager._apply_progress(
            job, {"phase": "denoise", "step": 1, "total_steps": 2}
        )
        assert job.progress == 35
        manager._apply_progress(
            job, {"phase": "denoise", "step": 2, "total_steps": 2}
        )
        assert job.progress == 65
        # Stitching pins the post-generation floor (denoise band top)
        manager._apply_progress(job, {"phase": "stitching"})
        assert job.progress == 65
        assert job.phase == "stitching"
        # Upscaling: 65 + int(32 * step / total), per-frame phase text
        manager._apply_progress(
            job, {"phase": "upscaling", "step": 5, "total_steps": 10}
        )
        assert job.progress == 81
        assert job.phase == "upscaling 5/10"
        manager._apply_progress(
            job, {"phase": "upscaling", "step": 10, "total_steps": 10}
        )
        assert job.progress == 97
        assert job.phase == "upscaling 10/10"

        # Upscaling without step totals holds the 65 floor
        floor_job = _make_job("video_prog2", upscale_resolution=1080)
        manager._apply_progress(floor_job, {"phase": "upscaling"})
        assert floor_job.progress == 65
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (19) wire shape: i2v / extend / upscale extension fields
# ---------------------------------------------------------------------------


def test_to_dict_includes_extension_fields():
    job = _make_job(
        "video_wire1",
        pipeline="i2v",
        extend_source_id="video_src9",
        upscale_resolution=1080,
        image_path="/tmp/ref.png",
    )
    wire = job.to_dict()
    assert wire["pipeline"] == "i2v"
    assert wire["extend_source_id"] == "video_src9"
    assert wire["upscale_resolution"] == 1080
    assert wire["has_input_reference"] is True

    # Plain t2v job: fields present, null/false defaults
    plain = _make_job("video_wire2").to_dict()
    assert plain["pipeline"] is None
    assert plain["extend_source_id"] is None
    assert plain["upscale_resolution"] is None
    assert plain["has_input_reference"] is False


# ---------------------------------------------------------------------------
# (20) Active Models card: live worker memory (0 when idle)
# ---------------------------------------------------------------------------


async def test_current_worker_memory_zero_when_idle(tmp_path):
    manager, _enf = _make_manager(tmp_path, "pass")
    assert manager.current_worker_memory_bytes() == 0
    await manager.shutdown()
