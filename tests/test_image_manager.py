# SPDX-License-Identifier: Apache-2.0
"""Tests for image jobs on MediaJobManager (omlx/video/manager.py).

One manager dispatches BOTH kinds: these tests run image jobs (and one
video regression job) through fake worker scripts written into tmp_path,
exactly like tests/test_video_manager.py. The manager spawns
[worker_python, -I, <kind worker script>, --spec, spec.json]; worker_python
points at sys.executable, the image script comes from the
image_worker_script constructor seam. No model / mflux / venv is needed.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

import omlx.video.manager as vm
from omlx.settings import ImageSettings, VideoSettings
from omlx.video.manager import MediaJob, MediaJobManager, QueueFullError

GB = 1024**3


# ---------------------------------------------------------------------------
# Fake worker scripts (stdlib only -- they run under python -I)
# ---------------------------------------------------------------------------

_PRELUDE = """\
import json, os, sys, time

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

spec_path = sys.argv[sys.argv.index("--spec") + 1]
with open(spec_path) as f:
    spec = json.load(f)
"""

# Image success: writes output-0.png .. output-{n-1}.png into
# spec["output_dir"] and a manifest listing them; emits the image worker's
# JSONL protocol including image_index.
_IMAGE_SUCCESS_BODY = """\
emit({"phase": "loading"})
emit({"phase": "loaded"})
n = int(spec.get("n") or 1)
total = int(spec.get("steps") or 2)
outputs = []
for i in range(n):
    for s in range(1, total + 1):
        emit({"phase": "denoise", "step": s, "total_steps": total,
              "image_index": i})
    emit({"phase": "saving", "image_index": i})
    name = "output-%d.png" % i
    with open(os.path.join(spec["output_dir"], name), "wb") as f:
        f.write(b"FAKE-PNG-BYTES")
    outputs.append(name)
emit({"phase": "done"})
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "completed", "outputs": outputs,
               "lifetime_max_phys_gb": 2.5}, f)
sys.exit(0)
"""

# Exit 0 with a manifest that lists a file the worker never wrote.
_IMAGE_MISSING_FILE_BODY = """\
emit({"phase": "loading"})
emit({"phase": "loaded"})
with open(os.path.join(spec["output_dir"], "output-0.png"), "wb") as f:
    f.write(b"FAKE-PNG-BYTES")
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "completed",
               "outputs": ["output-0.png", "output-1.png"]}, f)
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

# Video success body (copied shape from tests/test_video_manager.py) for
# the cross-kind regression test.
_VIDEO_SUCCESS_BODY = """\
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


def _write_worker(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(_PRELUDE + body)
    return script


# ---------------------------------------------------------------------------
# Fake enforcer (same seam as tests/test_video_manager.py)
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
        assert names.index("acquire") < names.index("set_pid")
        assert names[-1] == "release"


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make_video_settings(**overrides) -> VideoSettings:
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


def _make_image_settings(**overrides) -> ImageSettings:
    kwargs = dict(
        enabled=True,
        worker_python=sys.executable,
        memory_lease_gb=3.0,
        max_queued_jobs=4,
        job_timeout_seconds=60,
        progress_stall_timeout_seconds=30,
        artifacts_max_count=50,
        artifacts_max_gb=50.0,
    )
    kwargs.update(overrides)
    return ImageSettings(**kwargs)


def _make_image_job(job_id: str = "img_t1", **param_overrides) -> MediaJob:
    params = dict(prompt="a cat", width=256, height=256, steps=2, n=1,
                  seed=7, alias="z-image-turbo", pipeline="t2i")
    params.update(param_overrides)
    return MediaJob(id=job_id, kind="image", model_id="z-image-turbo-4bit",
                    model_dir="/nonexistent/model", params=params)


def _make_video_job(job_id: str = "video_t1", **param_overrides) -> MediaJob:
    params = dict(prompt="a cat", width=256, height=256, frames=5,
                  steps=2, fps=16, seed=7)
    params.update(param_overrides)
    return MediaJob(id=job_id, kind="video", model_id="wan-test",
                    model_dir="/nonexistent/model", params=params)


def _make_manager(
    tmp_path: Path,
    image_body: str = _IMAGE_SUCCESS_BODY,
    video_body: str = _VIDEO_SUCCESS_BODY,
    video_settings: VideoSettings | None = None,
    image_settings: ImageSettings | None = None,
    enforcer: FakeEnforcer | None = None,
) -> tuple[MediaJobManager, FakeEnforcer]:
    enforcer = enforcer or FakeEnforcer()
    video_script = _write_worker(tmp_path, "fake_video_worker.py", video_body)
    image_script = _write_worker(tmp_path, "fake_image_worker.py", image_body)
    manager = MediaJobManager(
        settings=video_settings or _make_video_settings(),
        base_path=tmp_path,
        enforcer=enforcer,
        worker_script=video_script,
        image_settings=image_settings or _make_image_settings(),
        image_worker_script=image_script,
    )
    return manager, enforcer


async def _wait_until(cond, timeout: float = 12.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(interval)
    return False


async def _wait_terminal(job: MediaJob, timeout: float = 12.0) -> None:
    ok = await _wait_until(
        lambda: job.status in ("completed", "failed"), timeout=timeout
    )
    assert ok, (
        f"job did not reach a terminal state within {timeout}s "
        f"(status={job.status}, phase={job.phase!r})"
    )


# ---------------------------------------------------------------------------
# (1) image happy path
# ---------------------------------------------------------------------------


async def test_image_success_completes_with_artifacts_and_lease(tmp_path):
    manager, enforcer = _make_manager(tmp_path)
    try:
        job = await manager.submit(_make_image_job("img_ok1"))
        await _wait_terminal(job)

        assert job.status == "completed"
        assert job.error is None
        assert job.progress == 100
        assert job.phase == "done"
        assert job.artifact_files == ["output-0.png"]
        blob_dir = manager.artifacts_dir_for("image") / job.id
        assert job.artifact_path == str(blob_dir / "output-0.png")
        artifact = Path(job.artifact_path)
        assert artifact.exists() and artifact.stat().st_size > 0
        assert job.peak_memory_gb == 2.5
        assert job.wall_seconds is not None

        # wire shape
        wire = job.to_dict()
        assert wire["object"] == "image.job"
        assert wire["status"] == "completed"
        assert wire["progress"] == 100
        assert wire["error"] is None
        assert wire["size"] == "256x256"
        assert wire["n"] == 1
        assert wire["outputs"] == 1
        assert wire["pipeline"] == "t2i"
        assert wire["has_input_image"] is False

        # The lease comes from the IMAGE settings default (3GB), not the
        # video settings (1GB) -- kind settings resolution.
        enforcer.assert_lease_cycle(lease_bytes=3 * GB)

        # persisted record reflects completion (in the image jobs dir)
        with open(manager.jobs_dir_for("image") / f"{job.id}.json") as f:
            persisted = json.load(f)
        assert persisted["status"] == "completed"
        assert persisted["kind"] == "image"
        assert persisted["artifact_files"] == ["output-0.png"]
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (2) n=2 outputs
# ---------------------------------------------------------------------------


async def test_image_n2_collects_both_outputs(tmp_path):
    manager, _ = _make_manager(tmp_path)
    try:
        job = await manager.submit(_make_image_job("img_n2", n=2))
        await _wait_terminal(job)

        assert job.status == "completed"
        assert job.artifact_files == ["output-0.png", "output-1.png"]
        blob_dir = manager.artifacts_dir_for("image") / job.id
        for name in job.artifact_files:
            f = blob_dir / name
            assert f.exists() and f.stat().st_size > 0
        # artifact_path points at the FIRST output
        assert job.artifact_path == str(blob_dir / "output-0.png")
        assert job.to_dict()["outputs"] == 2
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (3) manifest lists a file the worker never wrote
# ---------------------------------------------------------------------------


async def test_manifest_missing_file_is_output_invalid(tmp_path):
    manager, enforcer = _make_manager(tmp_path,
                                      image_body=_IMAGE_MISSING_FILE_BODY)
    try:
        job = await manager.submit(_make_image_job("img_missing1", n=2))
        await _wait_terminal(job)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_OUTPUT_INVALID
        assert "output-1.png" in job.error["message"]
        assert job.artifact_files == []
        assert job.artifact_path is None
        # lease released even on failure
        enforcer.assert_lease_cycle(lease_bytes=3 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (4) per-kind directories
# ---------------------------------------------------------------------------


async def test_per_kind_dirs_isolate_image_from_video(tmp_path):
    manager, _ = _make_manager(tmp_path)
    try:
        assert manager.jobs_dir_for("image") == tmp_path / "image-jobs"
        assert manager.artifacts_dir_for("image") == tmp_path / "image-artifacts"
        # Legacy video-named accessors stay pinned to the video dirs
        assert manager.jobs_dir == tmp_path / "video-jobs"
        assert manager.artifacts_dir == tmp_path / "video-artifacts"

        job = await manager.submit(_make_image_job("img_dirs1"))
        await _wait_terminal(job)

        assert (tmp_path / "image-jobs" / "img_dirs1.json").exists()
        assert (tmp_path / "image-artifacts" / "img_dirs1" / "output-0.png").exists()
        assert not (tmp_path / "video-jobs" / "img_dirs1.json").exists()
        assert not (tmp_path / "video-artifacts" / "img_dirs1").exists()
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (5) per-kind queue caps
# ---------------------------------------------------------------------------


async def test_image_queue_cap_independent_of_video(tmp_path):
    manager, _ = _make_manager(
        tmp_path,
        image_body=_CHATTY_BODY,
        image_settings=_make_image_settings(max_queued_jobs=1),
        video_settings=_make_video_settings(max_queued_jobs=4),
    )
    try:
        job_a = await manager.submit(_make_image_job("img_qa"))
        ok = await _wait_until(
            lambda: job_a.status == "in_progress"
            and manager.queue_depth("image") == 0
        )
        assert ok, "first image job never started"

        await manager.submit(_make_image_job("img_qb"))  # fills the image queue
        assert manager.queue_depth("image") == 1
        with pytest.raises(QueueFullError) as excinfo:
            await manager.submit(_make_image_job("img_qc"))
        assert "Image queue is full" in str(excinfo.value)
        assert manager.get("img_qc") is None

        # The video queue is NOT full: a video job is still accepted
        video_job = await manager.submit(_make_video_job("video_qok"))
        assert manager.get("video_qok") is video_job
        assert manager.queue_depth("video") == 1
        assert manager.queue_depth("image") == 1
        assert manager.queue_depth() == 2  # combined FIFO
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (6) kind settings resolution: image timeouts apply to image jobs
# ---------------------------------------------------------------------------


async def test_image_job_uses_image_timeout_not_video(tmp_path):
    # Video timeout is long (60s); the image timeout (2s) must be the one
    # that kills the chatty image worker.
    manager, enforcer = _make_manager(
        tmp_path,
        image_body=_CHATTY_BODY,
        image_settings=_make_image_settings(job_timeout_seconds=2),
        video_settings=_make_video_settings(job_timeout_seconds=60),
    )
    try:
        job = await manager.submit(_make_image_job("img_timeout1"))
        await _wait_terminal(job, timeout=12.0)

        assert job.status == "failed"
        assert job.error is not None
        assert job.error["code"] == vm.ERR_JOB_TIMEOUT
        enforcer.assert_lease_cycle(lease_bytes=3 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (7) per-job lease_bytes overrides the kind settings lease
# ---------------------------------------------------------------------------


async def test_job_lease_bytes_overrides_settings(tmp_path):
    manager, enforcer = _make_manager(tmp_path)
    try:
        job = _make_image_job("img_lease1")
        job.lease_bytes = 2 * GB  # image settings say 3GB; the job wins
        await manager.submit(job)
        await _wait_terminal(job)

        assert job.status == "completed"
        enforcer.assert_lease_cycle(lease_bytes=2 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (8) wait_terminal
# ---------------------------------------------------------------------------


async def test_wait_terminal_returns_on_completion(tmp_path):
    manager, _ = _make_manager(tmp_path)
    try:
        job = await manager.submit(_make_image_job("img_wait1"))
        finished = await manager.wait_terminal(job.id, timeout=15.0)

        assert finished is not None
        assert finished.id == job.id
        assert finished.status == "completed"
    finally:
        await manager.shutdown()


async def test_wait_terminal_returns_none_on_timeout(tmp_path):
    manager, _ = _make_manager(tmp_path, image_body=_CHATTY_BODY)
    try:
        job = await manager.submit(_make_image_job("img_wait2"))
        result = await manager.wait_terminal(job.id, timeout=0.3)

        assert result is None  # still running; caller polls GET
        assert manager.get(job.id) is not None
        assert manager.get(job.id).status in ("queued", "in_progress")
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (9) startup replay marks in-flight image jobs as failed
# ---------------------------------------------------------------------------


async def test_restart_replay_fails_inflight_image_jobs(tmp_path):
    jobs_dir = tmp_path / "image-jobs"
    jobs_dir.mkdir(parents=True)
    inflight = _make_image_job("img_replay1")
    inflight.status = "in_progress"
    inflight.started_at = time.time()
    with open(jobs_dir / "img_replay1.json", "w") as f:
        json.dump(inflight.to_persist(), f)
    done = _make_image_job("img_replay2")
    done.status = "completed"
    done.progress = 100
    done.completed_at = time.time()
    with open(jobs_dir / "img_replay2.json", "w") as f:
        json.dump(done.to_persist(), f)

    manager, _ = _make_manager(tmp_path)
    try:
        replayed = manager.get("img_replay1")
        assert replayed is not None
        assert replayed.kind == "image"
        assert replayed.status == "failed"
        assert replayed.error is not None
        assert replayed.error["code"] == vm.ERR_SERVER_RESTARTED
        assert replayed.completed_at is not None
        # the failure is persisted back to disk
        with open(jobs_dir / "img_replay1.json") as f:
            assert json.load(f)["status"] == "failed"
        # terminal jobs replay unchanged
        survivor = manager.get("img_replay2")
        assert survivor is not None and survivor.status == "completed"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (10) DELETE removes record + blobs and wakes wait_terminal
# ---------------------------------------------------------------------------


async def test_delete_running_image_job_wakes_waiter(tmp_path):
    manager, _ = _make_manager(tmp_path, image_body=_CHATTY_BODY)
    try:
        job = await manager.submit(_make_image_job("img_del1"))
        ok = await _wait_until(
            lambda: job.status == "in_progress"
            and manager._current_proc is not None
        )
        assert ok, "job never started"
        proc = manager._current_proc

        waiter = asyncio.create_task(
            manager.wait_terminal(job.id, timeout=30.0)
        )
        await asyncio.sleep(0.1)  # let the waiter block on the event

        assert await manager.delete(job.id) is True

        # The waiter wakes promptly (well under its 30s timeout) and sees
        # the record gone.
        result = await asyncio.wait_for(waiter, timeout=5.0)
        assert result is None

        assert proc.returncode is not None  # worker terminated
        assert manager.get(job.id) is None
        assert not (manager.jobs_dir_for("image") / f"{job.id}.json").exists()
        assert not (manager.artifacts_dir_for("image") / job.id).exists()
        # deleting again reports not found
        assert await manager.delete(job.id) is False
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (11) video regression: a video job through the same manager still works
# ---------------------------------------------------------------------------


async def test_video_job_still_works_through_same_manager(tmp_path):
    manager, enforcer = _make_manager(tmp_path)
    try:
        job = await manager.submit(_make_video_job("video_reg1"))
        await _wait_terminal(job)

        assert job.status == "completed"
        assert job.artifact_path is not None
        artifact = Path(job.artifact_path)
        assert artifact == manager.artifacts_dir / job.id / "output.mp4"
        assert artifact.exists() and artifact.stat().st_size > 0
        assert job.to_dict()["object"] == "video"
        # Video jobs use the VIDEO settings lease (1GB), not the image one
        enforcer.assert_lease_cycle(lease_bytes=1 * GB)
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (12) retention: image caps purge image blobs only
# ---------------------------------------------------------------------------


async def test_retention_image_cap_purges_image_blobs_only(tmp_path):
    manager, _ = _make_manager(
        tmp_path,
        image_settings=_make_image_settings(artifacts_max_count=1),
        video_settings=_make_video_settings(artifacts_max_count=50),
    )
    try:
        video_job = await manager.submit(_make_video_job("video_keep1"))
        await _wait_terminal(video_job)
        assert video_job.status == "completed"

        img1 = await manager.submit(_make_image_job("img_ret1"))
        await _wait_terminal(img1)
        assert img1.status == "completed"

        img2 = await manager.submit(_make_image_job("img_ret2"))
        await _wait_terminal(img2)
        assert img2.status == "completed"

        ok = await _wait_until(lambda: img1.artifact_path is None)
        assert ok, "retention sweep did not purge the older image artifact"
        assert img1.artifact_files == []
        assert img1.expires_at is not None
        assert img1.status == "completed"  # record kept, status unchanged
        assert manager.get(img1.id) is not None
        assert not (manager.artifacts_dir_for("image") / img1.id).exists()

        # newest image artifact survives
        assert img2.artifact_path is not None
        assert Path(img2.artifact_path).exists()
        assert img2.expires_at is None

        # the video artifact is untouched by the image cap (kind isolation)
        assert video_job.artifact_path is not None
        assert Path(video_job.artifact_path).exists()
        assert video_job.expires_at is None
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# (13) progress mapping: image denoise band split across n via image_index
# ---------------------------------------------------------------------------


async def test_apply_progress_image_band_split_across_n(tmp_path):
    manager, _ = _make_manager(tmp_path)
    try:
        job = _make_image_job("img_prog1", n=2)
        manager._apply_progress(job, {"phase": "loading"})
        assert job.progress == 2
        manager._apply_progress(job, {"phase": "loaded"})
        assert job.progress == 5

        # Image 0 of 2 at step 1/2: done_fraction (0 + 0.5) / 2 = 0.25
        manager._apply_progress(
            job, {"phase": "denoise", "step": 1, "total_steps": 2,
                  "image_index": 0}
        )
        assert job.progress == 5 + int(90 * 0.25)
        assert job.phase == "denoise 1/2"
        # Image 1 of 2 at step 2/2: done_fraction 1.0 -> top of band
        manager._apply_progress(
            job, {"phase": "denoise", "step": 2, "total_steps": 2,
                  "image_index": 1}
        )
        assert job.progress == 95
        assert job.phase == "denoise 2/2"

        manager._apply_progress(job, {"phase": "saving", "image_index": 1})
        assert job.progress == 97
        assert job.phase == "saving"

        # Single-image jobs keep the raw phase text (no x/n suffix)
        single = _make_image_job("img_prog2", n=1)
        manager._apply_progress(
            single, {"phase": "denoise", "step": 1, "total_steps": 2,
                     "image_index": 0}
        )
        assert single.progress == 5 + int(90 * 0.5)
        assert single.phase == "denoise"
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# Review regressions: infeasible-lease fail-fast + actual-dimension manifest
# ---------------------------------------------------------------------------


async def test_infeasible_lease_fails_fast_and_unblocks_queue(
    tmp_path, monkeypatch
):
    # A job whose lease exceeds the ceiling can never pass admission; the
    # dispatcher must fail it after the grace window instead of wedging the
    # shared FIFO (and starving feasible jobs of both kinds behind it).
    monkeypatch.setattr(vm, "_INFEASIBLE_FAIL_S", 0.2)
    manager, _ = _make_manager(
        tmp_path, enforcer=FakeEnforcer(ceiling_gb=10.0)
    )
    try:
        wedged = _make_image_job("img_wedge")
        wedged.lease_bytes = 12 * GB  # > 10GB ceiling: permanently infeasible
        feasible = _make_image_job("img_after")

        await manager.submit(wedged)
        await manager.submit(feasible)

        await _wait_terminal(wedged)
        assert wedged.status == "failed"
        assert wedged.error is not None
        assert wedged.error["code"] == vm.ERR_LEASE_INFEASIBLE

        # The queue is unblocked: the feasible job behind it completes
        await _wait_terminal(feasible)
        assert feasible.status == "completed"
    finally:
        await manager.shutdown()


_IMAGE_DIMS_BODY = """\
emit({"phase": "loading"})
emit({"phase": "loaded"})
with open(os.path.join(spec["output_dir"], "output-0.png"), "wb") as f:
    f.write(b"FAKE-PNG-BYTES")
with open(spec["manifest_path"], "w") as f:
    json.dump({"status": "completed", "outputs": ["output-0.png"],
               "width": 768, "height": 432}, f)
sys.exit(0)
"""


async def test_manifest_dimensions_override_requested_size(tmp_path):
    # With an input image mflux re-derives the canvas from the source
    # aspect, so the requested width/height can misreport the artifact;
    # the worker manifests the ACTUAL dimensions and _conclude adopts them.
    manager, _ = _make_manager(tmp_path, image_body=_IMAGE_DIMS_BODY)
    try:
        job = await manager.submit(_make_image_job("img_dims"))
        await _wait_terminal(job)
        assert job.status == "completed"
        assert job.params["width"] == 768
        assert job.params["height"] == 432
        assert job.to_dict()["size"] == "768x432"
    finally:
        await manager.shutdown()
