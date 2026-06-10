# SPDX-License-Identifier: Apache-2.0
"""VideoJobManager: async job queue for subprocess video generation.

Job shape follows the admin downloader/OQ patterns (task dict + status enum
+ cooperative cancel) with persistence (one JSON per job, atomic write) and
a memory lease held against the ProcessMemoryEnforcer for the duration of
each run. Design: docs/video-generation-engine-spec.md sections 4.2/4.4.

Wire status is exactly the OpenAI four-value enum: queued | in_progress |
completed | failed. Cancellation is not a wire state -- DELETE kills the
worker and removes the record entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..utils.proc_memory import get_phys_footprint

logger = logging.getLogger(__name__)

GB = 1024**3

# Stable error codes (spec 4.2). The worker failure manifest uses the same
# {code, message, detail?} schema and is passed through.
ERR_WORKER_CRASHED = "worker_crashed"
ERR_WORKER_STALLED = "worker_stalled"
ERR_JOB_TIMEOUT = "job_timeout"
ERR_LEASE_EXCEEDED = "memory_lease_exceeded"
ERR_MONITOR_FAILED = "monitor_failed"
ERR_SERVER_RESTARTED = "server_restarted"
ERR_OUTPUT_INVALID = "output_invalid"
ERR_EXTEND_SOURCE_MISSING = "extend_source_missing"

_WATCHDOG_INTERVAL_S = 2.0
_ADMISSION_RECHECK_S = 5.0
_SIGTERM_GRACE_S = 5.0


@dataclass
class VideoJob:
    """One video generation job."""

    id: str
    model_id: str
    model_dir: str
    params: dict[str, Any]  # prompt, width, height, frames, steps, fps, seed, ...
    status: str = "queued"  # queued | in_progress | completed | failed
    progress: int = 0  # 0-100
    phase: str = ""
    error: Optional[dict[str, str]] = None  # {code, message} when failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    expires_at: Optional[float] = None  # Set when the artifact blob is purged
    artifact_path: Optional[str] = None
    wall_seconds: Optional[float] = None
    peak_memory_gb: Optional[float] = None  # Worker lifetime-max, for records

    def to_dict(self) -> dict[str, Any]:
        """Wire shape: OpenAI video object fields + fmlx extensions."""
        return {
            "id": self.id,
            "object": "video",
            "model": self.model_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": int(self.created_at),
            "completed_at": int(self.completed_at) if self.completed_at else None,
            "expires_at": int(self.expires_at) if self.expires_at else None,
            "error": self.error,
            "seconds": str(self.params.get("seconds", "")),
            "size": f"{self.params.get('width')}x{self.params.get('height')}",
            # fmlx extensions
            "phase": self.phase,
            "prompt": self.params.get("prompt", ""),
            "frames": self.params.get("frames"),
            "fps": self.params.get("fps"),
            "steps": self.params.get("steps"),
            "seed": self.params.get("seed"),
            "wall_seconds": self.wall_seconds,
            "pipeline": self.params.get("pipeline") or None,
            "extend_source_id": self.params.get("extend_source_id"),
            "upscale_resolution": self.params.get("upscale_resolution"),
            "has_input_reference": bool(self.params.get("image_path")),
        }

    def to_persist(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "params": self.params,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "artifact_path": self.artifact_path,
            "wall_seconds": self.wall_seconds,
            "peak_memory_gb": self.peak_memory_gb,
        }

    @classmethod
    def from_persist(cls, data: dict[str, Any]) -> "VideoJob":
        return cls(
            id=str(data["id"]),
            model_id=str(data.get("model_id", "")),
            model_dir=str(data.get("model_dir", "")),
            params=dict(data.get("params") or {}),
            status=str(data.get("status", "failed")),
            progress=int(data.get("progress") or 0),
            phase=str(data.get("phase", "") or ""),
            error=data.get("error"),
            created_at=float(data.get("created_at") or time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            expires_at=data.get("expires_at"),
            artifact_path=data.get("artifact_path"),
            wall_seconds=data.get("wall_seconds"),
            peak_memory_gb=data.get("peak_memory_gb"),
        )


class VideoJobManager:
    """Serializes video generation jobs against a memory lease.

    Constructed in server lifespan AFTER the ProcessMemoryEnforcer so the
    enforcer can be constructor-injected (testability seam, spec 4.2).
    """

    def __init__(
        self,
        *,
        settings: Any,  # VideoSettings
        base_path: Path,
        enforcer: Any | None,  # ProcessMemoryEnforcer | None
        worker_script: Path | None = None,
    ):
        self._settings = settings
        self._base_path = Path(base_path)
        self._enforcer = enforcer
        self._worker_script = worker_script or (
            Path(__file__).parent / "worker.py"
        )
        self._jobs: dict[str, VideoJob] = {}
        self._queue: list[str] = []  # FIFO of queued job ids
        self._dispatcher: asyncio.Task | None = None
        self._current_proc: asyncio.subprocess.Process | None = None
        self._current_job_id: str | None = None
        self._wake = asyncio.Event()
        self._shutdown = False
        self._venv_probe_result: tuple[bool, str] | None = None

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._replay_persisted()

    # -- paths ---------------------------------------------------------------

    @property
    def jobs_dir(self) -> Path:
        return self._base_path / "video-jobs"

    @property
    def artifacts_dir(self) -> Path:
        return self._base_path / "video-artifacts"

    def worker_python(self) -> Path:
        return self._settings.get_worker_python(self._base_path)

    def upscaler_dir(self) -> Path:
        return self._settings.get_upscaler_dir(self._base_path)

    def extend_source_path(self, job: VideoJob) -> Path | None:
        """Resolve the mp4 a continuation should grow from.

        Upscaled jobs keep their pre-upscale stitchable video at
        output_raw.mp4 -- extending from the upscaled frames would force
        generating the new segment at the upscaled resolution (the 720p+
        multi-hour trap). Falls back to the artifact itself.
        """
        raw = self.artifacts_dir / job.id / "output_raw.mp4"
        if raw.exists():
            return raw
        if job.artifact_path and Path(job.artifact_path).exists():
            return Path(job.artifact_path)
        return None

    # -- persistence ---------------------------------------------------------

    def _persist(self, job: VideoJob) -> None:
        path = self.jobs_dir / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(job.to_persist(), f, indent=1)
            os.replace(tmp, path)
        except OSError as e:
            logger.error(f"[video] failed to persist job {job.id}: {e}")

    def _replay_persisted(self) -> None:
        """Reload job records at startup; in-flight jobs become failed."""
        for path in sorted(self.jobs_dir.glob("video_*.json")):
            try:
                with open(path) as f:
                    job = VideoJob.from_persist(json.load(f))
            except Exception as e:
                logger.warning(f"[video] skipping unreadable job file {path}: {e}")
                continue
            if job.status in ("queued", "in_progress"):
                job.status = "failed"
                job.error = {
                    "code": ERR_SERVER_RESTARTED,
                    "message": "Server restarted while the job was active",
                }
                job.completed_at = time.time()
                self._persist(job)
            self._jobs[job.id] = job

    # -- venv probe ----------------------------------------------------------

    async def probe_worker_venv(self, force: bool = False) -> tuple[bool, str]:
        """Check the worker venv is usable (cached after first success)."""
        if self._venv_probe_result and self._venv_probe_result[0] and not force:
            return self._venv_probe_result
        py = self.worker_python()
        install_hint = (
            "Install with: uv venv -p 3.12 {base}/venvs/video && "
            "uv pip sync --python {base}/venvs/video/bin/python "
            "omlx/video/requirements.lock".format(base=self._base_path)
        )
        if not py.exists():
            self._venv_probe_result = (
                False, f"Video worker python not found at {py}. {install_hint}"
            )
            return self._venv_probe_result
        try:
            proc = await asyncio.create_subprocess_exec(
                str(py), "-c", "import mflux",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                self._venv_probe_result = (
                    False,
                    f"Video worker venv at {py} cannot import mflux: "
                    f"{(stderr or b'').decode()[-300:]}. {install_hint}",
                )
                return self._venv_probe_result
        except Exception as e:
            self._venv_probe_result = (False, f"Video worker venv probe failed: {e}")
            return self._venv_probe_result
        self._venv_probe_result = (True, "")
        return self._venv_probe_result

    # -- memory admission ----------------------------------------------------

    def _lease_bytes(self) -> int:
        return int(float(self._settings.memory_lease_gb) * GB)

    def guard_available(self) -> tuple[bool, str]:
        """Submission-time check: refuse jobs without a live memory guard."""
        enf = self._enforcer
        if enf is None or not getattr(enf, "is_running", False):
            return False, (
                "Video jobs require the process memory guard, which is not "
                "running on this server"
            )
        if enf.get_final_ceiling() <= 0:
            return False, (
                "Video jobs require memory.prefill_memory_guard to be "
                "enabled (the guard is currently disabled)"
            )
        return True, ""

    def _memory_admission(self) -> tuple[bool, str]:
        """Dispatch-time predicate (spec 4.4): the lease must land with the
        system already at ok pressure and resident load below both the
        post-lease soft watermark and the post-lease prefill-gate trip."""
        enf = self._enforcer
        ok, reason = self.guard_available()
        if not ok:
            return False, reason
        assert enf is not None  # guard_available() established this
        ceiling = enf.get_final_ceiling()
        lease = self._lease_bytes()
        post = ceiling - lease
        if post <= 0:
            return False, (
                f"memory lease {lease / GB:.0f}GB does not fit under the "
                f"ceiling {ceiling / GB:.1f}GB"
            )
        soft_ratio = float(getattr(enf, "_soft_threshold", 0.85) or 0.85)
        margin = int(getattr(enf, "_prefill_transient_margin_bytes", 12 * GB)
                     or 12 * GB)
        budget = min(int(post * soft_ratio), post - margin)
        if budget <= 0:
            budget = int(post * soft_ratio)
        peak = int(enf.recent_peak_bytes() or 0)
        if peak > budget:
            return False, (
                f"waiting for memory: resident usage {peak / GB:.1f}GB above "
                f"post-lease budget {budget / GB:.1f}GB "
                f"(ceiling {ceiling / GB:.1f}GB, lease {lease / GB:.0f}GB)"
            )
        return True, ""

    # -- public API ----------------------------------------------------------

    def get(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def active_model_id(self) -> str | None:
        """Model id of the currently running job (None when idle).

        Consumed by the model-status endpoints so a video model shows as
        active while its worker is generating, even though it is never
        pool-loaded.
        """
        if self._current_job_id:
            job = self._jobs.get(self._current_job_id)
            if job is not None:
                return job.model_id
        return None

    def current_worker_memory_bytes(self) -> int:
        """Live phys footprint of the running worker (0 when idle).

        The Active Models card sums pool-loaded engine memory, which a
        video worker never appears in -- without this the card shows
        0 GB used while a generation occupies tens of GB.
        """
        proc = self._current_proc
        if proc is None or proc.returncode is not None:
            return 0
        try:
            return max(0, int(get_phys_footprint(proc.pid)))
        except Exception:
            return 0

    def list_jobs(
        self, limit: int = 20, after: str | None = None, order: str = "desc"
    ) -> tuple[list[VideoJob], bool]:
        jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=(order != "asc"),
        )
        if after:
            ids = [j.id for j in jobs]
            try:
                start = ids.index(after) + 1
                jobs = jobs[start:]
            except ValueError:
                pass
        page = jobs[:limit]
        return page, len(jobs) > limit

    def queue_depth(self) -> int:
        return len(self._queue)

    async def submit(
        self,
        job: VideoJob,
        input_reference: tuple[bytes, str] | None = None,
    ) -> VideoJob:
        """Accept a job into the queue (caller validates params + caps).

        input_reference is (bytes, suffix) of the I2V conditioning image;
        it lands in the job's blob dir so the worker reads it from disk.
        """
        if len(self._queue) >= int(self._settings.max_queued_jobs):
            raise QueueFullError(
                f"Video queue is full ({len(self._queue)}/"
                f"{self._settings.max_queued_jobs})"
            )
        if input_reference is not None:
            data, suffix = input_reference
            blob_dir = self.artifacts_dir / job.id
            blob_dir.mkdir(parents=True, exist_ok=True)
            ref_path = blob_dir / f"input_reference{suffix or '.png'}"
            ref_path.write_bytes(data)
            job.params["image_path"] = str(ref_path)
        self._jobs[job.id] = job
        self._queue.append(job.id)
        self._persist(job)
        self._ensure_dispatcher()
        self._wake.set()
        logger.info(f"[video] queued {job.id} ({job.model_id})")
        return job

    async def delete(self, job_id: str) -> bool:
        """DELETE semantics (spec 4.3): kill if running, drop record+blobs."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job_id in self._queue:
            self._queue.remove(job_id)
        if self._current_job_id == job_id and self._current_proc is not None:
            await self._terminate_proc(self._current_proc)
        self._jobs.pop(job_id, None)
        try:
            (self.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
        except OSError:
            pass
        blob_dir = self.artifacts_dir / job_id
        if blob_dir.exists():
            shutil.rmtree(blob_dir, ignore_errors=True)
        logger.info(f"[video] deleted {job_id}")
        return True

    async def shutdown(self) -> None:
        self._shutdown = True
        self._wake.set()
        if self._current_proc is not None:
            await self._terminate_proc(self._current_proc)
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except (asyncio.CancelledError, Exception):
                pass

    # -- dispatcher ----------------------------------------------------------

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def _dispatch_loop(self) -> None:
        while not self._shutdown:
            if not self._queue:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=60)
                except asyncio.TimeoutError:
                    continue
                continue
            job_id = self._queue[0]
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                self._queue.pop(0)
                continue
            ok, reason = self._memory_admission()
            if not ok:
                if job.phase != reason:
                    job.phase = reason
                    self._persist(job)
                await asyncio.sleep(_ADMISSION_RECHECK_S)
                continue
            self._queue.pop(0)
            try:
                await self._run_job(job)
            except Exception as e:  # noqa: BLE001 -- dispatcher must survive
                logger.exception(f"[video] job {job.id} runner crashed: {e}")
                if job.status == "in_progress":
                    self._finish(job, "failed", ERR_WORKER_CRASHED, str(e))

    # -- job execution -------------------------------------------------------

    def _finish(
        self, job: VideoJob, status: str, code: str | None = None,
        message: str | None = None,
    ) -> None:
        job.status = status
        job.completed_at = time.time()
        if job.started_at:
            job.wall_seconds = round(job.completed_at - job.started_at, 1)
        if status == "failed":
            job.error = {"code": code or ERR_WORKER_CRASHED,
                         "message": message or ""}
        else:
            job.progress = 100
            job.phase = "done"
        self._persist(job)
        logger.info(
            f"[video] {job.id} -> {status}"
            + (f" ({code}: {message})" if code else "")
        )

    async def _terminate_proc(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=_SIGTERM_GRACE_S)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    def _worker_env(self) -> dict[str, str]:
        """Explicit whitelist -- never inherit the full server env."""
        env = {}
        for key in ("PATH", "HOME", "TMPDIR", "USER", "LANG"):
            if key in os.environ:
                env[key] = os.environ[key]
        # The worker loads everything from the local model dir; forbid
        # accidental network fetches.
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        return env

    async def _run_job(self, job: VideoJob) -> None:
        enf = self._enforcer
        lease = self._lease_bytes()
        blob_dir = self.artifacts_dir / job.id
        blob_dir.mkdir(parents=True, exist_ok=True)
        output_path = blob_dir / "output.mp4"
        manifest_path = blob_dir / "manifest.json"
        spec_path = blob_dir / "spec.json"

        spec = dict(job.params)
        spec.update(
            model_dir=job.model_dir,
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            lease_bytes=lease,
        )

        # Extend sources resolve at dispatch, not submit -- the source may
        # be deleted or retention-purged while this job queues. Retention
        # skips sources referenced by pending jobs, but DELETE does not.
        src_id = spec.get("extend_source_id")
        if src_id:
            src_job = self._jobs.get(str(src_id))
            src_path = self.extend_source_path(src_job) if src_job else None
            if src_path is None:
                self._finish(
                    job, "failed", ERR_EXTEND_SOURCE_MISSING,
                    f"source video '{src_id}' no longer has an artifact",
                )
                return
            spec["extend_from"] = str(src_path)

        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=1)

        job.status = "in_progress"
        job.started_at = time.time()
        job.phase = "starting"
        self._persist(job)

        if enf is not None:
            enf.acquire_video_lease(lease)
        try:
            proc = await asyncio.create_subprocess_exec(
                str(self.worker_python()), "-I", str(self._worker_script),
                "--spec", str(spec_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._worker_env(),
            )
            self._current_proc = proc
            self._current_job_id = job.id
            if enf is not None:
                enf.set_video_worker_pid(proc.pid)

            kill_reason: list[tuple[str, str]] = []
            last_line_at = time.time()

            async def watchdog() -> None:
                zero_reads = 0
                while proc.returncode is None:
                    await asyncio.sleep(_WATCHDOG_INTERVAL_S)
                    if proc.returncode is not None:
                        return
                    now = time.time()
                    # Per-run timeout (clock starts at spawn, spec 4.2)
                    if (now - (job.started_at or now)
                            > int(self._settings.job_timeout_seconds)):
                        kill_reason.append((
                            ERR_JOB_TIMEOUT,
                            f"exceeded job_timeout_seconds="
                            f"{self._settings.job_timeout_seconds}",
                        ))
                        await self._terminate_proc(proc)
                        return
                    # Stall detection: no JSONL line for too long
                    if (now - last_line_at
                            > int(self._settings.progress_stall_timeout_seconds)):
                        kill_reason.append((
                            ERR_WORKER_STALLED,
                            "no progress output for "
                            f"{int(now - last_line_at)}s",
                        ))
                        await self._terminate_proc(proc)
                        return
                    # Footprint vs lease (secondary cleanup; layer 1 is the
                    # worker's own Metal wired limit)
                    footprint = get_phys_footprint(proc.pid)
                    if footprint <= 0:
                        zero_reads += 1
                        if zero_reads >= 3:
                            kill_reason.append((
                                ERR_MONITOR_FAILED,
                                "cannot read worker memory footprint",
                            ))
                            await self._terminate_proc(proc)
                            return
                        continue
                    zero_reads = 0
                    if footprint > lease:
                        kill_reason.append((
                            ERR_LEASE_EXCEEDED,
                            f"worker footprint {footprint / GB:.1f}GB "
                            f"exceeded lease {lease / GB:.1f}GB",
                        ))
                        try:
                            proc.kill()  # immediate, no grace
                        except ProcessLookupError:
                            pass
                        return

            wd_task = asyncio.create_task(watchdog())
            try:
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    last_line_at = time.time()
                    try:
                        ev = json.loads(raw.decode().strip())
                    except (ValueError, UnicodeDecodeError):
                        continue
                    self._apply_progress(job, ev)
                await proc.wait()
            finally:
                wd_task.cancel()
                try:
                    await wd_task
                except (asyncio.CancelledError, Exception):
                    pass

            self._conclude(job, proc.returncode or 0, kill_reason,
                           output_path, manifest_path)
        finally:
            self._current_proc = None
            self._current_job_id = None
            if enf is not None:
                enf.set_video_worker_pid(None)
                enf.release_video_lease()
            self._retention_sweep()

    def _apply_progress(self, job: VideoJob, ev: dict[str, Any]) -> None:
        phase = str(ev.get("phase", "") or "")
        if phase:
            job.phase = phase
        total = int(ev.get("total_steps") or 0)
        step = int(ev.get("step") or 0)
        if phase == "loading":
            job.progress = max(job.progress, 2)
        elif phase == "loaded":
            job.progress = max(job.progress, 5)
        elif phase == "upscaling":
            # Post-generation stage: denoise already drove progress to ~95
            if total > 0:
                job.progress = max(job.progress, 95 + int(2 * step / total))
            else:
                job.progress = max(job.progress, 95)
        elif phase == "stitching":
            job.progress = max(job.progress, 95)
        elif total > 0 and step > 0:
            job.progress = max(job.progress, 5 + int(90 * step / total))
        elif phase == "saving":
            job.progress = max(job.progress, 97)
        # Persist sparsely: every phase change and ~every 5 progress points
        if phase in (
            "loading", "loaded", "stitching", "saving", "done", "failed"
        ) or (job.progress % 5 == 0):
            self._persist(job)

    def _conclude(
        self, job: VideoJob, returncode: int,
        kill_reason: list[tuple[str, str]],
        output_path: Path, manifest_path: Path,
    ) -> None:
        # Job may have been deleted mid-run (DELETE endpoint kills the proc)
        if job.id not in self._jobs:
            return
        if kill_reason:
            code, message = kill_reason[0]
            self._finish(job, "failed", code, message)
            return
        manifest: dict[str, Any] = {}
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            pass
        if returncode != 0:
            self._finish(
                job, "failed",
                str(manifest.get("code", ERR_WORKER_CRASHED)),
                str(manifest.get("message",
                                 f"worker exited with code {returncode}")),
            )
            return
        if not output_path.exists() or output_path.stat().st_size == 0:
            self._finish(job, "failed", ERR_OUTPUT_INVALID,
                         "worker exited 0 but produced no output file")
            return
        job.artifact_path = str(output_path)
        if isinstance(manifest.get("lifetime_max_phys_gb"), (int, float)):
            job.peak_memory_gb = float(manifest["lifetime_max_phys_gb"])
        self._finish(job, "completed")

    # -- retention -----------------------------------------------------------

    def _retention_sweep(self) -> None:
        """LRU-purge artifact blobs beyond count/bytes caps. Records stay;
        expires_at marks the purge time (spec 4.2)."""
        max_count = int(self._settings.artifacts_max_count)
        max_bytes = int(float(self._settings.artifacts_max_gb) * GB)
        # Artifacts referenced by pending extend jobs must survive until
        # those jobs dispatch (they read the source's last frame + frames).
        protected = {
            str(j.params.get("extend_source_id"))
            for j in self._jobs.values()
            if j.status in ("queued", "in_progress")
            and j.params.get("extend_source_id")
        }
        holders = [
            j for j in self._jobs.values()
            if j.artifact_path and j.expires_at is None
            and j.id not in protected
            and Path(j.artifact_path).exists()
        ]
        holders.sort(key=lambda j: j.completed_at or j.created_at)  # oldest first
        total = sum(
            Path(j.artifact_path).stat().st_size for j in holders  # type: ignore[arg-type]
        )
        while holders and (len(holders) > max_count or total > max_bytes):
            victim = holders.pop(0)
            blob_dir = self.artifacts_dir / victim.id
            try:
                size = Path(victim.artifact_path).stat().st_size  # type: ignore[arg-type]
            except OSError:
                size = 0
            shutil.rmtree(blob_dir, ignore_errors=True)
            victim.artifact_path = None
            victim.expires_at = time.time()
            self._persist(victim)
            total -= size
            logger.info(f"[video] purged artifact of {victim.id} (retention)")


class QueueFullError(Exception):
    """Submission rejected: queue depth cap reached (HTTP 503)."""
