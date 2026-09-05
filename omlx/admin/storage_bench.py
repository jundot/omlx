# SPDX-License-Identifier: Apache-2.0
"""Storage-roofline bench jobs.

Background uncached SSD measurement (see
:mod:`omlx.utils.storage_roofline`) exposed to the admin UI. One job at a
time: a storage run saturates the volume's queue, so concurrent runs —
storage or inference benches — would corrupt each other's numbers. The
measurement itself runs in a worker thread (``asyncio.to_thread``); the
event loop only polls the job record.

There is no cancel: a default run lasts ~1-2 min and each phase is short.
Clients should disable Run while ``status == "running"``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_jobs: dict[str, "StorageBenchJob"] = {}
_lock = threading.Lock()


class StorageBenchRequest(BaseModel):
    model_id: Optional[str] = None
    file_gb: float = 2.0
    samples: int = 256
    read_mb: int = 2
    seed: int = 7
    tok_per_cycle: float = 1.0
    verify_mult: float = 2.3
    measured_base_tok_s: Optional[float] = None

    @field_validator("file_gb")
    @classmethod
    def _file_gb(cls, v: float) -> float:
        if v < 0.1 or v > 8.0:
            raise ValueError("file_gb must be between 0.1 and 8")
        return v

    @field_validator("samples")
    @classmethod
    def _samples(cls, v: int) -> int:
        if v < 8 or v > 2048:
            raise ValueError("samples must be between 8 and 2048")
        return v

    @field_validator("read_mb")
    @classmethod
    def _read_mb(cls, v: int) -> int:
        if v < 1 or v > 16:
            raise ValueError("read_mb must be between 1 and 16")
        return v

    @field_validator("tok_per_cycle", "verify_mult")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v


@dataclass
class StorageBenchJob:
    job_id: str
    request: StorageBenchRequest
    target_dir: str
    status: str = "running"  # running | completed | failed
    progress: dict = field(default_factory=lambda: {"phase": "queued", "done": 0, "total": 0})
    report: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


def get_active_job() -> Optional[StorageBenchJob]:
    with _lock:
        for job in _jobs.values():
            if job.status == "running":
                return job
    return None


def get_job(job_id: str) -> Optional[StorageBenchJob]:
    with _lock:
        return _jobs.get(job_id)


def _set_progress(job: StorageBenchJob, phase: str, done: int, total: int) -> None:
    with _lock:
        job.progress = {"phase": phase, "done": done, "total": total}


def create_job(request: StorageBenchRequest, target_dir: str) -> StorageBenchJob:
    job = StorageBenchJob(
        job_id=f"storage_{uuid.uuid4().hex[:12]}",
        request=request,
        target_dir=target_dir,
    )
    with _lock:
        _jobs[job.job_id] = job
    return job


def job_to_response(job: StorageBenchJob) -> dict:
    resp: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
    }
    if job.report is not None:
        resp["report"] = job.report
    return resp


def run_storage_benchmark(job: StorageBenchJob) -> None:
    """Blocking worker body — run via ``asyncio.to_thread``."""
    from omlx.utils.storage_roofline import (
        build_report,
        measure_storage,
        moe_step_profile,
        predict_roofline,
        save_report,
        volume_info_for,
    )

    req = job.request
    try:
        volume = volume_info_for(job.target_dir)
        free_need = int(req.file_gb * 1024**3) + 1024**3
        if 0 < volume.free_bytes < free_need:
            raise RuntimeError(
                f"not enough free space: need {free_need / 1024**3:.1f} GiB, "
                f"have {volume.free_bytes / 1024**3:.1f} GiB"
            )
        meas = measure_storage(
            job.target_dir,
            file_gb=req.file_gb,
            read_mb=req.read_mb,
            samples=req.samples,
            seed=req.seed,
            progress=lambda ph, d, t: _set_progress(job, ph, d, t),
        )
        profile = None
        prediction = None
        if req.model_id:
            prof = moe_step_profile(job.target_dir)
            if prof.supported:
                profile = prof
                prediction = predict_roofline(
                    prof, meas,
                    tok_per_cycle=req.tok_per_cycle,
                    verify_byte_mult=req.verify_mult,
                )
        report = build_report(
            volume, meas, profile, prediction,
            measured_base_tok_s=req.measured_base_tok_s,
        )
        slug = f"{(req.model_id or 'volume').replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
        saved = save_report(report, slug)
        report["path"] = str(saved)
        with _lock:
            job.report = report
            job.status = "completed"
            job.progress = {"phase": "done", "done": 1, "total": 1}
    except Exception as e:
        logger.exception("storage benchmark %s failed", job.job_id)
        with _lock:
            job.status = "failed"
            job.error = str(e)


def latest_measurement() -> Optional[dict]:
    """Most recent completed report's measurement, for /predict.

    In-memory first (this process), then the latest saved report on disk
    (survives server restarts).
    """
    with _lock:
        done = [j for j in _jobs.values() if j.status == "completed" and j.report]
    if done:
        done.sort(key=lambda j: j.report.get("timestamp", ""), reverse=True)  # type: ignore[union-attr]
        return done[0].report["measurement"]  # type: ignore[index]
    try:
        from omlx.utils.storage_roofline import latest_saved_report

        saved = latest_saved_report()
        if saved is not None:
            return saved.get("measurement")
    except Exception:
        pass
    return None
