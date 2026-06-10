# SPDX-License-Identifier: Apache-2.0
"""/v1/videos -- OpenAI-style async video generation job API.

Endpoints (design: docs/video-generation-engine-spec.md section 4.3):
- POST   /v1/videos               submit, returns job object immediately
- GET    /v1/videos               cursor-paginated list
- GET    /v1/videos/{id}          poll job object
- GET    /v1/videos/{id}/content  download the mp4 (Range supported)
- DELETE /v1/videos/{id}          cancel/delete job + artifacts

The router is mounted UNCONDITIONALLY at import time (settings are not
initialized yet at that point); all gating happens per-request:
settings.video.enabled off -> 503, manager missing -> 503, worker venv
unusable -> 503 with install guidance.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .video_models import VideoCreateParams

logger = logging.getLogger(__name__)

router = APIRouter()

GB = 1024**3

# Per-request peak predictor, calibrated from the P0 low-RAM measurement
# matrix on m5max (2026-06-11, mlx-gen==0.18.14 lock; recalibrate on every
# lock bump, spec 9.1). Empirical findings: peak scales with PER-FRAME
# spatial latent tokens (W/16 * H/16) and is invariant to frame count and
# step count (measured: 480x272 49f==101f within 0.2GB; 20 vs 40 steps
# byte-identical). Low-RAM mode (the worker default): 510 tok -> 18.83GB,
# 1560 tok -> 21.88GB => BASE 17.5, COEF 0.0029 GB/token. Margin covers
# the worst observed sub-poll transient (5.29GB per 0.5s) padded.
_PEAK_BASE_GB = 17.5
_PEAK_COEF_GB_PER_SPATIAL_TOKEN = 0.0029
_PEAK_MARGIN_GB = 6.0


def _get_video_manager():
    """Active VideoJobManager from server state (test-patchable)."""
    from omlx.server import _server_state

    settings = getattr(_server_state, "global_settings", None)
    video_settings = getattr(settings, "video", None) if settings else None
    if video_settings is None or not video_settings.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Video generation is disabled. Enable settings.video.enabled "
                "and configure the worker venv "
                "(docs/video-generation-engine-spec.md)."
            ),
        )
    manager = getattr(_server_state, "video_job_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503, detail="Video job manager not initialized"
        )
    return manager


def _get_engine_pool():
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _record_video_request(model_id: str) -> None:
    """Record request count without treating anything as tokens."""
    try:
        from omlx.server import get_server_metrics

        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record video metrics for %s: %s", model_id, exc)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _normalize_params(
    params: VideoCreateParams, video_settings: Any
) -> dict[str, Any]:
    """Apply defaults, dimension rules (W/H multiples of 16, frames 4n+1)
    and UX caps. Raises HTTPException 400 on violations."""
    width = params.width
    height = params.height
    if (width is None or height is None) and params.size:
        try:
            w_str, h_str = params.size.lower().split("x", 1)
            width = width or int(w_str)
            height = height or int(h_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid size '{params.size}', expected 'WxH'",
            )
    width = width or 480
    height = height or 272
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size must be positive")
    width = _round_up(width, 16)
    height = _round_up(height, 16)

    fps = params.fps or int(video_settings.default_fps)
    steps = params.steps or int(video_settings.default_steps)

    frames = params.frames
    if frames is None:
        seconds = params.seconds if params.seconds is not None else 3.0
        if seconds <= 0:
            raise HTTPException(status_code=400, detail="seconds must be positive")
        frames = int(round(seconds * fps))
    # Wan requires 4n+1 frames
    frames = max(5, 4 * math.ceil((frames - 1) / 4) + 1)

    if frames > int(video_settings.max_frames):
        raise HTTPException(
            status_code=400,
            detail=f"frames {frames} exceeds max_frames "
                   f"{video_settings.max_frames}",
        )
    if steps > int(video_settings.max_steps):
        raise HTTPException(
            status_code=400,
            detail=f"steps {steps} exceeds max_steps {video_settings.max_steps}",
        )
    if width * height > int(video_settings.max_pixels_per_frame):
        raise HTTPException(
            status_code=400,
            detail=f"{width}x{height} exceeds max_pixels_per_frame "
                   f"{video_settings.max_pixels_per_frame}",
        )

    # Memory bound: predicted peak must fit the lease (spec 4.3/4.4). The
    # static caps above are UX bounds only. Peak is frame-count-invariant
    # (P0 measured), so only per-frame spatial tokens enter the formula.
    spatial_tokens = (width / 16) * (height / 16)
    predicted_gb = _PEAK_BASE_GB + _PEAK_COEF_GB_PER_SPATIAL_TOKEN * spatial_tokens
    lease_gb = float(video_settings.memory_lease_gb)
    if predicted_gb + _PEAK_MARGIN_GB > lease_gb:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Predicted memory peak {predicted_gb:.1f}GB (+{_PEAK_MARGIN_GB}GB "
                f"margin) exceeds video.memory_lease_gb {lease_gb:.0f}GB. "
                "Reduce resolution/frames or raise the lease."
            ),
        )

    seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
    normalized: dict[str, Any] = {
        "prompt": params.prompt,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "fps": fps,
        "seed": int(seed),
        "seconds": round(frames / fps, 2),
    }
    if params.negative_prompt:
        normalized["negative_prompt"] = params.negative_prompt
    if params.guidance is not None:
        normalized["guidance"] = float(params.guidance)
    if params.guidance_2 is not None:
        normalized["guidance_2"] = float(params.guidance_2)
    return normalized


async def _parse_create_body(request: Request) -> VideoCreateParams:
    """Accept JSON or multipart (openai SDK sends multipart, all-string
    fields; pydantic lax coercion converts them)."""
    content_type = (request.headers.get("content-type") or "").lower()
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            data = {k: v for k, v in form.items() if isinstance(v, str)}
        else:
            data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed request body")
    try:
        return VideoCreateParams.model_validate(data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/videos")
async def create_video(request: Request):
    manager = _get_video_manager()
    params = await _parse_create_body(request)

    pool = _get_engine_pool()
    resolved = _resolve_model(params.model)
    entry = pool.get_entry(resolved) if hasattr(pool, "get_entry") else None
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{params.model}' not found",
        )
    if getattr(entry, "model_type", "") != "video":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{params.model}' is not a video generation model "
                f"(model_type={getattr(entry, 'model_type', '?')})"
            ),
        )

    ok, reason = manager.guard_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)
    venv_ok, venv_reason = await manager.probe_worker_venv()
    if not venv_ok:
        raise HTTPException(status_code=503, detail=venv_reason)

    from omlx.server import _server_state

    video_settings = _server_state.global_settings.video
    normalized = _normalize_params(params, video_settings)

    from omlx.video.manager import QueueFullError, VideoJob

    job = VideoJob(
        id=f"video_{uuid.uuid4().hex}",
        model_id=resolved,
        model_dir=str(entry.model_path),
        params=normalized,
    )
    try:
        await manager.submit(job)
    except QueueFullError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _record_video_request(resolved)
    return job.to_dict()


@router.get("/v1/videos")
async def list_videos(
    limit: int = 20, after: str | None = None, order: str = "desc"
):
    manager = _get_video_manager()
    limit = max(1, min(int(limit), 100))
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be asc|desc")
    page, has_more = manager.list_jobs(limit=limit, after=after, order=order)
    data = [j.to_dict() for j in page]
    return {
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


@router.get("/v1/videos/{video_id}")
async def get_video(video_id: str):
    manager = _get_video_manager()
    job = manager.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return job.to_dict()


@router.get("/v1/videos/{video_id}/content")
async def get_video_content(video_id: str):
    manager = _get_video_manager()
    job = manager.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Video '{video_id}' is {job.status}, content not available",
        )
    if not job.artifact_path or not Path(job.artifact_path).exists():
        # Artifact purged by retention (spec 4.3): record outlives the blob.
        raise HTTPException(
            status_code=404,
            detail={
                "code": "artifact_expired",
                "message": (
                    f"The artifact for '{video_id}' was purged by the "
                    "retention policy"
                ),
                "expires_at": int(job.expires_at) if job.expires_at else None,
            },
        )
    return FileResponse(
        job.artifact_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


@router.delete("/v1/videos/{video_id}")
async def delete_video(video_id: str):
    manager = _get_video_manager()
    deleted = await manager.delete(video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return {"id": video_id, "object": "video.deleted", "deleted": True}
