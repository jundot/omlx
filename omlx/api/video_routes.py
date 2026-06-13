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

from .prompt_extend import extend_video_prompt
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
    params: VideoCreateParams, video_settings: Any, model_settings: Any = None
) -> dict[str, Any]:
    """Apply defaults, dimension rules (W/H multiples of 16, frames 4n+1)
    and UX caps. Raises HTTPException 400 on violations.

    Default resolution order: explicit request value > per-model settings
    (video_default_*) > global video settings."""
    ms = model_settings
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
    if (width is None or height is None) and ms is not None:
        default_size = getattr(ms, "video_default_size", None)
        if default_size:
            try:
                w_str, h_str = default_size.lower().split("x", 1)
                width = width or int(w_str)
                height = height or int(h_str)
            except ValueError:
                pass  # validated at save time; never block the request
    width = width or 480
    height = height or 272
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size must be positive")
    width = _round_up(width, 16)
    height = _round_up(height, 16)

    fps = (
        params.fps
        or (getattr(ms, "video_default_fps", None) if ms else None)
        or int(video_settings.default_fps)
    )
    steps = (
        params.steps
        or (getattr(ms, "video_default_steps", None) if ms else None)
        or int(video_settings.default_steps)
    )

    frames = params.frames
    if frames is None:
        seconds = params.seconds
        if seconds is None and ms is not None:
            seconds = getattr(ms, "video_default_seconds", None)
        if seconds is None:
            seconds = 3.0
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


# I2V conditioning image limits. The magic sniff keeps obviously-wrong
# uploads out of the worker; PIL in the worker venv is the real decoder.
_MAX_REFERENCE_BYTES = 16 * 1024 * 1024
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
)


def _sniff_image(data: bytes) -> str:
    """Return a file suffix for supported image bytes, raise 400 otherwise."""
    if len(data) > _MAX_REFERENCE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input_reference exceeds {_MAX_REFERENCE_BYTES // (1024*1024)}MB"
            ),
        )
    for magic, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise HTTPException(
        status_code=400,
        detail="input_reference must be a PNG, JPEG or WebP image",
    )


def _decode_reference_string(value: str) -> bytes:
    """Decode a JSON-body input_reference: data URL or raw base64."""
    import base64

    payload = value
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "input_reference must be a data URL or base64-encoded image "
                "in JSON bodies (or a multipart file field)"
            ),
        )


async def _parse_create_body(
    request: Request,
) -> tuple[VideoCreateParams, bytes | None, str]:
    """Accept JSON or multipart (openai SDK sends multipart with
    input_reference as a file field; pydantic lax coercion converts the
    all-string form values). Returns (params, reference_bytes, suffix)."""
    content_type = (request.headers.get("content-type") or "").lower()
    ref_bytes: bytes | None = None
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            data = {k: v for k, v in form.items() if isinstance(v, str)}
            upload = form.get("input_reference")
            if upload is not None and not isinstance(upload, str):
                # Bounded read: an unbounded read() would materialize an
                # arbitrarily large upload in RAM before the size check
                # (the sniff rejects anything past the cap either way).
                ref_bytes = await upload.read(_MAX_REFERENCE_BYTES + 1)
            elif isinstance(upload, str) and upload:
                # Symmetric with the JSON body: a string form field holds a
                # data URL / base64 image instead of a file part.
                data.pop("input_reference", None)
                ref_bytes = _decode_reference_string(upload)
        else:
            data = await request.json()
            ref_value = data.pop("input_reference", None) if isinstance(
                data, dict
            ) else None
            if isinstance(ref_value, str) and ref_value:
                ref_bytes = _decode_reference_string(ref_value)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed request body")
    if ref_bytes is not None and not ref_bytes:
        # b"" would slip past the is-None pipeline gates yet submit no
        # image -- an i2v job would then load 40GB of weights only for
        # mflux to refuse generation.
        raise HTTPException(
            status_code=400,
            detail="input_reference is empty (zero-byte upload)",
        )
    suffix = _sniff_image(ref_bytes) if ref_bytes else ""
    try:
        return VideoCreateParams.model_validate(data), ref_bytes, suffix
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/videos")
async def create_video(request: Request):
    manager = _get_video_manager()
    params, ref_bytes, ref_suffix = await _parse_create_body(request)

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

    # Pipeline kind gates the conditioning image: i2v REQUIRES one (mflux
    # hard-refuses without), ti2v accepts one, t2v rejects one. Entries
    # discovered before this field existed default to t2v.
    pipeline = getattr(entry, "video_pipeline", "") or "t2v"
    extend_id = (params.extend_video_id or "").strip() or None
    if ref_bytes is not None and extend_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "input_reference and extend_video_id are mutually exclusive "
                "(extending always conditions on the source's last frame)"
            ),
        )
    if (ref_bytes is not None or extend_id) and pipeline == "t2v":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{params.model}' is text-to-video only and does not "
                "accept an input image; use an image-to-video (i2v) model"
            ),
        )
    if pipeline == "i2v" and ref_bytes is None and not extend_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{params.model}' is image-to-video and requires "
                "input_reference (an image) or extend_video_id"
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
    settings_manager = getattr(_server_state, "settings_manager", None)
    model_settings = (
        settings_manager.get_settings(resolved) if settings_manager else None
    )

    # Extending pins the new segment to the source's frame geometry so the
    # stitch is seamless; user-supplied size/fps are overridden.
    src_job = None
    if extend_id:
        src_job = manager.get(extend_id)
        if src_job is None:
            raise HTTPException(
                status_code=404,
                detail=f"extend_video_id '{extend_id}' not found",
            )
        if src_job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"extend_video_id '{extend_id}' is {src_job.status}; "
                    "only completed videos can be extended"
                ),
            )
        if manager.extend_source_path(src_job) is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "artifact_expired",
                    "message": (
                        f"The artifact for '{extend_id}' was purged by the "
                        "retention policy and cannot be extended"
                    ),
                },
            )
        sp = src_job.params
        if not (sp.get("width") and sp.get("height") and sp.get("fps")):
            raise HTTPException(
                status_code=400,
                detail=f"'{extend_id}' lacks frame geometry; cannot extend",
            )
        params.width = int(sp["width"])
        params.height = int(sp["height"])
        params.fps = int(sp["fps"])
        params.size = None

    normalized = _normalize_params(params, video_settings, model_settings)
    normalized["pipeline"] = pipeline
    if src_job is not None:
        normalized["extend_source_id"] = src_job.id

    # LLM prompt extension (Wan's --use_prompt_extend equivalent): expand a
    # terse prompt into an explicit motion description server-side before
    # dispatch -- the highest-leverage fix for weak instruction following.
    # Runs here (not in the mflux-only worker venv) via the engine pool.
    # Best-effort: any failure falls back to the original prompt.
    extend_model = (getattr(video_settings, "prompt_extend_model", "") or "").strip()
    want_extend = (
        params.prompt_extend if params.prompt_extend is not None
        else bool(extend_model)
    )
    if params.prompt_extend and not extend_model:
        raise HTTPException(
            status_code=400,
            detail=(
                "prompt_extend requested but video.prompt_extend_model is not "
                "configured"
            ),
        )
    if want_extend and extend_model:
        final_prompt, extended = await extend_video_prompt(
            normalized["prompt"],
            model_id=_resolve_model(extend_model),
            engine_pool=pool,
        )
        if extended:
            normalized["original_prompt"] = normalized["prompt"]
            normalized["prompt"] = final_prompt
            normalized["prompt_extended"] = True

    # SeedVR2 upscale stage: per-frame, runs in the worker after generation
    # (and after the stitch, for extends). Weights are a separate download;
    # missing weights are a 400 with the fetch hint, not a 503 -- the rest
    # of the video engine is healthy.
    # Resolution order: explicit request (0 = explicitly off) > per-model
    # video_default_upscale_resolution > off. A model default with missing
    # weights is skipped with a warning rather than failing every request.
    upscale_resolution = params.upscale_resolution
    if upscale_resolution is None and model_settings is not None:
        default_upscale = getattr(
            model_settings, "video_default_upscale_resolution", None
        )
        if default_upscale:
            upscaler_dir = manager.upscaler_dir()
            if (
                upscaler_dir / "transformer" / "model.safetensors.index.json"
            ).is_file():
                upscale_resolution = int(default_upscale)
            else:
                logger.warning(
                    "Model %s sets video_default_upscale_resolution=%s but "
                    "the SeedVR2 weights are missing at %s; skipping",
                    resolved, default_upscale, upscaler_dir,
                )
    if upscale_resolution is not None and int(upscale_resolution) > 0:
        target = int(upscale_resolution)
        max_res = int(getattr(video_settings, "max_upscale_resolution", 2160))
        if not (480 <= target <= max_res):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"upscale_resolution {target} out of range "
                    f"(480..{max_res})"
                ),
            )
        upscaler_dir = manager.upscaler_dir()
        # The index file is how the mflux CLI detects this layout; a bare
        # transformer/ dir from an aborted download must not pass.
        if not (
            upscaler_dir / "transformer" / "model.safetensors.index.json"
        ).is_file():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"SeedVR2 upscaler weights not found at {upscaler_dir}. "
                    "Download AbstractFramework/seedvr2-3b-8bit there or set "
                    "settings.video.upscaler_model_path."
                ),
            )
        normalized["upscale_resolution"] = target
        normalized["upscaler_model_dir"] = str(upscaler_dir)

    from omlx.video.manager import QueueFullError, VideoJob

    job = VideoJob(
        id=f"video_{uuid.uuid4().hex}",
        model_id=resolved,
        model_dir=str(entry.model_path),
        params=normalized,
    )
    try:
        await manager.submit(
            job,
            input_reference=(ref_bytes, ref_suffix) if ref_bytes else None,
        )
    except QueueFullError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _record_video_request(resolved)
    return job.to_dict()


def _get_video_job(manager, video_id: str):
    """Kind guard: the manager is shared with the image engine, so image
    jobs must be invisible on the video wire surface (and vice versa)."""
    job = manager.get(video_id)
    if job is None or getattr(job, "kind", "video") != "video":
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return job


@router.get("/v1/videos")
async def list_videos(
    limit: int = 20, after: str | None = None, order: str = "desc"
):
    manager = _get_video_manager()
    limit = max(1, min(int(limit), 100))
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be asc|desc")
    page, has_more = manager.list_jobs(
        limit=limit, after=after, order=order, kind="video"
    )
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
    return _get_video_job(manager, video_id).to_dict()


@router.get("/v1/videos/{video_id}/content")
async def get_video_content(video_id: str):
    manager = _get_video_manager()
    job = _get_video_job(manager, video_id)
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
    _get_video_job(manager, video_id)
    await manager.delete(video_id)
    return {"id": video_id, "object": "video.deleted", "deleted": True}
