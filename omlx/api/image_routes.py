# SPDX-License-Identifier: Apache-2.0
"""/v1/images -- OpenAI-style image generation API (sync or job-based).

Endpoints:
- POST   /v1/images               generate; sync=true (default) blocks and
                                  returns OpenAI image-API shape (b64_json),
                                  sync=false returns a job object to poll
- POST   /v1/images/generations   OpenAI SDK compat alias (always sync)
- GET    /v1/images               cursor-paginated job list
- GET    /v1/images/{id}          poll job object
- GET    /v1/images/{id}/content  download one PNG (?index=N for n>1)
- DELETE /v1/images/{id}          cancel/delete job + artifacts

The router is mounted UNCONDITIONALLY at import time (settings are not
initialized yet at that point); all gating happens per-request:
settings.image.enabled off -> 503, manager missing -> 503, worker venv
unusable -> 503 with install guidance. Jobs run on the shared
MediaJobManager so image and video generations serialize against the
single enforcer memory lease.
"""

from __future__ import annotations

import base64
import logging
import random
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .image_models import ImageCreateParams

logger = logging.getLogger(__name__)

router = APIRouter()

GB = 1024**3

_MAX_INPUT_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_INPUT_IMAGES = 4
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
)

# Pre-calibration lease defaults by mflux alias (GB). Conservative until
# the m5max measurement pass lands real peaks (then recalibrate here and
# in docs, mirroring the video peak-predictor discipline). 4bit weights:
# z-image-turbo 5.9GB, qwen-image 17.4GB, qwen-image-edit-2511 18.3GB --
# activations ride on top, tiled VAE keeps decode flat.
_DEFAULT_LEASE_GB = {
    "z-image-turbo": 12.0,
    "z-image": 12.0,
    "qwen-image": 30.0,
    "qwen-image-edit": 32.0,
    "qwen-image-edit-2509": 32.0,
    "qwen-image-edit-2511": 32.0,
}
_FALLBACK_LEASE_GB = 32.0

# Per-alias step defaults (mflux signature defaults are unsafe: 4 steps
# everywhere except edit). Turbo is distilled for ~8 steps; the Qwen
# models want their full schedule for quality.
_DEFAULT_STEPS = {
    "z-image-turbo": 9,
    "z-image": 28,
    "qwen-image": 40,
    "qwen-image-edit": 40,
    "qwen-image-edit-2509": 40,
    "qwen-image-edit-2511": 40,
}
_FALLBACK_STEPS = 30

# Multi-reference editing needs an edit-plus ModelConfig; the python API
# does not guard this (the CLI does) -- wrong conditioning otherwise.
_MULTI_REF_ALIASES = {"qwen-image-edit-2509", "qwen-image-edit-2511"}


def _get_image_manager():
    """Active MediaJobManager from server state (test-patchable)."""
    from omlx.server import _server_state

    settings = getattr(_server_state, "global_settings", None)
    image_settings = getattr(settings, "image", None) if settings else None
    if image_settings is None or not image_settings.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Image generation is disabled. Enable settings.image.enabled "
                "and configure the worker venv (shared with the video "
                "engine, docs/video-generation-engine-spec.md section 4.5)."
            ),
        )
    manager = getattr(_server_state, "video_job_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503, detail="Media job manager not initialized"
        )
    return manager


def _image_settings():
    from omlx.server import _server_state

    settings = getattr(_server_state, "global_settings", None)
    return getattr(settings, "image", None) if settings else None


def _get_engine_pool():
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _record_image_request(model_id: str) -> None:
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
        logger.warning("Failed to record image metrics for %s: %s", model_id, exc)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _sniff_image(data: bytes) -> str:
    """Return a file suffix for supported image bytes, raise 400 otherwise."""
    if len(data) > _MAX_INPUT_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input image exceeds {_MAX_INPUT_IMAGE_BYTES // (1024*1024)}MB"
            ),
        )
    for magic, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise HTTPException(
        status_code=400,
        detail="input images must be PNG, JPEG or WebP",
    )


def _decode_image_string(value: str) -> bytes:
    """Decode a JSON-body input image: data URL or raw base64."""
    payload = value
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "image must be a data URL or base64-encoded image in JSON "
                "bodies (or multipart file fields)"
            ),
        )


async def _parse_create_body(
    request: Request,
) -> tuple[ImageCreateParams, list[tuple[bytes, str]]]:
    """Accept JSON or multipart. Returns (params, [(bytes, suffix), ...])."""
    content_type = (request.headers.get("content-type") or "").lower()
    images: list[bytes] = []
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            data: dict[str, Any] = {
                k: v for k, v in form.items() if isinstance(v, str)
            }
            data.pop("image", None)
            for upload in form.getlist("image"):
                if isinstance(upload, str):
                    if upload:
                        images.append(_decode_image_string(upload))
                else:
                    # Bounded read: an unbounded read() would materialize an
                    # arbitrarily large upload in RAM before the size check.
                    images.append(await upload.read(_MAX_INPUT_IMAGE_BYTES + 1))
        else:
            data = await request.json()
            if not isinstance(data, dict):
                raise HTTPException(
                    status_code=400, detail="Malformed request body"
                )
            value = data.pop("image", None)
            values = value if isinstance(value, list) else (
                [value] if value else []
            )
            for item in values:
                if isinstance(item, str) and item:
                    images.append(_decode_image_string(item))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed request body")
    if len(images) > _MAX_INPUT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"At most {_MAX_INPUT_IMAGES} input images per request",
        )
    for data_bytes in images:
        if not data_bytes:
            # b"" slips past truthiness gates upstream but would load GBs
            # of weights only for mflux to refuse generation.
            raise HTTPException(
                status_code=400, detail="image is empty (zero-byte upload)"
            )
    pairs = [(b, _sniff_image(b)) for b in images]
    try:
        return ImageCreateParams.model_validate(data), pairs
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _pick_default_model(pool, want_edit: bool) -> str | None:
    """Auto-pick an image model when the request omits one.

    With input images prefer an edit model; otherwise prefer z-image-turbo
    (the fast default), then any t2i, then anything image-typed.
    Deterministic: candidates scan in sorted id order.
    """
    candidates = []
    for model_id in sorted(pool.get_model_ids()):
        entry = pool.get_entry(model_id)
        if getattr(entry, "model_type", "") != "image":
            continue
        candidates.append(
            (
                model_id,
                getattr(entry, "image_pipeline", ""),
                getattr(entry, "image_alias", ""),
            )
        )
    if not candidates:
        return None
    if want_edit:
        for model_id, pipeline, _alias in candidates:
            if pipeline == "edit":
                return model_id
        return None  # Images attached but no edit model: caller 400s
    for model_id, pipeline, alias in candidates:
        if alias == "z-image-turbo":
            return model_id
    for model_id, pipeline, _alias in candidates:
        if pipeline == "t2i":
            return model_id
    # Only edit models installed: auto-picking one for a text-only request
    # guarantees a confusing "requires an input image" 400 -- 404 instead.
    return None


def _normalize_params(
    params: ImageCreateParams,
    entry: Any,
    image_settings: Any,
    model_settings: Any,
    input_count: int,
) -> dict[str, Any]:
    """Resolution order: explicit request > per-model settings > global
    image settings > per-alias defaults. UX caps 400 past the static
    bounds; the memory bound is the per-alias lease."""
    alias = getattr(entry, "image_alias", "") or ""
    pipeline = getattr(entry, "image_pipeline", "") or "t2i"

    # n/steps distinguish None (use defaults) from explicit 0 (a 400):
    # silent coercion of 0 to a default would mask client bugs.
    n = 1 if params.n is None else int(params.n)
    max_n = int(getattr(image_settings, "max_n", 4))
    if n < 1 or n > max_n:
        raise HTTPException(
            status_code=400, detail=f"n must be between 1 and {max_n}"
        )

    steps = params.steps
    if steps is None:
        steps = getattr(model_settings, "image_default_steps", None) or None
        if steps is None:
            steps = int(getattr(image_settings, "default_steps", 0) or 0) or None
        if steps is None:
            steps = _DEFAULT_STEPS.get(alias, _FALLBACK_STEPS)
    steps = int(steps)
    max_steps = int(getattr(image_settings, "max_steps", 60))
    if steps < 1 or steps > max_steps:
        raise HTTPException(
            status_code=400, detail=f"steps must be between 1 and {max_steps}"
        )

    # Size. Edit models default to the source aspect at ~1MP inside mflux
    # (recommended: forced square output degrades qwen-edit quality), so
    # they only get explicit dimensions. t2i always resolves dimensions.
    width = params.width
    height = params.height
    size = (params.size or "").strip().lower()
    if width is None and height is None and size and size != "auto":
        try:
            w_str, h_str = size.split("x", 1)
            width, height = int(w_str), int(h_str)
        except ValueError:
            raise HTTPException(
                status_code=400, detail='size must be "WxH" (e.g. "1024x1024")'
            )
    if (width is None) != (height is None):
        raise HTTPException(
            status_code=400, detail="width and height must be given together"
        )
    if width is None and pipeline != "edit":
        default_size = (
            getattr(model_settings, "image_default_size", None)
            or getattr(image_settings, "default_size", "1024x1024")
        )
        try:
            w_str, h_str = str(default_size).lower().split("x", 1)
            width, height = int(w_str), int(h_str)
        except ValueError:
            width, height = 1024, 1024
    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail="width and height must be positive"
            )
        width = _round_up(max(64, width), 16)
        height = _round_up(max(64, height), 16)
        max_pixels = int(getattr(image_settings, "max_pixels", 2048 * 2048))
        if width * height > max_pixels:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{width}x{height} exceeds max_pixels={max_pixels} "
                    "(settings.image.max_pixels)"
                ),
            )

    if pipeline == "edit":
        if input_count < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{entry.model_id}' is an image edit model and "
                    "requires at least one input image"
                ),
            )
        if input_count > 1 and alias not in _MULTI_REF_ALIASES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{entry.model_id}' ({alias}) supports a single "
                    "reference image; multi-reference editing needs "
                    "qwen-image-edit-2509/2511"
                ),
            )
    else:
        if input_count > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Text-to-image models accept at most one input image "
                    "(img2img source)"
                ),
            )
        if params.image_strength is not None and not (
            0 < params.image_strength <= 1
        ):
            raise HTTPException(
                status_code=400, detail="image_strength must be in (0, 1]"
            )

    response_format = (params.response_format or "b64_json").lower()
    if response_format not in ("b64_json", "url"):
        raise HTTPException(
            status_code=400, detail="response_format must be b64_json or url"
        )

    seed = params.seed if params.seed is not None else random.randint(0, 2**31)

    normalized: dict[str, Any] = {
        "prompt": params.prompt,
        "alias": alias,
        "pipeline": pipeline,
        "n": n,
        "steps": steps,
        "seed": int(seed),
        "width": width,
        "height": height,
        "response_format": response_format,
    }
    if params.negative_prompt:
        normalized["negative_prompt"] = params.negative_prompt
    if params.guidance is not None:
        normalized["guidance"] = float(params.guidance)
    if params.image_strength is not None and pipeline != "edit":
        normalized["image_strength"] = float(params.image_strength)
    if params.lora_paths:
        # Request strings must not become arbitrary local filesystem paths
        # in the worker (file-existence oracle via reflected loader errors).
        # Allowed forms are mflux references: HF repo ids, collection syntax
        # "org/repo:file.safetensors" and registry names -- never absolute
        # paths or parent traversal.
        for p in params.lora_paths:
            p = str(p)
            if p.startswith(("/", "~")) or ".." in p or "\\" in p:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "lora_paths entries must be HF references (e.g. "
                        "org/repo:file.safetensors), not filesystem paths"
                    ),
                )
        normalized["lora_paths"] = [str(p) for p in params.lora_paths]
        if params.lora_scales:
            normalized["lora_scales"] = [float(s) for s in params.lora_scales]
    return normalized


def _lease_bytes_for(alias: str, image_settings: Any) -> int:
    configured = float(getattr(image_settings, "memory_lease_gb", 0) or 0)
    if configured > 0:
        return int(configured * GB)
    return int(_DEFAULT_LEASE_GB.get(alias, _FALLBACK_LEASE_GB) * GB)


def _sync_response(
    job: Any, response_format: str, request: Request
) -> dict[str, Any]:
    """OpenAI images API shape from a completed job."""
    data = []
    base = str(request.base_url).rstrip("/")
    for i, name in enumerate(job.artifact_files):
        if response_format == "url":
            # Absolute URL so OpenAI-style clients can fetch it directly.
            # The content endpoint sits behind the same API-key auth as
            # this one -- callers reuse their credentials.
            data.append(
                {"url": f"{base}/v1/images/{job.id}/content?index={i}"}
            )
        else:
            file_path = (
                Path(job.artifact_path).parent / name
                if job.artifact_path
                else None
            )
            if file_path is None or not file_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="completed job has a missing artifact file",
                )
            data.append(
                {"b64_json": base64.b64encode(file_path.read_bytes()).decode()}
            )
    return {
        "created": int(job.completed_at or 0),
        "id": job.id,  # fmlx extension: poll/content/delete handle
        "data": data,
    }


async def _create_image_impl(request: Request, force_sync: bool):
    manager = _get_image_manager()
    image_settings = _image_settings()
    params, input_images = await _parse_create_body(request)

    pool = _get_engine_pool()
    if params.model:
        resolved = _resolve_model(params.model)
    else:
        picked = _pick_default_model(pool, want_edit=bool(input_images))
        if picked is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No suitable image model found"
                    + (" (input images need an edit model)" if input_images
                       else "")
                ),
            )
        resolved = picked
    entry = pool.get_entry(resolved) if hasattr(pool, "get_entry") else None
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{params.model}' not found"
        )
    if getattr(entry, "model_type", "") != "image":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{resolved}' is not an image generation model "
                f"(model_type={getattr(entry, 'model_type', '?')})"
            ),
        )

    ok, reason = manager.guard_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)
    ok, reason = await manager.probe_worker_venv(kind="image")
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    model_settings = None
    try:
        from omlx.server import _server_state

        sm = getattr(_server_state, "settings_manager", None)
        if sm is not None:
            model_settings = sm.get_settings(resolved)
    except Exception:  # noqa: BLE001
        model_settings = None

    normalized = _normalize_params(
        params, entry, image_settings, model_settings, len(input_images)
    )
    response_format = normalized.pop("response_format")
    alias = normalized["alias"]
    lease_bytes = _lease_bytes_for(alias, image_settings)

    # A lease at or above the machine ceiling would never pass dispatch
    # admission -- reject now instead of accepting a job that can only
    # wedge the queue until the dispatcher fails it.
    ok, reason = manager.lease_fits_ceiling(lease_bytes)
    if not ok:
        raise HTTPException(status_code=413, detail=reason)

    from ..video.manager import MediaJob, QueueFullError

    job = MediaJob(
        id=f"img_{uuid.uuid4().hex}",
        kind="image",
        model_id=resolved,
        model_dir=str(entry.model_path),
        params=normalized,
        lease_bytes=lease_bytes,
    )
    try:
        await manager.submit(job, input_images=input_images or None)
    except QueueFullError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _record_image_request(resolved)

    sync = force_sync or (params.sync is not False)
    if not sync:
        return job.to_dict()

    timeout = float(getattr(image_settings, "sync_timeout_seconds", 900))
    finished = await manager.wait_terminal(job.id, timeout=timeout)
    if finished is None:
        existing = manager.get(job.id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Job was deleted")
        if force_sync:
            # The generations alias serves OpenAI SDK clients that cannot
            # poll and auto-retry 5xx responses -- an orphaned job would
            # hold the queue while duplicates pile up behind it.
            await manager.delete(job.id)
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Image generation exceeded {int(timeout)}s and was "
                    "cancelled; retry with fewer steps or a smaller size"
                ),
            )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Image generation still running after {int(timeout)}s; "
                f"poll GET /v1/images/{job.id}"
            ),
        )
    if finished.status != "completed":
        error = finished.error or {}
        raise HTTPException(
            status_code=500,
            detail=(
                f"Image generation failed "
                f"({error.get('code', 'unknown')}): "
                f"{error.get('message', '')}"
            ),
        )
    return _sync_response(finished, response_format, request)


@router.post("/v1/images")
async def create_image(request: Request):
    return await _create_image_impl(request, force_sync=False)


@router.post("/v1/images/generations")
async def create_image_generations(request: Request):
    """OpenAI SDK compat alias (client.images.generate): always sync."""
    return await _create_image_impl(request, force_sync=True)


@router.get("/v1/images")
async def list_images(
    limit: int = 20, after: str | None = None, order: str = "desc"
):
    manager = _get_image_manager()
    limit = max(1, min(int(limit), 100))
    order = "asc" if order == "asc" else "desc"
    page, has_more = manager.list_jobs(
        limit=limit, after=after, order=order, kind="image"
    )
    return {
        "object": "list",
        "data": [job.to_dict() for job in page],
        "has_more": has_more,
        "last_id": page[-1].id if page else None,
    }


def _get_image_job(manager, job_id: str):
    job = manager.get(job_id)
    if job is None or job.kind != "image":
        raise HTTPException(status_code=404, detail=f"Image job '{job_id}' not found")
    return job


@router.get("/v1/images/{job_id}")
async def get_image(job_id: str):
    manager = _get_image_manager()
    return _get_image_job(manager, job_id).to_dict()


@router.get("/v1/images/{job_id}/content")
async def get_image_content(job_id: str, index: int = 0):
    manager = _get_image_manager()
    job = _get_image_job(manager, job_id)
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status}; content is available once completed",
        )
    # Expiry check must run BEFORE the index range check: retention clears
    # artifact_files to [] alongside artifact_path, so a purged job would
    # otherwise always 404 as "index out of range" and clients would never
    # see the artifact_expired code they key the expired-state UI on.
    if not job.artifact_path:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "artifact_expired",
                "message": "Artifact was purged by retention",
                "expires_at": int(job.expires_at) if job.expires_at else None,
            },
        )
    if index < 0 or index >= len(job.artifact_files):
        raise HTTPException(
            status_code=404,
            detail=f"index {index} out of range (job has "
                   f"{len(job.artifact_files)} images)",
        )
    file_path = Path(job.artifact_path).parent / job.artifact_files[index]
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "artifact_expired",
                "message": "Artifact was purged by retention",
                "expires_at": int(job.expires_at) if job.expires_at else None,
            },
        )
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=f"{job_id}-{index}.png",
    )


@router.delete("/v1/images/{job_id}")
async def delete_image(job_id: str):
    manager = _get_image_manager()
    _get_image_job(manager, job_id)
    await manager.delete(job_id)
    return {"id": job_id, "object": "image.deleted", "deleted": True}
