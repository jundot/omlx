# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible image endpoints (FLUX.2 Klein via mflux):

- POST /v1/images/generations — text-to-image (and img2img via Klein's
  native image_path when an input image is supplied)
- POST /v1/images/edits       — edits (JSON data URIs or multipart files)

All tunables (seed, num_inference_steps, guidance, image_strength) are
request parameters validated by the engine — no per-model server defaults.
"""

import base64
import binascii
import logging
import os
import random
import tempfile
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from ..server_metrics import get_server_metrics
from ..utils.formatting import make_json_safe
from .image_models import (
    ImageEditRequest,
    ImageEditResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Client-tunable parameters forwarded to the engine after validation.
_EDIT_DEFAULT_STRENGTH = 0.5


def _get_engine_pool():
    """Active EnginePool (lazy import; patchable in tests)."""
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


async def _get_image_engine(model_id: str):
    """Resolve an image model through the pool, or 404/400."""
    resolved = _resolve_model(model_id)
    try:
        engine = await _get_engine_pool().get_engine(resolved)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved}' not found or failed to load: {exc}",
        ) from exc
    if not hasattr(engine, "generate"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved}' is not an image generation model",
        )
    return resolved, engine


def _record_image_request(model_id: str) -> None:
    """Count the request without inventing token numbers for images."""
    try:
        get_server_metrics().record_request_complete(
            prompt_tokens=0, completion_tokens=0, cached_tokens=0, model_id=model_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record image metrics for %s: %s", model_id, exc)


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------


def _size_to_dims(size: str) -> tuple[int, int]:
    """Parse '1024x768' into (width, height); fall back to 1024x1024."""
    try:
        width, height = size.lower().split("x", 1)
        return int(width), int(height)
    except ValueError:
        return 1024, 1024


def _tunable_params(body: dict[str, Any]) -> dict[str, Any]:
    """Extract the tunable request parameters (None values dropped)."""
    params: dict[str, Any] = {}
    for key in (
        "seed",
        "num_inference_steps",
        "guidance",
        "image_strength",
        "use_kv_cache",
        "width",
        "height",
    ):
        if body.get(key) is not None:
            params[key] = body[key]
    size = body.get("size")
    if isinstance(size, str) and size:
        params["width"], params["height"] = _size_to_dims(size)
    return params


# ---------------------------------------------------------------------------
# Image reference handling (data URI / multipart → temp files)
# ---------------------------------------------------------------------------


def _decode_data_uri(value: str) -> bytes:
    prefix, separator, encoded = value.strip().partition(",")
    low = prefix.lower()
    if separator != "," or not low.startswith("data:image/") or ";base64" not in low:
        raise HTTPException(
            400,
            detail="Image references must be base64 data URIs (data:image/...;base64,...).",
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            400, detail="Image data URI contains invalid base64 data."
        ) from exc


def _guess_image_suffix(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _write_temp_image(data: bytes) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=_guess_image_suffix(data))
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp_path


def _extract_data_uris(body: dict[str, Any]) -> list[str]:
    """Collect image references from ``image``/``images`` body fields."""
    refs: list[str] = []
    raw = body.get("images") or body.get("image")
    if isinstance(raw, (str, dict)):
        raw = [raw]
    for item in raw or []:
        if isinstance(item, str):
            refs.append(item)
        elif isinstance(item, dict):
            url = item.get("image_url") or item.get("url")
            if url:
                refs.append(str(url))
    return refs


def _cleanup_temp_images(paths: list[str]) -> None:
    tmpdir = os.path.realpath(tempfile.gettempdir())
    for path in paths:
        try:
            real = os.path.realpath(path)
            if os.path.commonpath([real, tmpdir]) == tmpdir:
                os.unlink(real)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _validate_params(engine, kind: str, body: dict[str, Any]):
    tunables = _tunable_params(body)
    tunables["prompt"] = body["prompt"]
    try:
        if kind == "generate":
            return engine.validate_generate(tunables)
        return engine.validate_edit(tunables)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=make_json_safe(exc.errors())
        ) from exc


async def _generate_images(
    engine,
    resolved_model: str,
    body: dict[str, Any],
    image_paths: list[str] | None = None,
):
    """Run n generations and return the base64 response items.

    Each image gets its own seed: ``seed + i`` when the caller supplied one,
    otherwise a fresh random seed — always reported back in the response.
    """
    n = body.get("n") or 1
    caller_seed = body.get("seed")
    params = _validate_params(engine, "edit" if image_paths else "generate", body)

    results: list[ImageResponse] = []
    for i in range(n):
        seed = (
            caller_seed + i if caller_seed is not None else random.randint(0, 2**32 - 1)
        )
        params.seed = seed
        if image_paths:
            img_bytes = await engine.edit(params, image_paths)
        else:
            img_bytes = await engine.generate(params)
        results.append(
            ImageResponse(
                b64_json=base64.b64encode(img_bytes).decode("utf-8"), seed=seed
            )
        )
    return results


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    body = request.model_dump(exclude_none=True)
    body.update({k: v for k, v in request.model_extra.items() if v is not None})
    body["prompt"] = request.prompt

    resolved_model, engine = await _get_image_engine(request.model)
    try:
        data = await _generate_images(engine, resolved_model, body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {exc}"
        ) from exc

    _record_image_request(resolved_model)
    return ImageGenerationResponse(created=int(time.time()), data=data)


@router.post("/v1/images/edits", response_model=ImageEditResponse)
async def edit_image(request: Request):
    """Image edits: JSON body with data-URI images, or multipart file parts
    named ``image`` / ``image[]``."""
    if "multipart/form-data" in (request.headers.get("content-type") or "").lower():
        return await _edit_image_multipart(request)
    return await _edit_image_json(request)


async def _edit_image_json(request: Request) -> ImageEditResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid JSON body: {exc}"
        ) from exc
    try:
        ImageEditRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=make_json_safe(exc.errors())
        ) from exc

    refs = _extract_data_uris(body)
    if not refs:
        raise HTTPException(
            status_code=400,
            detail="An input image is required — send 'images' as base64 data URIs.",
        )

    resolved_model, engine = await _get_image_engine(body["model"])
    image_paths = [_write_temp_image(_decode_data_uri(ref)) for ref in refs]
    try:
        return await _run_edit(engine, resolved_model, body, image_paths)
    finally:
        _cleanup_temp_images(image_paths)


async def _edit_image_multipart(request: Request) -> ImageEditResponse:
    form = await request.form()
    model = form.get("model")
    prompt = form.get("prompt")
    if not model or not str(prompt).strip():
        raise HTTPException(
            status_code=400,
            detail="'model' and a non-empty 'prompt' form fields are required.",
        )

    image_paths: list[str] = []
    for part in form.getlist("image") + form.getlist("image[]"):
        data = await part.read() if hasattr(part, "read") else b""
        if data:
            image_paths.append(_write_temp_image(data))
    if not image_paths:
        raise HTTPException(
            status_code=400,
            detail="An image file part ('image' or 'image[]') is required for image editing.",
        )

    def _num(key: str, cast):
        raw = form.get(key)
        if raw in (None, ""):
            return None
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return None

    body: dict[str, Any] = {
        "model": str(model),
        "prompt": str(prompt),
        "n": _num("n", int) or 1,
        "seed": _num("seed", int),
        "size": form.get("size"),
        "num_inference_steps": _num("num_inference_steps", int),
        "guidance": _num("guidance", float),
        "image_strength": _num("image_strength", float),
        "use_kv_cache": (
            None
            if form.get("use_kv_cache") in (None, "")
            else str(form.get("use_kv_cache")).lower() in ("1", "true", "yes", "on")
        ),
    }

    resolved_model, engine = await _get_image_engine(str(model))
    try:
        return await _run_edit(engine, resolved_model, body, image_paths)
    finally:
        _cleanup_temp_images(image_paths)


async def _run_edit(
    engine, resolved_model: str, body: dict[str, Any], image_paths: list[str]
) -> ImageEditResponse:
    if getattr(engine, "edits_params", None) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' does not support image editing",
        )
    if body.get("image_strength") is None:
        body["image_strength"] = _EDIT_DEFAULT_STRENGTH
    try:
        data = await _generate_images(
            engine, resolved_model, body, image_paths=image_paths
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image edit failed")
        raise HTTPException(
            status_code=500, detail=f"Image edit failed: {exc}"
        ) from exc

    _record_image_request(resolved_model)
    return ImageEditResponse(created=int(time.time()), data=data)
