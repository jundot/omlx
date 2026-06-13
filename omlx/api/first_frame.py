# SPDX-License-Identifier: Apache-2.0
"""Text-to-image first-frame generation for the T2I->I2V video workflow.

Pure text-to-video follows fine-grained actions poorly. A stronger path is
to generate a high-fidelity first frame from the prompt with an image model
(image models follow prompts far better than video models), then let an
image-to-video model animate that frame. This is the orchestration seam: it
generates ONE image by reusing the EXISTING image engine (the same
MediaJobManager + image worker that /v1/images uses) and returns the bytes
so the caller can pass them as the I2V conditioning image.

Loosely coupled by design: the video path owns no image-gen code, it just
submits an image job and waits. Opt-in (a per-request flag) and
configurable (video.first_frame_model); when off, the video path is
unchanged.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class FirstFrameError(Exception):
    """First-frame generation could not produce an image. Carries a
    client-facing message; the route maps it to an HTTP error (the caller
    opted in, so there is no silent fallback -- I2V needs a frame)."""


async def generate_first_frame(
    prompt: str,
    *,
    width: int,
    height: int,
    model_id: str,
    manager,
    engine_pool,
    settings_manager,
    image_settings,
    timeout_s: float = 300.0,
) -> tuple[bytes, str]:
    """Generate a first frame from ``prompt`` via the image engine.

    Returns ``(image_bytes, suffix)``. Raises :class:`FirstFrameError` with a
    client-facing message on any failure (unknown/again non-image model,
    queue full, generation failure, timeout, missing artifact).

    Reuses image_routes' model resolution + parameter normalization so the
    image job is built exactly like a real /v1/images request -- no
    duplicated image logic here.
    """
    # Lazy import avoids a video_routes <-> image_routes import cycle.
    from .image_models import ImageCreateParams
    from .image_routes import _lease_bytes_for as _img_lease_bytes
    from .image_routes import _normalize_params as _img_normalize
    from .image_routes import _resolve_model as _img_resolve
    from ..video.manager import MediaJob, QueueFullError

    resolved = _img_resolve(model_id)
    entry = engine_pool.get_entry(resolved) if hasattr(
        engine_pool, "get_entry"
    ) else None
    if entry is None:
        raise FirstFrameError(
            f"first_frame_model '{model_id}' not found"
        )
    if getattr(entry, "model_type", "") != "image":
        raise FirstFrameError(
            f"first_frame_model '{model_id}' is not an image model "
            f"(model_type={getattr(entry, 'model_type', '?')})"
        )
    pipeline = getattr(entry, "image_pipeline", "") or "t2i"
    if pipeline != "t2i":
        raise FirstFrameError(
            f"first_frame_model '{model_id}' is an image-edit model; a "
            "text-to-image (t2i) model is required to generate a first frame"
        )

    model_settings = (
        settings_manager.get_settings(resolved) if settings_manager else None
    )
    img_params = ImageCreateParams(
        model=resolved,
        prompt=prompt,
        width=int(width),
        height=int(height),
        n=1,
        sync=True,
    )
    try:
        normalized = _img_normalize(
            img_params, entry, image_settings, model_settings, 0
        )
    except Exception as e:
        raise FirstFrameError(
            f"could not build first-frame image request: {e}"
        )
    normalized.pop("response_format", None)

    # Image jobs need a positive memory lease (the worker refuses lease 0).
    # Reuse the same per-alias lease sizing /v1/images uses.
    lease_bytes = _img_lease_bytes(normalized.get("alias", ""), image_settings)
    ok, reason = manager.lease_fits_ceiling(lease_bytes)
    if not ok:
        raise FirstFrameError(f"first-frame image lease too large: {reason}")

    job = MediaJob(
        id=f"image_{uuid.uuid4().hex}",
        kind="image",
        model_id=resolved,
        model_dir=str(entry.model_path),
        params=normalized,
        lease_bytes=lease_bytes,
    )
    try:
        await manager.submit(job)
    except QueueFullError as e:
        raise FirstFrameError(f"image queue is full: {e}")

    finished = await manager.wait_terminal(job.id, timeout=timeout_s)
    if finished is None:
        raise FirstFrameError(
            f"first-frame generation timed out after {timeout_s:.0f}s"
        )
    if finished.status != "completed":
        raise FirstFrameError(
            f"first-frame generation {finished.status}"
            + (f": {finished.error}" if getattr(finished, "error", None) else "")
        )
    if not finished.artifact_files or not finished.artifact_path:
        raise FirstFrameError("first-frame job completed with no artifact")

    name = finished.artifact_files[0]
    file_path = Path(finished.artifact_path).parent / name
    if not file_path.exists():
        raise FirstFrameError("first-frame artifact file is missing")
    data = file_path.read_bytes()
    suffix = file_path.suffix or ".png"
    logger.info(
        "Generated first frame via %s (%dx%d, %d bytes) for I2V",
        resolved, width, height, len(data),
    )
    return data, suffix
