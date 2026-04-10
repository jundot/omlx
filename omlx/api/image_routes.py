# SPDX-License-Identifier: Apache-2.0
"""
API routes for OpenAI-compatible Images API.

Provides endpoints for image generation:
- POST /v1/images/generations
"""

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from ..api.image_models import (
    ImageRequest,
    ImageResponse,
    ImageData,
    get_inference_steps,
    get_guidance_scale,
    parse_size,
)

if TYPE_CHECKING:
    from ..engine_pool import EnginePool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Images"])


# ---------------------------------------------------------------------------
# Engine pool accessor — patched in tests via omlx.api.image_routes._get_engine_pool
# ---------------------------------------------------------------------------


def _get_engine_pool() -> "EnginePool":
    """Return the active EnginePool from server state.

    Imported lazily to avoid a circular import at module load time.
    Can be replaced in tests via patch('omlx.api.image_routes._get_engine_pool').
    """
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve a model alias to its real model ID."""
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


@router.post("/v1/images/generations")
async def create_image(request: ImageRequest) -> ImageResponse:
    """
    Create an image given a prompt.

    OpenAI-compatible endpoint for image generation (Text-to-Image, Image-to-Image).
    """
    from ..engine.image import ImageEngine
    from ..exceptions import ModelNotFoundError

    # Early rejection: URL response format not supported
    if request.response_format == "url":
        raise HTTPException(
            status_code=501,
            detail="URL response format not yet supported. Use b64_json.",
        )

    pool = _get_engine_pool()
    model_id = _resolve_model(request.model)

    # Get the engine
    try:
        engine = await pool.get_engine(model_id)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{request.model}': {str(exc)}",
        ) from exc

    # Verify it's an image engine
    if not isinstance(engine, ImageEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not an image generation model (type: {type(engine).__name__})",
        )

    # Parse parameters
    width, height = parse_size(request.size)
    num_inference_steps = get_inference_steps(request)
    guidance_scale = get_guidance_scale(request)

    # Handle strength default: use 0.8 for I2I if not specified
    strength = request.strength if request.strength is not None else 0.8

    # Generate images
    images_data: list[ImageData] = []
    created_time = int(time.time())

    for i in range(request.n):
        try:
            # Use seed for reproducibility
            seed = request.seed
            if seed is not None and request.n > 1:
                # For batch generation, increment seed for each image
                seed = seed + i

            output = await engine.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                image=request.image,
                strength=strength,
            )

            # Format response (only b64_json supported)
            b64_data = output.to_base64()
            images_data.append(ImageData(
                b64_json=b64_data,
                revised_prompt=request.prompt,  # Could be modified by model
            ))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {str(e)}",
            ) from e

    return ImageResponse(
        created=created_time,
        data=images_data,
    )