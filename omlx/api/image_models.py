# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible Images API.

These models define the request and response schemas for:
- /v1/images/generations endpoint (Text-to-Image, Image-to-Image)
"""

import base64
import re
import time
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


def parse_size(size: str) -> Tuple[int, int]:
    """
    Parse size string to (width, height) tuple.

    Args:
        size: Size string like "1024x1024"

    Returns:
        Tuple of (width, height)
    """
    width, height = size.split("x")
    return int(width), int(height)


class ImageRequest(BaseModel):
    """
    Request for creating images (OpenAI-compatible).

    Supports Text-to-Image (T2I) and Image-to-Image (I2I) generation.
    """

    model: str
    """ID of the model to use for image generation."""

    prompt: str
    """A text description of the desired image(s)."""

    n: int = Field(default=1, ge=1, le=4)
    """The number of images to generate. Must be between 1 and 4. Default is 1."""

    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    """The size of the generated images. Default is 1024x1024."""

    quality: Literal["standard", "hd"] = "standard"
    """The quality of the generated image. 'hd' uses more inference steps."""

    response_format: Literal["url", "b64_json"] = "b64_json"
    """The format in which the generated images are returned."""

    style: Optional[Literal["vivid", "natural"]] = None
    """The style of the generated images. Only supported by some models."""

    user: Optional[str] = None
    """A unique identifier for the end-user."""

    # omlx extensions
    negative_prompt: Optional[str] = None
    """Negative prompt for image generation (exclusions)."""

    seed: Optional[int] = None
    """Random seed for reproducible generation."""

    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    """Number of denoising steps. Overrides quality setting."""

    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    """Guidance scale for classifier-free guidance."""

    image: Optional[str] = None
    """Base64-encoded image for Image-to-Image transformation."""

    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Strength of transformation for I2I (0.0-1.0). Higher = more change."""

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: Optional[str]) -> Optional[str]:
        """Validate that image field is valid base64 or None/empty."""
        if v is None or v == "" or v == "string":
            # None, empty string, or placeholder "string" -> treat as None
            return None
        # Strip data URI prefix if present
        if v.startswith("data:"):
            v = v.split(",", 1)[1]
        # Validate base64 format (allow padding variants)
        # Base64 chars: A-Z, a-z, 0-9, +, /, and optional = padding
        if not re.match(r"^[A-Za-z0-9+/]+={0,2}$", v):
            raise ValueError(
                "image field must be valid base64-encoded data. "
                "Leave empty or omit for text-to-image generation."
            )
        return v


class ImageData(BaseModel):
    """A single generated image result."""

    b64_json: Optional[str] = None
    """The base64-encoded JSON of the generated image."""

    url: Optional[str] = None
    """The URL of the generated image."""

    revised_prompt: Optional[str] = None
    """The prompt that was used for generation, if modified."""


class ImageResponse(BaseModel):
    """Response from image generation endpoint."""

    created: int
    """The Unix timestamp when the images were created."""

    data: List[ImageData]
    """List of generated images."""


# Quality to inference steps mapping
QUALITY_STEPS = {
    "standard": 20,
    "hd": 40,
}

# Style to guidance scale mapping
STYLE_GUIDANCE = {
    "vivid": 5.0,
    "natural": 3.5,
    None: 3.5,  # Default
}


def get_inference_steps(request: ImageRequest) -> int:
    """
    Determine the number of inference steps based on request parameters.

    Priority: num_inference_steps > quality mapping

    Args:
        request: The image generation request

    Returns:
        Number of inference steps
    """
    if request.num_inference_steps is not None:
        return request.num_inference_steps
    return QUALITY_STEPS.get(request.quality, 20)


def get_guidance_scale(request: ImageRequest) -> float:
    """
    Determine the guidance scale based on request parameters.

    Priority: guidance_scale > style mapping

    Args:
        request: The image generation request

    Returns:
        Guidance scale value
    """
    if request.guidance_scale is not None:
        return request.guidance_scale
    return STYLE_GUIDANCE.get(request.style, 3.5)