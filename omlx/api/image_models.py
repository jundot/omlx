# SPDX-License-Identifier: Apache-2.0
"""
Request/response models for the OpenAI-compatible image API.

Tunable generation parameters (seed, num_inference_steps, guidance, ...) are
request-body fields validated by the engine's parameter models — there are
no per-model server-side defaults for image generation.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible text-to-image request."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    n: int | None = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: Literal["b64_json"] = "b64_json"


class ImageEditRequest(BaseModel):
    """OpenAI-compatible image edit request (JSON form).

    Input images are base64 data URIs, plain strings, or
    ``{"image_url": ...}`` objects (the ``image_url`` of a remote URL is
    not fetched — only data URIs are accepted).
    """

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    n: int | None = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: Literal["b64_json"] = "b64_json"
    image: str | dict[str, Any] | list[str | dict[str, Any]] | None = None
    images: list[str | dict[str, Any]] | None = None


class ImageResponse(BaseModel):
    """One generated image.

    ``seed`` is an oMLX extension: the seed that produced this image
    (caller-supplied or freshly generated), so clients can reproduce it.
    """

    b64_json: str
    seed: int | None = None


class ImageResponseEnvelope(BaseModel):
    created: int
    data: list[ImageResponse]


class ImageGenerationResponse(ImageResponseEnvelope):
    """Response of POST /v1/images/generations."""


class ImageEditResponse(ImageResponseEnvelope):
    """Response of POST /v1/images/edits."""
