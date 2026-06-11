# SPDX-License-Identifier: Apache-2.0
"""Request/response models for the /v1/images API.

POST /v1/images accepts BOTH application/json and multipart/form-data
(mirroring /v1/videos: the official openai SDK sends multipart for image
edits, all fields as strings -- pydantic v2 lax coercion converts).
Input images travel outside this model: multipart "image" file fields
(repeatable) or a JSON "image" string / list of strings (data URL or
base64); the route extracts either form into raw bytes before validation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageCreateParams(BaseModel):
    """Normalized create-image parameters (JSON or multipart source).

    OpenAI-compatible core: model (optional -- the route auto-picks an
    image model by pipeline when omitted), prompt, n, size ("WxH" or
    "auto"), response_format ("b64_json" default, or "url" pointing at the
    content endpoint). fmlx extensions: sync (default true: the request
    blocks until the job finishes and returns OpenAI image-API shape;
    false returns the job object immediately for polling -- the chat UI
    uses that for progress), negative_prompt, steps, seed, guidance,
    width/height (explicit override beats size), image_strength (img2img
    on t2i models), lora_paths/lora_scales (mflux LoRA references; HF
    collection syntax "org/repo:file.safetensors" -- must be pre-cached,
    the worker runs HF-offline). Extension collision policy: if OpenAI
    later claims an extension name, fmlx semantics yield and the extension
    moves to an fmlx_ prefix (video spec 4.3).
    """

    model: Optional[str] = None
    prompt: str = Field(min_length=1)
    n: Optional[int] = None
    size: Optional[str] = None  # "WxH" | "auto"
    response_format: Optional[str] = None  # "b64_json" (default) | "url"
    sync: Optional[bool] = None  # Default true
    negative_prompt: Optional[str] = None
    width: Optional[int] = None  # Explicit override beats size
    height: Optional[int] = None
    steps: Optional[int] = None
    # Bounded: mlx random keys take uint64, and the worker derives per-image
    # seeds as seed+i. An out-of-range seed would crash only AFTER the full
    # multi-GB model load -- reject it at validation time instead.
    seed: Optional[int] = Field(None, ge=0, lt=2**31)
    guidance: Optional[float] = None
    image_strength: Optional[float] = None  # img2img only (t2i models)
    lora_paths: Optional[list[str]] = None
    lora_scales: Optional[list[float]] = None
