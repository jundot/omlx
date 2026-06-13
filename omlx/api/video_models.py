# SPDX-License-Identifier: Apache-2.0
"""Request/response models for the /v1/videos API.

POST /v1/videos accepts BOTH application/json and multipart/form-data --
the official openai SDK sends multipart (all fields as strings), so the
route normalizes either body into VideoCreateParams here. Pydantic v2 lax
coercion handles the string-to-number conversion ("4" -> 4).
Design: docs/video-generation-engine-spec.md section 4.3.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VideoCreateParams(BaseModel):
    """Normalized create-video parameters (JSON or multipart source).

    OpenAI-compatible core: model, prompt, size ("WxH"), seconds (the SDK
    sends string literals like "4"), input_reference (the I2V conditioning
    image; multipart file field from the SDK, or a base64/data-URL string in
    JSON bodies -- the route extracts either form into raw bytes before
    validation, so this model never carries it). fmlx extensions:
    negative_prompt, frames/steps/fps/seed/guidance/guidance_2,
    extend_video_id (continue a completed job's video from its last frame;
    needs an i2v-capable model), upscale_resolution (SeedVR2 per-frame
    upscale of the final video to this short-side resolution). Extension
    collision policy: if OpenAI later claims an extension name, fmlx
    semantics yield and the extension moves to an fmlx_ prefix (spec 4.3).
    """

    model: str
    prompt: str = Field(min_length=1)
    size: Optional[str] = None  # "WxH", e.g. "480x272"
    seconds: Optional[float] = None
    negative_prompt: Optional[str] = None
    width: Optional[int] = None  # Explicit override beats size
    height: Optional[int] = None
    frames: Optional[int] = None  # Explicit override beats seconds*fps
    steps: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None
    guidance: Optional[float] = None
    guidance_2: Optional[float] = None
    extend_video_id: Optional[str] = None  # Continue this completed job
    upscale_resolution: Optional[int] = None  # SeedVR2 target short side
    # Per-request override for LLM prompt extension. None = follow the
    # server setting (extend iff video.prompt_extend_model is configured);
    # True = force extend; False = skip extension for this request.
    prompt_extend: Optional[bool] = None
