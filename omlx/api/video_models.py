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
    sends string literals like "4"). fmlx extensions: negative_prompt,
    frames/steps/fps/seed/guidance/guidance_2. Extension collision policy:
    if OpenAI later claims an extension name, fmlx semantics yield and the
    extension moves to an fmlx_ prefix (spec 4.3).
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
