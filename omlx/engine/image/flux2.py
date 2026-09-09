# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Klein engines.

Wraps mflux's ``Flux2Klein`` (text-to-image, plus native img2img via
``image_path``) and ``Flux2KleinEdit`` (multi-image editing via
``image_paths``). FLUX models do not take a negative prompt.
"""

from typing import Any

from pydantic import BaseModel, Field

from .base import BaseImageEngine


class FluxGenerateParams(BaseModel):
    """Text-to-image request parameters (FLUX.2 Klein)."""

    prompt: str
    seed: int | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=1000)
    height: int | None = Field(default=None, ge=16)
    width: int | None = Field(default=None, ge=16)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)


class FluxEditParams(FluxGenerateParams):
    """Edit request parameters (input images arrive separately as image parts
    / data URIs and are injected post-validation by the route)."""

    image_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    use_kv_cache: bool | None = None


class Flux2KleinImageEngine(BaseImageEngine):
    """FLUX.2 Klein engine (txt2img / img2img through ``image_path``)."""

    generates_params = FluxGenerateParams
    edits_params = FluxEditParams

    def edit_kwargs(self, image_paths: list[str]) -> dict[str, Any]:
        # Klein does img2img natively: the model takes a single image_path.
        return {"image_path": image_paths[0]} if image_paths else {}


class Flux2KleinEditImageEngine(BaseImageEngine):
    """FLUX.2 Klein edit engine (multi-image editing through ``image_paths``)."""

    generates_params = FluxGenerateParams
    edits_params = FluxEditParams
