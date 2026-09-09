# SPDX-License-Identifier: Apache-2.0
"""
Image generation engines for oMLX (mflux, FLUX.2 Klein).
"""

from typing import Any

from ...utils.mflux import resolve_mflux_config, resolve_mflux_family
from .base import BaseImageEngine
from .flux2 import Flux2KleinEditImageEngine, Flux2KleinImageEngine

__all__ = ["BaseImageEngine", "REGISTRY", "get_image_engine"]

#: mflux family class name → engine class.
REGISTRY: dict[str, type[BaseImageEngine]] = {
    "Flux2Klein": Flux2KleinImageEngine,
    "Flux2KleinEdit": Flux2KleinEditImageEngine,
}


def get_image_engine(model_name: str, **kwargs: Any) -> BaseImageEngine:
    """Instantiate the engine for an mflux model directory."""
    model_config = resolve_mflux_config(model_name)
    family_cls = resolve_mflux_family(model_config)
    engine_cls = REGISTRY.get(family_cls.__name__)
    if engine_cls is None:
        raise ValueError(
            f"Unsupported image model family '{family_cls.__name__}' "
            f"for model '{model_name}' — only FLUX.2 Klein is supported."
        )
    return engine_cls(model_name=model_name, **kwargs)
