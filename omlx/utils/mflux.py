# SPDX-License-Identifier: Apache-2.0
"""
mflux helpers for the image API (FLUX.2 Klein only).

mflux is imported lazily inside these functions so the module imports
cleanly when mflux is not installed.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_mflux_config(model_name: str):
    """Resolve the mflux ``ModelConfig`` for a model directory (name-based)."""
    from mflux.models.common.config import ModelConfig

    dir_name = model_name.rstrip("/").rsplit("/", 1)[-1]
    return ModelConfig.from_name(model_name=dir_name)


def resolve_mflux_family(model_config):
    """Resolve the mflux variant class (txt2img or edit) for a ModelConfig.

    Edit checkpoints are identified by an ``edit`` marker in the resolved
    name or aliases (e.g. ``flux2-klein-9b-edit``).
    """
    from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

    haystack = " ".join(
        [
            (model_config.model_name or "").lower(),
            *(a.lower() for a in model_config.aliases),
        ]
    )
    if "flux2-klein" in haystack and "edit" in haystack:
        return Flux2KleinEdit
    if "flux2-klein" in haystack:
        return Flux2Klein

    raise ValueError(
        f"Unsupported mflux model '{model_config.model_name}' — only "
        "FLUX.2 Klein checkpoints are supported by the image API."
    )
