# SPDX-License-Identifier: Apache-2.0
"""Runtime-only optimizations for mlx-vlm's native Prism Hadamard model.

The upstream loader owns the checkpoint format, precision and processor. This
module only preserves KV capacity after a batch shrinks to an unpadded singleton.
"""

import logging

from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

from .decode import CapacityPreservingModel

logger = logging.getLogger(__name__)
MODEL_TYPE = "prism_hadamard_qwen35"


def apply_runtime_patches(model):
    """Patch a loaded native Prism VLM, leaving other model classes untouched."""
    if getattr(getattr(model, "config", None), "model_type", None) != MODEL_TYPE:
        return False
    language_model = getattr(model, "language_model", None)
    backbone = getattr(language_model, "model", None)
    if type(backbone) is CapacityPreservingModel:
        return True
    if type(backbone) is not Qwen3_5Model:
        logger.warning("Prism runtime patches skipped: unfamiliar decoder class")
        return False

    backbone.__class__ = CapacityPreservingModel
    logger.info("Prism unpadded singleton KV capacity preservation enabled")
    return True
