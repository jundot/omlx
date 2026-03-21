# SPDX-License-Identifier: Apache-2.0
"""Patch qwen3_5 GatedDeltaNet to call cache.advance(S).

mlx-lm's qwen3_5.py GatedDeltaNet is missing cache.advance(S) after
updating cache[1] = state. Every other hybrid model in mlx-lm
(qwen3_next, falcon_h1, mamba2, plamo2, etc.) calls advance() in their
forward pass. Without it, ArraysCache.left_padding / lengths are never
decremented between prefill chunks, causing incorrect SSM masks when
batch_size > 1 with different prompt lengths.

Supported model types: qwen3_5
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_TYPES = {"qwen3_5"}

# Track whether the class-level patch has been applied
_class_patch_applied = False


def _get_model_type(model: Any) -> str | None:
    """Extract model_type string from a loaded mlx-lm model."""
    for attr in ("model_type", "args"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if isinstance(obj, str):
            return obj
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def _make_patched_gdn_call(original_call):
    """Create a patched __call__ for GatedDeltaNet.

    Adds cache.advance(S) after the original forward pass, matching
    the pattern used by Qwen3NextGatedDeltaNet and all other hybrid
    models in mlx-lm.
    """
    def patched_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        result = original_call(self, inputs, mask=mask, cache=cache)
        if cache is not None:
            cache.advance(inputs.shape[1])
        return result

    return patched_call


def apply_gated_delta_advance_patch(model: Any) -> bool:
    """Apply advance() monkey-patch to qwen3_5 GatedDeltaNet.

    Args:
        model: A loaded mlx-lm model instance.

    Returns:
        True if the patch was applied, False if the model is not
        supported or the patch is not needed.
    """
    global _class_patch_applied

    model_type = _get_model_type(model)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        return False

    if _class_patch_applied:
        return True

    try:
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
    except ImportError:
        logger.debug("GatedDeltaNet advance patch: qwen3_5 module not found")
        return False

    # Forward compatibility: skip if mlx-lm already added advance()
    try:
        source = inspect.getsource(GatedDeltaNet.__call__)
        if ".advance(" in source:
            logger.debug(
                "GatedDeltaNet advance patch: upstream already has advance(), skipping"
            )
            _class_patch_applied = True
            return False
    except (OSError, TypeError):
        pass

    original_call = GatedDeltaNet.__call__
    GatedDeltaNet.__call__ = _make_patched_gdn_call(original_call)

    _class_patch_applied = True
    logger.info("GatedDeltaNet advance() patch applied for qwen3_5")
    return True
