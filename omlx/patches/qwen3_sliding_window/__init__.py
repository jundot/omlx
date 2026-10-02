# SPDX-License-Identifier: Apache-2.0
"""Qwen3 sliding-window attention monkey-patch for the pinned mlx-lm dependency.

Unlike the laguna/mimo_v2 patches, which register a vendored module under a
model_type that mlx-lm doesn't ship at all yet (see omlx/patches/laguna/__init__.py),
this patches two attributes, ModelArgs and Qwen3Model, directly on the real,
already-existing mlx_lm.models.qwen3 module. That's safe here because
the patched classes are a strict backward-compatible superset of the originals.

mlx_lm.utils._get_classes() resolves arch.Model / arch.ModelArgs
via direct attribute access on the live module object, at call time, on every
model load. So patching the module's attributes before any load is
enough for both Jina and any other Qwen3 model to pick them up.
self.model = Qwen3Model(args) in Model.__init__ resolves the bare name
against that same module's namespace at instantiation time, so Model
itself needs no changes and is reused verbatim, along with Attention, MLP,
and TransformerBlock.

There's no known upstream proposal for Qwen3 sliding-window support
at the time of writing, so there's no PR reference to carry here
the way laguna/mimo_v2 do.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def _patch_qwen3_module() -> None:
    from mlx_lm.models import qwen3 as _qwen3

    from . import qwen3_model

    _qwen3.ModelArgs = qwen3_model.ModelArgs
    _qwen3.Qwen3Model = qwen3_model.Qwen3Model


def apply_qwen3_sliding_window_patch() -> bool:
    """Patch mlx_lm.models.qwen3 to honor layer_types/sliding_window.

    Returns True the first time the patch is applied, False on any
    subsequent call (idempotent).
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _patch_qwen3_module()
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("mlx_lm"):
            logger.debug(
                "mlx_lm not importable - qwen3 sliding-window patch skipped"
            )
            return False
        raise

    _APPLIED = True
    logger.info("Qwen3 sliding-window attention patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_qwen3_sliding_window_patch", "is_applied"]
