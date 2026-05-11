# SPDX-License-Identifier: Apache-2.0
"""Native MTP monkey-patches for mlx-vlm.

mlx-vlm ships its own ``Model.sanitize`` for VLM checkpoints
(Qwen3_5ForConditionalGeneration etc). The stock body unconditionally
strips ``mtp.*`` weights and only shifts a fixed set of backbone norm
keys by +1 — the MTP head's own norms (``mtp.norm.weight``,
``mtp.pre_fc_norm_*.weight`` and the per-block layernorms) miss the
shift, so MTP layers run with raw RMSNorm weights after quantization.

This package patches the affected mlx-vlm Model classes so the sanitize
output keeps ``mtp.*`` and applies the +1 norm shift to every MTP norm,
matching what mlx-lm PR 990 does on the LLM side.

Activation: oQ quantization invokes ``apply_mlx_vlm_mtp_patch`` before
building the VLM sanitizer. The patches are idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mlx_vlm_mtp_patch() -> bool:
    """Apply the mlx-vlm MTP sanitize monkey-patches.

    Returns True on success (or if already applied), False if the
    sub-step refused (mlx-vlm not importable, etc.).
    """
    global _PATCHED
    if _PATCHED:
        return True

    from . import qwen35_vlm_model, qwen35_moe_vlm_model

    if not qwen35_vlm_model.apply():
        logger.debug("Qwen3.5 VLM MTP sanitize patch did not apply")
    if not qwen35_moe_vlm_model.apply():
        logger.debug("Qwen3.5 MoE VLM MTP sanitize patch did not apply")

    _PATCHED = True
    logger.info("mlx-vlm MTP sanitize patch applied")
    return True


_RUNTIME_PATCHED = False


def apply_mlx_vlm_mtp_runtime_patch() -> bool:
    """Apply the mlx-vlm runtime MTP patches (attach MTPModule, mtp_forward).

    Distinct from ``apply_mlx_vlm_mtp_patch``: that one only patches
    ``sanitize`` for conversion-time MTP preservation. This one builds
    the runtime infrastructure so VLMBatchedEngine can actually invoke
    the MTP head at inference time.

    Should be called *before* ``mlx_vlm.utils.load(...)`` so the
    instantiated LanguageModel picks up the patched ``__init__``.
    """
    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return True

    from . import qwen35_moe_vlm_runtime

    if not qwen35_moe_vlm_runtime.apply():
        logger.debug("Qwen3.5-MoE VLM runtime MTP patch did not apply")
        return False

    _RUNTIME_PATCHED = True
    logger.info("mlx-vlm MTP runtime patch applied")
    return True
