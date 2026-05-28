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


# Process-wide gate read by ``LanguageModel.__init__`` (in both
# ``qwen35_moe_vlm_runtime`` and ``qwen35_vlm_runtime``) to decide whether
# to attach ``MTPModule`` when the config declares ``mtp_num_hidden_layers
# > 0``. Default True preserves PR #1404 behavior: VLM checkpoints that
# actually ship persisted ``mtp.*`` weights need the head bound even with
# ``mtp_enabled=False`` so strict ``load_weights`` succeeds.
#
# Caller (``utils/model_loading.py::maybe_apply_pre_load_patches``) flips
# this to False right before ``mlx_vlm.utils.load()`` runs when the
# checkpoint declares MTP heads in config but ships no ``mtp.*`` weights
# (e.g. unsloth Qwen3.6 UD MLX builds, issue #1426). Without the gate,
# strict load fails with "Missing N parameters" and the engine silently
# falls back to LLM, dropping vision.
#
# Single-thread MLX executor serializes loads, so this is race-free.
_MTP_ATTACH_ENABLED = True


def set_mtp_attach_enabled(enabled: bool) -> None:
    """Toggle whether subsequent ``mlx_vlm.utils.load()`` calls attach the
    VLM MTPModule.

    Independent of ``mlx_lm_mtp.set_mtp_active``: attach controls module
    presence so strict load can bind persisted ``mtp.*`` weights, active
    controls whether BatchGenerator actually invokes the MTP head during
    decode.
    """
    global _MTP_ATTACH_ENABLED
    _MTP_ATTACH_ENABLED = bool(enabled)


def is_mtp_attach_enabled() -> bool:
    return _MTP_ATTACH_ENABLED

def apply_mlx_vlm_mtp_patch() -> bool:
    """Apply the mlx-vlm MTP sanitize monkey-patches.

    Returns True on success (or if already applied), False if the
    sub-step refused (mlx-vlm not importable, etc.). Idempotency is
    handled by each sub-patcher via class-level flags — no module-wide
    cache flag, keeping behavior consistent with mlx_lm_mtp.
    """
    from . import qwen35_moe_vlm_model, qwen35_vlm_model

    if not qwen35_vlm_model.apply():
        logger.debug("Qwen3.5 VLM MTP sanitize patch did not apply")
    if not qwen35_moe_vlm_model.apply():
        logger.debug("Qwen3.5 MoE VLM MTP sanitize patch did not apply")

    return True


def apply_mlx_vlm_mtp_runtime_patch() -> bool:
    """Apply the mlx-vlm runtime MTP patches (attach MTPModule, mtp_forward).

    Distinct from ``apply_mlx_vlm_mtp_patch``: that one only patches
    ``sanitize`` for conversion-time MTP preservation. This one builds
    the runtime infrastructure so VLMBatchedEngine can actually invoke
    the MTP head at inference time.

    Covers both Qwen3.5-MoE (qwen3_5_moe) and dense Qwen3.5/3.6
    (qwen3_5) VLM families. Each sub-patch tracks its own ``_APPLIED``
    flag, so calling repeatedly is cheap once both have settled.
    Returns True if at least one sub-patch applied successfully — a
    given model only needs whichever matches its model_type.

    Should be called *before* ``mlx_vlm.utils.load(...)`` so the
    instantiated LanguageModel picks up the patched ``__init__``.
    """
    from . import qwen35_moe_vlm_runtime, qwen35_vlm_runtime

    moe_ok = qwen35_moe_vlm_runtime.apply()
    if not moe_ok:
        logger.debug("Qwen3.5-MoE VLM runtime MTP patch did not apply")
    dense_ok = qwen35_vlm_runtime.apply()
    if not dense_ok:
        logger.debug("Qwen3.5 (dense) VLM runtime MTP patch did not apply")

    return moe_ok or dense_ok
