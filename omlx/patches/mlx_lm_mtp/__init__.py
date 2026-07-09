# SPDX-License-Identifier: Apache-2.0
"""Native MTP (Multi-Token Prediction) monkey-patches for mlx-lm.

This package adapts two upstream PRs into runtime monkey-patches:

- ml-explore/mlx-lm#990 — Qwen3.5 / Qwen3.6 native MTP heads (dense + MoE)
- Blaizzy/mlx-lm#15    — DeepSeek-V4-Flash native MTP heads

Both PRs follow the same shape: a model gains an extra ``mtp`` module + a
``mtp_forward`` method and an enhanced ``__call__`` that returns hidden
states alongside logits. A separate ``mtp_generate_step`` generator drives
the draft/verify loop using those hooks.

This package implements the model-side hooks as in-place monkey-patches and
folds the draft/verify loop into mlx-lm's ``GenerationBatch.next()`` so the
existing oMLX paged + prefix + SSD cache stack keeps working unchanged.

Activation gate: caller (utils/model_loading.py) checks
``model_settings.mtp_enabled`` and the model's ``config.json`` for MTP
heads + a supported ``model_type`` before invoking ``apply_mlx_lm_mtp_patch``.
The patches are idempotent.

Concurrency model: the BatchGenerator patch uses the singleton MTP path for
one active sequence, and a row-wise MTP controller for multi-sequence batches
only when every row is at the same target cache position. Late-join or otherwise
unaligned batches fall through to standard continuous batching.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Process-wide construction flag read by the patched ``Model.__init__``
# (Qwen3.5/3.6 + DeepSeek-V4) to decide whether to attach the MTP head module.
# Caller (``utils/model_loading.py::maybe_apply_pre_load_patches``) sets this
# right before ``mlx_lm.load()`` runs based on ``model_settings.mtp_enabled``.
# Decode-time eligibility is stored on each loaded model instance instead.
# Default False keeps newly-loaded models MTP-free unless explicitly opted in.
_MTP_ACTIVE = False


def set_mtp_active(active: bool) -> None:
    """Toggle whether subsequent ``mlx_lm.load()`` calls attach the MTP head.

    Affects ``self.mtp`` attachment in patched ``Model.__init__`` (and
    DeepSeek-V4 equivalent). Patched model instances persist the load-time
    decode decision on ``_omlx_mtp_decode_enabled``; BatchGenerator reads that
    per-instance marker so later model loads cannot change existing models.
    Single-thread MLX executor serializes loads, so the construction flag is
    race-free.
    """
    global _MTP_ACTIVE
    _MTP_ACTIVE = bool(active)


def is_mtp_active() -> bool:
    return _MTP_ACTIVE


# Draft depth (number of chained MTP draft tokens per verify cycle) for the
# next model load. Same construction-time-flag pattern as _MTP_ACTIVE: the
# patched ``TextModel.__init__`` copies it onto the instance
# (``_omlx_mtp_depth``) so decode never reads the global. Depth > 1 only
# engages on models whose patch marks ``_omlx_mtp_chain`` (Qwen3.5/3.6);
# DeepSeek-V4 stays on the depth-1 legacy cycle.
_MTP_DEPTH = 1


def set_mtp_depth(depth: int) -> None:
    global _MTP_DEPTH
    _MTP_DEPTH = max(1, min(8, int(depth)))


def get_mtp_depth() -> int:
    return _MTP_DEPTH


def apply_mlx_lm_mtp_patch() -> bool:
    """Apply the model-side and BatchGenerator monkey-patches.

    Self-healing: re-runs each sub-patch every call. Sub-patches use
    marker-based identity checks on the live class state, so a no-op
    when our patch is already installed and a re-apply when something
    else (e.g. dflash hooks) has overwritten ``__call__`` since the
    last load. Without this, a sequence like (dflash → mtp) on the
    same process leaves the linear-attention class patched by dflash
    and the MTP draft cycle crashes with a TypeError on n_confirmed
    (issue #1388).

    Must be invoked before ``mlx_lm.load()`` so the patched
    ``__init__`` / ``sanitize`` / ``from_dict`` paths see MTP weights.

    Returns:
        True if the patch is now active. False if a sub-step refused
        to apply (mlx-lm not importable, missing prerequisite patch).
    """
    from . import (
        batch_generator,
        cache_rollback,
        deepseek_v4_model,
        glm_moe_dsa_model,
        qwen35_model,
    )

    if not cache_rollback.apply():
        return False
    if not qwen35_model.apply():
        # Qwen models are the main target; if the qwen patch refuses we
        # still continue so DeepSeek-V4 users aren't blocked.
        logger.debug("Qwen3.5/3.6 MTP patch did not apply (likely import error)")
    if not deepseek_v4_model.apply():
        logger.debug("DeepSeek-V4 MTP patch did not apply (likely missing base patch)")
    if not glm_moe_dsa_model.apply():
        logger.debug("GLM-5.2 MTP patch did not apply (likely missing base patch)")
    if not batch_generator.apply():
        logger.warning(
            "BatchGenerator MTP dispatch patch failed; MTP path will be inactive"
        )
        return False

    # Verify-shape qmm kernels. Optional: inert unless armed around an MTP
    # verify forward, and strictly shape-gated at call time.
    try:
        from ..qwen35_verify_qmm import apply_verify_qmm_patch

        apply_verify_qmm_patch()
    except Exception:
        logger.debug("verify qmm patch not applied", exc_info=True)

    return True
