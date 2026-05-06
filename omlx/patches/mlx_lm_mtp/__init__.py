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

Concurrency model: the BatchGenerator patch only takes the MTP path when
exactly one sequence is active in the generation batch. Concurrent requests
fall through to the standard continuous-batching path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False

# Process-wide flag read by the patched ``Model.__init__`` (Qwen3.5/3.6 +
# DeepSeek-V4) to decide whether to attach the MTP head module. Caller
# (``utils/model_loading.py::maybe_apply_pre_load_patches``) sets this
# right before ``mlx_lm.load()`` runs based on ``model_settings.mtp_enabled``.
# Default False keeps newly-loaded models MTP-free unless explicitly opted in.
_MTP_ACTIVE = False


def set_mtp_active(active: bool) -> None:
    """Toggle whether subsequent ``mlx_lm.load()`` calls attach the MTP head.

    Affects ``self.mtp`` attachment in patched ``Model.__init__`` (and
    DeepSeek-V4 equivalent) and is checked by BatchGenerator's
    ``_is_mtp_eligible`` (via the presence of the ``mtp`` attribute).
    Single-thread MLX executor serializes loads, so this is race-free.
    """
    global _MTP_ACTIVE
    _MTP_ACTIVE = bool(active)


def is_mtp_active() -> bool:
    return _MTP_ACTIVE


def apply_mlx_lm_mtp_patch() -> bool:
    """Apply the model-side and BatchGenerator monkey-patches.

    Idempotent. Must be invoked before ``mlx_lm.load()`` so the patched
    ``__init__`` / ``sanitize`` / ``from_dict`` paths see MTP weights.

    Returns:
        True if the patch is now active (newly applied or already applied
        on a previous call). False if a sub-step refused to apply (mlx-lm
        not importable, missing prerequisite patch, etc.).
    """
    global _PATCHED
    if _PATCHED:
        return True

    from . import batch_generator, cache_rollback, deepseek_v4_model, qwen35_model

    if not cache_rollback.apply():
        return False
    if not qwen35_model.apply():
        # Qwen models are the main target; if the qwen patch refuses we
        # still continue so DeepSeek-V4 users aren't blocked.
        logger.debug("Qwen3.5/3.6 MTP patch did not apply (likely import error)")
    if not deepseek_v4_model.apply():
        logger.debug("DeepSeek-V4 MTP patch did not apply (likely missing base patch)")
    if not batch_generator.apply():
        logger.warning(
            "BatchGenerator MTP dispatch patch failed; MTP path will be inactive"
        )
        return False

    _PATCHED = True
    logger.info("mlx-lm MTP patch applied (PR 990 + PR 15)")
    return True
