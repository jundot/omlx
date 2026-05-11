# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def maybe_apply_pre_load_patches(
    model_name: str,
    model_settings: Any | None = None,
) -> None:
    """Apply patches that need to run *before* mlx_lm.load() runs.

    Dispatches:

    - DeepSeek V4 patch (PR 1192) when ``config.json`` declares
      ``model_type == "deepseek_v4"``.
    - Native MTP patch (PR 990 + PR 15) when ``model_settings.mtp_enabled``
      is True AND the config declares MTP heads on a supported model_type.

    Both patches inject modules into ``sys.modules`` and replace mlx-lm
    internals; gating keeps non-affected models at zero cost.

    Safe to call repeatedly; the patches are idempotent.
    """
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for pre-load patch dispatch: %s", config_path, e
        )
        return

    model_type = config.get("model_type")
    if model_type == "deepseek_v4":
        from ..patches.deepseek_v4 import apply_deepseek_v4_patch

        if apply_deepseek_v4_patch():
            logger.info("DeepSeek V4 pre-load patch applied for %s", model_name)

    # Apply the MTP patch whenever the model has MTP heads on a compatible
    # model_type — even when mtp_enabled is False. The patch is required
    # for *sanitize correctness*: stock mlx-lm Model.sanitize triggers a
    # +1 norm shift whenever it sees mtp.* keys (assuming a raw HF
    # checkpoint), which double-shifts an already-converted MLX model and
    # corrupts the output (garbage tokens). PR 990's sanitize gates the
    # shift on "unsanitized conv1d" instead.
    #
    # Whether the model actually attaches an MTP head — and therefore
    # whether BatchGenerator runs the MTP draft+verify cycle — is gated
    # by a process-wide flag set just before mlx_lm.load() runs. With
    # mtp_enabled=False the patch is still active so sanitize behaves
    # correctly, but Model.__init__ skips ``self.mtp = MTPModule(args)``;
    # the resulting model is indistinguishable from a stock model that
    # never had MTP heads.
    if _is_mtp_compatible(config, model_type):
        mtp_enabled = bool(
            model_settings is not None
            and getattr(model_settings, "mtp_enabled", False)
        )
        from ..patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active

        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)
            if mtp_enabled:
                logger.info(
                    "Native MTP patch applied for %s (model_type=%s, active)",
                    model_name,
                    model_type,
                )
            else:
                logger.debug(
                    "Native MTP patch applied for %s for sanitize correctness "
                    "(model has MTP heads but mtp_enabled=False; head not attached)",
                    model_name,
                )

        # mlx-vlm side: when the model loads via VLMBatchedEngine
        # (e.g. ``qwen3_5_moe`` with vision_config), the mlx-lm patch
        # alone can't attach an MTP head to the mlx-vlm classes.
        # Apply the parallel runtime patch on mlx-vlm so the MTPModule is
        # instantiated on ``LanguageModel.__init__``.
        if mtp_enabled:
            try:
                from ..patches.mlx_vlm_mtp import (
                    apply_mlx_vlm_mtp_runtime_patch,
                )
            except Exception:
                pass
            else:
                if apply_mlx_vlm_mtp_runtime_patch():
                    logger.info(
                        "mlx-vlm runtime MTP patch applied for %s",
                        model_name,
                    )
    elif (
        model_settings is not None
        and getattr(model_settings, "mtp_enabled", False)
    ):
        logger.warning(
            "mtp_enabled=True for %s but model is incompatible "
            "(model_type=%r, mtp_heads=%s); MTP path will be inactive",
            model_name,
            model_type,
            _has_mtp_heads(config),
        )


def _has_mtp_heads(config: dict) -> bool:
    """True iff the model config declares any MTP head layers."""
    if int(config.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(config.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    text_cfg = config.get("text_config") or {}
    if int(text_cfg.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(text_cfg.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    return False


def _is_mtp_compatible(config: dict, model_type: str | None) -> bool:
    """Decide whether the native MTP patch can be applied to this model.

    Phase 1 supports Qwen3.5/3.6 (mlx-lm PR 990) and DeepSeek-V4-Flash
    (Blaizzy/mlx-lm fork PR 15). The model also has to declare MTP heads
    in the config; otherwise the patch is a no-op.
    """
    if not _has_mtp_heads(config):
        return False
    if not model_type:
        return False
    return (
        model_type.startswith("qwen3_5")
        or model_type.startswith("qwen3_6")
        or model_type.startswith("deepseek_v4")
    )


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_settings: Any | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    from mlx_lm import load

    return load(model_name, tokenizer_config=tokenizer_config)


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    """Apply optional post-load model transforms based on settings.

    Currently supports:
    - IndexCache: skip redundant indexer computation in DSA layers
    - GatedDeltaNet advance: fix missing cache.advance() in qwen3_5

    Args:
        model: A loaded mlx-lm model instance.
        model_settings: A ModelSettings instance (or None).

    Returns:
        The (possibly patched) model.
    """
    # GatedDeltaNet advance patch: always applied for qwen3_5 models
    # (no settings needed — auto-detected by model type)
    from ..patches.gated_delta_advance import apply_gated_delta_advance_patch
    from ..patches.qwen3_5_attention import apply_qwen3_5_attention_patch

    if apply_gated_delta_advance_patch(model):
        logger.info("GatedDeltaNet advance() patch applied")
    if apply_qwen3_5_attention_patch(model):
        logger.info("Qwen3_5Attention plain-rope patch applied")

    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model
