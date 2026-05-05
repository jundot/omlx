# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def maybe_apply_pre_load_patches(model_name: str) -> None:
    """Apply patches that need to run *before* mlx_lm.load() runs.

    Currently dispatches the DeepSeek V4 patch (PR 1192) when
    ``config.json`` declares ``model_type == "deepseek_v4"``. The patch
    injects new modules into ``sys.modules`` and replaces a couple of
    mlx-lm internals; gating on model type keeps other models at zero
    cost.

    Safe to call repeatedly; the patches are idempotent.
    """
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        # HF repo IDs and other non-local paths fall through. mlx_lm.load
        # downloads the config first and we'll rely on that path being
        # taken before anyone calls a deepseek_v4 model anyway. Safe
        # default: do nothing here.
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


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name)
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
