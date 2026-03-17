# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
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

    if apply_gated_delta_advance_patch(model):
        logger.info("GatedDeltaNet advance() patch applied")

    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model
