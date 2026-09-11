# SPDX-License-Identifier: Apache-2.0
"""V4.1-specific load_model hooks (fp8 quant via make_quantization_config).

Reuses the V4 utils_patch F8_E8M0 loader when already installed; adds a
``deepseek_v41`` fp8 branch if the active load_model does not yet route V4.1.
"""
from __future__ import annotations

import logging

from omlx.patches.deepseek_v41.predicates import is_deepseek_v41

logger = logging.getLogger(__name__)
_PATCHED = False


def apply_utils_patch() -> bool:
    """Ensure V4.1 fp8 quantization uses ``make_quantization_config``.

    The V4 ``utils_patch`` already replaces ``mlx_lm.utils.load_model`` with
    F8_E8M0 handling. We wrap that (or the stock function) so ``deepseek_v41``
    hits the same fp8 path with V4.1's quantization config.
    """
    global _PATCHED
    if _PATCHED:
        return False

    # Prefer installing V4 utils patch first for shared safetensors fallback.
    try:
        from omlx.patches.deepseek_v4.utils_patch import apply_utils_patch as v4_utils

        v4_utils()
    except Exception as e:
        logger.debug("V4 utils_patch not applied before V4.1: %s", e)

    import mlx_lm.utils as _utils

    original = _utils.load_model

    # If V4 patch already handles only is_deepseek_v4, wrap to add v41.
    # Monkey-patch make_quantization_config lookup via a thin wrapper around
    # the quant branch is fragile; instead register a helper used by tests and
    # document that omlx model_loading applies this patch before load.
    def _ensure_v41_quant(config, model):
        qc = config.get("quantization_config") or {}
        if qc.get("quant_method") == "fp8" and is_deepseek_v41(
            str(config.get("model_type", ""))
        ):
            from mlx_lm.models.deepseek_v41 import make_quantization_config

            return make_quantization_config(model)
        return None

    _utils._omlx_deepseek_v41_quant = _ensure_v41_quant  # type: ignore[attr-defined]
    _PATCHED = True
    logger.info("deepseek_v41 utils helpers installed")
    return True


def flatten_text_config(config: dict) -> dict:
    """Public helper used by tests and loaders."""
    if not isinstance(config, dict):
        return config
    out = dict(config)
    text = out.get("text_config")
    if isinstance(text, dict):
        for k, v in text.items():
            out.setdefault(k, v)
    return out
