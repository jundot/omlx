# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 (CSA2) monkey-patch for mlx-lm.

Registers ``mlx_lm.models.deepseek_v41`` from this package so omlx can load
``deepseek-ai/DeepSeek-V4.1-Flash`` (``model_type=deepseek_v41``).

Architecture notes (see DeepSeek_V41_Tech_Report.pdf + inference/model.py):
- Compressed Sparse Attention 2 (CSA2) with cross-layer KV / indexer reuse
- Hierarchical Sparse Indexer + candidate block prefilter
- compress_ratios in {0,1,2} (no V4 ratio-4/128 / APE / overlap)
- Deferred single-pass mHC (attn uses prior FFN pre_mix)
- Engram conditional memory; DSpark draft head; optional vision

Critical: must NOT share the naive ``startswith("deepseek_v4")`` gate with
the V4 patch — that predicate matches ``deepseek_v41``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from .predicates import is_deepseek_v41

logger = logging.getLogger(__name__)

_APPLIED = False


def is_applied() -> bool:
    return _APPLIED


def _register_module(qualname: str, file_name: str) -> None:
    if qualname in sys.modules:
        return
    here = Path(__file__).parent
    file_path = here / file_name
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    logger.info("Registered %s from %s", qualname, file_path.name)


def _register_model_type_aliases() -> None:
    try:
        import mlx_lm.utils as _utils
    except Exception as e:  # pragma: no cover
        logger.debug("mlx_lm.utils unavailable for V4.1 alias: %s", e)
        return
    remapping = getattr(_utils, "MODEL_REMAPPING", None)
    if not isinstance(remapping, dict):
        return
    remapping.setdefault("deepseek_v41", "deepseek_v41")
    remapping.setdefault("deepseek_v41_text", "deepseek_v41")
    remapping.setdefault("deepseek_v41_mtp", "deepseek_v41")


def apply_deepseek_v41_patch() -> bool:
    """Idempotent. Returns True if this call performed the registration."""
    global _APPLIED
    if _APPLIED:
        return False

    # Reuse V4 PoolingCache injection / handlers when present; V4.1 also needs
    # shared-pool cache types but PoolingCache remains useful for ratio-1/2.
    try:
        from omlx.patches.deepseek_v4 import apply_pooling_cache_support

        apply_pooling_cache_support()
    except Exception as e:
        logger.warning("V4.1: PoolingCache support not applied: %s", e)


    # HyperConnection Sinkhorn kernels — shared with V4.
    hc_path = Path(__file__).resolve().parent.parent / "deepseek_v4" / "hyper_connection.py"
    if "mlx_lm.models.hyper_connection" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "mlx_lm.models.hyper_connection", str(hc_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load hyper_connection from {hc_path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "mlx_lm.models"
        sys.modules["mlx_lm.models.hyper_connection"] = module
        spec.loader.exec_module(module)
        logger.info("Registered mlx_lm.models.hyper_connection from deepseek_v4")

    _register_module("mlx_lm.models.deepseek_v41", "deepseek_v41_model.py")
    _register_model_type_aliases()

    try:
        from .utils_patch import apply_utils_patch

        apply_utils_patch()
    except Exception as e:
        logger.warning("V4.1 utils_patch skipped: %s", e)

    try:
        from .tokenizer_patch import apply_tokenizer_patch

        apply_tokenizer_patch()
    except Exception as e:
        logger.warning("V4.1 tokenizer_patch skipped: %s", e)

    _APPLIED = True
    logger.info("DeepSeek-V4.1 patch applied")
    return True


def maybe_apply_for_config(config: dict | None) -> bool:
    if not isinstance(config, dict):
        return False
    mt = config.get("model_type")
    if mt is None and isinstance(config.get("text_config"), dict):
        mt = config["text_config"].get("model_type")
    if is_deepseek_v41(mt):
        return apply_deepseek_v41_patch()
    return False


__all__ = [
    "apply_deepseek_v41_patch",
    "is_applied",
    "is_deepseek_v41",
    "maybe_apply_for_config",
]
