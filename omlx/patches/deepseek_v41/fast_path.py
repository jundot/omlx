# SPDX-License-Identifier: Apache-2.0
"""Opt-in fast DeepSeek-V4.1 path (our SwitchGLU + wsdpa + deferred HC stack).

Enable with ``OMLX_DSV41_FAST=1``. Uses mlx_lm.load of ``deepseek_v41_model.py``
instead of the #3574 language.py reference port. Engram via
``OMLX_DSV41_ENGRAM=stub|mmap|full``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_APPLIED = False


def fast_path_enabled() -> bool:
    return os.environ.get("OMLX_DSV41_FAST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _inject_pooling_cache() -> None:
    import mlx_lm.models.cache as cache_mod

    from omlx.patches.deepseek_v4 import cache_extras as extras

    if not hasattr(cache_mod, "PoolingCache"):
        cache_mod.PoolingCache = extras.PoolingCache
        cache_mod.__dict__["PoolingCache"] = extras.PoolingCache
        extras.PoolingCache.__module__ = "mlx_lm.models.cache"
    if not hasattr(cache_mod, "BatchPoolingCache"):
        cache_mod.BatchPoolingCache = extras.BatchPoolingCache
        cache_mod.__dict__["BatchPoolingCache"] = extras.BatchPoolingCache
        extras.BatchPoolingCache.__module__ = "mlx_lm.models.cache"
    try:
        from omlx.patches.deepseek_v4.generate_patch import apply_generate_patch

        apply_generate_patch()
    except Exception as e:
        logger.warning("generate_patch skipped: %s", e)


def _register_module(qualname: str, file_path: Path, package: str) -> None:
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    logger.info("Registered fast %s from %s", qualname, file_path.name)


def apply_fast_path() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    os.environ.setdefault("OMLX_DSV41_ENGRAM", "mmap")
    _inject_pooling_cache()
    v4 = Path(__file__).resolve().parent.parent / "deepseek_v4"
    _register_module(
        "mlx_lm.models.hyper_connection",
        v4 / "hyper_connection.py",
        "mlx_lm.models",
    )
    # utils_patch may import deepseek_v4 for make_quantization_config
    if (v4 / "deepseek_v4_model.py").exists():
        _register_module(
            "mlx_lm.models.deepseek_v4",
            v4 / "deepseek_v4_model.py",
            "mlx_lm.models",
        )
    here = Path(__file__).parent
    _register_module(
        "mlx_lm.models.deepseek_v41",
        here / "deepseek_v41_model.py",
        "mlx_lm.models",
    )
    try:
        import mlx_lm.utils as _utils

        remapping = getattr(_utils, "MODEL_REMAPPING", None)
        if isinstance(remapping, dict):
            remapping.setdefault("deepseek_v41", "deepseek_v41")
            remapping.setdefault("deepseek_v41_text", "deepseek_v41")
    except Exception as e:
        logger.debug("MODEL_REMAPPING skip: %s", e)
    for name, attr in (
        ("utils_patch", "apply_utils_patch"),
        ("tokenizer_patch", "apply_tokenizer_patch"),
    ):
        try:
            mod = __import__(f"omlx.patches.deepseek_v41.{name}", fromlist=[attr])
            getattr(mod, attr)()
        except Exception as e:
            logger.warning("fast path %s skipped: %s", name, e)
    _APPLIED = True
    logger.info("DeepSeek-V4.1 FAST path applied (OMLX_DSV41_FAST)")
    return True
