# SPDX-License-Identifier: Apache-2.0
"""Activation hook: replace standard KVCache with PlanarQuantKVCache.

When :func:`enable_planarquant_cache` is called, ``mlx_lm.models.cache.make_prompt_cache``
returns a list of :class:`PlanarQuantKVCache` instances instead of the default
``KVCache``. This is the counterpart to ``omlx/patches/turboquant_attention.py``
— the attention patch routes attention through PlanarQuant's decode/prefill
code path, and this patch ensures that per-layer caches are instantiated as
PlanarQuant types in the first place so the isinstance check matches.

Scope limitations (Stage 2 MVP):

* Applies only to layers whose default cache is a ``KVCache``. Models that
  override ``make_cache()`` and return ``RotatingKVCache`` / ``ChunkedKVCache`` /
  ``MambaCache`` are passed through unchanged.
* No-op for models whose head_dim is not a multiple of ``PLANAR_D`` (128).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVE_BITS: float | None = None
_QUANTIZE_V: bool = True
_ORIGINAL_MAKE_PROMPT_CACHE = None
_PATCHED = False
_PATCH_ATTR = "_omlx_planarquant_patched"
_ORIGINAL_ATTR = "_omlx_original_make_prompt_cache"
_MODEL_CONFIG_ATTR = "_omlx_planarquant_cache_config"
_MODEL_DISABLED_ATTR = "_omlx_planarquant_cache_disabled"


def enable_planarquant_cache(bits: float = 3.0, quantize_v: bool = True) -> None:
    """Activate the PlanarQuant cache hook globally.

    Args:
        bits: quantization bits (3.0 for PlanarQuant3).
        quantize_v: If False, V stays FP16 while only K is quantized.
            This gives zero PPL loss at 5.1x K-compression (upstream's best config).
    """
    global _ACTIVE_BITS, _QUANTIZE_V, _ORIGINAL_MAKE_PROMPT_CACHE, _PATCHED

    _ACTIVE_BITS = float(bits)
    _QUANTIZE_V = quantize_v

    try:
        from ..cache.planarquant.metal_kernels import warm_planarquant_kernels

        warm_planarquant_kernels()
    except Exception:
        logger.debug("PlanarQuant kernel warmup unavailable", exc_info=True)

    if _PATCHED:
        return

    try:
        from mlx_lm.models import cache as mlx_cache_mod
    except ImportError:
        logger.warning("mlx_lm.models.cache not importable — PlanarQuant hook skipped")
        return

    current_make_prompt_cache = mlx_cache_mod.make_prompt_cache
    _ORIGINAL_MAKE_PROMPT_CACHE = getattr(
        current_make_prompt_cache, _ORIGINAL_ATTR, current_make_prompt_cache
    )

    def patched_make_prompt_cache(model, max_kv_size: int | None = None):
        original = _ORIGINAL_MAKE_PROMPT_CACHE(model, max_kv_size=max_kv_size)
        if getattr(model, _MODEL_DISABLED_ATTR, False):
            return original
        model_config = getattr(model, _MODEL_CONFIG_ATTR, None)
        if model_config is not None:
            model_bits, model_quantize_v = model_config
            return _wrap_cache_list(
                original, bits=float(model_bits), quantize_v=bool(model_quantize_v)
            )
        if _ACTIVE_BITS is None:
            return original
        return _wrap_cache_list(original, bits=_ACTIVE_BITS, quantize_v=_QUANTIZE_V)

    setattr(patched_make_prompt_cache, _PATCH_ATTR, True)
    setattr(patched_make_prompt_cache, _ORIGINAL_ATTR, _ORIGINAL_MAKE_PROMPT_CACHE)
    mlx_cache_mod.make_prompt_cache = patched_make_prompt_cache

    # Also patch every module that already did `from mlx_lm.models.cache import make_prompt_cache`
    # — Python's from-import captures the reference at import time, so a
    # module-attribute patch alone is invisible to those callers.
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name.startswith("mlx_lm.models.cache"):
            continue
        existing = vars(mod).get("make_prompt_cache")
        if existing is _ORIGINAL_MAKE_PROMPT_CACHE:
            setattr(mod, "make_prompt_cache", patched_make_prompt_cache)

    _PATCHED = True
    logger.info("PlanarQuant cache hook installed (%.1f-bit)", _ACTIVE_BITS)


def disable_planarquant_cache() -> None:
    """Disable the PlanarQuant cache hook globally."""
    global _ACTIVE_BITS, _PATCHED
    _ACTIVE_BITS = None
    try:
        from mlx_lm.models import cache as mlx_cache_mod

        patched = mlx_cache_mod.make_prompt_cache
        original = getattr(patched, _ORIGINAL_ATTR, _ORIGINAL_MAKE_PROMPT_CACHE)
        if getattr(patched, _PATCH_ATTR, False) and original is not None:
            mlx_cache_mod.make_prompt_cache = original
            import sys

            for mod_name, mod in list(sys.modules.items()):
                if mod is None or mod_name.startswith("mlx_lm.models.cache"):
                    continue
                if vars(mod).get("make_prompt_cache") is patched:
                    mod.make_prompt_cache = original
    except ImportError:
        pass
    _PATCHED = False


def is_planarquant_active() -> bool:
    return _ACTIVE_BITS is not None


def active_bits() -> float | None:
    return _ACTIVE_BITS


def mark_model_for_planarquant(
    model: Any, bits: float = 3.0, quantize_v: bool = True
) -> Any:
    setattr(model, _MODEL_CONFIG_ATTR, (float(bits), bool(quantize_v)))
    if hasattr(model, _MODEL_DISABLED_ATTR):
        delattr(model, _MODEL_DISABLED_ATTR)
    return model


def mark_model_without_planarquant(model: Any) -> Any:
    if hasattr(model, _MODEL_CONFIG_ATTR):
        delattr(model, _MODEL_CONFIG_ATTR)
    setattr(model, _MODEL_DISABLED_ATTR, True)
    return model


def _wrap_cache_list(cache_list: list[Any], bits: float, quantize_v: bool = True) -> list[Any]:
    """Replace each ``KVCache`` in ``cache_list`` with a ``PlanarQuantKVCache``."""
    from mlx_lm.models.cache import KVCache

    from ..cache.planarquant.kv_cache import PlanarQuantKVCache

    wrapped: list[Any] = []
    replaced = 0
    for entry in cache_list:
        if type(entry) is KVCache:
            wrapped.append(PlanarQuantKVCache(bits=bits, quantize_v=quantize_v))
            replaced += 1
        else:
            wrapped.append(entry)
    if replaced > 0:
        logger.debug(
            "PlanarQuant: wrapped %d/%d cache layers (quantize_v=%s)", replaced, len(cache_list), quantize_v
        )
    return wrapped
