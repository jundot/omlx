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

    if _PATCHED:
        return

    try:
        from mlx_lm.models import cache as mlx_cache_mod
    except ImportError:
        logger.warning("mlx_lm.models.cache not importable — PlanarQuant hook skipped")
        return

    _ORIGINAL_MAKE_PROMPT_CACHE = mlx_cache_mod.make_prompt_cache

    def patched_make_prompt_cache(model, max_kv_size: int | None = None):
        original = _ORIGINAL_MAKE_PROMPT_CACHE(model, max_kv_size=max_kv_size)
        if _ACTIVE_BITS is None:
            return original
        return _wrap_cache_list(original, bits=_ACTIVE_BITS, quantize_v=_QUANTIZE_V)

    mlx_cache_mod.make_prompt_cache = patched_make_prompt_cache

    # Also patch every module that already did `from mlx_lm.models.cache import make_prompt_cache`
    # — Python's from-import captures the reference at import time, so a
    # module-attribute patch alone is invisible to those callers.
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name.startswith("mlx_lm.models.cache"):
            continue
        if hasattr(mod, "make_prompt_cache"):
            existing = getattr(mod, "make_prompt_cache")
            if existing is _ORIGINAL_MAKE_PROMPT_CACHE:
                setattr(mod, "make_prompt_cache", patched_make_prompt_cache)

    _PATCHED = True
    logger.info("PlanarQuant cache hook installed (%.1f-bit)", _ACTIVE_BITS)


def disable_planarquant_cache() -> None:
    """Disable the PlanarQuant cache hook globally."""
    global _ACTIVE_BITS, _PATCHED
    _ACTIVE_BITS = None
    if not _PATCHED:
        return
    try:
        from mlx_lm.models import cache as mlx_cache_mod

        if _ORIGINAL_MAKE_PROMPT_CACHE is not None:
            patched = mlx_cache_mod.make_prompt_cache
            mlx_cache_mod.make_prompt_cache = _ORIGINAL_MAKE_PROMPT_CACHE
            import sys

            for mod_name, mod in list(sys.modules.items()):
                if mod is None or mod_name.startswith("mlx_lm.models.cache"):
                    continue
                if getattr(mod, "make_prompt_cache", None) is patched:
                    mod.make_prompt_cache = _ORIGINAL_MAKE_PROMPT_CACHE
    except ImportError:
        pass
    _PATCHED = False


def is_planarquant_active() -> bool:
    return _ACTIVE_BITS is not None


def active_bits() -> float | None:
    return _ACTIVE_BITS


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
