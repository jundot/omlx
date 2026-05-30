# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Prefill (L>1): cache.prefill_attention() fast path, fallback to
    dequantize + mx.fast.scaled_dot_product_attention
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_RHT_DECODE_FIXED = False


def _fix_masked_decode_rht() -> None:
    """Work around mlx-vlm's L=1 value Metal kernels that ignore RHT (Bug 2).

    `_TurboQuantMSECodec.weighted_sum` and `weighted_sum_stats_from_scores` call
    the single-query (L=1) value kernels (`_metal_mse_weighted_sum`,
    `_metal_mse_weighted_sum_sum_from_scores`) WITHOUT the `if not self.use_rht`
    guard that the sibling `weighted_sum_from_scores` has. Those kernels undo the
    codec rotation with `matmul(., rotation)`, but TurboQuant KV codecs use
    `use_rht=True` (randomized Hadamard transform) whose inverse is
    `_rht_inverse` — so they corrupt the masked decode path (~140% error). That
    is what makes B>1 continuous-batching decode (which passes a per-request
    left-padding array mask) produce garbage.

    We disable those two kernels so the codec falls back to the correct einsum +
    `_rotate_inverse` path. Since our KV codecs are always `use_rht=True`, this is
    equivalent to the upstream `not self.use_rht` guard, at the cost of one matmul
    instead of a fused kernel on the slow decode path — negligible.

    Temporary until the upstream fix lands; see
    docs/upstream/mlx-vlm-turboquant-rht-decode-PR.md. (Bug 1, the fused
    single-token quantize kernel, is already fixed on the pinned mlx-vlm main.)
    """
    global _RHT_DECODE_FIXED
    if _RHT_DECODE_FIXED:
        return
    try:
        import mlx_vlm.turboquant as _tq
    except ImportError:
        return

    def _decline(*args, **kwargs):
        return None

    _tq._metal_mse_weighted_sum = _decline
    _tq._metal_mse_weighted_sum_sum_from_scores = _decline
    _RHT_DECODE_FIXED = True
    logger.info(
        "TurboQuant decode fix applied: disabled RHT-incompatible L=1 value "
        "kernels (mlx-vlm Bug 2 workaround)"
    )


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache
        from ..turboquant_kv import BatchTurboQuantKVCache

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if queries.shape[-2] == 1:
                # Decode (B=1 and B>1). With the masked decode path corrected by
                # _fix_masked_decode_rht(), continuous-batching decode using a
                # per-request left-padding array mask runs the quantized kernels
                # directly — no full-batch dequantize per step.
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            # Prefill: try quantized fast path, fallback to dequantize+SDPA
            result = real_cache.prefill_attention(
                queries, scale=scale, mask=mask,
            )
            if result is not None:
                return result
            dequantized_keys, dequantized_values = real_cache.dequantize()
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    # Without this, B>1 (masked) decode is corrupt and TurboQuant batching
    # produces garbage (see _fix_masked_decode_rht).
    _fix_masked_decode_rht()

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
