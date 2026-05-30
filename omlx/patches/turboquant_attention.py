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
_DECODE_QUANT_FIXED = False


def _fix_decode_single_token_quantize() -> None:
    """Disable mlx-vlm's broken fused single-token KV-quantize kernel.

    mlx-vlm's TurboQuantKVCache._try_fused_kv_quantize takes a fused Metal
    kernel path ONLY when keys.shape[-2] == 1 — i.e. exactly the decode step.
    In the pinned mlx-vlm (f96138e) that kernel is broken: it produces ~140%
    reconstruction error on the appended token at every bit depth, while the
    non-fused codec.quantize() path used for T>=2 (prefill) is correct. The
    result is garbage generation once TurboQuant decode is actually engaged.

    Forcing _try_fused_kv_quantize to decline (return (None, None)) routes T=1
    through the correct non-fused path. Cost: one extra Metal dispatch per
    decode step (separate K and V quantize) — negligible. Forward-compatible:
    if upstream fixes the kernel this only loses the fused micro-optimization.

    NOTE: fixed on mlx-vlm main (fea81522) but not in our pinned f96138e nor
    the v0.5.0 release tag — drop this workaround once the pin bumps past the
    fix. Bug #2 (the masked decode path) is still broken on main; see the B>1
    dequantize+SDPA route in apply_turboquant_attention_patch().
    """
    global _DECODE_QUANT_FIXED
    if _DECODE_QUANT_FIXED:
        return
    try:
        from mlx_vlm.turboquant import TurboQuantKVCache
    except ImportError:
        return

    def _decline_fused_kv_quantize(self, keys, values):
        return None, None

    TurboQuantKVCache._try_fused_kv_quantize = _decline_fused_kv_quantize
    _DECODE_QUANT_FIXED = True
    logger.info(
        "TurboQuant decode fix applied: disabled broken fused single-token "
        "quantize kernel (mlx-vlm f96138e)"
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
                # Continuous-batching decode (B>1) passes an array mask for
                # per-request left-padding. mlx-vlm f96138e's masked
                # decode_attention path is broken (~140% error), so route the
                # array-mask case through dequantize + standard SDPA — the same
                # approach mlx-vlm uses for its own BatchTurboQuantKVCache.
                # B=1 keeps mask=None/"causal" and the correct fused kernel.
                if isinstance(mask, mx.array):
                    dq_keys, dq_values = real_cache.dequantize(keys, values)
                    return mx.fast.scaled_dot_product_attention(
                        queries,
                        dq_keys.astype(queries.dtype),
                        dq_values.astype(queries.dtype),
                        scale=scale,
                        mask=mask,
                    )
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

    # Without this, decode-step KV quantization is corrupt and TurboQuant
    # produces garbage even at 8-bit (see _fix_decode_single_token_quantize).
    _fix_decode_single_token_quantize()

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
