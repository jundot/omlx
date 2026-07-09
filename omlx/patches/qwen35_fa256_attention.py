# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5/3.6 head_dim=256 prefill attention to a steel FA kernel.

This patch is intentionally narrow:
  - Qwen3.5/3.6 dense VLM attention layout: q heads=24, kv heads=4, D=256
  - causal prefill/chunked-prefill only (q_len >= 16; decode and MTP verify
    stay on the stock path)
  - no array masks and no sinks

The native op uses MLX's steel attention template through the oMLX custom
kernel extension, so scores are never materialized and QK/PV use simdgroup MMA.
If the native extension is not present or the shape does not match, the prior
SDPA implementation is called unchanged.
"""

from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False

# Decode-shaped multi-row calls (MTP verify: q_len = 1 + draft depth <= 9)
# must not take the steel prefill kernel: it scans the full KV per call, so
# at tiny q_len it is 3-16x slower than the stock fused/unfused path and
# collapses long-context MTP throughput (issue #2127). Genuine prefill
# chunks are always far above this floor.
_MIN_ROUTE_Q_LEN = 16


def _native_kernel():
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return None
    if not fast.has_symbol("qwen35_fa256_attention"):
        return None
    return fast.qwen35_fa256_attention


def _has_quantized_cache(cache) -> bool:
    return cache is not None and hasattr(cache, "bits")


def _should_route(queries, keys, cache, mask, sinks, min_kv_len: int) -> bool:
    # Query-shape gates first: this runs on every SDPA call of every decode
    # step, so the common (decode / MTP verify) case must exit on the q_len
    # check before touching Metal state or cache attributes (issue #2132).
    # ``keys`` may be a TurboQuant state proxy without ``ndim``/``dtype``, so
    # it must not be inspected before the quantized-cache check declines.
    if queries.ndim != 4 or queries.shape[-2] < _MIN_ROUTE_Q_LEN:
        return False
    if not mx.metal.is_available() or _has_quantized_cache(cache):
        return False
    if sinks is not None:
        return False
    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return False
    if keys.ndim != 4:
        return False
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return False
    if queries.dtype != keys.dtype:
        return False

    q_heads = queries.shape[-3]
    kv_heads = keys.shape[-3]
    q_len = queries.shape[-2]
    kv_len = keys.shape[-2]
    head_dim = queries.shape[-1]
    return (
        head_dim == 256
        and q_heads == 24
        and kv_heads == 4
        and kv_len >= min_kv_len
        and q_len <= kv_len
    )


def apply_qwen35_fa256_attention_patch(min_kv_len: int | None = None) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OMLX_FA256_STEEL", "1") == "0":
        return False

    kernel = _native_kernel()
    if kernel is None:
        logger.debug("Qwen FA-256 steel kernel unavailable; patch skipped")
        return False

    min_kv_len = int(os.environ.get("OMLX_FA256_MIN_KV_LEN", min_kv_len or 2048))
    q_block = int(os.environ.get("OMLX_FA256_Q_BLOCK", "32"))
    k_block = int(os.environ.get("OMLX_FA256_K_BLOCK", "8"))
    debug = os.environ.get("OMLX_FA256_DEBUG", "0") == "1"

    patched_any = False

    try:
        from mlx_lm.models import base as mlx_base

        original_lm_sdpa = mlx_base.scaled_dot_product_attention

        def patched_lm_sdpa(
            queries,
            keys,
            values,
            cache,
            scale: float,
            mask: mx.array | None,
            sinks: mx.array | None = None,
        ) -> mx.array:
            routed = _should_route(queries, keys, cache, mask, sinks, min_kv_len)
            if debug:
                logger.info(
                    "fa256 steel lm route=%s q=%s k=%s mask=%s",
                    routed,
                    queries.shape,
                    keys.shape,
                    type(mask).__name__ if not isinstance(mask, str) else mask,
                )
            if routed:
                try:
                    return kernel(
                        queries,
                        keys,
                        values,
                        scale,
                        causal=True,
                        q_block=q_block,
                        k_block=k_block,
                    )
                except Exception:
                    logger.warning("fa256 steel lm kernel failed", exc_info=True)
            return original_lm_sdpa(queries, keys, values, cache, scale, mask, sinks)

        mlx_base.scaled_dot_product_attention = patched_lm_sdpa
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("mlx_lm.models."):
                continue
            if getattr(mod, "scaled_dot_product_attention", None) is original_lm_sdpa:
                mod.scaled_dot_product_attention = patched_lm_sdpa
        patched_any = True
    except ImportError:
        pass

    try:
        from mlx_vlm.models import base as vlm_base

        original_vlm_sdpa = getattr(vlm_base, "scaled_dot_product_attention", None)
        if original_vlm_sdpa is not None:

            def patched_vlm_sdpa(
                queries,
                keys,
                values,
                cache,
                scale: float,
                mask=None,
                sinks=None,
            ):
                routed = _should_route(queries, keys, cache, mask, sinks, min_kv_len)
                if debug:
                    logger.info(
                        "fa256 steel vlm route=%s q=%s k=%s mask=%s",
                        routed,
                        queries.shape,
                        keys.shape,
                        type(mask).__name__ if not isinstance(mask, str) else mask,
                    )
                if routed:
                    try:
                        return kernel(
                            queries,
                            keys,
                            values,
                            scale,
                            causal=True,
                            q_block=q_block,
                            k_block=k_block,
                        )
                    except Exception:
                        logger.warning("fa256 steel vlm kernel failed", exc_info=True)
                return original_vlm_sdpa(
                    queries, keys, values, cache, scale, mask, sinks
                )

            vlm_base.scaled_dot_product_attention = patched_vlm_sdpa
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or not mod_name.startswith("mlx_vlm.models."):
                    continue
                if (
                    getattr(mod, "scaled_dot_product_attention", None)
                    is original_vlm_sdpa
                ):
                    mod.scaled_dot_product_attention = patched_vlm_sdpa
            patched_any = True
    except ImportError:
        pass

    if patched_any:
        _PATCHED = True
        logger.info(
            "Qwen3.5/3.6 FA-256 steel attention patch applied "
            "(min_kv_len=%d, q_block=%d, k_block=%d)",
            min_kv_len,
            q_block,
            k_block,
        )
    return patched_any
