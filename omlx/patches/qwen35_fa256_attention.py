# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5/3.6 head_dim=256 prefill attention to a steel FA kernel.

This patch is intentionally narrow:
  - Qwen3.5/3.6 head_dim=256 GQA attention (dense 24/4, MoE 16/2, and any
    other q%kv==0 layout; gqa_factor is a runtime kernel param)
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
import time

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available
from omlx.custom_kernels.qwen35_prefill import fast as _fa256_fast

logger = logging.getLogger(__name__)

_PATCHED = False

# Decode-shaped multi-row calls (MTP verify: q_len = 1 + draft depth <= 9)
# must not take the steel prefill kernel: it scans the full KV per call, so
# at tiny q_len it is 3-16x slower than the stock fused/unfused path and
# collapses long-context MTP throughput (issue #2127). Genuine prefill
# chunks are always far above this floor.
_MIN_ROUTE_Q_LEN = 16

# Per-dispatch work ceiling (batch * heads * q_len * keys) for the native
# kernel. A single dispatch scanning the whole KV monopolizes the GPU long
# enough at high kv_len to trip the macOS IOGPU interactivity preemption on
# pre-NAX GPUs, collapsing long-context prefill (issue #2225; the M3 Max
# report cliffs near 24 heads * 2048 q * 30k keys ~= 1.5e9). The kernel
# splits the key axis into separately dispatched chunks above this budget.
# This fixed value (~6x below that cliff) is only the fallback when
# calibration fails; 0 disables chunking (pre-#2225 single-dispatch
# behavior, benchmarking only).
_DEFAULT_DISPATCH_BUDGET = 250_000_000

# The preemption threshold is wallclock-based, so a fixed work budget leaves
# uneven margins across GPU tiers (the same work runs ~5-8x longer on an M1
# than on an M3 Ultra). When OMLX_FA256_DISPATCH_BUDGET is unset, the patch
# measures the kernel's own throughput once and sizes the budget so one
# chunk dispatch targets this wallclock; the M3 Max report puts the cliff
# above ~50ms per dispatch, so 10ms keeps a wide margin on any tier while
# chunks stay far too coarse to cost throughput. The calibration shape is
# small and eval overhead inflates its measured time, so the derived budget
# errs low (real dispatches run shorter than the target) — the safe side.
_TARGET_DISPATCH_SECONDS = 0.010
_CALIB_HEADS = 16
_CALIB_KV_HEADS = 2
_CALIB_Q_LEN = 1024
_CALIB_KV_LEN = 8192
# Clamp for calibration outliers (timer glitches, busy GPU at startup).
# Bounds correspond to ~10ms dispatches on a GPU ~8x slower / ~13x faster
# than an M3 Ultra (measured 1.55e10 work/s at the calibration shape).
_MIN_AUTO_BUDGET = 20_000_000
_MAX_AUTO_BUDGET = 2_000_000_000


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


def _auto_dispatch_budget(kernel, q_block: int, k_block: int) -> int:
    """Size the dispatch budget from the kernel's measured throughput.

    Runs the native kernel once as a single dispatch on a small synthetic
    shape (one warmup for pipeline compilation, then best of three timed
    runs) and converts the throughput into the work that fits in
    ``_TARGET_DISPATCH_SECONDS``. Falls back to the fixed default when
    measurement fails."""
    try:
        work = _CALIB_HEADS * _CALIB_Q_LEN * _CALIB_KV_LEN
        q = mx.random.normal((1, _CALIB_HEADS, _CALIB_Q_LEN, 256)).astype(
            mx.bfloat16
        )
        k = mx.random.normal((1, _CALIB_KV_HEADS, _CALIB_KV_LEN, 256)).astype(
            mx.bfloat16
        )
        v = mx.random.normal((1, _CALIB_KV_HEADS, _CALIB_KV_LEN, 256)).astype(
            mx.bfloat16
        )
        mx.eval(q, k, v)
        best = 0.0
        for i in range(4):
            start = time.perf_counter()
            mx.eval(
                kernel(
                    q,
                    k,
                    v,
                    256**-0.5,
                    causal=True,
                    q_block=q_block,
                    k_block=k_block,
                    dispatch_budget=0,
                )
            )
            elapsed = time.perf_counter() - start
            if i > 0:
                best = elapsed if best == 0.0 else min(best, elapsed)
        if best <= 0:
            raise ValueError(f"non-positive calibration time {best}")
        budget = int(work / best * _TARGET_DISPATCH_SECONDS)
        budget = max(_MIN_AUTO_BUDGET, min(_MAX_AUTO_BUDGET, budget))
        logger.info(
            "Qwen FA-256 dispatch budget auto-calibrated: %.2e work/s -> "
            "%d (target %dms/dispatch)",
            work / best,
            budget,
            int(_TARGET_DISPATCH_SECONDS * 1000),
        )
        return budget
    except Exception:
        logger.warning(
            "Qwen FA-256 dispatch budget calibration failed; using default %d",
            _DEFAULT_DISPATCH_BUDGET,
            exc_info=True,
        )
        return _DEFAULT_DISPATCH_BUDGET


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
    # Any D=256 GQA layout, not just the dense 24/4 one: the steel kernel
    # takes gqa_factor as a runtime param, and the 16/2 MoE models
    # (Qwen3.6-35B-A3B etc.) otherwise fall into the tiled sdpa256 prefill
    # path, which is ~2x slower than even the stock unfused fallback at
    # long context (issue #2155).
    return (
        head_dim == 256
        and kv_heads > 0
        and q_heads % kv_heads == 0
        and kv_len >= min_kv_len
        and q_len <= kv_len
    )


def apply_qwen35_fa256_attention_patch(min_kv_len: int | None = None) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    steel_env = os.environ.get("OMLX_FA256_STEEL", "").strip()
    if steel_env == "0":
        return False
    if steel_env != "1" and is_nax_available():
        # Auto: on NAX GPUs (M5 family) stock SDPA's head_dim-256 fallback
        # runs its matmuls on the tensor units and beats this pre-NAX steel
        # kernel, so routing it would regress prefill (M5 Max report:
        # 4k pp 828 -> 400 tok/s on 0.5.0). OMLX_FA256_STEEL=1 forces the
        # kernel on for benchmarking; a NAX port of this kernel is the
        # tracked follow-up.
        logger.info(
            "Qwen FA-256 steel patch skipped: NAX GPU, stock SDPA is faster"
        )
        return False

    kernel = _native_kernel()
    if kernel is None:
        logger.debug("Qwen FA-256 steel kernel unavailable; patch skipped")
        return False

    min_kv_len = int(os.environ.get("OMLX_FA256_MIN_KV_LEN", min_kv_len or 2048))
    q_block = int(os.environ.get("OMLX_FA256_Q_BLOCK", "32"))
    k_block = int(os.environ.get("OMLX_FA256_K_BLOCK", "8"))
    debug = os.environ.get("OMLX_FA256_DEBUG", "0") == "1"
    budget_env = os.environ.get("OMLX_FA256_DISPATCH_BUDGET", "").strip()
    if not _fa256_fast.fa256_supports_dispatch_budget():
        if budget_env != "0":
            logger.warning(
                "Qwen FA-256 steel kernel predates chunked dispatch (issue "
                "#2225); long-context prefill may hit the IOGPU preemption "
                "slowdown on pre-NAX GPUs. Rebuild the native extension to "
                "enable it."
            )
        dispatch_budget = 0
    elif budget_env:
        dispatch_budget = int(budget_env)
    else:
        dispatch_budget = _auto_dispatch_budget(kernel, q_block, k_block)

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
                        dispatch_budget=dispatch_budget,
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
                            dispatch_budget=dispatch_budget,
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
            "(min_kv_len=%d, q_block=%d, k_block=%d, dispatch_budget=%d)",
            min_kv_len,
            q_block,
            k_block,
            dispatch_budget,
        )
    return patched_any
