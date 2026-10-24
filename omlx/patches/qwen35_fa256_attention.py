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

import functools
import logging
import sys
import time

import mlx.core as mx

from omlx.custom_kernels.attention import fast as _attention_fast
from omlx.custom_kernels.nax import is_nax_available

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
# calibration fails.
_DEFAULT_DISPATCH_BUDGET = 250_000_000

# The preemption threshold is wallclock-based, so a fixed work budget leaves
# uneven margins across GPU tiers (the same work runs ~5-8x longer on an M1
# than on an M3 Ultra). The patch measures the kernel's own throughput once
# and sizes the budget so one chunk dispatch targets this wallclock; the M3
# Max report puts the cliff above ~50ms per dispatch, so 10ms keeps a wide
# margin on any tier while chunks stay far too coarse to cost throughput. The
# calibration shape is small and eval overhead inflates its measured time, so
# the derived budget errs low (real dispatches run shorter than the target) -
# the safe side.
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

# Steel kernel tile sizes (queries x keys per block).
_Q_BLOCK = 32
_K_BLOCK = 8

# M1-class GPUs (applegpu_g13): MLX's own SDPA outruns the steel kernel here
# by 22-40% (M1 Max, fp16, 2048-row prefill chunks over 32k-98k keys) as long
# as each dispatch stays short; one long dispatch hits the IOGPU
# preemption slowdown (#2225) instead. The query rows are split so a dispatch
# covers at most _M1_SDPA_WORK (heads * rows * keys, ~15-20ms there); below
# _M1_SDPA_MIN_ROWS rows the GPU is under-occupied, so when even that many
# rows exceed _M1_SDPA_MAX_WORK (~128k keys) the steel kernel keeps the call.
#
# Both limits were tuned on an M1 Max, where the steel kernel's calibrated
# dispatch budget is _M1_REF_BUDGET; they scale with the budget calibrated on
# the machine at hand (about 2-4x smaller on an 8-core M1, 2x larger on an
# Ultra), so a dispatch lasts about the same time everywhere.
_M1_SDPA_WORK = 120_000_000
_M1_SDPA_MAX_WORK = 160_000_000
_M1_REF_BUDGET = 51_000_000
_M1_WORK_SCALE = 1.0
_M1_SDPA_MIN_ROWS = 64
# The first prefill chunk (keys == queries == 2048) is 10-19% faster on the
# steel kernel; from 3072 keys the chunked route wins by 6-32% (M1 Max,
# 16/8, 24/4 and 16/2 layouts).
_M1_SDPA_MIN_KV = 3072
# Head_dim 512 (Gemma 4 global layers) has no fused MLX route at all, so the
# stock path materializes every score: one 2048-row chunk over 128k keys
# peaked at +8.9 GB and ran at 4.4 TFLOP/s on M1 Max; the chunked route ran
# it at 7.0 TFLOP/s (-37%) within +0.6 GB. Routed from this many keys on
# (below it the two tie on time; the chunking still bounds memory).
_M1_D512_MIN_KV = 32768
_M1_K_BLOCK = 16


def _native_kernel():
    try:
        from omlx.custom_kernels.attention import fast
    except Exception:
        return None
    if not fast.has_symbol("fa256_attention"):
        return None
    return fast.fa256_attention


def _has_quantized_cache(cache) -> bool:
    return cache is not None and hasattr(cache, "bits")


def _bounded_sdpa_fallback(queries, keys, values, scale, mask, sinks):
    """Keep the registered O(L) contract if the steel kernel itself fails."""
    from .sdpa256_attention import _flash_sdpa256

    return _flash_sdpa256(queries, keys, values, scale, mask, sinks)


def _calibration_dtype():
    """M1-class GPUs have no bf16 ALUs: bf16 runs at about half the fp16 rate
    there (5.1e9 vs 2.7e9 work/s on an M1 Max), and the models served on them
    are fp16, so calibrating in bf16 would halve every dispatch's budget."""
    from omlx.utils.hardware import get_apple_gpu_generation

    return mx.float16 if get_apple_gpu_generation() == 13 else mx.bfloat16


def _auto_dispatch_budget(kernel, q_block: int, k_block: int) -> int:
    """Size the dispatch budget from the kernel's measured throughput.

    Runs the native kernel once as a single dispatch on a small synthetic
    shape (one warmup for pipeline compilation, then best of three timed
    runs) and converts the throughput into the work that fits in
    ``_TARGET_DISPATCH_SECONDS``. Falls back to the fixed default when
    measurement fails."""
    try:
        work = _CALIB_HEADS * _CALIB_Q_LEN * _CALIB_KV_LEN
        dtype = _calibration_dtype()
        q = mx.random.normal((1, _CALIB_HEADS, _CALIB_Q_LEN, 256)).astype(dtype)
        k = mx.random.normal((1, _CALIB_KV_HEADS, _CALIB_KV_LEN, 256)).astype(dtype)
        v = mx.random.normal((1, _CALIB_KV_HEADS, _CALIB_KV_LEN, 256)).astype(dtype)
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


@functools.cache
def _m1_sdpa_enabled() -> bool:
    from omlx.utils.hardware import get_apple_gpu_generation

    return get_apple_gpu_generation() == 13


def _m1_query_rows(q_heads: int, q_len: int, kv_len: int) -> int:
    """Query rows per MLX SDPA dispatch on M1, or 0 to keep the steel kernel."""
    work = _M1_SDPA_WORK * _M1_WORK_SCALE
    rows = q_len
    while rows > _M1_SDPA_MIN_ROWS and q_heads * rows * kv_len > work:
        rows = max(_M1_SDPA_MIN_ROWS, rows // 2)
    if q_heads * rows * kv_len > _M1_SDPA_MAX_WORK * _M1_WORK_SCALE:
        return 0
    return rows


def _m1_chunked_sdpa(queries, keys, values, scale: float, rows: int) -> mx.array:
    """Causal SDPA for queries at the end of the keys, ``rows`` per dispatch.

    Each block materializes its scores (head_dim 256 is not a fused SDPA
    size on these GPUs), so at most two blocks are kept in flight: block
    ``i`` is submitted, then block ``i-1`` is waited on and its scratch
    released.

    Every block attends to all the keys through an explicit causal mask,
    rather than to its own end-aligned prefix. The prefix would skip only the
    last few thousand keys, but it gives every block a different scratch
    size, and MLX's buffer cache cannot reuse those: after one call it held
    4-8 GB of them (10 GiB above the steel route on an M1 Ultra at 134k
    keys). With one size for all blocks the cache stays below 0.6 GB, at the
    same speed within 2-5%.
    """
    q_len = queries.shape[2]
    kv_len = keys.shape[2]
    if q_len <= rows:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask="causal"
        )
    cols = mx.arange(kv_len)[None, :]
    outs = []
    for start in range(0, q_len, rows):
        stop = min(q_len, start + rows)
        last = mx.arange(start, stop)[:, None] + (kv_len - q_len)
        out = mx.fast.scaled_dot_product_attention(
            queries[:, :, start:stop], keys, values, scale=scale, mask=cols <= last
        )
        mx.async_eval(out)
        if outs:
            mx.eval(outs[-1])
        outs.append(out)
    return mx.concatenate(outs, axis=2)


def _m1_d512_attention(queries, keys, values, cache, scale, mask, sinks):
    """Causal head_dim-512 prefill attention through the M1 chunked route, or
    None when the call is not that shape (or not on an M1-class GPU)."""
    if (
        queries.ndim != 4
        or queries.shape[-1] != 512
        or queries.shape[-2] < _MIN_ROUTE_Q_LEN
        or sinks is not None
        or not (isinstance(mask, str) and mask == "causal")
        or _has_quantized_cache(cache)
        or getattr(keys, "ndim", 0) != 4
        or getattr(values, "ndim", 0) != 4
        or keys.shape[-1] != 512
        or values.shape[-1] != 512
        or queries.dtype not in (mx.float16, mx.bfloat16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or keys.shape[-3] == 0
        or queries.shape[-3] % keys.shape[-3]
        or keys.shape[-2] < max(_M1_D512_MIN_KV, queries.shape[-2])
        or not _m1_sdpa_enabled()
    ):
        return None
    rows = _m1_query_rows(queries.shape[1], queries.shape[2], keys.shape[2])
    return _m1_chunked_sdpa(queries, keys, values, scale, rows or _M1_SDPA_MIN_ROWS)


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
    # Route only on an explicit "causal" mask, never on None. MLX's own
    # scaled_dot_product_attention treats mask=None as "no mask at all"
    # (unrestricted attention), but this kernel is always invoked below
    # with causal=True -- silently causal-izing a caller that genuinely
    # intended unrestricted attention would be a correctness bug. No
    # currently-reachable head_dim=256 caller in this codebase actually
    # passes mask=None at a routable q_len (audited: every mask=None call
    # site is either single-token decode, already excluded by the q_len
    # gate above, or a truncated-key loop where None correctly means "no
    # mask over the already-causal-truncated keys"), so this only closes a
    # latent hazard for future callers -- it does not change today's
    # routing decisions.
    # See docs/qwen35-hardening-and-optimization.md E4.
    if not (isinstance(mask, str) and mask == "causal"):
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
    global _PATCHED, _M1_WORK_SCALE
    if _PATCHED:
        return True
    if is_nax_available():
        # MLX 0.32.2 selects its native split-D fused kernel for NAX
        # head-dim-256 causal prefills with at least 1024 queries. It beats
        # this pre-NAX steel kernel, so intercepting it would regress prefill.
        logger.info(
            "Qwen FA-256 steel patch skipped: NAX GPU, MLX native fused SDPA "
            "is faster"
        )
        return False

    kernel = _native_kernel()
    if kernel is None:
        logger.debug("Qwen FA-256 steel kernel unavailable; patch skipped")
        return False

    min_kv_len = min_kv_len or 2048
    q_block = _Q_BLOCK
    k_block = _K_BLOCK
    m1_sdpa = _m1_sdpa_enabled()
    if m1_sdpa:
        # 16-key tiles measured 3-7% faster than 8 on M1 Max (98k keys).
        k_block = _M1_K_BLOCK
    if not _attention_fast.fa256_supports_dispatch_budget():
        logger.warning(
            "Qwen FA-256 steel kernel predates chunked dispatch (issue "
            "#2225); long-context prefill may hit the IOGPU preemption "
            "slowdown on pre-NAX GPUs. Rebuild the native extension to "
            "enable it."
        )
        dispatch_budget = 0
    else:
        dispatch_budget = _auto_dispatch_budget(kernel, q_block, k_block)
        if m1_sdpa and dispatch_budget != _DEFAULT_DISPATCH_BUDGET:
            _M1_WORK_SCALE = min(4.0, max(0.1, dispatch_budget / _M1_REF_BUDGET))
            logger.info("M1 chunked SDPA work scale %.2f", _M1_WORK_SCALE)

    def attend(queries, keys, values, scale):
        if m1_sdpa and keys.shape[2] >= _M1_SDPA_MIN_KV:
            rows = _m1_query_rows(queries.shape[1], queries.shape[2], keys.shape[2])
            if rows:
                return _m1_chunked_sdpa(queries, keys, values, scale, rows)
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
            m1_out = _m1_d512_attention(queries, keys, values, cache, scale, mask, sinks)
            if m1_out is not None:
                return m1_out
            if _should_route(queries, keys, cache, mask, sinks, min_kv_len):
                try:
                    return attend(queries, keys, values, scale)
                except Exception:
                    logger.warning(
                        "fa256 steel lm kernel failed; using bounded MLX SDPA",
                        exc_info=True,
                    )
                    return _bounded_sdpa_fallback(
                        queries, keys, values, scale, mask, sinks
                    )
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
                m1_out = _m1_d512_attention(
                    queries, keys, values, cache, scale, mask, sinks
                )
                if m1_out is not None:
                    return m1_out
                if _should_route(queries, keys, cache, mask, sinks, min_kv_len):
                    try:
                        return attend(queries, keys, values, scale)
                    except Exception:
                        logger.warning(
                            "fa256 steel vlm kernel failed; using bounded MLX SDPA",
                            exc_info=True,
                        )
                        return _bounded_sdpa_fallback(
                            queries, keys, values, scale, mask, sinks
                        )
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
        try:
            from .. import memory_monitor

            memory_monitor.register_tiled_prefill_head_dim(
                256,
                min_query_len=_MIN_ROUTE_Q_LEN,
                min_kv_len=min_kv_len,
                kv_tile=1024,
            )
        except Exception:
            logger.debug(
                "could not register Qwen FA-256 with memory_monitor",
                exc_info=True,
            )
        _PATCHED = True
        logger.info(
            "Qwen3.5/3.6 FA-256 steel attention patch applied "
            "(min_kv_len=%d, q_block=%d, k_block=%d, dispatch_budget=%d%s)",
            min_kv_len,
            q_block,
            k_block,
            dispatch_budget,
            ", M1 chunked MLX SDPA up to ~128k keys" if m1_sdpa else "",
        )
    return patched_any
