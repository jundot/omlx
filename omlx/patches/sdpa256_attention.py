# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to fix head_dim=256 long-context prefill.

MLX's fused SDPA kernel supports head_dim in {64, 80, 128} only, so head_dim=256
(e.g. Qwen3.6-27B) multi-token prefill falls back to an unfused path that
materializes the full ``[n_q, query_len, kv_len]`` score matrix -> O(L^2) memory,
OOMing / tripping the prefill guard far below the context window. Decode
(query_len == 1) is unaffected (MLX has a fused vector kernel for 256).

This routes head_dim=256 causal prefill to a flash-style online-softmax pass in
pure MLX array ops (tiled over KV; running max/sum/accumulator) that never
materializes the score matrix -> peak memory O(L). It rides MLX's GEMM, so speed
is on par with the fallback; the win is memory. ``register_tiled_prefill_head_dim``
flips the prefill-guard estimator to O(L) in lockstep (else it keeps rejecting).

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False

HEAD_DIM = 256
# Engage the tiled kernel only once the context is long enough that the unfused
# fallback's O(L^2) score matrix becomes a memory problem. Below this, the
# fused-GEMM fallback is faster and fits comfortably. Tunable.
_SDPA256_MIN_KV_LEN = 8192
# Tile sizes for the online-softmax kernel (tuned on M2 Max).
_Q_TILE = 512
_KV_TILE = 1024

_NEG_INF = -1e30  # fp32 sentinel for masked logits (exp -> 0)


def _flash_sdpa256(queries, keys, values, scale, mask):
    """Flash-style online-softmax attention for head_dim=256 prefill.

    queries: [B, n_q, Lq, D]   keys/values: [B, n_kv, Lk, D]   (n_q % n_kv == 0)
    mask: "causal" or None. Returns [B, n_q, Lq, D] in queries.dtype.

    Tiles over Q and KV, keeping a running (max m, sum denom, accumulator acc) per
    query row so the [q x full_kv] score matrix is never materialized. fp32
    accumulators; output cast back to the input dtype. GQA via reshape+broadcast.

    MLX is lazy: without forcing materialization the whole tiled graph would stay
    live until eval (peak dominated by graph buildup, not the O(L) working set),
    so the running carry is eval'd per KV step / per finished Q tile to bound the
    live graph to ~one tile -> true O(L) peak.
    """
    B, n_q, Lq, D = queries.shape
    _, n_kv, Lk, _ = keys.shape
    G = n_q // n_kv
    causal = mask == "causal"

    qr = queries.reshape(B, n_kv, G, Lq, D)
    kr = keys.reshape(B, n_kv, 1, Lk, D)
    vr = values.reshape(B, n_kv, 1, Lk, D)

    # MLX 'causal' aligns queries to the END of the key axis: with a cached
    # prefix (Lk > Lq, chunked prefill) local query i is global position
    # i + offset and attends keys 0..(i + offset). offset == 0 for square.
    offset = Lk - Lq

    out_q_tiles = []
    for qi0 in range(0, Lq, _Q_TILE):
        qi1 = min(qi0 + _Q_TILE, Lq)
        qb = qr[:, :, :, qi0:qi1, :].astype(mx.float32)
        qt = qi1 - qi0
        q_pos = mx.arange(qi0 + offset, qi1 + offset).reshape(1, 1, 1, qt, 1)

        m = mx.full((B, n_kv, G, qt, 1), _NEG_INF, dtype=mx.float32)
        denom = mx.zeros((B, n_kv, G, qt, 1), dtype=mx.float32)
        acc = mx.zeros((B, n_kv, G, qt, D), dtype=mx.float32)

        kv_end = min(qi1 + offset, Lk) if causal else Lk
        for kj0 in range(0, kv_end, _KV_TILE):
            kj1 = min(kj0 + _KV_TILE, kv_end)
            kb = kr[:, :, :, kj0:kj1, :].astype(mx.float32)
            vb = vr[:, :, :, kj0:kj1, :].astype(mx.float32)
            kt = kj1 - kj0

            s = (qb @ mx.swapaxes(kb, -1, -2)) * scale
            if causal:
                k_pos = mx.arange(kj0, kj1).reshape(1, 1, 1, 1, kt)
                s = mx.where(k_pos > q_pos, _NEG_INF, s)

            m_tile = mx.max(s, axis=-1, keepdims=True)
            m_new = mx.maximum(m, m_tile)
            p = mx.exp(s - m_new)
            corr = mx.exp(m - m_new)
            denom = denom * corr + mx.sum(p, axis=-1, keepdims=True)
            acc = acc * corr + (p @ vb)
            m = m_new
            mx.eval(m, denom, acc)  # bound the live graph -> O(L) peak

        out_tile = (acc / denom).astype(queries.dtype)
        mx.eval(out_tile)
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(B, n_q, Lq, D)


def _should_route(queries, keys, cache, mask, sinks) -> bool:
    # Never raise: any unexpected input must fall through to the original SDPA,
    # never break a request. Worst case we decline to engage.
    try:
        if sinks is not None:
            return False
        # Quantized KV cache (TurboQuant etc.): keys/values are packed state,
        # not plain [.., kv, hd] arrays. MLX's own dispatcher detects this via
        # hasattr(cache, "bits"); let the quant-aware path handle it.
        if cache is not None and hasattr(cache, "bits"):
            return False
        if queries.shape[-1] != HEAD_DIM:
            return False
        if queries.shape[-2] <= 1:  # decode -> fused vector kernel handles 256
            return False
        if not (mask is None or (isinstance(mask, str) and mask == "causal")):
            return False
        if keys.shape[-2] < _SDPA256_MIN_KV_LEN:
            return False
        n_q = queries.shape[-3]
        n_kv = keys.shape[-3]
        if n_kv <= 0 or n_q % n_kv != 0:
            return False
        return True
    except Exception:
        return False


def apply_sdpa256_attention_patch(min_kv_len: int = _SDPA256_MIN_KV_LEN) -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for head_dim=256
    long-context prefill, and register the O(L) cost with the memory monitor."""
    global _PATCHED, _SDPA256_MIN_KV_LEN
    if _PATCHED:
        return False
    _SDPA256_MIN_KV_LEN = min_kv_len

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
        if _should_route(queries, keys, cache, mask, sinks):
            try:
                return _flash_sdpa256(queries, keys, values, scale, mask)
            except Exception:
                logger.warning(
                    "sdpa256 prefill kernel failed; falling back to MLX SDPA",
                    exc_info=True,
                )
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Rebind already-imported model modules that did
    # `from .base import scaled_dot_product_attention` at import time. Only
    # rebind modules whose attribute IS the base function we wrapped — a model
    # that defined its own SDPA keeps it untouched (don't silently redirect a
    # model we never intended to patch).
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (
            mod_name.startswith("mlx_lm.models.")
            or mod_name.startswith("mlx_vlm.models.")
        ):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    try:
        from mlx_vlm.models import base as vlm_base

        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    # Keep the prefill memory guard in lockstep: tell the monitor head_dim 256
    # prefill is now O(L), so it stops charging the O(L^2) score matrix.
    try:
        from .. import memory_monitor

        memory_monitor.register_tiled_prefill_head_dim(
            HEAD_DIM, min_kv_len=min_kv_len, kv_tile=_KV_TILE
        )
    except Exception:
        logger.debug("could not register sdpa256 with memory_monitor", exc_info=True)

    _PATCHED = True
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d -> "
        "O(L) tiled kernel)",
        min_kv_len,
    )
    return True
