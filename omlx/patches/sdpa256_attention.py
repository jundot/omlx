# SPDX-License-Identifier: Apache-2.0
"""Keep head-dim-256 long-context prefill bounded on MLX 0.32.3.

MLX 0.32.3 ships fused full-attention kernels for head dimension 256, but on
pre-NAX GPUs its default dispatch still takes the unfused path for chunked
prefill (more keys than queries) and for array masks. That path materializes
the full ``[n_q, query_len, kv_len]`` score matrix, quadratic in context
length, and can exceed oMLX's memory-guard ceiling.

Qualifying calls use the bounded route: MLX's fused kernel on Metal for causal,
no-mask and array-mask calls in any float dtype. Pre-NAX GPUs run array-mask
and FP32 calls in query chunks of about 10 ms each (issue #2225). On NAX, MLX's
default already selects its split-D head-dim-256 kernel for causal and
array-mask prefills with at least 1024 queries.

CUDA retains the array-tiled implementation because MLX 0.32.3's CUDA fused
kernel does not support head_dim 256.

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
import time

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available

logger = logging.getLogger(__name__)

_PATCHED = False

HEAD_DIM = 256
# Force the bounded kernel only once the context is long enough that the
# default unfused route's O(L^2) score matrix becomes a memory problem.
_SDPA256_MIN_KV_LEN = 8192
# Decode-shaped multi-row calls (MTP verify: q_len = 1 + draft depth <= 9)
# do not need the forced full-attention route. Below this floor the stock path's
# score matrix is at most n_q * 15 * kv_len and is not a memory problem.
_SDPA256_MIN_Q_LEN = 16
_Q_TILE = 512
# A deliberately conservative score-tile width used only by the admission
# estimate. MLX's fused kernel keeps a smaller on-chip block, so this does not
# understate the bounded route's score working set.
_KV_TILE = 1024
_NEG_INF = -1e30
# Per-dispatch wallclock target and clamp for the pre-NAX chunked route; the
# same values as qwen35_fa256_attention's budget (issue #2225).
_TARGET_DISPATCH_SECONDS = 0.010
_DEFAULT_DISPATCH_BUDGET = 250_000_000
_MIN_DISPATCH_BUDGET = 20_000_000
_MAX_DISPATCH_BUDGET = 2_000_000_000
_DISPATCH_BUDGET: int | None = None

# Bounded-route reasons already logged. The first engagement per reason logs at
# INFO; repeats stay silent to keep the hot path quiet.
_TILED_ROUTE_LOGGED: "set[str]" = set()


def _note_tiled_route(reason: str, detail: str) -> None:
    if reason in _TILED_ROUTE_LOGGED:
        return
    _TILED_ROUTE_LOGGED.add(reason)
    logger.info(
        "sdpa256: head-dim-256 long-context prefill is using the "
        "memory-bounded path: %s.",
        detail,
    )


def _broadcast_mask_5d(mask, batch, n_kv, group_size, q_len, k_len):
    """Reshape an array mask for the tiled GQA attention layout."""
    if mask.ndim == 4:
        pass
    elif mask.ndim == 3:
        # Preserve mlx-lm's convention: [batch, query, key].
        mask = mask[:, None, :, :]
    elif mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 1:
        mask = mask[None, None, None, :]
    else:
        raise ValueError(f"unsupported attention mask ndim: {mask.ndim}")
    n_q = n_kv * group_size
    mask = mx.broadcast_to(mask, (batch, n_q, q_len, k_len))
    return mask.reshape(batch, n_kv, group_size, q_len, k_len)


def _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Portable bounded fallback for shapes without a native fused kernel."""
    output_dtype = mx.result_type(queries.dtype, keys.dtype, values.dtype)
    batch, n_q, q_len, head_dim = queries.shape
    _, n_kv, k_len, _ = keys.shape
    value_dim = values.shape[-1]
    group_size = n_q // n_kv
    causal = isinstance(mask, str) and mask == "causal"
    array_mask = None
    if isinstance(mask, mx.array):
        array_mask = _broadcast_mask_5d(mask, batch, n_kv, group_size, q_len, k_len)

    qr = queries.reshape(batch, n_kv, group_size, q_len, head_dim)
    kr = keys.reshape(batch, n_kv, 1, k_len, head_dim)
    vr = values.reshape(batch, n_kv, 1, k_len, value_dim)
    offset = k_len - q_len

    out_q_tiles = []
    for qi0 in range(0, q_len, _Q_TILE):
        qi1 = min(qi0 + _Q_TILE, q_len)
        qb = qr[:, :, :, qi0:qi1, :].astype(mx.float32)
        qt = qi1 - qi0
        q_pos = mx.arange(qi0 + offset, qi1 + offset).reshape(1, 1, 1, qt, 1)

        state_shape = (batch, n_kv, group_size, qt, 1)
        if sinks is None:
            m = mx.full(state_shape, _NEG_INF, dtype=mx.float32)
            denom = mx.zeros(state_shape, dtype=mx.float32)
        else:
            sink_logits = sinks.astype(mx.float32).reshape(1, n_kv, group_size, 1, 1)
            m = mx.broadcast_to(sink_logits, state_shape)
            denom = mx.ones(state_shape, dtype=mx.float32)
        acc = mx.zeros((batch, n_kv, group_size, qt, value_dim), dtype=mx.float32)

        kv_end = min(qi1 + offset, k_len) if causal else k_len
        for kj0 in range(0, kv_end, _KV_TILE):
            kj1 = min(kj0 + _KV_TILE, kv_end)
            kb = kr[:, :, :, kj0:kj1, :].astype(mx.float32)
            vb = vr[:, :, :, kj0:kj1, :].astype(mx.float32)
            kt = kj1 - kj0

            scores = (qb @ mx.swapaxes(kb, -1, -2)) * scale
            if causal:
                k_pos = mx.arange(kj0, kj1).reshape(1, 1, 1, 1, kt)
                scores = mx.where(k_pos > q_pos, _NEG_INF, scores)
            elif array_mask is not None:
                tile_mask = array_mask[..., qi0:qi1, kj0:kj1]
                if tile_mask.dtype == mx.bool_:
                    scores = mx.where(tile_mask, scores, _NEG_INF)
                else:
                    scores = scores + tile_mask.astype(mx.float32)

            tile_max = mx.max(scores, axis=-1, keepdims=True)
            new_max = mx.maximum(m, tile_max)
            probabilities = mx.exp(scores - new_max)
            correction = mx.exp(m - new_max)
            denom = denom * correction + mx.sum(probabilities, axis=-1, keepdims=True)
            acc = acc * correction + (probabilities @ vb)
            m = new_max
            mx.eval(m, denom, acc)

        out_tile = (acc / denom).astype(output_dtype)
        mx.eval(out_tile)
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(batch, n_q, q_len, value_dim)


# ``force_fused=`` arrived in MLX 0.32.2. On an older runtime the keyword is a
# TypeError, and retrying without it is not a safe substitute -- MLX would then
# be free to pick the unfused fp32 score matrix, which is the O(L^2) spike this
# patch exists to bound. Such a runtime routes to the array-tiled path instead,
# which is bounded by construction.
#
# Probed by use rather than by signature: MLX's nanobind functions report
# ``(*args, **kwargs)``, so the keyword is only visible in the docstring, and
# ``python -OO`` strips that. One TypeError on the first call is cheaper than a
# fragile capability check, and the answer is latched.
_NATIVE_FORCE_FUSED = True


def _fused_dispatch_budget() -> int:
    """Work (heads x query rows x keys) one fused dispatch runs in about
    ``_TARGET_DISPATCH_SECONDS`` on this GPU, measured once."""
    global _DISPATCH_BUDGET
    if _DISPATCH_BUDGET is not None:
        return _DISPATCH_BUDGET
    try:
        q = mx.zeros((1, 16, 1024, HEAD_DIM), dtype=mx.bfloat16)
        kv = mx.zeros((1, 2, 8192, HEAD_DIM), dtype=mx.bfloat16)
        mx.eval(q, kv)
        best = None
        for i in range(4):
            start = time.perf_counter()
            mx.eval(
                mx.fast.scaled_dot_product_attention(
                    q, kv, kv, scale=HEAD_DIM**-0.5, mask="causal", force_fused=True
                )
            )
            elapsed = time.perf_counter() - start
            if i > 0:
                best = elapsed if best is None else min(best, elapsed)
        budget = int(16 * 1024 * 8192 / best * _TARGET_DISPATCH_SECONDS)
    except Exception:
        # Traced (mx.compile) or no fused kernel: decide on a later eager call.
        return _DEFAULT_DISPATCH_BUDGET
    _DISPATCH_BUDGET = max(_MIN_DISPATCH_BUDGET, min(_MAX_DISPATCH_BUDGET, budget))
    return _DISPATCH_BUDGET


def _chunked_fused_sdpa256(queries, keys, values, scale, mask, sinks):
    """Fused SDPA as query chunks of one dispatch budget each.

    A single dispatch over a long KV can trip the IOGPU interactivity
    preemption on pre-NAX GPUs (issue #2225). A mask with one row per query
    is sliced per chunk. A causal chunk ends its keys at its own last row,
    since the "causal" mask aligns the queries to the end of the keys.
    """
    heads, q_len, kv_len = queries.shape[-3], queries.shape[-2], keys.shape[-2]
    rows = max(16, _fused_dispatch_budget() // max(1, heads * kv_len))
    if rows >= q_len:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
        )
    per_row_mask = (
        isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == q_len
    )
    causal = isinstance(mask, str) and mask == "causal"
    outs = []
    for start in range(0, q_len, rows):
        stop = min(q_len, start + rows)
        end = kv_len - q_len + stop if causal else kv_len
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[..., start:stop, :],
                keys[..., :end, :],
                values[..., :end, :],
                scale=scale,
                mask=mask[..., start:stop, :] if per_row_mask else mask,
                sinks=sinks,
                force_fused=True,
            )
        )
    return mx.concatenate(outs, axis=-2)


def _flash_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Use MLX 0.32.3 native fused SDPA on Metal, portable tiling elsewhere.

    The fused kernel keeps causal, no-mask and array-mask calls O(L) in every
    float dtype. Pre-NAX GPUs run array-mask and FP32 calls in query chunks
    (see ``_chunked_fused_sdpa256``). Calls the fused kernel rejects (more
    queries than keys under a causal mask, a mask that does not promote to
    the output dtype) take the array-tiled route."""
    global _NATIVE_FORCE_FUSED

    native_shape = values.shape[-1] == HEAD_DIM and not (
        isinstance(mask, str)
        and mask == "causal"
        and queries.shape[-2] > keys.shape[-2]
    )
    if mx.metal.is_available() and native_shape and _NATIVE_FORCE_FUSED:
        chunked = not is_nax_available() and (
            isinstance(mask, mx.array)
            or mx.result_type(queries.dtype, keys.dtype, values.dtype) == mx.float32
        )
        try:
            if chunked:
                return _chunked_fused_sdpa256(queries, keys, values, scale, mask, sinks)
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                sinks=sinks,
                force_fused=True,
            )
        except TypeError:
            # Falling through to the tiled path rather than re-raising: it is a
            # correct implementation of the same op, so a genuinely malformed
            # call still fails there rather than being swallowed here.
            _NATIVE_FORCE_FUSED = False
            logger.warning(
                "sdpa256: mlx %s has no force_fused= (0.32.2+); using the "
                "array-tiled bounded route instead of the native fused kernel",
                getattr(mx, "__version__", "?"),
            )
        except ValueError:
            # A layout the fused kernel rejects; the tiled route covers it.
            pass
    return _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks)


def _should_route(queries, keys, cache, mask, sinks) -> bool:
    # Never raise: any unexpected input must fall through to the original SDPA,
    # never break a request. Worst case we decline to engage.
    # Shape gates first: this wrapper is installed unconditionally and runs
    # on every SDPA call of every decode step, so the common (decode / MTP
    # verify) case must exit on the q_len check alone (issue #2132).
    try:
        if queries.shape[-2] < _SDPA256_MIN_Q_LEN:  # decode / MTP verify
            return False
        if queries.shape[-1] != HEAD_DIM:
            return False
        if keys.shape[-2] < _SDPA256_MIN_KV_LEN:
            return False
        # Quantized KV cache (TurboQuant etc.): keys/values are packed state,
        # not plain [.., kv, hd] arrays. MLX's own dispatcher detects this via
        # hasattr(cache, "bits"); let the quant-aware path handle it.
        if cache is not None and hasattr(cache, "bits"):
            return False
        if not (
            mask is None
            or (isinstance(mask, str) and mask == "causal")
            or (isinstance(mask, mx.array) and 1 <= mask.ndim <= 4)
        ):
            return False
        n_q = queries.shape[-3]
        n_kv = keys.shape[-3]
        if n_kv <= 0 or n_q % n_kv != 0:
            return False
        _note_tiled_route(
            "long-context",
            "bounded attention keeps execution consistent with prefill memory pricing",
        )
        return True
    except Exception:
        return False


def _register_bounded_route(min_kv_len: int) -> bool:
    """Publish the bounded route's O(L) cost; False if registration fails."""
    try:
        from .. import memory_monitor

        memory_monitor.register_tiled_prefill_head_dim(
            HEAD_DIM,
            min_query_len=_SDPA256_MIN_Q_LEN,
            min_kv_len=min_kv_len,
            kv_tile=_KV_TILE,
            supports_array_mask=True,
        )
    except Exception:
        logger.debug("could not register sdpa256 with memory_monitor", exc_info=True)
        return False
    return True


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
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        if _should_route(queries, keys, cache, mask, sinks):
            return _flash_sdpa256(queries, keys, values, scale, mask, sinks)
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Rebind already-imported model modules that did
    # `from .base import scaled_dot_product_attention` at import time. Only
    # rebind modules whose attribute IS the base function we wrapped — a model
    # that defined its own SDPA keeps it untouched (don't silently redirect a
    # model we never intended to patch).
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            mod.scaled_dot_product_attention = patched_sdpa

    # mlx-vlm carries its own base SDPA (a distinct function, TurboQuant-aware
    # cache handling included), and model modules like qwen3_5.language copy
    # the reference at import time. It needs its own capture + wrapper +
    # submodule rebind, mirroring qwen35_fa256_attention: checking mlx-vlm
    # modules against the mlx-lm original can never match, which left the VLM
    # engine on the unfused O(L^2) path and — because this patch installs
    # first — polluted the fa256 patch's "original" capture so its rebind
    # missed the VLM submodules too.
    try:
        from mlx_vlm.models import base as vlm_base
    except ImportError:
        vlm_base = None

    if vlm_base is not None:
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
            ) -> mx.array:
                if _should_route(queries, keys, cache, mask, sinks):
                    return _flash_sdpa256(queries, keys, values, scale, mask, sinks)
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

    # Keep the prefill memory guard in lockstep: tell the monitor head_dim 256
    # prefill is now O(L), so it stops charging the O(L^2) score matrix.
    _register_bounded_route(min_kv_len)

    _PATCHED = True
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d, "
        "always force bounded for qualifying calls (#2025))",
        min_kv_len,
    )
    return True
