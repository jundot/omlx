# SPDX-License-Identifier: Apache-2.0
"""Keep head-dim-256 long-context prefill bounded on MLX 0.32.2.

MLX 0.32.2 ships a fused full-attention kernel for head dimensions 192 and 256,
but deliberately keeps the faster unfused path as the default on pre-NAX GPUs.
That default materializes the full ``[n_q, query_len, kv_len]`` score matrix —
O(L^2) in context length — and can still exceed oMLX's memory-guard ceiling.

Qualifying calls use the bounded route. Metal uses native fused attention for supported FP16/BF16 calls, while FP32 and array masks use array tiling.
The native FP32 full-attention kernel exceeds 32 KiB of threadgroup memory.
On NAX, MLX's default selects its split-D head-dim-256 kernel for causal prefills with at least 1024 queries.

``OMLX_SDPA256_TILED=1/0`` remains accepted for compatibility and now forces or
disables the bounded route. Metal uses the native fused kernel; CUDA retains
the prior array-tiled implementation because MLX 0.32.2's CUDA fused kernel
does not support head_dim 256.

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
import os

import mlx.core as mx

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

# Backward-compatible override: True = force the bounded route, False = never
# force it (opt out of the #2025 memory fix), None = the default below.
_FORCE_TILED: bool | None = None
# Bounded-route reasons already logged. The first engagement per reason logs at
# INFO; repeats stay silent to keep the hot path quiet.
_TILED_ROUTE_LOGGED: "set[str]" = set()


def _note_tiled_route(reason: str, detail: str) -> None:
    if reason in _TILED_ROUTE_LOGGED:
        return
    _TILED_ROUTE_LOGGED.add(reason)
    logger.info(
        "sdpa256: head-dim-256 long-context prefill is using the "
        "memory-bounded path: %s. OMLX_SDPA256_TILED=0 opts out.",
        detail,
    )


def _parse_force_tiled_env() -> bool | None:
    value = os.environ.get("OMLX_SDPA256_TILED", "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _tiled_route_required(queries, keys) -> bool:
    """Take the memory-bounded route for every qualifying call (True = force).

    This is #2025's original contract. head-dim-256 multi-token prefill has no
    fused MLX kernel, so the stock fallback materializes the whole
    ``[n_q, query_len, kv_len]`` fp32 score matrix. For the shapes
    ``_should_route`` admits, the bounded native fused kernel runs instead and
    peak prefill memory stays O(L).

    #2204 made this conditional on live guard headroom, to keep the faster
    unfused path wherever its matrix fits. Deciding per call from live process
    memory made the choice depend on process history: the same request saw
    headroom differ by gigabytes between otherwise identical processes
    (resident pages of the mmap'd weights, a previous request's pooled
    buffers), switched routes at a different kv_len, and — because the two
    routes are different floating-point reductions — produced different
    temperature-0 output. The prefill memory estimator was not made
    conditional alongside it: ``memory_monitor`` prices a registered head dim
    as O(L) for every shape the route covers, so a chunk could be admitted on
    the bounded route's transient and then run the unfused one.

    Deciding from the call shape alone keeps the route identical in every
    process and keeps the estimator's assumption true by construction.
    """
    if _FORCE_TILED is not None:
        if _FORCE_TILED:
            _note_tiled_route("forced", "forced by OMLX_SDPA256_TILED=1")
        return _FORCE_TILED
    _note_tiled_route(
        "long-context",
        "head_dim 256 has no fused kernel for multi-token prefill, so the "
        "stock fallback would materialize the full fp32 score matrix",
    )
    return True


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


def _flash_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Use MLX 0.32.2 native fused SDPA on Metal, portable tiling elsewhere.

    Explicit array masks never take the native fused call: MLX's fused
    array-mask support is unproven and may silently fall back to the
    unfused fp32 score matrix, which is exactly the O(L^2) spike this patch
    exists to bound. The array-tiled implementation already handles bool and
    additive masks, so route them there directly."""
    global _NATIVE_FORCE_FUSED

    if isinstance(mask, mx.array):
        return _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks)
    # MLX promotes Q/K/V together; any FP32 input selects the 53,760-byte kernel.
    if mx.result_type(queries.dtype, keys.dtype, values.dtype) == mx.float32:
        return _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks)
    native_shape = values.shape[-1] == HEAD_DIM and not (
        isinstance(mask, str)
        and mask == "causal"
        and queries.shape[-2] > keys.shape[-2]
    )
    if mx.metal.is_available() and native_shape and _NATIVE_FORCE_FUSED:
        try:
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
        return _tiled_route_required(queries, keys)
    except Exception:
        return False


def _register_bounded_route(min_kv_len: int) -> bool:
    """Publish only a runtime guarantee that is actually enabled."""
    if _FORCE_TILED is False:
        return False
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
    global _PATCHED, _SDPA256_MIN_KV_LEN, _FORCE_TILED
    if _PATCHED:
        return False
    _SDPA256_MIN_KV_LEN = min_kv_len
    _FORCE_TILED = _parse_force_tiled_env()

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
    # prefill is now O(L), so it stops charging the O(L^2) score matrix. The
    # explicit benchmark override disables this guarantee, so registering it
    # in that mode would under-estimate the same unfused path the user forced.
    _register_bounded_route(min_kv_len)

    _PATCHED = True
    if _FORCE_TILED is None:
        routing = "always force bounded for qualifying calls (#2025)"
    elif _FORCE_TILED:
        routing = "always force bounded (OMLX_SDPA256_TILED=1)"
    else:
        routing = "never force bounded (OMLX_SDPA256_TILED=0)"
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d, %s)",
        min_kv_len,
        routing,
    )
    return True
