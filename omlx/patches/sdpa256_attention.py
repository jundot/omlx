# SPDX-License-Identifier: Apache-2.0
"""Keep head-dim-256 long-context prefill bounded on MLX 0.32.2.

MLX 0.32.2 ships a fused full-attention kernel for head dimensions 192 and 256,
but deliberately keeps the faster unfused path as the default on pre-NAX GPUs.
That default materializes the full ``[n_q, query_len, kv_len]`` score matrix and
can still exceed oMLX's memory-guard ceiling.

When the unfused transient fits, this patch preserves MLX's default routing. If
it does not fit (or no guard ceiling is available), it calls MLX 0.32.2 with
``force_fused=True``. This replaces oMLX's old pure-array tiled implementation:
the bounded route is now an upstream native fused kernel instead of the slow
sequential tile loop. On NAX, MLX's default already selects its fast split-D
head-dim-256 kernel for causal prefills with at least 1024 queries.

``OMLX_SDPA256_TILED=1/0`` remains accepted for compatibility and now forces or
disables the bounded route. Metal uses the native fused kernel; CUDA retains
the prior array-tiled implementation because MLX 0.32.2's CUDA fused kernel
does not support head_dim 256. The default is memory-aware.

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
import os
import weakref

import mlx.core as mx

from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

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

# Live guard-headroom provider for memory-aware routing (issue #2204).
# Registered by Scheduler.__init__ as a bound method returning the bytes left
# under the adaptive-prefill-throttle target (hard ceiling x headroom safety,
# clamped by the abort cap), or a negative value when no ceiling is active.
# Held as a WeakMethod so a torn-down Scheduler auto-unregisters and the route
# falls back to the memory-bounded native fused default.
_HEADROOM_PROVIDER: "weakref.WeakMethod | None" = None
# Backward-compatible override: True = always force fused, False = never force,
# None = memory-aware auto.
_FORCE_TILED: bool | None = None
# Bounded-route reasons already logged. The first engagement per reason logs at
# INFO; repeats stay silent to keep the hot path quiet.
_TILED_ROUTE_LOGGED: "set[str]" = set()


def _note_tiled_route(reason: str, detail: str) -> None:
    if reason in _TILED_ROUTE_LOGGED:
        return
    _TILED_ROUTE_LOGGED.add(reason)
    logger.info(
        "sdpa256: head-dim-256 prefill forcing the memory-bounded path: %s. "
        "The default fast path resumes when guard "
        "headroom allows; "
        "OMLX_SDPA256_TILED=1/0 forces the route.",
        detail,
    )


def set_unfused_headroom_provider(method) -> None:
    """Register a bound method returning the prefill guard's live headroom in
    bytes (negative when no ceiling is active). Lets ``_should_route`` prefer
    the faster unfused fallback whenever its O(L^2) transient fits."""
    global _HEADROOM_PROVIDER
    _HEADROOM_PROVIDER = weakref.WeakMethod(method)


def _parse_force_tiled_env() -> bool | None:
    value = os.environ.get("OMLX_SDPA256_TILED", "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _tiled_route_required(queries, keys) -> bool:
    """Decide forced-fused vs default for a matched call (True = force).

    The stock unfused fallback is faster wherever its score matrix fits
    (issues #2155 / #2204), so force the fused path only when the unfused
    transient would not fit under the guard ceiling — or when no headroom
    info is available, keeping the memory-safe #2025 behavior."""
    if _FORCE_TILED is not None:
        if _FORCE_TILED:
            _note_tiled_route("forced", "forced by OMLX_SDPA256_TILED=1")
        return _FORCE_TILED
    try:
        provider = _HEADROOM_PROVIDER() if _HEADROOM_PROVIDER is not None else None
        if provider is None:
            _note_tiled_route(
                "no-provider",
                "no guard headroom provider registered "
                "(engine without a scheduler, or scheduler gone)",
            )
            return True
        headroom = provider()
        if headroom is None or headroom < 0:
            _note_tiled_route(
                "no-ceiling",
                "memory ceiling not available (enforcer state not yet "
                "propagated)",
            )
            return True
        batch, n_q, q_len, _ = queries.shape
        transient = estimate_unfused_sdpa_call_bytes(
            batch * n_q,
            q_len,
            keys.shape[-2],
            HEAD_DIM,
            score_dtype_size=queries.dtype.size,
        )
        if transient > headroom:
            _note_tiled_route(
                "insufficient-headroom",
                f"unfused transient ~{transient / 2**20:.0f}MiB exceeds live "
                f"guard headroom ~{headroom / 2**20:.0f}MiB at "
                f"kv_len={keys.shape[-2]}",
            )
            return True
        return False
    except Exception:
        _note_tiled_route("probe-error", "guard headroom probe failed")
        logger.debug("sdpa256 headroom probe failed", exc_info=True)
        return True  # headroom info unavailable -> memory-safe default


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

        out_tile = (acc / denom).astype(queries.dtype)
        mx.eval(out_tile)
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(batch, n_q, q_len, value_dim)


def _flash_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Use MLX 0.32.2 native fused SDPA on Metal, portable tiling elsewhere."""
    native_shape = values.shape[-1] == HEAD_DIM and not (
        isinstance(mask, str)
        and mask == "causal"
        and queries.shape[-2] > keys.shape[-2]
    )
    if mx.metal.is_available() and native_shape:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
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
        routing = "force bounded when unfused exceeds guard headroom"
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
