# SPDX-License-Identifier: Apache-2.0
"""Prefill attention fast paths that MLX's fused SDPA does not cover.

Two gaps in ``mx.fast.scaled_dot_product_attention`` (mlx 0.32) cost a lot of
prefill time on large GPUs:

* Sliding-window attention is expressed as an array mask over the full
  ``[L, S]`` score matrix, so every query still scores every key and the
  window only masks the result. For a 128-token window at 4K context that is
  ~97% wasted work (and ~4 GB of scores per layer on the unfused path).
  ``blocked_sliding_window_attention`` tiles the queries into blocks that only
  see their ``block + window`` key span.
* Prefill with different query/key and value head dims (e.g. 192/128) has no
  fused kernel, so MLX materialises the full score matrix. On NAX (M5) GPUs
  ``omlx.utils.nax_attention`` runs MLX's tensor-unit flash-attention kernel
  with a separate value head dim as a JIT kernel. Otherwise zero-padding V
  (or Q/K/V) makes one of MLX's fused kernels applicable; the extra output
  columns are exactly zero and sliced away. MLX also routes head dim 192/256
  prefill to the unfused path by default, which measures slower on NAX (M5)
  GPUs, so the fused kernel is requested explicitly there.

Both helpers compute the same attention as the masked full computation (up to
floating-point summation order).
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available
from omlx.utils.nax_attention import nax_mixed_head_dim_attention, uses_key_passes

# Kill switch for A/B comparisons: OMLX_FAST_ATTENTION=0 keeps MLX's default
# SDPA routing everywhere.
_ENABLED = os.environ.get("OMLX_FAST_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "off",
}


@lru_cache(maxsize=1)
def _nax_available() -> bool:
    try:
        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=None)
def _native_mixed_dims_supported(qk_dim: int, v_dim: int) -> bool:
    """True when MLX's fused prefill kernel accepts ``qk_dim``/``v_dim`` directly."""
    try:
        q = mx.zeros((1, 1, 16, qk_dim), mx.float16)
        v = mx.zeros((1, 1, 16, v_dim), mx.float16)
        out = mx.fast.scaled_dot_product_attention(
            q, q, v, scale=1.0, mask="causal", force_fused=True
        )
        mx.eval(out)
        return True
    except Exception:  # noqa: BLE001 - older MLX raises ValueError
        return False


def _pad_last(x: mx.array, width: int) -> mx.array:
    pad = [(0, 0)] * (x.ndim - 1) + [(0, width - x.shape[-1])]
    return mx.pad(x, pad)


def _block_sdpa(queries, keys, values, *, scale, mask, sinks):
    """Fused SDPA over window blocks, mixed head dims included.

    MLX has no fused kernel for mixed head dims (e.g. 192/128) unless it
    carries the NAX value-head-dim kernel; without one the block attention
    runs as the JIT NAX kernel on M5 GPUs instead of MLX's unfused fallback.
    """
    qk_dim, v_dim = queries.shape[-1], values.shape[-1]
    if qk_dim != v_dim and not _native_mixed_dims_supported(qk_dim, v_dim):
        out = nax_mixed_head_dim_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )
        if out is not None:
            return out
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=sinks
    )


def mixed_head_dim_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask,
    sinks: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Fused SDPA for prefill with ``qk_dim > v_dim``; None when not applicable.

    Four exact routes, best first:

    * MLX builds whose fused kernel takes the mixed head dims natively
      (NAX kernel with a separate value head dim) are called directly,
      except for long key ranges: the oMLX JIT kernel below runs those in
      several key-range dispatches that keep the K/V stream on chip (same
      arithmetic; its head-dim split variant only reorders the fp32 sums).
    * Otherwise, on NAX (M5) GPUs, the same NAX kernel runs as an oMLX JIT
      kernel (``nax_mixed_head_dim_attention``, 192/128 head dims).
    * Otherwise, on NAX GPUs, Q/K/V are zero-padded to 256 so the
      tensor-unit head-dim-split kernel runs: padded query/key columns add
      exactly zero to every score and padded value columns are sliced away.
      This beats padding V to 192 (which lands on the classic kernel, ~3x
      slower) despite the extra multiply-adds.
    * Elsewhere V is zero-padded to the query head dim for the classic kernel.
    """
    qk_dim, v_dim = queries.shape[-1], values.shape[-1]
    if (
        not _ENABLED
        or qk_dim <= v_dim
        or queries.shape[2] <= 8
        or not _nax_available()
        or qk_dim not in (64, 72, 80, 96, 128, 192, 256)
        or not (mask is None or isinstance(mask, str) or mask.dtype == mx.bool_)
    ):
        return None
    native = _native_mixed_dims_supported(qk_dim, v_dim)
    if not native or uses_key_passes(queries, keys):
        out = nax_mixed_head_dim_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )
        if out is not None:
            return out
    if native:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
        )
    if qk_dim in (96, 128, 192) and qk_dim not in (64, 96, 128):
        # No NAX kernel for this width: pad to the 256-wide split kernel.
        width = 256
        out = mx.fast.scaled_dot_product_attention(
            _pad_last(queries, width),
            _pad_last(keys, width),
            _pad_last(values, width),
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
        )
        return out[..., :v_dim]
    out = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        _pad_last(values, qk_dim),
        scale=scale,
        mask=mask,
        sinks=sinks,
        force_fused=True,
    )
    return out[..., :v_dim]


# Block masks are identical for every sliding-window layer of one forward (the
# model passes the same mask array to all of them), so the last one is reused
# instead of being rebuilt per layer. Keyed by thread and by the identity of
# the caller's mask; the entry keeps that mask alive so its id cannot be
# recycled while cached.
_BLOCK_MASK_CACHE: list = [None]


def _window_block_mask(
    *,
    nb: int,
    block: int,
    window: int,
    lead: int,
    user_mask: Optional[mx.array],
    col_start: int,
    pad_q: int,
) -> mx.array:
    key = (
        threading.get_ident(),
        nb,
        block,
        window,
        lead,
        col_start,
        pad_q,
        None if user_mask is None else id(user_mask),
    )
    entry = _BLOCK_MASK_CACHE[0]
    if entry is not None and entry[0] == key:
        return entry[2]
    span = block + window
    # In block coordinates query r sits at key index r + window; it sees keys
    # j with r < j <= r + window. Keys in the zero padding (global index below
    # `lead`) are masked; only the first blocks can contain padding.
    r = mx.arange(block)[:, None]
    j = mx.arange(span)[None, :]
    base = (j > r) & (j <= r + window)
    starts = (mx.arange(nb) * block)[:, None, None]
    block_mask = base[None] & ((starts + j[None]) >= lead)
    if user_mask is not None:
        # Re-index the caller's key columns to the padded block layout.
        L = user_mask.shape[-2]
        cols = user_mask.reshape(L, -1)[:, col_start:]
        if pad_q or lead:
            cols = mx.pad(cols, [(0, pad_q), (lead, pad_q)])
        rows = cols.reshape(nb, block, cols.shape[-1])
        idx = (mx.arange(nb) * block)[:, None] + mx.arange(span)[None, :]
        user_blocks = mx.take_along_axis(
            rows, mx.broadcast_to(idx[:, None, :], (nb, block, span)), axis=2
        )
        block_mask = block_mask & user_blocks
    block_mask = block_mask[:, None]  # broadcast over heads
    _BLOCK_MASK_CACHE[0] = (key, user_mask, block_mask)
    return block_mask


def window_query_padding(num_queries: int, *, block: int = 128) -> int:
    """Rows a caller can append to ``num_queries`` sliding-window queries.

    ``blocked_sliding_window_attention`` runs whole ``block``-query blocks and
    otherwise pads the queries itself, copying every query head. A caller can
    instead pad a narrower tensor upstream (e.g. the query projection's input,
    whose rows the projection and RoPE treat independently) and pass the
    padded queries with ``query_len=num_queries``. 0 when no padding is needed
    or the blocked path does not run for this many queries.
    """
    if not _ENABLED or num_queries < 2 * block:
        return 0
    return (-num_queries) % block


def blocked_sliding_window_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    window: int,
    sinks: Optional[mx.array] = None,
    mask=None,
    block: int = 128,
    query_len: Optional[int] = None,
) -> Optional[mx.array]:
    """Causal sliding-window attention computed per query block.

    ``keys``/``values`` hold ``P >= 0`` prefix positions immediately preceding
    the ``L`` queries followed by the queries' own positions (the layout a
    KVCache / RotatingKVCache returns during prefill). Query ``i`` attends to
    keys at positions ``(i - window, i]``, matching mlx-lm's
    ``create_causal_mask(window_size=window)``. A boolean ``mask`` of shape
    ``[..., L, S]`` (e.g. one carrying batch-cache left padding) is honoured
    by slicing it per block; it must not allow keys outside the window.
    Returns None when the inputs do not fit this layout (batched inputs,
    short prompts, uneven blocks, additive masks).

    With ``query_len`` set, ``queries`` holds ``L = query_len`` real rows
    followed by the ``window_query_padding(L)`` padding rows the blocked
    layout needs (their outputs are dropped), so no query copy is made here.

    The query blocks and their overlapping ``block + window`` key spans are
    strided views of one contiguous copy of the key/value rows, so the fused
    kernel reads them in place (no per-block gather or reshape copies).
    """
    B, H, Lq, D = queries.shape
    L = Lq if query_len is None else query_len
    S = keys.shape[2]
    prefix = S - L
    if (
        not _ENABLED
        or B != 1
        or window <= 0
        or prefix < 0
        or L < 2 * block
        or Lq not in (L, L + (-L) % block)
        or keys.shape[2] != values.shape[2]
    ):
        return None
    user_mask = None
    if isinstance(mask, mx.array):
        if mask.dtype != mx.bool_ or mask.shape[-2:] != (L, S) or mask.size != L * S:
            return None
        user_mask = mask
    elif mask is not None and mask != "causal":
        return None
    # Prompt chunks are rarely a multiple of the block (the scheduler keeps the
    # last prompt token for generation, so 4095 is typical): pad the queries
    # (unless the caller already did) and the corresponding key/value
    # positions and drop the padded rows at the end. Padded keys sit after
    # every real query position, so the causal window never lets a real query
    # see them.
    pad_q = (-L) % block
    Lp = L + pad_q
    Hk = keys.shape[1]
    v_dim = values.shape[-1]
    nb = Lp // block
    used = min(prefix, window)
    lead = window - used  # zero-padded (masked) positions before the prefix
    span = block + window
    rows = window + Lp  # key rows per head: lead + used + L + pad_q

    def key_rows(x):
        # One contiguous run per head: [lead zeros | prefix | chunk | pad].
        x = x[:, :, S - L - used :, :]
        if lead or pad_q:
            x = mx.pad(x, [(0, 0), (0, 0), (lead, pad_q), (0, 0)])
        return x

    k = key_rows(keys)
    v = key_rows(values)
    # Block b's keys are rows [b * block, b * block + span) of each head.
    kb = mx.as_strided(k, (nb, Hk, span, D), (block * D, rows * D, D, 1))
    vb = mx.as_strided(
        v, (nb, Hk, span, v_dim), (block * v_dim, rows * v_dim, v_dim, 1)
    )
    if Lq != Lp:
        queries = mx.pad(queries, [(0, 0), (0, 0), (0, pad_q), (0, 0)])
    qb = queries.reshape(H, nb, block, D).transpose(1, 0, 2, 3)

    block_mask = _window_block_mask(
        nb=nb,
        block=block,
        window=window,
        lead=lead,
        user_mask=user_mask,
        col_start=S - L - used,
        pad_q=pad_q,
    )
    out = _block_sdpa(qb, kb, vb, scale=scale, mask=block_mask, sinks=sinks)
    # The fused kernel writes [nb, block, H, v_dim] rows, so this regrouping
    # to [1, H, L, v_dim] (and the caller's transpose back) stays a view.
    out = out.transpose(1, 0, 2, 3).reshape(B, H, Lp, v_dim)
    return out[:, :, :L]
