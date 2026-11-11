# SPDX-License-Identifier: Apache-2.0
"""Reference math of the DeepSeek V4.1 prefill forward, in plain MLX ops.

Ported from oMLX (omlx/patches/deepseek_v41). The kernels in kernels.py are
checked against these functions.
"""

from __future__ import annotations

import logging
import math
from functools import cache, lru_cache

import mlx.core as mx
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

# Key rows gathered at once by sparse attention.
ATTENTION_KEYS = 1 << 18
# Scores held at once by the reference index scan.
_INDEX_SCORES = 1 << 26
_NEG_INF = -float("inf")
# Shapes of bfloat16 products already compared with the fp32 product.
_CHECKED_PRODUCTS: set[tuple[int, ...]] = set()
# A sum in another order moves well under 1% of the values by one unit.
_PROBE_ROWS, _PROBE_LIMIT = 16, 0.01


def matmul(x: mx.array, w: mx.array) -> mx.array:
    """x @ w in the types of the operands, as the host multiplies.

    The host rounds a bfloat16 product once from fp32. Each new shape is checked
    against the fp32 product on a few rows to report a build that rounds apart.
    """
    if x.dtype != mx.bfloat16 or w.dtype != mx.bfloat16:
        return x @ w
    out = x @ w
    key = (*x.shape[-2:], *w.shape)
    if key not in _CHECKED_PRODUCTS:
        _CHECKED_PRODUCTS.add(key)
        rows = x[..., :_PROBE_ROWS, :]
        want = (rows.astype(mx.float32) @ w.astype(mx.float32)).astype(x.dtype)
        differ = mx.mean(out[..., :_PROBE_ROWS, :] != want).item()
        if differ > _PROBE_LIMIT:
            logger.warning(
                "DeepSeek V4.1: a bfloat16 product of shape %s x %s differs from "
                "the fp32 product in %.1f%% of its values; this MLX build does "
                "not round it as the host does",
                tuple(x.shape[-2:]),
                tuple(w.shape),
                100 * differ,
            )
    return out


def norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMS norm computed in fp32, in the dtype of ``x``."""
    f = x.astype(mx.float32)
    f = f * mx.rsqrt(mx.mean(f * f, -1, keepdims=True) + eps) * weight
    return f.astype(x.dtype)


@cache
def _rope_freqs(d, base, original_seq_len, factor, beta_fast, beta_slow):
    """RoPE frequencies in fp32, with the YaRN blend when a length is given."""
    exps = (np.arange(0, d, 2, dtype=np.float32) / np.float32(d)).astype(np.float32)
    # float64 pow is within 1 ulp(double), so its fp32 rounding is exact.
    powers = np.power(np.float64(base), exps.astype(np.float64))
    freqs = (np.float32(1.0) / powers.astype(np.float32)).astype(np.float32)
    if original_seq_len > 0:

        def corrected(rotations):
            ratio = original_seq_len / (rotations * 2 * math.pi)
            return d * math.log(ratio) / (2 * math.log(base))

        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), d - 1)
        ramp = np.arange(d // 2, dtype=np.float32) - np.float32(low)
        ramp = (ramp / np.float32(max(high - low, 1e-3))).astype(np.float32)
        smooth = (np.float32(1) - np.clip(ramp, 0, 1).astype(np.float32)).astype(
            np.float32
        )
        scaled = (freqs / np.float32(factor)).astype(np.float32)
        freqs = (
            (scaled * (np.float32(1) - smooth)).astype(np.float32)
            + (freqs * smooth).astype(np.float32)
        ).astype(np.float32)
    return freqs


@lru_cache(maxsize=16)
def _rope_table(params, start, stop, step):
    """cos and sin of the fp32 angles, evaluated in float64."""
    positions = np.arange(start, stop, step).astype(np.float32)
    angles = (positions[:, None] * _rope_freqs(*params)[None, :]).astype(np.float32)
    angles = angles.astype(np.float64)
    return (
        mx.array(np.cos(angles).astype(np.float32)),
        mx.array(np.sin(angles).astype(np.float32)),
    )


def rope_table(positions: range, config: Config, compressed: bool):
    params = (
        config.rope_head_dim,
        float(config.compress_rope_theta if compressed else config.rope_theta),
        config.original_seq_len if compressed else 0,
        float(config.rope_factor),
        float(config.beta_fast),
        float(config.beta_slow),
    )
    return _rope_table(params, positions.start, positions.stop, positions.step)


def rope_apply(x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False):
    """Rotate the last values of x [1, L, ..., D] by tables [L, pairs]."""
    d = 2 * cos.shape[-1]
    shape = (1, cos.shape[0], *([1] * (x.ndim - 3)), d // 2)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if inverse:
        sin = -sin
    tail = x[..., -d:].astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], -1)
    rotated = rotated.reshape(*x.shape[:-1], d).astype(x.dtype)
    return mx.concatenate([x[..., :-d], rotated], -1)


def swiglu(gate, up, weights, dtype, limit):
    """silu(gate) * up in fp32 with the reference clamps, rounded to dtype."""
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    if limit:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    y = (gate * mx.sigmoid(gate)) * up
    if weights is not None:
        y = y * weights
    return y.astype(dtype)


def combine(routed, inverse, shared):
    """Routed rows [R, 1, D] sorted by expert back to tokens, summed over the
    routes of a token in fp32, plus the shared expert [L, D]."""
    rows = routed[inverse].reshape(shared.shape[0], -1, shared.shape[-1])
    total = mx.sum(rows.astype(mx.float32), axis=1) + shared.astype(mx.float32)
    return total.astype(shared.dtype)


def hc_mixes(x, fn_t, scale, base, config: Config):
    """Hyper-connection weights (pre, post, comb) of streams x [1, L, n, D]."""
    n, eps = config.hc_mult, config.hc_eps
    flat = x.flatten(-2).astype(mx.float32)
    mixes = (flat @ fn_t) * mx.rsqrt(
        mx.mean(flat * flat, -1, keepdims=True) + config.norm_eps
    )
    pre = mx.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + eps
    post = 2 * mx.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = mixes[..., 2 * n :] * scale[2] + base[2 * n :]
    comb = mx.softmax(comb.reshape(*mixes.shape[:-1], n, n), -1) + eps
    # Sinkhorn: columns first, then rows and columns in turn.
    comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, -1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre_norm(x, pre, weight, eps):
    """Mix the streams with ``pre``; the sum is rounded before the norm."""
    mixed = mx.sum(x.astype(mx.float32) * pre[..., None], axis=-2).astype(x.dtype)
    return norm(mixed, weight, eps)


def hc_post(x, residual, post, comb):
    """Streams after a sublayer: post * x + comb^T residual, per position."""
    f = residual.astype(mx.float32)
    mixed = post[..., None] * x[..., None, :]
    for i in range(residual.shape[-2]):
        mixed = mixed + comb[..., i, :, None] * f[..., i, None, :]
    return mixed.astype(x.dtype)


def window_rows(start: int, length: int, size: int, held: int) -> mx.array:
    """Window rows [L, W] each query attends over, -1 for masked slots.

    The window holds ``held`` rows from before ``start``, then the chunk's rows.
    Each query lists its causal local window, the current token last.
    """
    position = mx.arange(start, start + length)[:, None]
    if start == 0:
        local = mx.maximum(position - size + 1, 0) + mx.arange(min(length, size))
    else:
        local = position - size + 1 + mx.arange(size)
    seen = (local >= max(0, start - held)) & (local <= position)
    return mx.where(seen, local - (start - held), -1)


def _attend(q, keys, valid, sink, scale):
    """Softmax attention of q [B, H, D] over each row's keys [B, S, D].

    Keys go in tiles of 64: a tile's weights use the running maximum through it
    and are rounded to the activation dtype, while the denominator keeps fp32.
    """
    dtype = q.dtype
    b, h, _ = q.shape
    values = keys.astype(mx.float32)
    scores = (q.astype(mx.float32) @ values.swapaxes(-1, -2)) * scale
    scores = mx.where(valid[:, None, :], scores, _NEG_INF)
    pad = -scores.shape[-1] % 64
    if pad:
        scores = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=_NEG_INF)
        values = mx.pad(values, [(0, 0), (0, pad), (0, 0)])
    tiles = scores.shape[-1] // 64
    scores = scores.reshape(b, h, tiles, 64)
    tile_max = mx.max(scores, axis=-1)
    running, maxima = mx.full((b, h), -1e30), []
    for t in range(tiles):
        running = mx.maximum(running, tile_max[..., t])
        maxima.append(running)
    maxima = mx.stack(maxima, -1)
    p = mx.exp(scores - maxima[..., None])
    # Rescale every tile to the final maximum.
    carry = mx.exp(maxima - maxima[..., -1:])
    denominator = mx.sum(mx.sum(p, -1) * carry, -1) + mx.exp(sink - maxima[..., -1])
    weights = p.astype(dtype).astype(mx.float32) * carry[..., None]
    out = (weights.reshape(b, h, -1) @ values) / denominator[..., None]
    return out.astype(dtype)


def sparse_attention(q, window, pooled, wi, ci, sink, scale):
    """Attention over window rows ``wi`` then compressed rows ``ci`` per query.

    q [1, L, H, D]; window [NW, D] and pooled [NC, D] hold unpacked values;
    wi [L, W] and ci [L, C] hold row ids, negative for masked slots.
    """
    length = q.shape[1]
    slots = wi.shape[-1] + (ci.shape[-1] if ci is not None else 0)
    step = max(1, ATTENTION_KEYS // slots)
    out = []
    for begin in range(0, length, step):
        local = wi[begin : begin + step]
        keys, valid = [window[mx.maximum(local, 0)]], [local >= 0]
        if ci is not None and ci.shape[-1]:
            top = ci[begin : begin + step]
            ok = (top >= 0) & (top < pooled.shape[0])
            keys.append(pooled[mx.where(ok, top, 0)])
            valid.append(ok)
        part = _attend(
            q[0, begin : begin + step],
            mx.concatenate(keys, 1),
            mx.concatenate(valid, 1),
            sink,
            scale,
        )
        mx.eval(part)
        out.append(part)
    return mx.concatenate(out, 0)[None]


def select_topk(scores: mx.array, count: int) -> mx.array:
    """Ids of each row's ``count`` largest scores, ascending, -1 for -inf.

    Equal scores go to the lower id, as in a stable sort.
    """
    n = scores.shape[-1]
    kth = mx.sort(scores, axis=-1)[:, n - count : n - count + 1]
    above, ties = scores > kth, scores == kth
    budget = count - mx.sum(above, -1, keepdims=True)
    take = above | (ties & (mx.cumsum(ties.astype(mx.int32), axis=-1) <= budget))
    ids = mx.arange(n, dtype=mx.int32)
    chosen = mx.where(take, mx.where(scores > _NEG_INF, ids, -1), n)
    return mx.sort(chosen, axis=-1)[:, :count]


def index_scores(q, keys_t, weights, start, ratio):
    """Index scores of queries q [B, H, D] against keys_t [D, N], in fp32.

    The head dot products, the weighted ReLU and the head sum are each rounded
    to the activation dtype. Keys a query cannot see yet score -inf.
    """
    dtype = q.dtype
    dots = mx.maximum(q @ keys_t, 0).astype(mx.float32)
    heads = (dots * weights[..., None]).astype(dtype).astype(mx.float32)
    scores = mx.sum(heads, axis=1).astype(dtype).astype(mx.float32)
    visible = mx.arange(start + 1, start + q.shape[0] + 1) // ratio
    hidden = mx.arange(keys_t.shape[-1]) >= visible[:, None]
    return mx.where(hidden, _NEG_INF, scores)


def index_topk(q, keys, weights, start, ratio, count):
    """Compressed rows each query attends over: [1, L, min(count, N)] int32.

    q [1, L, H, D] quantized index queries, keys [N, D] unpacked index keys,
    weights [1, L, H] fp32 head weights; ``start`` is the first position.
    """
    length, heads = q.shape[1], q.shape[2]
    n = keys.shape[0]
    count = min(count, n)
    if not count:
        return mx.zeros((1, length, 0), mx.int32)
    keys_t = mx.contiguous(keys.T)
    step = max(1, _INDEX_SCORES // (heads * n))
    out = []
    for begin in range(0, length, step):
        scores = index_scores(
            q[0, begin : begin + step],
            keys_t,
            weights[0, begin : begin + step],
            start + begin,
            ratio,
        )
        part = select_topk(scores, count)
        mx.eval(part)
        out.append(part)
    return mx.concatenate(out, 0)[None]
