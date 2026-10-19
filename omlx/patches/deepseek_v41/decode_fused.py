# SPDX-License-Identifier: MIT
"""Decode paths for DeepSeek V4.1 built from the fused kernels.

Mirror language.Block.__call__ / Attention.__call__ for one decoded token (decode_block),
for one token of each of B requests (decode_block_nb) and for an MTP verify block of R
consecutive tokens of one request (decode_block_verify); never for CED or image spans.
Compressor, indexer and cache bookkeeping are the existing implementations; the
hyper-connection, projection, norm, RoPE, quantization, attention and MoE stages run as
fused kernels (decode_layer.py, decode_moe.py), each of which documents its rounding points
against the DeepSeek-V4.1-Flash-1017 reference.
"""

from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import decode_layer as DL
from . import decode_moe as DM
from . import growth
from .quantization import QuantizedProjection, pack_activation

_TPG = 256


def _cos_sin(c, compressed, start):
    from .language import _rope_table

    params = (
        c.rope_head_dim,
        c.compress_rope_theta if compressed else c.rope_theta,
        c.original_seq_len if compressed else 0,
        c.rope_factor,
        c.beta_fast,
        c.beta_slow,
    )
    cos, sin = _rope_table(params, (start, start + 1, 1))
    return cos.reshape(-1), sin.reshape(-1)


@lru_cache(maxsize=None)
def _window_idx(window, start, old_len):
    """Attention.__call__ window indices for one query at `start` (host-side, cached)."""
    local = start - window + 1 + np.arange(window)
    valid = (local >= max(0, start - old_len)) & (local <= start)
    return mx.array(np.where(valid, local - (start - old_len), -1).astype(np.int32)[None, None])


@lru_cache(maxsize=None)
def _const(values, dtype=mx.float32):
    """Small materialized constant (cached): a lazy mx.zeros / mx.full would cost a fill
    dispatch on every decode step."""
    a = mx.array(list(values), dtype)
    mx.eval(a)
    return a


@lru_cache(maxsize=None)
def _empty(shape, dtype):
    a = mx.zeros(shape, dtype)
    mx.eval(a)
    return a


def _dummy(dtype):
    return _const((0,), dtype)


def _scalars(c):
    return _const((c.norm_eps,)), _const((c.hc_eps,))


def _woa_mxfp8(att):
    """wo_a as MXFP8, or None. The checkpoint stores wo_a as FP8 with power-of-two block scales
    and the reference's convert.py dequantizes it to bf16, so every bf16 value is FP8 * 2^k per
    32-group: quantizing back is lossless (verified bitwise on all 40 layers of
    DeepSeek-V4.1-Flash and -1017), and the einsum's products are unchanged. Checked once per
    layer; a wo_a that does not round-trip exactly keeps the unfused path."""
    cached = getattr(att, "_woa_mx", None)
    if cached is None:
        w = att.wo_a.weight
        cached = False
        if not isinstance(att.wo_a, QuantizedProjection) and w.dtype == mx.bfloat16 and w.shape[-1] % 32 == 0:
            q, s = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")[:2]
            back = mx.dequantize(q, s, None, group_size=32, bits=8, mode="mxfp8", dtype=w.dtype)
            if bool(mx.array_equal(back, w).item()):
                cached = (q, s)
        object.__setattr__(att, "_woa_mx", cached)
    return cached or None


def _mxfp8_projection(p):
    """A projection the fused GEMVs can read: MXFP8, group 32, FP8-quantized input."""
    return (
        isinstance(p, QuantizedProjection)
        and p.mode == "mxfp8"
        and p.group_size == 32
        and getattr(p, "quantize_input", False)
        and "biases" not in p
    )


def mix(blk, kind, c, res, pre_in, sub=None, post=None, comb=None):
    """(optional hc_post of the previous sublayer) + hc_mixes + hc_pre + RMSNorm + act_quant.
    Returns (coefficients[24] = pre, post, comb), normed input bf16, its act_quant, h."""
    hc, dim = c.hc_mult, c.dim
    n = hc * dim
    nparts = (n + _TPG - 1) // _TPG
    mixw = (2 + hc) * hc
    eps, hc_eps = _scalars(c)
    dummy = _dummy(mx.float32)
    posted = sub is not None
    partial, hout = DL.kernel("mix_partial")(
        inputs=[res, blk[f"hc_{kind}_fn"], sub if posted else dummy, post if posted else dummy, comb if posted else dummy],
        template=[("HC", hc), ("DIM", dim), ("MIX", mixw), ("TPG", _TPG), ("POST", posted)],
        grid=(nparts * _TPG, 1, 1), threadgroup=(_TPG, 1, 1),
        output_shapes=[(nparts * (mixw + 1),), (n if posted else 1,)],
        output_dtypes=[mx.float32, mx.bfloat16])
    h = hout if posted else res
    norm = blk.attn_norm if kind == "attn" else blk.ffn_norm
    coef, xn, xq = DL.kernel("mix_final")(
        inputs=[partial, h, pre_in, norm.weight, blk[f"hc_{kind}_scale"], blk[f"hc_{kind}_base"], eps, hc_eps],
        template=[("HC", hc), ("DIM", dim), ("MIX", mixw), ("NPART", nparts), ("ITERS", c.hc_sinkhorn_iters)],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(mixw,), (dim,), (dim,)], output_dtypes=[mx.float32, mx.bfloat16, mx.float32])
    return coef, xn, xq, h


def gemv_fp8(x, a, b=None):
    na, k = a.weight.shape[0], a.weight.shape[1] * 4
    nb = 0 if b is None else b.weight.shape[0]
    assert na % 4 == 0 and nb % 4 == 0, "4-row simdgroups must not straddle the two matrices"
    dummy_w, dummy_s = _dummy(mx.uint32), _dummy(mx.uint8)
    ya, yb = DL.kernel("gemv_fp8")(
        inputs=[x, a.weight, a.scales, b.weight if b is not None else dummy_w, b.scales if b is not None else dummy_s],
        template=[("NA", na), ("NB", nb), ("K", k)],
        grid=((na + nb + 7) // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(na,), (max(nb, 1),)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    return ya, (yb if b is not None else None)


def attn_mid(att, qa, kv_in, cos, sin, c):
    eps, _ = _scalars(c)
    return DL.kernel("attn_mid")(
        inputs=[qa, kv_in, att.q_norm.weight, att.kv_norm.weight, cos, sin, eps],
        template=[("QR", c.q_lora_rank), ("HD", c.head_dim), ("RD", c.rope_head_dim)],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(c.q_lora_rank,), (c.q_lora_rank,), (c.head_dim + c.head_dim // 32,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.uint8])


def wqb_rope(att, qrq, cos, sin, c):
    nq = c.n_heads * c.head_dim
    return DL.kernel("wqb_rope")(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, cos, sin],
        template=[("N", nq), ("K", c.q_lora_rank), ("HD", c.head_dim), ("RD", c.rope_head_dim)],
        grid=(nq // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(nq,)], output_dtypes=[mx.bfloat16])[0]


def woa(att, oq, c):
    wq, ws = _woa_mxfp8(att)
    rows = c.o_groups * c.o_lora_rank
    return DL.kernel("woa")(
        inputs=[oq, wq, ws],
        template=[("R", c.o_lora_rank), ("K", c.n_heads * c.head_dim // c.o_groups)],
        grid=(rows // 32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows,)], output_dtypes=[mx.float32])[0]


def hc_post(f, h, post, comb, c):
    hc, dim = c.hc_mult, c.dim
    return DL.kernel("hc_post")(
        inputs=[f, h, post, comb], template=[("HC", hc), ("DIM", dim)],
        grid=(hc * dim, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(hc * dim,)], output_dtypes=[mx.bfloat16])[0]


def split_attention(q, window, pooled, wi, ci, sink, cos, sin, c):
    """Single-query sparse attention + o epilogue (see decode_layer._ATTN_SCORES / _ATTN_PV /
    _O_MERGE). q: [H*D] bf16; window / pooled: packed rows; wi / ci: [1, 1, W] / [1, 1, C]."""
    h, d = c.n_heads, c.head_dim
    nw, nc, w, cc = window.shape[1], pooled.shape[1], wi.shape[-1], ci.shape[-1]
    nb = max(1, (w + cc + 63) // 64)
    wi = wi.reshape(-1).astype(mx.int32) if wi.size else _dummy(mx.int32)
    ci = ci.reshape(-1).astype(mx.int32) if ci.size else _dummy(mx.int32)
    window = window.reshape(-1) if window.size else _dummy(mx.uint8)
    pooled = pooled.reshape(-1) if pooled.size else _dummy(mx.uint8)
    meta = mx.array([w, cc, nw, nc, nb], mx.int32)
    grid = (h // 8 * 1024, nb, 1)
    s, bmax = DL.kernel("attn_scores")(
        inputs=[q, window, pooled, wi, ci, meta, _const((d**-0.5,))],
        template=[("D", d)], grid=grid, threadgroup=(1024, 1, 1),
        output_shapes=[(h * nb * 64,), (h * nb,)], output_dtypes=[mx.float32, mx.float32])
    pv, den = DL.kernel("attn_pv")(
        inputs=[window, pooled, wi, ci, meta, s, bmax],
        template=[("D", d)], grid=grid, threadgroup=(1024, 1, 1),
        output_shapes=[(h * nb * d,), (h * nb,)], output_dtypes=[mx.float32, mx.float32])
    return DL.kernel("o_merge")(
        inputs=[pv, den, bmax, sink, meta, cos, sin],
        template=[("NH", h), ("HD", d), ("RD", c.rope_head_dim), ("TPG", 1024)],
        grid=((h * d // 32 + 31) // 32 * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(h * d,)], output_dtypes=[mx.float32])[0]


def compress_index(att, x, qr, cache, shared, start, c):
    """Compressor and indexer bookkeeping of one request at `start` (the compressed part of
    Attention.__call__). Returns its compressed rows and selected compressed indices."""
    layer = att._layer
    ratio = c.compress_ratios[layer]
    if not ratio:
        return _empty((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8), _empty((1, 1, 0), mx.int32)
    x3 = x.reshape(1, 1, -1)
    latent = None
    if layer in c.kv_source_layers:
        latent = att.compressor(x3, cache, start)
        if cache[2] is None:
            cache[2] = pack_activation(mx.zeros((1, 0, c.head_dim), x.dtype), bits=4, group_size=16, e4m3_scale=True)
        shared["kv"] = cache[2][:, : start // ratio]
    if layer in c.index_source_layers:
        shared["idx"] = att.indexer(x3, qr.reshape(1, 1, -1), latent, cache, shared, start, ratio)
    if latent is not None:
        from .language import rope

        compressed = rope(latent, range(start // ratio * ratio, (start + 1) // ratio * ratio, ratio), c, True)
        compressed = pack_activation(compressed, bits=4, group_size=16, e4m3_scale=True)
        shared["kv"] = cache[2] = growth.append(cache, 2, shared["kv"], compressed, start // ratio)
    return shared["kv"], shared["idx"]


def attention_core(att, x, qr, q, packed, cache, shared, start, cos, sin, c):
    """One request's attention after its projections: window KV update, compressor, indexer,
    sparse attention and o epilogue. x: [dim] bf16, qr: [q_lora] bf16, q: [H*D] bf16 (RoPE'd),
    packed: the new window row. Returns act_quant(o) [H*D] fp32 (the wo_a input)."""
    new = packed.reshape(1, 1, -1)
    old = cache[1]
    old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
    kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
    cache[1] = kv[:, -c.window_size:]
    idx = _window_idx(c.window_size, start, old_len)
    pooled, ci = compress_index(att, x, qr, cache, shared, start, c)
    return split_attention(q.reshape(-1), kv, pooled, idx, ci, att.attn_sink, cos, sin, c)


def decode_attention(att, x, xq, cache, shared, start, c):
    """Attention.__call__ for one token. x: [dim] bf16 (attn_norm output), xq: act_quant(x)."""
    ratio = c.compress_ratios[att._layer]
    qa, kv_in = gemv_fp8(xq, att.wq_a, att.wkv)
    cos, sin = _cos_sin(c, bool(ratio), start)
    qr, qrq, packed = attn_mid(att, qa, kv_in, cos, sin, c)
    q = wqb_rope(att, qrq, cos, sin, c)
    oq = attention_core(att, x, qr, q, packed, cache, shared, start, cos, sin, c)
    yq = woa(att, oq, c)
    return gemv_fp8(yq, att.wo_b)[0]


# ---- batched decode: B requests, one token each --------------------------------------------
MAX_BATCH = 8


def mix_nb(blk, kind, c, res, pre_in, sub=None, post=None, comb=None):
    """mix() for B requests (rows of res [B, hc*dim], pre_in [B, hc], sub [B, dim], post [B, hc],
    comb [B, hc*hc]); each request runs the single-request kernels on its own slice."""
    b_n = res.shape[0]
    hc, dim = c.hc_mult, c.dim
    n = hc * dim
    nparts = (n + _TPG - 1) // _TPG
    mixw = (2 + hc) * hc
    eps, hc_eps = _scalars(c)
    dummy = _dummy(mx.float32)
    posted = sub is not None
    partial, hout = DL.kernel("mix_partial_b")(
        inputs=[res, blk[f"hc_{kind}_fn"], sub if posted else dummy, post if posted else dummy, comb if posted else dummy],
        template=[("HC", hc), ("DIM", dim), ("MIX", mixw), ("TPG", _TPG), ("POST", posted)],
        grid=(nparts * _TPG, b_n, 1), threadgroup=(_TPG, 1, 1),
        output_shapes=[(b_n * nparts * (mixw + 1),), (b_n * n if posted else 1,)],
        output_dtypes=[mx.float32, mx.bfloat16])
    h = hout.reshape(b_n, n) if posted else res
    norm = blk.attn_norm if kind == "attn" else blk.ffn_norm
    coef, xn, xq = DL.kernel("mix_final_b")(
        inputs=[partial, h, pre_in, norm.weight, blk[f"hc_{kind}_scale"], blk[f"hc_{kind}_base"], eps, hc_eps],
        template=[("HC", hc), ("DIM", dim), ("MIX", mixw), ("NPART", nparts), ("ITERS", c.hc_sinkhorn_iters)],
        grid=(1024, b_n, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(b_n * mixw,), (b_n * dim,), (b_n * dim,)], output_dtypes=[mx.float32, mx.bfloat16, mx.float32])
    return coef.reshape(b_n, mixw), xn.reshape(b_n, dim), xq.reshape(b_n, dim), h


def _mma_split(tiles, groups):
    """K split for fp8_mma8: enough threadgroups to fill the GPU, whole scale groups each."""
    for ks in ((1,) if tiles >= 2048 else (2, 1) if tiles >= 512 else (4, 2, 1)):
        if groups % ks == 0:
            return ks
    return 1


def gemv_fp8_nb(x, a, b=None, mma=None):
    """gemv_fp8 for x [B, K]: one weight read for all B vectors. Bit for bit gemv_fp8 per
    vector without matrix tiles (mma=False); default DM.use_mma(B)."""
    b_n = x.shape[0]
    na, k = a.weight.shape[0], a.weight.shape[1] * 4
    nb = 0 if b is None else b.weight.shape[0]
    assert na % 4 == 0 and nb % 4 == 0
    if (DM.use_mma(b_n) if mma is None else mma) and na % 8 == 0 and nb % 8 == 0:
        # 8-element dummies: inputs under 8 elements are passed in constant address space.
        dw, ds = _const((0,) * 8, mx.uint32), _const((0,) * 8, mx.uint8)
        tiles = (na + nb) // 8
        ks = _mma_split(tiles, k // 32)
        ya, yb = DL.kernel("gemv_fp8_mma")(
            inputs=[x, a.weight, a.scales, b.weight if b is not None else dw, b.scales if b is not None else ds],
            template=[("NA", na), ("NB_ROWS", nb), ("K", k), ("NB", b_n), ("KS", ks)],
            grid=(tiles * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
            output_shapes=[(b_n * na,), (b_n * max(nb, 1),)], output_dtypes=[mx.bfloat16, mx.bfloat16])
        return ya.reshape(b_n, na), (yb.reshape(b_n, nb) if b is not None else None)
    dummy_w, dummy_s = _dummy(mx.uint32), _dummy(mx.uint8)
    ya, yb = DL.kernel("gemv_fp8_nb")(
        inputs=[x, a.weight, a.scales, b.weight if b is not None else dummy_w, b.scales if b is not None else dummy_s],
        template=[("NA", na), ("NB_ROWS", nb), ("K", k), ("NB", b_n)],
        grid=((na + nb + 7) // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(b_n * na,), (b_n * max(nb, 1),)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    return ya.reshape(b_n, na), (yb.reshape(b_n, nb) if b is not None else None)


def _cos_sin_nb(c, compressed, starts):
    trig = [_cos_sin(c, compressed, s) for s in starts]
    return mx.stack([t[0] for t in trig]), mx.stack([t[1] for t in trig])


def attn_mid_nb(att, qa, kv_in, cos, sin, c):
    """attn_mid for B rows (each row: the single-row kernel)."""
    b_n = qa.shape[0]
    eps, _ = _scalars(c)
    qr, qrq, packed = DL.kernel("attn_mid_b")(
        inputs=[qa, kv_in, att.q_norm.weight, att.kv_norm.weight, cos, sin, eps],
        template=[("QR", c.q_lora_rank), ("HD", c.head_dim), ("RD", c.rope_head_dim)],
        grid=(1024, b_n, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(b_n * c.q_lora_rank,), (b_n * c.q_lora_rank,), (b_n * (c.head_dim + c.head_dim // 32),)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.uint8])
    return qr.reshape(b_n, -1), qrq.reshape(b_n, -1), packed.reshape(b_n, -1)


def wqb_rope_nb(att, qrq, cos, sin, c, mma):
    """wqb_rope for B rows: bit for bit per row without matrix tiles (mma=False)."""
    b_n = qrq.shape[0]
    nq = c.n_heads * c.head_dim
    tmpl = [("N", nq), ("K", c.q_lora_rank), ("HD", c.head_dim), ("RD", c.rope_head_dim), ("NB", b_n)]
    if mma:
        ks = _mma_split(nq // 8, c.q_lora_rank // 32)
        return DL.kernel("wqb_rope_mma")(
            inputs=[qrq, att.wq_b.weight, att.wq_b.scales, cos, sin], template=tmpl + [("KS", ks)],
            grid=(nq // 8 * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
            output_shapes=[(b_n * nq,)], output_dtypes=[mx.bfloat16])[0].reshape(b_n, nq)
    return DL.kernel("wqb_rope_nb")(
        inputs=[qrq, att.wq_b.weight, att.wq_b.scales, cos, sin], template=tmpl,
        grid=(nq // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(b_n * nq,)], output_dtypes=[mx.bfloat16])[0].reshape(b_n, nq)


def woa_nb(att, oq, c, mma):
    """woa (with act_quant) for B rows: bit for bit per row without matrix tiles (mma=False)."""
    b_n = oq.shape[0]
    wq, ws = _woa_mxfp8(att)
    rows = c.o_groups * c.o_lora_rank
    tmpl = [("R", c.o_lora_rank), ("K", c.n_heads * c.head_dim // c.o_groups), ("G", c.o_groups), ("NB", b_n)]
    name, tmpl = ("woa_mma", tmpl + [("KS", 2)]) if mma else ("woa_nb", tmpl)
    return DL.kernel(name)(
        inputs=[oq, wq, ws], template=tmpl, grid=(rows // 32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b_n * rows,)], output_dtypes=[mx.float32])[0].reshape(b_n, rows)


def hc_post_nb(f, h, post, comb, c):
    b_n = f.shape[0]
    hc, dim = c.hc_mult, c.dim
    return DL.kernel("hc_post_b")(
        inputs=[f, h, post, comb], template=[("HC", hc), ("DIM", dim)],
        grid=(hc * dim, b_n, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b_n * hc * dim,)], output_dtypes=[mx.bfloat16])[0]


@lru_cache(maxsize=None)
def _arange_i32(n):
    a = mx.arange(n, dtype=mx.int32)
    mx.eval(a)
    return a


def batched_attention(att, q, packed, x, qr, caches, shareds, starts, cos, sin, c):
    """Attention for B requests whose windows are full (start >= window_size): one window
    shift for the batch, per-request compressor / indexer, then the split attention with a
    request grid axis. Each request's keys, their order and the arithmetic are those of
    attention_core (window rows old[1:] + new are exactly the rows it attends to). Returns
    act_quant(o) [B, H*D], or None when a request's window is not full yet."""
    b_n, w = q.shape[0], c.window_size
    if any(
        s < w or caches[r][1] is None or caches[r][1].shape[1] != w for r, s in enumerate(starts)
    ):
        return None
    old = mx.concatenate([caches[r][1] for r in range(b_n)], 0)
    win = mx.concatenate([old[:, 1:], packed[:, None, :]], 1)
    for r in range(b_n):
        caches[r][1] = win[r : r + 1]
    parts = None
    if c.compress_ratios[att._layer]:
        xs, qrs = mx.split(x, b_n), mx.split(qr, b_n)
        parts = [compress_index(att, xs[r].reshape(-1), qrs[r].reshape(-1), caches[r], shareds[r], starts[r], c)
                 for r in range(b_n)]
    return window_attention_bq(att, q, win, parts, shareds[0], cos, sin, c)


def window_attention_bq(att, q, win, parts, memo, cos, sin, c):
    """Split attention of B queries with a query grid axis: query b attends to the full
    window win[b] ([B, window_size, row bytes], oldest first) and to its compressed rows and
    selected indices parts[b] (None for a layer without compression). Per query the keys,
    their order and the arithmetic are attention_core's. memo (a dict that lives for the
    forward) keeps the concatenated compressed rows for the layers that reuse a source
    layer's arrays. Returns act_quant(o) [B, H*D]."""
    b_n, w = q.shape[0], c.window_size
    h, d = c.n_heads, c.head_dim
    if parts is not None:
        # Layers that reuse a source layer's rows get the same arrays back: concatenate once.
        # The entry holds those arrays, so an identity match can never be a recycled id().
        key = [a for p in parts for a in p]
        cat = memo.get("_batched_compressed")
        if cat is None or len(cat[0]) != len(key) or any(a is not b for a, b in zip(cat[0], key)):
            ncs = [int(p.shape[1]) for p, _ in parts]
            cs_ = [int(i.shape[-1]) for _, i in parts]
            offs = [sum(ncs[:r]) for r in range(b_n)]
            pooled = (mx.concatenate([p.reshape(-1, p.shape[-1]) for p, _ in parts if p.shape[1]], 0)
                      if sum(ncs) else _dummy(mx.uint8))
            cmax = max(max(cs_), 1)
            ci = mx.concatenate([mx.pad(i.reshape(1, -1).astype(mx.int32), [(0, 0), (0, cmax - i.shape[-1])], constant_values=-1)
                                 for _, i in parts], 0)
            cat = (key, pooled, ncs, cs_, offs, ci, cmax)
            memo["_batched_compressed"] = cat
        _, pooled, ncs, cs_, offs, ci, cmax = cat
    else:
        pooled, ncs, cs_, offs, ci, cmax = _dummy(mx.uint8), [0] * b_n, [0] * b_n, [0] * b_n, _dummy(mx.int32), 1
    nbs = [max(1, (w + cr + 63) // 64) for cr in cs_]
    nbmax = max(nbs)
    meta = mx.array([[w, cs_[r], w, ncs[r], nbs[r], offs[r], 0, 0] for r in range(b_n)], mx.int32)
    tmpl = [("D", d), ("H", h), ("NBMAX", nbmax), ("WSTRIDE", w * (d + d // 32)), ("WIS", 0), ("CMAX", cmax)]
    grid = (h // 8 * 1024, nbmax, b_n)
    s, bmax = DL.kernel("attn_scores_bq")(
        inputs=[q, win, pooled, _arange_i32(w), ci, meta, _const((d**-0.5,))],
        template=tmpl, grid=grid, threadgroup=(1024, 1, 1),
        output_shapes=[(b_n * h * nbmax * 64,), (b_n * h * nbmax,)], output_dtypes=[mx.float32, mx.float32])
    pv, den = DL.kernel("attn_pv_bq")(
        inputs=[win, pooled, _arange_i32(w), ci, meta, s, bmax],
        template=tmpl, grid=grid, threadgroup=(1024, 1, 1),
        output_shapes=[(b_n * h * nbmax * d,), (b_n * h * nbmax,)], output_dtypes=[mx.float32, mx.float32])
    return DL.kernel("o_merge_bq")(
        inputs=[pv, den, bmax, att.attn_sink, meta, cos, sin],
        template=[("NH", h), ("HD", d), ("RD", c.rope_head_dim), ("TPG", 1024), ("NBMAX", nbmax)],
        grid=((h * d // 32 + 31) // 32 * 1024, b_n, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(b_n * h * d,)], output_dtypes=[mx.float32])[0].reshape(b_n, h * d)


def decode_block_nb(blk, h, pre, caches, shareds, starts, c):
    """decode_block for B requests at their own positions. h: [B, 1, hc, dim] bf16, pre:
    [B, 1, hc]; caches / shareds / starts: per request. Weight-streaming kernels run once for
    the batch; window, compressor, indexer and attention run per request."""
    b_n = h.shape[0]
    hc, dim = c.hc_mult, c.dim
    att = blk.attn
    ratio = c.compress_ratios[att._layer]
    hf = h.reshape(b_n, hc * dim)
    coef_a, x, xq, _ = mix_nb(blk, "attn", c, hf, pre.reshape(b_n, hc))
    ap, ao, ac = coef_a[:, :hc], coef_a[:, hc:2 * hc], coef_a[:, 2 * hc:]
    qa, kv_in = gemv_fp8_nb(xq, att.wq_a, att.wkv)
    cos, sin = _cos_sin_nb(c, bool(ratio), starts)
    qr, qrq, packed = attn_mid_nb(att, qa, kv_in, cos, sin, c)
    q = wqb_rope_nb(att, qrq, cos, sin, c, DM.use_mma(b_n))
    oq = batched_attention(att, q, packed, x, qr, caches, shareds, starts, cos, sin, c)
    if oq is None:   # a request still filling its window: per-request attention
        xs, qrs, qs, ps = mx.split(x, b_n), mx.split(qr, b_n), mx.split(q, b_n), mx.split(packed, b_n)
        cs, ss = mx.split(cos, b_n), mx.split(sin, b_n)
        oq = mx.concatenate([attention_core(att, xs[r].reshape(-1), qrs[r].reshape(-1), qs[r].reshape(-1), ps[r].reshape(-1),
                                            caches[r], shareds[r], starts[r], cs[r].reshape(-1), ss[r].reshape(-1), c)[None]
                             for r in range(b_n)])
    yq = woa_nb(att, oq, c, DM.use_mma(b_n))
    a, _ = gemv_fp8_nb(yq, att.wo_b)
    coef_f, x2, _, h2 = mix_nb(blk, "ffn", c, hf, ap, sub=a, post=ao, comb=ac)
    fp, fo, fc = coef_f[:, :hc], coef_f[:, hc:2 * hc], coef_f[:, 2 * hc:]
    f = DM.fused_decode_moe_nb(blk.ffn, x2, c)[0]
    h3 = hc_post_nb(f, h2, fo, fc, c)
    return h3.reshape(b_n, 1, hc, dim), fp.reshape(b_n, 1, hc)


def decode_block_verify(blk, h, pre, cache, shared, start, c):
    """Block.__call__ for an MTP verify block: R consecutive tokens of one request at
    positions start .. start + R - 1 (h: [1, R, hc, dim] bf16, pre: [1, R, hc]).

    Every token gets bit for bit what decode_block computes for it when the tokens are
    decoded one at a time, so speculative decoding cannot change greedy output:
      * the stages that stream weights run once for the R rows with the batch-invariant
        kernels (no matrix tiles: each row is the single-token arithmetic);
      * the compressor and indexer run token by token in order, each token with its own
        shared-state dict, since a token only sees the cache up to itself;
      * the sparse attention runs once for the block, each token over the window
        attention_core would build for it (window_attention_bq), or token by token while
        the window is still filling.
    The verify rollback states that Attention and Compressor record for a block
    (DSparkMixin.mtp_partial_rollback) are rebuilt from the same rows.
    """
    r_n = h.shape[1]
    hc, dim = c.hc_mult, c.dim
    att = blk.attn
    layer = att._layer
    ratio = c.compress_ratios[layer]
    hf = h.reshape(r_n, hc * dim)
    coef_a, x, xq, _ = mix_nb(blk, "attn", c, hf, pre.reshape(r_n, hc))
    ap, ao, ac = coef_a[:, :hc], coef_a[:, hc:2 * hc], coef_a[:, 2 * hc:]
    qa, kv_in = gemv_fp8_nb(xq, att.wq_a, att.wkv, mma=False)
    cos, sin = _cos_sin_nb(c, bool(ratio), range(start, start + r_n))
    qr, qrq, packed = attn_mid_nb(att, qa, kv_in, cos, sin, c)
    q = wqb_rope_nb(att, qrq, cos, sin, c, False)
    state = getattr(cache, "_mtp_verify_state", None)
    old = cache[1]
    old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
    pooling = state is not None and ratio > 1 and layer in c.kv_source_layers
    if pooling:
        rem = start % ratio
        pending = [(cache[4][:, :rem], cache[5][:, :rem])] if rem else []
    tokens = shared.setdefault("_verify_tokens", [{} for _ in range(r_n)])
    w = c.window_size
    if old_len == w:
        # Full window: token j attends to rows j+1 .. j+w of (kept window + block rows), the
        # window attention_core builds for it; one dispatch set serves the whole block.
        full = mx.concatenate([old[:, :old_len], packed.reshape(1, r_n, -1)], 1)
        win = mx.concatenate([full[:, j + 1 : j + 1 + w] for j in range(r_n)], 0)
        cache[1] = full[:, -w:]
        parts = None
        if ratio:
            parts = []
            for j in range(r_n):
                parts.append(compress_index(att, x[j], qr[j], cache, tokens[j], start + j, c))
                if pooling:
                    # Compressor.__call__ recorded (pending rows + this token's projection).
                    kv, gate = state["compressor"]
                    pending.append((kv[:, -1:], gate[:, -1:]))
        oq = window_attention_bq(att, q, win, parts, tokens[0], cos, sin, c)
    else:
        oq = []
        for j in range(r_n):
            oq.append(attention_core(att, x[j], qr[j], q[j], packed[j], cache, tokens[j], start + j, cos[j], sin[j], c))
            if pooling:
                kv, gate = state["compressor"]
                pending.append((kv[:, -1:], gate[:, -1:]))
        oq = mx.stack(oq)
    if state is not None:
        # Attention.__call__'s verify-block window: the kept rows, then the block's rows.
        new = packed.reshape(1, r_n, -1)
        state["window"] = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        if pooling:
            state["compressor"] = tuple(mx.concatenate([p[i] for p in pending], 1) for i in (0, 1))
    yq = woa_nb(att, oq, c, False)
    a, _ = gemv_fp8_nb(yq, att.wo_b, mma=False)
    coef_f, x2, _, h2 = mix_nb(blk, "ffn", c, hf, ap, sub=a, post=ao, comb=ac)
    fp, fo, fc = coef_f[:, :hc], coef_f[:, hc:2 * hc], coef_f[:, 2 * hc:]
    f = DM.fused_decode_moe_nb(blk.ffn, x2, c, mma=False)[0]
    h3 = hc_post_nb(f, h2, fo, fc, c)
    return h3.reshape(1, r_n, hc, dim), fp.reshape(1, r_n, hc)


def supported(blk, c):
    """Static support check for a decoder block (cached on the block): the layouts the fused
    kernels read (MXFP8 attention projections, an exactly MXFP8-representable bf16 wo_a, MXFP4
    routed and MXFP8 shared experts) and the shapes they assume. Anything else, e.g. an oQ
    export below oQ4 with affine-requantized projections, keeps the unfused path."""
    ok = getattr(blk, "_fused_decode_ok", None)
    if ok is None:
        att = blk.attn
        ok = bool(
            c.hc_mult == 4
            and c.dim % 1024 == 0
            and c.head_dim % 256 == 0   # split_attention's layout (V4.1: 512)
            and c.n_heads % 8 == 0
            and all(_mxfp8_projection(getattr(att, n)) for n in ("wq_a", "wkv", "wq_b", "wo_b"))
            and DM.supported(blk.ffn, _empty((1, 1, c.dim), mx.bfloat16), c)
            and _woa_mxfp8(att) is not None
        )
        object.__setattr__(blk, "_fused_decode_ok", ok)
    return ok


def decode_block(blk, h, pre, cache, shared, start, c):
    """Block.__call__ for one token. h: [1, 1, hc, dim] bf16, pre: [1, 1, hc] fp32."""
    hc, dim = c.hc_mult, c.dim
    hf = h.reshape(-1)
    coef_a, x, xq, _ = mix(blk, "attn", c, hf, pre.reshape(-1))
    ap, ao, ac = coef_a[:hc], coef_a[hc:2 * hc], coef_a[2 * hc:]
    a = decode_attention(blk.attn, x, xq, cache, shared, start, c)
    coef_f, x2, _, h2 = mix(blk, "ffn", c, hf, ap, sub=a, post=ao, comb=ac)
    fp, fo, fc = coef_f[:hc], coef_f[hc:2 * hc], coef_f[2 * hc:]
    f = DM.fused_decode_moe(blk.ffn, x2.reshape(1, 1, -1), c)[0].reshape(-1)
    h3 = hc_post(f, h2, fo, fc, c)
    return h3.reshape(1, 1, hc, dim), fp.reshape(1, 1, hc)
