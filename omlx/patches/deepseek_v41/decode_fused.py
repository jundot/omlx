# SPDX-License-Identifier: MIT
"""Single-token decode path for DeepSeek V4.1 built from the fused kernels.

Mirrors language.Block.__call__ / Attention.__call__ for one decoded token (no CED, no
MTP verify, no capture). Compressor, indexer and cache bookkeeping are the existing
implementations; the hyper-connection, projection, norm, RoPE, quantization, attention and
MoE stages run as fused kernels (decode_layer.py, decode_moe.py), each of which documents
its rounding points against the DeepSeek-V4.1-Flash-1017 reference.
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


def o_epi(out, cos, sin, c):
    nq = c.n_heads * c.head_dim
    return DL.kernel("o_epi")(
        inputs=[out, cos, sin],
        template=[("NH", c.n_heads), ("HD", c.head_dim), ("RD", c.rope_head_dim), ("TPG", 1024)],
        grid=((nq // 32 + 31) // 32 * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(nq,)], output_dtypes=[mx.float32])[0]


def woa(att, oq, c, raw=False):
    wq, ws = _woa_mxfp8(att)
    rows = c.o_groups * c.o_lora_rank
    return DL.kernel("woa")(
        inputs=[oq, wq, ws],
        template=[("R", c.o_lora_rank), ("K", c.n_heads * c.head_dim // c.o_groups), ("RAW", raw)],
        grid=(rows // 32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows,)], output_dtypes=[mx.float32])[0]


def hc_post(f, h, post, comb, c):
    hc, dim = c.hc_mult, c.dim
    return DL.kernel("hc_post")(
        inputs=[f, h, post, comb], template=[("HC", hc), ("DIM", dim)],
        grid=(hc * dim, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(hc * dim,)], output_dtypes=[mx.bfloat16])[0]


def split_attention_supported(c):
    return c.head_dim % 256 == 0 and c.n_heads % 8 == 0


def split_attention(q, window, pooled, wi, ci, sink, cos, sin, c, raw=False):
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
        template=[("NH", h), ("HD", d), ("RD", c.rope_head_dim), ("TPG", 1024), ("RAWO", raw)],
        grid=((h * d // 32 + 31) // 32 * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(h * d,)], output_dtypes=[mx.float32])[0]


def decode_attention(att, x, xq, cache, shared, start, c):
    """Attention.__call__ for one token. x: [dim] bf16 (attn_norm output), xq: act_quant(x)."""
    from .language import packed_sparse_attention

    layer = att._layer
    ratio = c.compress_ratios[layer]
    qa, kv_in = gemv_fp8(xq, att.wq_a, att.wkv)
    cos, sin = _cos_sin(c, bool(ratio), start)
    qr, qrq, packed = attn_mid(att, qa, kv_in, cos, sin, c)
    q = wqb_rope(att, qrq, cos, sin, c).reshape(1, 1, c.n_heads, c.head_dim)
    new = packed.reshape(1, 1, -1)
    old = cache[1]
    old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
    kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
    cache[1] = kv[:, -c.window_size:]
    idx = _window_idx(c.window_size, start, old_len)
    pooled = _empty((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
    ci = _empty((1, 1, 0), mx.int32)
    x3 = x.reshape(1, 1, -1)
    if ratio:
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
        ci, pooled = shared["idx"], shared["kv"]
    if split_attention_supported(c):
        oq = split_attention(q.reshape(-1), kv, pooled, idx, ci, att.attn_sink, cos, sin, c)
    else:
        out = packed_sparse_attention(q, kv, pooled, idx, ci, att.attn_sink, c.head_dim**-0.5)
        oq = o_epi(out.reshape(-1), cos, sin, c)
    yq = woa(att, oq, c)
    return gemv_fp8(yq, att.wo_b)[0]


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
            and c.head_dim % 32 == 0
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
