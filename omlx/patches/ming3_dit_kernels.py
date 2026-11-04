# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernels for Ming-3.0's DiT stacks (CFM flow head, aggregator).

At decode time the flow head runs on 6 rows (CFG batch 2 x seq 3) and the
aggregator on 2, so every linear is a bandwidth-bound GEMV and dispatch count
dominates. These kernels run a DiT block as 5 dispatches, and each kernel reads
its weights once:

- QKV: rmsnorm1 + q/k/v linear + bias + interleaved-pair rotary.
- ATT: attention over the L positions of each CFG half.
- OUT: out linear + bias + residual.
- FF1: rmsnorm2 + ff1 linear + bias + tanh GELU.
- FF2: ff2 linear + bias + residual.

Each CFM Euler step also uses EMBED (x embedder and row assembly) and FINAL
(final norm and linear on the last row of each half, CFG combine, Euler
update).

Storage is bf16 and accumulation f32. Values that MLX rounds to bf16 are
rounded at the same points. These parts differ from MLX:
- RMSNorm is folded into the GEMV.
- f32 summation order.
- Softmax is two-pass.
The output is therefore close to the MLX graph but not bit-identical.

GEMV layout: a threadgroup of SG simdgroups owns NO output columns for all R
rows. The simdgroups take 128-wide blocks of K round robin, and each lane does
one 8-byte load (4 bf16) per weight row. Partial sums meet in threadgroup
memory.
"""

# Copyright (c) 2026 Stéphane Busso
# Copyright (c) 2026 David Dalcu
#
# The kernel sources below are adapted from sbusso/ming-tts-serve's
# MIT-licensed ``src/ming_fused.zig`` at release ``v0.2.0``. Preserve this
# scoped notice with those kernels.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import struct
from functools import cache

import mlx.core as mx
import mlx.nn as nn

D = 1024
FF = 4096
HEADS = 16
MAX_ROWS = 6
# GEMV tile (output columns per threadgroup, simdgroups per threadgroup),
# tuned on an M5 Max by ming-tts-serve.
_NO = 2
_SG = 4

_HEADER = """
inline float4 bf4(uint2 v) {
    return float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xffff0000u),
                  as_type<float>(v.y << 16), as_type<float>(v.y & 0xffff0000u));
}
inline float rb(float x) { return float(bfloat16_t(x)); }
#define LDW(p, i) bf4(*(const device uint2*)((p) + (i)))
inline float rsum_(threadgroup const float* red, int sgn, int no, int rows, int n, int r) {
    float a = 0.0f;
    for (int g = 0; g < sgn; g++) a += red[(g * no + n) * rows + r];
    return a;
}
inline float rinv_(threadgroup const float* ssr, int sgn, int rows, int r, int kd) {
    float a = 0.0f;
    for (int g = 0; g < sgn; g++) a += ssr[g * rows + r];
    return metal::precise::rsqrt(a / float(kd) + 1e-6f);
}
"""

_PRELUDE = """
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int ob = int(threadgroup_position_in_grid.x) * NO;
threadgroup float red[SG * NO * R];
threadgroup float ssr[SG * R];
"""


def _gemv(k: int, wrow: str, xv: str, norm: bool) -> str:
    # With norm, the sum of squares rides along and 1/rms is applied in the
    # epilogue through rinv(r).
    norm_load = (
        "const float4 nw = LDW(NW, i); for (int r = 0; r < R; r++) "
        "{ ss[r] += dot(xv[r], xv[r]); xv[r] *= nw; }"
        if norm
        else ""
    )
    norm_store = (
        "{ const float v = simd_sum(ss[r]); if (lane == 0) ssr[int(sg) * R + r] = v; }"
        if norm
        else ""
    )
    return f"""
{{
float acc[NO][R];
float ss[R];
for (int r = 0; r < R; r++) {{ ss[r] = 0.0f; for (int n = 0; n < NO; n++) acc[n][r] = 0.0f; }}
for (int i = int(sg) * 128 + int(lane) * 4; i < {k}; i += SG * 128) {{
    float4 xv[R];
    for (int r = 0; r < R; r++) xv[r] = {xv};
    {norm_load}
    for (int n = 0; n < NO; n++) {{
        const float4 w = LDW({wrow}, i);
        for (int r = 0; r < R; r++) acc[n][r] += dot(w, xv[r]);
    }}
}}
for (int r = 0; r < R; r++) {{
    for (int n = 0; n < NO; n++) {{
        const float v = simd_sum(acc[n][r]);
        if (lane == 0) red[(int(sg) * NO + n) * R + r] = v;
    }}
    {norm_store}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
}}
#define rsum(n, r) rsum_(red, SG, NO, R, n, r)
#define rinv(r) rinv_(ssr, SG, R, r, {k})
"""


_SRC_QKV = (
    _PRELUDE
    + """
const int which = ob / 1024, col0 = ob % 1024;
const device bfloat16_t* W = which == 0 ? WQ : (which == 1 ? WK : WV);
const device bfloat16_t* Bv = which == 0 ? BQ : (which == 1 ? BK : BV);
"""
    + _gemv(1024, "W + (col0 + n) * 1024", "LDW(X, r * 1024 + i)", True)
    + """
for (int ei = int(t); ei < NO / 2 * R; ei += 32 * SG) {
    const int n = (ei / R) * 2, r = ei % R, col = col0 + n;
    const float inv = rinv(r);
    float e = rb(rsum(n, r) * inv + float(Bv[col]));
    float od = rb(rsum(n + 1, r) * inv + float(Bv[col + 1]));
    if (which < 2) {
        // Interleaved-pair rotary in f32, rounded once (MLX rounds per op).
        const int pi = (col % 64) / 2, l = r % L;
        const float c = float(COS[l * 32 + pi]), s = float(SIN[l * 32 + pi]);
        const float ne = e * c - od * s, no = e * s + od * c;
        e = ne; od = no;
    }
    OUT[(which * R + r) * 1024 + col] = bfloat16_t(e);
    OUT[(which * R + r) * 1024 + col + 1] = bfloat16_t(od);
}
"""
)

# One threadgroup of 64 threads per head: 8 lanes per row, 8 dims per lane.
_SRC_ATT = """
const uint t = thread_position_in_threadgroup.x;
const int h = int(threadgroup_position_in_grid.x);
const int r = int(t) >> 3;
if (r < R) {
    const int b = r / L, d = (int(t) & 7) * 8;
    const int qo = r * 1024 + h * 64 + d;
    const float4 qa = 0.125f * LDW(QKV, qo), qb = 0.125f * LDW(QKV, qo + 4);
    float sc[L];
    float m = -INFINITY;
    for (int j = 0; j < L; j++) {
        const int ko = (R + b * L + j) * 1024 + h * 64 + d;
        float v = dot(qa, LDW(QKV, ko)) + dot(qb, LDW(QKV, ko + 4));
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 4);
        sc[j] = v;
        m = max(m, v);
    }
    float sum = 0.0f;
    float4 oa = 0.0f, oc = 0.0f;
    for (int j = 0; j < L; j++) {
        const float p = metal::exp(sc[j] - m);
        const int vo = (2 * R + b * L + j) * 1024 + h * 64 + d;
        sum += p; oa += p * LDW(QKV, vo); oc += p * LDW(QKV, vo + 4);
    }
    for (int k = 0; k < 4; k++) {
        OUT[qo + k] = bfloat16_t(oa[k] / sum);
        OUT[qo + 4 + k] = bfloat16_t(oc[k] / sum);
    }
}
"""

_SRC_OUT = (
    _PRELUDE
    + _gemv(1024, "WO + (ob + n) * 1024", "LDW(ATT, r * 1024 + i)", False)
    + """
for (int e = int(t); e < NO * R; e += 32 * SG) {
    const int n = e / R, r = e % R, o = ob + n;
    OUT[r * 1024 + o] = bfloat16_t(float(X[r * 1024 + o]) + rb(rsum(n, r) + float(BO[o])));
}
"""
)

# mlx.nn.gelu_approx with each bf16 op rounded and bf16 constants, like the
# Python weak scalars.
_SRC_FF1 = (
    _PRELUDE + _gemv(1024, "W1 + (ob + n) * 1024", "LDW(X, r * 1024 + i)", True) + """
for (int e = int(t); e < NO * R; e += 32 * SG) {
    const int n = e / R, r = e % R, o = ob + n;
    const float x = rb(rsum(n, r) * rinv(r) + float(B1[o]));
    const float x3 = rb(x * x * x);
    const float inner = rb(x + rb(rb(0.044715f) * x3));
    const float th = rb(metal::precise::tanh(rb(rb(0.7978845608f) * inner)));
    OUT[r * 4096 + o] = bfloat16_t(rb(0.5f * x) * rb(1.0f + th));
}
"""
)

_SRC_FF2 = (
    _PRELUDE + _gemv(4096, "W2 + (ob + n) * 4096", "LDW(HH, r * 4096 + i)", False) + """
for (int e = int(t); e < NO * R; e += 32 * SG) {
    const int n = e / R, r = e % R, o = ob + n;
    OUT[r * 1024 + o] = bfloat16_t(float(X[r * 1024 + o]) + rb(rsum(n, r) + float(B2[o])));
}
"""
)

# CFM input rows [t + c, x_embed(hist), x_embed(y)] for both CFG halves.
# TE: [2, 1024] timestep embedding, CE: [2, 1024] condition embedding (cond,
# zeros), XH: [1024] x_embed(hist). One thread per column.
_SRC_EMBED = """
const int o = int(thread_position_in_grid.x);
float a = 0.0f;
for (int k = 0; k < 64; k += 4)
    a += dot(LDW(WX + o * 64, k), float4(float(Y[k]), float(Y[k + 1]), float(Y[k + 2]), float(Y[k + 3])));
const bfloat16_t xe = bfloat16_t(a + float(BX[o]));
for (int b = 0; b < 2; b++) {
    OUT[(b * 3 + 0) * 1024 + o] = bfloat16_t(float(TE[b * 1024 + o]) + float(CE[b * 1024 + o]));
    OUT[(b * 3 + 1) * 1024 + o] = XH[o];
    OUT[(b * 3 + 2) * 1024 + o] = xe;
}
"""

# Final rmsnorm + linear on each CFG half's last row, CFG combine, Euler step
# (+ SDE noise when SH != 0). One simdgroup per latent channel; S is the
# Euler step and CFG is an f32 bit pattern. SH is an input so that requests
# with different temperatures share one pipeline.
_SRC_FINAL = """
const uint lane = thread_index_in_simdgroup;
const int o = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
const int r0 = (L - 1) * 1024, r1 = (2 * L - 1) * 1024;
float a0 = 0.0f, a1 = 0.0f, s0 = 0.0f, s1 = 0.0f;
for (int b = 0; b < 1024; b += 128) {
    const int i = b + int(lane) * 4;
    const float4 w = LDW(WF + o * 1024, i) * LDW(NW, i);
    const float4 x0 = LDW(X, r0 + i), x1 = LDW(X, r1 + i);
    s0 += dot(x0, x0); s1 += dot(x1, x1);
    a0 += dot(w, x0); a1 += dot(w, x1);
}
a0 = simd_sum(a0); a1 = simd_sum(a1); s0 = simd_sum(s0); s1 = simd_sum(s1);
if (lane == 0) {
    a0 *= metal::precise::rsqrt(s0 / 1024.0f + 1e-6f);
    a1 *= metal::precise::rsqrt(s1 / 1024.0f + 1e-6f);
    const float pc = rb(a0 + float(BF[o])), pu = rb(a1 + float(BF[o]));
    const float f0 = rb(pc + rb(rb(pc - pu) * rb(as_type<float>(CFG))));
    float y = rb(float(Y[o]) + rb(float(DTT[S]) * f0));
    const float sh = float(SH);
    if (sh != 0.0f) y = rb(y + rb(rb(sh * float(SQ[S])) * float(SDE[S * 64 + o])));
    OUT[o] = bfloat16_t(y);
}
"""


def _kernel(name, inputs, source, outputs=("OUT",)):
    return mx.fast.metal_kernel(
        name=name,
        input_names=list(inputs),
        output_names=list(outputs),
        source=source,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


@cache
def _kernels():
    return {
        "qkv": _kernel(
            "omlx_ming3_dit_qkv",
            ["X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "COS", "SIN"],
            _SRC_QKV,
        ),
        "att": _kernel("omlx_ming3_dit_att", ["QKV"], _SRC_ATT),
        "out": _kernel("omlx_ming3_dit_out", ["ATT", "X", "WO", "BO"], _SRC_OUT),
        "ff1": _kernel("omlx_ming3_dit_ff1", ["X", "NW", "W1", "B1"], _SRC_FF1),
        "ff2": _kernel("omlx_ming3_dit_ff2", ["HH", "X", "W2", "B2"], _SRC_FF2),
        "embed": _kernel(
            "omlx_ming3_cfm_embed", ["Y", "WX", "BX", "TE", "CE", "XH"], _SRC_EMBED
        ),
        "final": _kernel(
            "omlx_ming3_cfm_final",
            ["X", "NW", "WF", "BF", "Y", "DTT", "SQ", "SDE", "SH"],
            _SRC_FINAL,
        ),
    }


def _f32_bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


def _gemv_call(name, inputs, out_shape, n_cols, rows, seq_len):
    tg = 32 * _SG
    return _kernels()[name](
        inputs=inputs,
        template=[("NO", _NO), ("SG", _SG), ("R", rows), ("L", seq_len)],
        grid=(n_cols // _NO * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _bf16_linear(layer, out_dim: int, in_dim: int, bias: bool = True) -> bool:
    if type(layer) is not nn.Linear or layer.weight.dtype != mx.bfloat16:
        return False
    if layer.weight.shape != (out_dim, in_dim):
        return False
    return ("bias" in layer) == bias and (not bias or layer.bias.dtype == mx.bfloat16)


def stack_supported(blocks) -> bool:
    """Whether every block is a bf16, unquantized 1024-wide DiT block."""
    for block in blocks:
        attn = block.attn
        if attn.heads != HEADS or attn.dim_head * attn.heads != D:
            return False
        linears = (
            (attn.to_q, D, D),
            (attn.to_k, D, D),
            (attn.to_v, D, D),
            (attn.to_out[0], D, D),
            (block.mlp.ff[0][0], FF, D),
            (block.mlp.ff[2], D, FF),
        )
        if not all(_bf16_linear(layer, o, i) for layer, o, i in linears):
            return False
        for norm in (block.norm1, block.norm2):
            if norm.eps != 1e-6 or norm.weight.dtype != mx.bfloat16:
                return False
    return True


def cfm_supported(dit) -> bool:
    """Whether the flow head's embed and final layers fit EMBED and FINAL."""
    final = dit.final_layer
    return (
        stack_supported(dit.blocks)
        and _bf16_linear(dit.x_embedder, D, dit.in_channels)
        and dit.in_channels == 64
        and _bf16_linear(final.linear, 64, D)
        and final.norm_final.eps == 1e-6
        and final.norm_final.weight.dtype == mx.bfloat16
    )


def dit_block(block, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """One DiT block on x [B, L, 1024] (B x L <= 6), 5 dispatches."""
    batch, seq_len, _ = x.shape
    rows = batch * seq_len
    if rows > MAX_ROWS:
        raise ValueError(f"Fused DiT block supports {MAX_ROWS} rows, got {rows}")
    attn = block.attn
    qkv = _gemv_call(
        "qkv",
        [
            x,
            block.norm1.weight,
            attn.to_q.weight,
            attn.to_k.weight,
            attn.to_v.weight,
            attn.to_q.bias,
            attn.to_k.bias,
            attn.to_v.bias,
            cos,
            sin,
        ],
        (3, rows, D),
        3 * D,
        rows,
        seq_len,
    )
    att = _kernels()["att"](
        inputs=[qkv],
        template=[("R", rows), ("L", seq_len)],
        grid=(HEADS * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    out = attn.to_out[0]
    x1 = _gemv_call("out", [att, x, out.weight, out.bias], x.shape, D, rows, seq_len)
    ff1, ff2 = block.mlp.ff[0][0], block.mlp.ff[2]
    hh = _gemv_call(
        "ff1",
        [x1, block.norm2.weight, ff1.weight, ff1.bias],
        (rows, FF),
        FF,
        rows,
        seq_len,
    )
    return _gemv_call("ff2", [hh, x1, ff2.weight, ff2.bias], x.shape, D, rows, seq_len)


def cfm_embed(y, x_embedder, t_emb, c_emb, x_hist) -> mx.array:
    """CFM input rows [2, 3, 1024] from latent y [1, 1, 64]."""
    return _kernels()["embed"](
        inputs=[y, x_embedder.weight, x_embedder.bias, t_emb, c_emb, x_hist],
        grid=(D, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(2, 3, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


def cfm_final(h, final_layer, y, dts, sqs, sde, shift, step, cfg) -> mx.array:
    """Euler step ``step`` from DiT output h [2, L, 1024] -> latent [1, 1, 64].

    ``shift`` is the 0-d bf16 SDE scale sigma * sqrt(temperature).
    """
    return _kernels()["final"](
        inputs=[
            h,
            final_layer.norm_final.weight,
            final_layer.linear.weight,
            final_layer.linear.bias,
            y,
            dts,
            sqs,
            sde,
            shift,
        ],
        template=[("L", h.shape[1]), ("S", step), ("CFG", _f32_bits(cfg))],
        grid=(64 // 4 * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 64)],
        output_dtypes=[mx.bfloat16],
    )[0]
