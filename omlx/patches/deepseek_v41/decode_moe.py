# SPDX-License-Identifier: MIT
"""Fused single-token MoE for DeepSeek V4.1 decode: 3 dispatches instead of ~30.

Follows DeepSeek-V4.1-Flash-1017 inference/model.py (Gate, Expert, MoE) and kernel.py:
  gate:    scores = sqrt(softplus(linear(x.float(), W.float()) / gate_temp)), softplus with
           torch's threshold 20; top-k of (scores + bias) picks experts, weights are the
           unbiased scores, w / (sum(w) + 1e-20) * route_scale
  expert:  act_quant(x) -> fp4/fp8 GEMM -> bf16 gate, up; clamp(up, +-limit),
           min(gate, limit); silu(gate) * up * weight in fp32 -> bf16 -> act_quant ->
           w2 GEMM -> bf16
  combine: fp32 sum of the routed bf16 outputs plus the shared expert's, stored as bf16
Every bf16/fp8 rounding point above is reproduced exactly. Free (unspecified in the
reference): fp32 accumulation order of each GEMM row and of the expert sum, the order of
tied top-k candidates, and the transcendentals (exp, log1p, sqrt). Weight times activation
products are exact in fp32 (fp4 or fp8 times a power-of-two scale times bf16), so each
GEMM row differs from any other fp32 order by at most gamma_K * sum|x_i w_i| before its
bf16 rounding (K = 5120 or 2304, gamma_K = K 2^-24 / (1 - K 2^-24)).

Why this differs from upstream oMLX's decode MoE (language.MoE): same math, but gate, top-k,
quantization, the seven expert GEMMs, SwiGLU and the combine run in three kernels with the
selected experts' packed MXFP4 / MXFP8 weights read in place.
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0(uchar s) { return s == 0 ? 0x1p-127f : as_type<float>(uint(s) << 23); }
inline float e4m3(uint code) {
    const uint a = code & 127u, e = a >> 3, m = a & 7u;
    const float v = e == 0 ? float(m) * 0x1p-9f : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (code & 128u) ? -v : v;
}
// act_quant scale for an amax already floored at 1e-4 (kernel.py fast_round_scale).
inline float act_scale(float amax) {
    const uint b = as_type<uint>(amax * (1.0f / 448.0f));
    const int e = int((b >> 23) & 0xFFu) - 127 + int((b & 0x7FFFFFu) != 0u);
    return as_type<float>(uint(e + 127) << 23);
}
// E4M3FN round-to-nearest-even of a value already clamped to [-448, 448].
inline float round_e4m3(float v) {
    const float a = abs(v);
    const int step_e = max(int((as_type<uint>(a) >> 23) & 0xFFu) - 127 - 3, -9);
    const float step = as_type<float>(uint(step_e + 127) << 23);
    const float q = min(rint(a / step) * step, 448.0f);
    return v < 0.0f ? -q : q;
}
// One act_quant(inplace) element: x / s clamped, cast to fp8, times s, cast to bf16.
inline float act_quant_value(float x, float s) {
    return float(bfloat16_t(round_e4m3(clamp(x / s, -448.0f, 448.0f)) * s));
}
// Exact FP8 E4M3 / FP4 E2M1 decode via fp16 bit patterns: shifting the code's exponent and
// mantissa bits into an fp16 encoding gives value * 2^-8 (E4M3) or value * 2^-14 (E2M1) for
// normal and subnormal codes alike. Products with act_quant-ed inputs (<= 4 significant bits,
// magnitude >= 2^-31) are exact in fp32 either way, and multiplying the group partial by 2^8 /
// 2^14 before the e8m0 scale is exact (no overflow or underflow at these magnitudes), so the
// per-group partial equals the reference's partial * scale for the same summation order.
inline float fp8_shifted(uint b) {
    return float(as_type<half>(ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7))));
}
inline float fp4_shifted(uint n) {
    return float(as_type<half>(ushort(((n & 8u) << 12) | ((n & 7u) << 9))));
}
// ROWS rows of an MXFP8 matrix ([*, K] bytes as uint32, one e8m0 scale per 32) dotted with an
// fp32 vector; each lane covers 8 contiguous values per step (MLX qmv_fast layout), x is
// loaded once per step and reused for all ROWS rows. fp32 accumulation order is free.
template <int K, int ROWS, typename W, typename S, typename X>
inline void fp8_rows(W w, S s, uint row0, X x, uint lane, thread float* acc) {
    static_assert(K % 256 == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8; k < uint(K); k += 256) {
        float xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = x[k + i];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = w + size_t(row0 + r) * (K / 4) + k / 4;
            const uint p0 = wr[0], p1 = wr[1];
            float a = 0.0f;
            for (int j = 0; j < 4; ++j) a += fp8_shifted((p0 >> (8 * j)) & 255u) * xv[j];
            for (int j = 0; j < 4; ++j) a += fp8_shifted((p1 >> (8 * j)) & 255u) * xv[4 + j];
            acc[r] += (a * 256.0f) * e8m0(s[size_t(row0 + r) * (K / 32) + k / 32]);
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// Same for MXFP4 (8 values per uint32): 16 contiguous values per lane per step when K allows
// (K % 512 == 0), else 8 (e.g. the 2304-wide expert down projection).
template <int K, int ROWS, typename W, typename S, typename X>
inline void fp4_rows(W w, S s, uint row0, X x, uint lane, thread float* acc) {
    constexpr int NW = (K % 512 == 0) ? 2 : 1;      // uint32 words per lane per step
    static_assert(K % (256 * NW) == 0, "K must be a multiple of 32 lanes x 8 values");
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8 * NW; k < uint(K); k += 256 * NW) {
        float xv[8 * NW];
        for (int i = 0; i < 8 * NW; ++i) xv[i] = x[k + i];
        for (int r = 0; r < ROWS; ++r) {
            const auto wr = w + size_t(row0 + r) * (K / 8) + k / 8;
            float a = 0.0f;
            for (int q = 0; q < NW; ++q) {
                const uint p = wr[q];
                for (int j = 0; j < 8; ++j) a += fp4_shifted((p >> (4 * j)) & 15u) * xv[8 * q + j];
            }
            acc[r] += (a * 16384.0f) * e8m0(s[size_t(row0 + r) * (K / 32) + k / 32]);
        }
    }
    for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
// Top-k of scores + bias with ties to the lower index (any tie order is spec-conformant).
template <int NE, int K>
inline void topk_select(const device float* scores, const device float* bias, threadgroup int* chosen,
                        threadgroup float* scratch_v, threadgroup int* scratch_i, uint tid, uint tpg) {
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY; int bi = 0x7FFFFFFF;
        for (uint e = tid; e < uint(NE); e += tpg) {
            bool taken = false;
            for (int j = 0; j < k; ++j) taken |= chosen[j] == int(e);
            const float v = taken ? -INFINITY : scores[e] + bias[e];
            if (v > best || (v == best && int(e) < bi)) { best = v; bi = int(e); }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(best, off); const int oi = simd_shuffle_xor(bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (tid % 32 == 0) { scratch_v[tid / 32] = best; scratch_i[tid / 32] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = scratch_v[0]; int ix = scratch_i[0];
            for (uint s = 1; s < (tpg + 31) / 32; ++s)
                if (scratch_v[s] > b || (scratch_v[s] == b && scratch_i[s] < ix)) { b = scratch_v[s]; ix = scratch_i[s]; }
            chosen[k] = ix;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""

# M1: gate scores (one simdgroup per expert row) + act_quant(x) shared by all experts.
_GATE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * (TPG / 32) + simdgroup_index_in_threadgroup;
    if (row < uint(NE)) {
        float dot = 0.0f;
        for (uint k = lane * 4; k < uint(DIM); k += 128) {
            const float4 a = float4(float(x[k]), float(x[k + 1]), float(x[k + 2]), float(x[k + 3]));
            const float4 b = float4(float(w[row * DIM + k]), float(w[row * DIM + k + 1]),
                                    float(w[row * DIM + k + 2]), float(w[row * DIM + k + 3]));
            dot += a.x * b.x; dot += a.y * b.y; dot += a.z * b.z; dot += a.w * b.w;
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const float raw = dot / temp[0];
            // torch F.softplus: x above threshold 20, else log1p(exp(x)) (accurate for tiny exp(x)).
            const float y = metal::precise::exp(raw);
            const float sp = raw > 20.0f ? raw
                : (y < 1e-4f ? y * (1.0f - y * (0.5f - y * (1.0f / 3.0f))) : metal::precise::log(1.0f + y));
            scores[row] = metal::precise::sqrt(sp);
        }
    }
    // act_quant(x) for the expert GEMMs: groups of 32, spread over the gate grid.
    const uint group = threadgroup_position_in_grid.x * (TPG / 32) + simdgroup_index_in_threadgroup;
    for (uint g = group; g < uint(DIM / 32); g += NGROUPS_STRIDE) {
        const float v = float(x[g * 32 + lane]);
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        xq[g * 32 + lane] = act_quant_value(v, s);
    }
"""

# M2: up projections of the K routed experts (MXFP4) + shared expert (MXFP8), SwiGLU,
# routing weight, bf16, and act_quant over each 32-row block (= one threadgroup).
_UP = r"""
    threadgroup int chosen[8];
    threadgroup float sv[32]; threadgroup int si[32];
    threadgroup float hblock[32];
    const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    topk_select<NE, TOPK>(scores, bias, chosen, sv, si, tid, 256);
    const uint slot = threadgroup_position_in_grid.y;           // 0..TOPK-1 routed, TOPK shared
    // routing weights (unbiased scores, normalized, scaled) -- identical in every threadgroup
    float wsum = 0.0f;
    for (int j = 0; j < TOPK; ++j) wsum += scores[chosen[j]];
    const float rw = slot < uint(TOPK) ? scores[chosen[slot]] / (wsum + 1e-20f) * route_scale[0] : 1.0f;
    if (tid == 0 && threadgroup_position_in_grid.x == 0 && slot == 0)
        for (int j = 0; j < TOPK; ++j) {
            out_idx[j] = chosen[j];
            out_w[j] = scores[chosen[j]] / (wsum + 1e-20f) * route_scale[0];
        }
    float g4[4], u4[4];
    const uint row0 = threadgroup_position_in_grid.x * 32 + sg * 4;
    if (slot < uint(TOPK)) {
        const uint e = uint(chosen[slot]);
        fp4_rows<DIM, 4>(w1 + size_t(e) * INTER * (DIM / 8), s1p + size_t(e) * INTER * (DIM / 32), row0, xq, lane, g4);
        fp4_rows<DIM, 4>(w3 + size_t(e) * INTER * (DIM / 8), s3p + size_t(e) * INTER * (DIM / 32), row0, xq, lane, u4);
    } else {
        fp8_rows<DIM, 4>(sw1, ss1, row0, xq, lane, g4);
        fp8_rows<DIM, 4>(sw3, ss3, row0, xq, lane, u4);
    }
    if (lane < 4) {
        float gate = float(bfloat16_t(g4[lane])), up = float(bfloat16_t(u4[lane]));   // bf16 GEMM outputs
        if (limit[0] > 0.0f) { up = clamp(up, -limit[0], limit[0]); gate = min(gate, limit[0]); }
        float h = gate / (1.0f + metal::precise::exp(-gate)) * up;               // F.silu(gate) * up
        if (slot < uint(TOPK)) h = rw * h;                                       // weights * x
        hblock[sg * 4 + lane] = float(bfloat16_t(h));                            // x.to(dtype)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {   // act_quant over this 32-row block (the w2 input's quantization group)
        const float v = hblock[lane];
        const float s = act_scale(max(simd_max(abs(v)), 1e-4f));
        hq[slot * INTER + threadgroup_position_in_grid.x * 32 + lane] = act_quant_value(v, s);
    }
"""

# M3: down projections of all K+1 experts per output row, bf16 per expert, fp32 sum, bf16.
_DOWN = r"""
    // simdgroup s computes expert slot s (s == TOPK: shared expert) for 4 output rows; the
    // per-expert bf16 outputs are then summed in slot order, as in the serial formulation.
    threadgroup float part[(TOPK + 1) * 4];
    const uint lane = thread_index_in_simdgroup, slot = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.x * 4;
    if (slot <= uint(TOPK)) {
        float acc[4];
        const device float* hrow = hq + slot * INTER;
        if (slot < uint(TOPK)) {
            const uint e = uint(idx[slot]);
            fp4_rows<INTER, 4>(w2 + size_t(e) * DIM * (INTER / 8), s2p + size_t(e) * DIM * (INTER / 32), row0, hrow, lane, acc);
        } else {
            fp8_rows<INTER, 4>(sw2, ss2, row0, hrow, lane, acc);
        }
        if (lane < 4) part[slot * 4 + lane] = float(bfloat16_t(acc[lane]));   // each expert's w2 output is bf16
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4) {
        float total = 0.0f;
        for (int s = 0; s <= TOPK; ++s) total += part[s * 4 + lane];
        out[row0 + lane] = bfloat16_t(total);
    }
"""


@cache
def _const(values):
    """Materialized scalar constant (cached; no per-step array construction)."""
    a = mx.array(list(values), mx.float32)
    mx.eval(a)
    return a


@cache
def _kernels():
    gate = mx.fast.metal_kernel(name="v41_decode_moe_gate", input_names=["x", "w", "temp"],
                                output_names=["scores", "xq"], source=_GATE, header=_HEADER)
    up = mx.fast.metal_kernel(
        name="v41_decode_moe_up",
        input_names=["scores", "bias", "route_scale", "limit", "xq", "w1", "s1p", "w3", "s3p",
                     "sw1", "ss1", "sw3", "ss3"],
        output_names=["hq", "out_idx", "out_w"], source=_UP, header=_HEADER)
    down = mx.fast.metal_kernel(
        name="v41_decode_moe_down",
        input_names=["hq", "idx", "w2", "s2p", "sw2", "ss2"],
        output_names=["out"], source=_DOWN, header=_HEADER)
    return gate, up, down


def supported(moe, x, config):
    e, s = moe.experts, moe.shared_experts
    return (
        x.size == config.dim
        and x.dtype == mx.bfloat16
        and getattr(config, "score_func", "sqrtsoftplus") == "sqrtsoftplus"
        and all(getattr(p, "mode", None) == "mxfp4" and p.group_size == 32 and "biases" not in p for p in (e.w1, e.w2, e.w3))
        and all(getattr(p, "mode", None) == "mxfp8" and p.group_size == 32 and "biases" not in p for p in (s.w1, s.w2, s.w3))
        and config.n_activated_experts <= 31
        and config.dim % 256 == 0 and config.moe_inter_dim % 256 == 0
    )


def fused_decode_moe(moe, x, config):
    """x: [1, 1, dim] bf16 (ffn_norm output). Returns ([1, 1, dim] bf16, idx, weights)."""
    gate_k, up_k, down_k = _kernels()
    dim, inter, ne, topk = config.dim, config.moe_inter_dim, config.n_routed_experts, config.n_activated_experts
    xf = x.reshape(-1)
    tpg = 256
    n_tg = (ne + tpg // 32 - 1) // (tpg // 32)
    scores, xq = gate_k(
        inputs=[xf, moe.gate.weight, _const((getattr(config, "gate_temp", 1.0),))],
        template=[("NE", ne), ("DIM", dim), ("TPG", tpg), ("NGROUPS_STRIDE", n_tg * (tpg // 32))],
        grid=(n_tg * tpg, 1, 1), threadgroup=(tpg, 1, 1),
        output_shapes=[(ne,), (dim,)], output_dtypes=[mx.float32, mx.float32])
    e, s = moe.experts, moe.shared_experts
    hq, idx, w = up_k(
        inputs=[scores, moe.gate.bias, _const((config.route_scale,)),
                _const((config.swiglu_limit,)), xq,
                e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales,
                s.w1.weight, s.w1.scales, s.w3.weight, s.w3.scales],
        template=[("NE", ne), ("TOPK", topk), ("DIM", dim), ("INTER", inter)],
        grid=(inter // 32 * 256, topk + 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[((topk + 1) * inter,), (topk,), (topk,)],
        output_dtypes=[mx.float32, mx.int32, mx.float32])
    out = down_k(
        inputs=[hq, idx, e.w2.weight, e.w2.scales, s.w2.weight, s.w2.scales],
        template=[("TOPK", topk), ("DIM", dim), ("INTER", inter)],
        grid=(dim // 4 * (topk + 1) * 32, 1, 1), threadgroup=((topk + 1) * 32, 1, 1),
        output_shapes=[(dim,)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(x.shape), idx, w
