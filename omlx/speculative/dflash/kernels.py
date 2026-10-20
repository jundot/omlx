# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for the DFlash2 dynamic convolutions and candidate selector."""

from __future__ import annotations

import mlx.core as mx

# ``_grouped_dynamic_convolve`` for one of the two kernel sets in
# ``dynamic`` (B, L, 2, KS, G), one thread per output element, with the bf16
# rounding of each op in the composed version.
_CONV_SOURCE = """
    uint e = thread_position_in_grid.x;
    uint c = e % H;
    uint t = (e / H) % L;
    uint bl = e / H;
    uint g = c / GS;
    T acc = T(0);
    for (int o = 0; o < KS; ++o) {
        T xv = int(t) >= o ? x[e - uint(o) * H] : T(0);
        T k = static_cast<T>(
            float(base[(SET * KS + o) * H + c]) + float(dyn[((bl * 2 + SET) * KS + o) * G + g]));
        T prod = static_cast<T>(float(k) * float(xv));
        acc = static_cast<T>(float(acc) + float(prod));
    }
    out[e] = acc;
"""
_CONV_KERNEL = None


def _convolve(conv, hidden: mx.array, dynamic: mx.array, which: int) -> mx.array:
    global _CONV_KERNEL
    if _CONV_KERNEL is None:
        _CONV_KERNEL = mx.fast.metal_kernel(
            name="omlx_dflash2_conv",
            input_names=["x", "dyn", "base"],
            output_names=["out"],
            source=_CONV_SOURCE,
        )
    batch, length, width = hidden.shape
    (out,) = _CONV_KERNEL(
        inputs=[hidden, dynamic, conv.base_kernel],
        template=[
            ("T", hidden.dtype),
            ("H", width),
            ("L", length),
            ("GS", conv.group_size),
            ("G", width // conv.group_size),
            ("KS", conv.kernel_size),
            ("SET", which),
        ],
        grid=(hidden.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
    )
    return out


def conv_prepare(conv, hidden: mx.array) -> tuple[mx.array, mx.array]:
    """``GroupedDynamicCausalConv.prepare``; returns both kernel sets."""
    groups = hidden.shape[-1] // conv.group_size
    dynamic = conv.kernel_projection(hidden).reshape(
        *hidden.shape[:-1], 2, conv.kernel_size, groups
    )
    return _convolve(conv, hidden, dynamic, 0), dynamic


def conv_finish(conv, hidden: mx.array, dynamic: mx.array) -> mx.array:
    return _convolve(conv, hidden, dynamic, 1)


# One threadgroup per row walks the positions: simdgroup c scores candidate
# c against the previous pick, then simdgroup 0 applies the row's filters
# (as ``selection.sample_candidates``) and draws by inverse CDF. Temperature <= 0
# takes the first argmax.
_SELECT_SOURCE = """
    constexpr int C = 16;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint b = threadgroup_position_in_grid.x;
    threadgroup float scores[C];
    threadgroup int pred_tg;
    if (sg == 0 && lane == 0)
        pred_tg = anchors[b];
    float temp = params[b * 4 + 0];
    float top_p = params[b * 4 + 1];
    int top_k = int(params[b * 4 + 2]);
    float min_p = params[b * 4 + 3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < P; ++p) {
        int row = int(b) * P + p;
        int pred = pred_tg;
        int cid = cand[row * C + int(sg)];
        auto pc = pcb + long(pred) * R;
        auto sc = scb + long(cid) * R;
        auto pj = proj + long(row) * R;
        float acc = 0.0f;
        for (int i = 0; i < R / 32; ++i) {
            int d = int(lane) + i * 32;
            acc += float(pc[d]) * float(pj[d]) * float(sc[d]);
        }
        acc = simd_sum(acc);
        if (lane == 0)
            scores[sg] = float(unary[row * C + int(sg)]) + acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bool live = int(lane) < C;
            float s = live ? scores[lane] : -INFINITY;
            float m = simd_max(s);
            int pick;
            float q = -INFINITY;
            if (temp <= 0.0f) {
                pick = simd_min((live && s == m) ? int(lane) : C);
            } else {
                float lp = s - m - log(simd_sum(live ? exp(s - m) : 0.0f));
                if (!live)
                    lp = -INFINITY;
                if (top_k > 0 && top_k < C) {
                    int greater = 0;
                    for (int j = 0; j < C; ++j)
                        greater += simd_shuffle(lp, ushort(j)) > lp ? 1 : 0;
                    if (greater >= top_k)
                        lp = -INFINITY;
                }
                if (top_p > 0.0f && top_p < 1.0f) {
                    float prefix = 0.0f;
                    for (int j = 0; j < C; ++j) {
                        float o = simd_shuffle(lp, ushort(j));
                        if (o > lp || (o == lp && j < int(lane)))
                            prefix += exp(o);
                    }
                    if (!(prefix < top_p))
                        lp = -INFINITY;
                }
                if (min_p > 0.0f) {
                    float floor_ = simd_max(lp) + log(min_p);
                    if (lp < floor_)
                        lp = -INFINITY;
                }
                float scaled = lp / temp;
                float sm = simd_max(scaled);
                float e = scaled == -INFINITY ? 0.0f : exp(scaled - sm);
                float total = simd_sum(e);
                q = scaled == -INFINITY ? -INFINITY : scaled - sm - log(total);
                float cdf = simd_prefix_inclusive_sum(e) / total;
                float u = uniforms[row];
                int first = simd_min((e > 0.0f && cdf > u) ? int(lane) : C);
                int last = simd_max(e > 0.0f ? int(lane) : -1);
                pick = first < C ? first : last;
            }
            if (live)
                logq[row * C + int(lane)] = q;
            if (lane == 0) {
                int token = cand[row * C + pick];
                tokens[row] = token;
                pred_tg = token;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_SELECT_KERNEL = None


def select_kernel():
    global _SELECT_KERNEL
    if _SELECT_KERNEL is None:
        _SELECT_KERNEL = mx.fast.metal_kernel(
            name="omlx_dflash2_select",
            input_names=[
                "cand",
                "unary",
                "proj",
                "anchors",
                "pcb",
                "scb",
                "params",
                "uniforms",
            ],
            output_names=["tokens", "logq"],
            source=_SELECT_SOURCE,
        )
    return _SELECT_KERNEL
