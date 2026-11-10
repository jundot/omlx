# SPDX-License-Identifier: Apache-2.0
"""Routed FP4 expert kernels for decode and short verify calls.

For a forward of a few token rows, MLX runs every (row, expert) pair of a
switch layer as its own ``gather_qmv``: gate and up are two passes over the
same activations, the activation is a third kernel, and an expert picked by
several rows of a verify forward is streamed once per row. These kernels
compute the same numbers with fewer passes:

* ``gate_up``: the gate and up mat-vecs of all pairs and the model's
  activation in one kernel, one weight pass per distinct expert.
* ``down``: the down mat-vec of all pairs, one weight pass per distinct expert.

Bit-exactness: each output row is the reduction of MLX's ``fp_qmv_fast_impl``.
Lane ``l`` owns the 16 inputs ``[512 b + 16 l, +16)`` of every 512-wide block
``b``, accumulates ``scale * sum_i(((x0 w0 + x1 w1) + x2 w2) + x3 w3)`` block by
block, the lanes are combined with ``simd_sum`` and the result is rounded to
the activation dtype once. MXFP4 (an E8M0 scale per 32 values) and NVFP4 (an
E4M3 scale per 16) differ only in the scale. The activation repeats the
model's own ops in the same order and dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import mlx.core as mx

BLOCK = 512  # K values per simdgroup step (16 per lane)

_HEADER = r"""
// MLX fp4_e2m1 nibble -> float (via half).
inline float fp4_value(uint bits) {
    half value = as_type<half>(ushort((bits & 7) << 9));
    value *= 16384.0;
    return float((bits & 8) ? -value : value);
}

// MLX fp8_e8m0 (MXFP4 scale) and fp8_e4m3 (NVFP4 scale, finite values).
inline float fp4_scale_e8m0(uint8_t bits) {
    return as_type<float>(bits == 0 ? 0x400000u : (uint(bits) << 23));
}
inline float fp4_scale_e4m3(uint8_t bits) {
    half value = as_type<half>(ushort((bits & 127) << 7)) * 256.0h;
    return float((bits & 128) ? -value : value);
}
inline float fp4_scale(uint8_t bits, int group_size) {
    return group_size == 32 ? fp4_scale_e8m0(bits) : fp4_scale_e4m3(bits);
}

// MLX unary_ops.h Sigmoid and binary_ops.h Minimum / Maximum.
template <typename U>
inline U fp4_sigmoid(U x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
inline float fp4_minimum(float x, float y) {
    if (metal::isnan(x)) {
        return x;
    }
    return x < y ? x : y;
}
inline float fp4_maximum(float x, float y) {
    if (metal::isnan(x)) {
        return x;
    }
    return x > y ? x : y;
}

// MLX fp_quantized.h qdot<float, 16, 4> on 8 packed bytes (4 x uint16).
inline float fp4_qdot16(uint2 w, thread const float* x, float scale) {
    ushort ws[4] = {
        ushort(w.x & 0xFFFF), ushort(w.x >> 16), ushort(w.y & 0xFFFF), ushort(w.y >> 16)};
    float accum = 0;
    for (int i = 0; i < 4; ++i) {
        accum += (x[4 * i] * fp4_value(ws[i]) + x[4 * i + 1] * fp4_value(ws[i] >> 4)
                  + x[4 * i + 2] * fp4_value(ws[i] >> 8)
                  + x[4 * i + 3] * fp4_value(ws[i] >> 12));
    }
    return scale * accum;
}
"""

# Pair p = row * TOPK + k uses expert inds[p]. The threadgroup of the first
# pair that picked an expert computes every pair that picked it and the later
# pairs' threadgroups exit. An expert appears at most once per row, so MAXDUP
# (the row count) always covers its pairs.
_PAIRS = r"""
    const uint pid = threadgroup_position_in_grid.x;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint e = inds[pid];
    for (uint q = 0; q < pid; ++q) {
        if (inds[q] == e) {
            return;
        }
    }
    uint members[MAXDUP];
    int count = 0;
    for (uint q = pid; q < NPAIRS && count < MAXDUP; ++q) {
        if (inds[q] == e) {
            members[count++] = q;
        }
    }
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(simdgroup)) * RPS;
    constexpr int KW = KDIM / 8;
    constexpr int KS = KDIM / SCALE_GS;
    constexpr int SB = BLOCK / SCALE_GS;
    const int lane_scale = int(lane) * 16 / SCALE_GS;
"""

# ACTIVATION writes act[out] from the float sums g and u of one (pair, row).
_GATE_UP_SOURCE = _PAIRS + r"""
    const device uint32_t* gw = wg + (size_t(e) * GROWS + n0) * KW + lane * 2;
    const device uint32_t* uw = wu + (size_t(e) * UROWS + UOFF + n0) * KW + lane * 2;
    const device uint8_t* gs = sg + (size_t(e) * GROWS + n0) * KS + lane_scale;
    const device uint8_t* us = su + (size_t(e) * UROWS + UOFF + n0) * KS + lane_scale;

    float acc_g[MAXDUP][RPS];
    float acc_u[MAXDUP][RPS];
    for (int j = 0; j < MAXDUP; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc_g[j][r] = 0;
            acc_u[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / BLOCK; ++b) {
        uint2 wgv[RPS];
        uint2 wuv[RPS];
        float gscale[RPS];
        float uscale[RPS];
        for (int r = 0; r < RPS; ++r) {
            wgv[r] = *(const device uint2*)(gw + r * KW + b * 64);
            wuv[r] = *(const device uint2*)(uw + r * KW + b * 64);
            gscale[r] = fp4_scale(gs[r * KS + b * SB], SCALE_GS);
            uscale[r] = fp4_scale(us[r * KS + b * SB], SCALE_GS);
        }
        for (int j = 0; j < MAXDUP; ++j) {
            if (j < count) {
                const device T* xp =
                    x + size_t(members[j] / TOPK) * KDIM + b * BLOCK + lane * 16;
                float xt[16];
                for (int i = 0; i < 16; ++i) {
                    xt[i] = float(xp[i]);
                }
                for (int r = 0; r < RPS; ++r) {
                    acc_g[j][r] += fp4_qdot16(wgv[r], xt, gscale[r]);
                    acc_u[j][r] += fp4_qdot16(wuv[r], xt, uscale[r]);
                }
            }
        }
    }
    for (int j = 0; j < MAXDUP; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float g = simd_sum(acc_g[j][r]);
                const float u = simd_sum(acc_u[j][r]);
                if (lane == 0) {
                    const size_t out = size_t(members[j]) * NOUT + n0 + r;
                    ACTIVATION
                }
            }
        }
    }
"""

_DOWN_SOURCE = _PAIRS + r"""
    const device uint32_t* dw = w + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint8_t* ds = s + (size_t(e) * NOUT + n0) * KS + lane_scale;

    float acc[MAXDUP][RPS];
    for (int j = 0; j < MAXDUP; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / BLOCK; ++b) {
        uint2 wv[RPS];
        float scale[RPS];
        for (int r = 0; r < RPS; ++r) {
            wv[r] = *(const device uint2*)(dw + r * KW + b * 64);
            scale[r] = fp4_scale(ds[r * KS + b * SB], SCALE_GS);
        }
        for (int j = 0; j < MAXDUP; ++j) {
            if (j < count) {
                const device T* ap =
                    a + size_t(members[j]) * KDIM + b * BLOCK + lane * 16;
                float at[16];
                for (int i = 0; i < 16; ++i) {
                    at[i] = float(ap[i]);
                }
                for (int r = 0; r < RPS; ++r) {
                    acc[j][r] += fp4_qdot16(wv[r], at, scale[r]);
                }
            }
        }
    }
    for (int j = 0; j < MAXDUP; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float v = simd_sum(acc[j][r]);
                if (lane == 0) {
                    y[size_t(members[j]) * NOUT + n0 + r] = static_cast<T>(v);
                }
            }
        }
    }
"""


@dataclass(frozen=True)
class Activation:
    """A model's gate/up activation as Metal statements.

    ``source`` reads the float sums ``g`` and ``u``, the expert ``e`` and the
    names in ``inputs`` (extra kernel inputs), and writes ``act[out]``.
    """

    name: str
    source: str
    inputs: tuple[str, ...] = ()


@cache
def _gate_up_kernel(activation: Activation):
    return mx.fast.metal_kernel(
        name=f"omlx_fp4_gate_up_{activation.name}",
        input_names=["x", "inds", "wg", "sg", "wu", "su", *activation.inputs],
        output_names=["act"],
        header=_HEADER,
        source=_GATE_UP_SOURCE.replace("ACTIVATION", activation.source),
    )


@cache
def _down_kernel():
    return mx.fast.metal_kernel(
        name="omlx_fp4_down",
        input_names=["a", "inds", "w", "s"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SOURCE,
    )


def _launch(kernel, inputs, template, inds, n_out, dtype, rps):
    topk = int(inds.shape[-1])
    pairs = int(inds.size)
    rows = pairs // topk
    nsg = 2
    assert n_out % (nsg * rps) == 0
    return kernel(
        inputs=inputs,
        template=template
        + [
            ("TOPK", topk),
            ("NPAIRS", pairs),
            ("MAXDUP", rows),
            ("NSG", nsg),
            ("RPS", rps),
            ("BLOCK", BLOCK),
        ],
        grid=(32 * pairs, nsg * (n_out // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(pairs, n_out)],
        output_dtypes=[dtype],
    )[0]


def gate_up(
    x,
    inds,
    wg,
    sg,
    wu,
    su,
    *,
    n_out,
    scale_group_size,
    activation,
    extra=(),
    up_offset=0,
    rps=4,
):
    """``activation(gate(x), up(x))`` per (row, expert) pair, ``(pairs, n_out)``.

    ``x (..., K)`` rows match ``inds (..., TOPK)``; ``wg``/``wu`` are packed FP4
    ``(E, rows, K / 8)`` with up rows starting at ``up_offset`` (a fused
    ``[gate; up]`` tensor passes itself twice with ``up_offset=n_out``).
    ``extra`` holds the arrays named by ``activation.inputs``. ``rps`` is the
    output rows per simdgroup.
    """
    return _launch(
        _gate_up_kernel(activation),
        [x, inds, wg, sg, wu, su, *extra],
        [
            ("T", x.dtype),
            ("KDIM", int(x.shape[-1])),
            ("NOUT", int(n_out)),
            ("GROWS", int(wg.shape[1])),
            ("UROWS", int(wu.shape[1])),
            ("UOFF", int(up_offset)),
            ("SCALE_GS", int(scale_group_size)),
        ],
        inds,
        int(n_out),
        x.dtype,
        rps,
    )


def down(a, inds, w, s, *, scale_group_size, rps=4):
    """Per-pair down projection: ``a (pairs, K)`` -> ``(pairs, N)``."""
    n_out = int(w.shape[1])
    return _launch(
        _down_kernel(),
        [a, inds, w, s],
        [
            ("T", a.dtype),
            ("KDIM", int(a.shape[-1])),
            ("NOUT", n_out),
            ("SCALE_GS", int(scale_group_size)),
        ],
        inds,
        n_out,
        a.dtype,
        rps,
    )
