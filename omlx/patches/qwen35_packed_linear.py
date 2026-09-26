# SPDX-License-Identifier: Apache-2.0
"""Tile-repacked 4-bit projections for dense Qwen3.5-family models on M5.

A Lightning MTP verify forward multiplies 1 + draft-depth rows against every
target projection. The SIMD quantized matmuls become ALU-bound from about 4
rows, so deep drafts never pay. The NAX tensor unit keeps the cost flat up to
8 rows, but only reaches DRAM bandwidth when a tile's weights are contiguous.

At load time this module replaces eligible 4-bit ``QuantizedLinear`` layers
with ``PackedLinear``, which holds the same codes, scales and biases in a
tile-contiguous layout. The originals are freed, so resident memory does not
grow. Layout, per 128-column tile and 64-wide quant group:

- Codes: [tile][group][column][32 B], the nibbles unchanged.
- Scales/biases: [tile][group / 4][column][4], so a thread reads its 4
  columns x 4 groups in one 32-byte load.

Projections that read the same input (GDN qkv/z, attention q/k/v, MLP
gate/up) share one buffer, so a verify forward serves them in one dispatch.
Every row count runs on packed kernels: one row on a SIMD matvec, 2..8 rows
on a tensor-unit kernel, more rows on a tensor-unit GEMM that dequantizes
each group tile into threadgroup memory. ``PackedLinear`` does not subclass
``nn.QuantizedLinear``, so code that reads stock quantized weights directly
skips it and calls the module instead.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

TILE_N = 128
_SMALL_ROWS = 8
_KERNELS: dict = {}

_SMALL_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_SMALL_SOURCE = """
    // A threadgroup owns one TILE_N-column tile. Its PARTS partitions of SG
    // simdgroups each stream an equal K range through their own tensor-unit
    // op; partials meet in threadgroup memory.
    constexpr int GROUPS = K / 64;
    constexpr int GPS = GROUPS / KSPLIT;     // quant groups per K split
    constexpr int GPP = GPS / PARTS;         // quant groups per partition
    // Rows come from the input shape so one pipeline serves 2..8 rows.
    const int M = x_shape[0];
    threadgroup float sums[GPS * 8];
    threadgroup float partials[PARTS * 8 * TILE_N];
    const uint tile = threadgroup_position_in_grid.x;
    const uint split = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint part = simd / SG;
    const uint gs = split * GPS;
    const uint g0 = gs + part * GPP;
    const uint n0 = tile * TILE_N;
    const uint tid = simd * 32 + lane;
    constexpr uint NT = PARTS * SG * 32;

    // Per-row sums of x over each quant group feed the bias term.
    for (uint idx = simd; idx < 8 * GPS; idx += PARTS * SG) {
        const uint row = idx % 8;
        const uint gl = idx / 8;
        float v = 0.0f;
        if (row < M) {
            const device bfloat* xr = (const device bfloat*)x + row * K + (gs + gl) * 64;
            v = simd_sum(float(xr[lane]) + float(xr[lane + 32]));
        }
        if (lane == 0) {
            sums[gl * 8 + row] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device uchar* wtile = (const device uchar*)w + ulong(tile) * GROUPS * TILE_N * 32;
    const device bfloat* stile = (const device bfloat*)sc + ulong(tile) * GROUPS * TILE_N;
    const device bfloat* btile = (const device bfloat*)bi + ulong(tile) * GROUPS * TILE_N;
    auto a = tensor((device bfloat*)x, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(8, TILE_N, 64, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bfirst(
        (device uchar*)wtile, dextents<int, 2>{64, TILE_N}, array<int, 2>{1, 64});
    auto a_first = a.template slice<64, 8>(0, 0);
    auto b_first = bfirst.template slice<64, TILE_N>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<
        decltype(a_first), decltype(b_first), float>();
    // 16 columns per simdgroup: every thread owns 4 consecutive columns of one row.
    for (ushort i = 0; i < 4; ++i) {
        acc[i] = 0.0f;
    }
    const auto first = acc.get_multidimensional_index(ushort(0));
    const uint my_col = first[0];
    const uint my_row = first[1];

    for (uint gi = 0; gi < GPP; gi += 4) {
        const uint g = g0 + gi;
        // Issue the affine loads before the matmuls so they overlap.
        const ulong at = (ulong(g / 4) * TILE_N + my_col) * 4;
        const vec<bfloat, 16> sv = *(const device vec<bfloat, 16>*)(stile + at);
        const vec<bfloat, 16> bv = *(const device vec<bfloat, 16>*)(btile + at);
        decltype(acc) p0, p1, p2, p3;
        auto run = [&](int j, thread decltype(acc)& part_acc) {
            auto as = a.template slice<64, 8>((g + j) * 64, 0);
            tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt(
                (device uchar*)wtile + ulong(g + j) * TILE_N * 32,
                dextents<int, 2>{64, TILE_N}, array<int, 2>{1, 64});
            auto bs = bt.template slice<64, TILE_N>(0, 0);
            op.run(as, bs, part_acc);
        };
        run(0, p0);
        run(1, p1);
        run(2, p2);
        run(3, p3);
        const uint gl = g - gs;
        const float x0 = sums[(gl + 0) * 8 + my_row];
        const float x1 = sums[(gl + 1) * 8 + my_row];
        const float x2 = sums[(gl + 2) * 8 + my_row];
        const float x3 = sums[(gl + 3) * 8 + my_row];
        for (ushort i = 0; i < 4; ++i) {
            acc[i] += p0[i] * float(sv[i * 4 + 0]) + x0 * float(bv[i * 4 + 0]);
            acc[i] += p1[i] * float(sv[i * 4 + 1]) + x1 * float(bv[i * 4 + 1]);
            acc[i] += p2[i] * float(sv[i * 4 + 2]) + x2 * float(bv[i * 4 + 2]);
            acc[i] += p3[i] * float(sv[i * 4 + 3]) + x3 * float(bv[i * 4 + 3]);
        }
    }
    *(threadgroup float4*)(partials + (part * 8 + my_row) * TILE_N + my_col) =
        float4(acc[0], acc[1], acc[2], acc[3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(M) * TILE_N; i += NT) {
        const uint row = i / TILE_N;
        const uint col = i % TILE_N;
        float v = 0.0f;
        for (uint p = 0; p < PARTS; ++p) {
            v += partials[(p * 8 + row) * TILE_N + col];
        }
        if (KSPLIT == 1) {
            ((device bfloat*)y)[row * N + n0 + col] = bfloat(v);
        } else {
            ((device float*)y)[split * M * N + row * N + n0 + col] = v;
        }
    }
"""

_MATVEC_SOURCE = """
    // One output column per thread: each quant group is 32 contiguous bytes
    // of that column, and x is uniform across the simdgroup.
    constexpr int GROUPS = K / 64;
    constexpr int GPS = GROUPS / KSPLIT;
    const uint tile = threadgroup_position_in_grid.x;
    const uint split = threadgroup_position_in_grid.y;
    const uint col = thread_position_in_threadgroup.x;
    const uint g0 = split * GPS;
    const device uchar* wt = (const device uchar*)w + ulong(tile) * GROUPS * TILE_N * 32;
    const device bfloat* st = (const device bfloat*)sc + ulong(tile) * GROUPS * TILE_N;
    const device bfloat* bt = (const device bfloat*)bi + ulong(tile) * GROUPS * TILE_N;
    float acc[M];
    for (int r = 0; r < M; ++r) {
        acc[r] = 0.0f;
    }
    for (uint gq = 0; gq < GPS; gq += 4) {
        const uint g = g0 + gq;
        const ulong at = (ulong(g / 4) * TILE_N + col) * 4;
        const vec<bfloat, 4> s4 = *(const device vec<bfloat, 4>*)(st + at);
        const vec<bfloat, 4> b4 = *(const device vec<bfloat, 4>*)(bt + at);
        for (int j = 0; j < 4; ++j) {
            const device uint4* wp = (const device uint4*)(wt + (ulong(g + j) * TILE_N + col) * 32);
            const uint4 w0 = wp[0];
            const uint4 w1 = wp[1];
            const uint words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            for (int r = 0; r < M; ++r) {
                const device bfloat* xr = (const device bfloat*)x + r * K + (g + j) * 64;
                float dot = 0.0f;
                float sum = 0.0f;
                for (int q = 0; q < 8; ++q) {
                    const vec<float, 8> xv = vec<float, 8>(*(const device vec<bfloat, 8>*)(xr + q * 8));
                    for (int e = 0; e < 8; ++e) {
                        dot += float((words[q] >> (4 * e)) & 15u) * xv[e];
                        sum += xv[e];
                    }
                }
                acc[r] += float(s4[j]) * dot + float(b4[j]) * sum;
            }
        }
    }
    const uint n = tile * TILE_N + col;
    for (int r = 0; r < M; ++r) {
        if (KSPLIT == 1) {
            ((device bfloat*)y)[r * N + n] = bfloat(acc[r]);
        } else {
            ((device float*)y)[(split * M + r) * N + n] = acc[r];
        }
    }
"""

_REDUCE_SOURCE = """
    const uint base = thread_position_in_grid.x * 4;
    if (base >= M * N) {
        return;
    }
    float4 v = 0.0f;
    for (int k = 0; k < KSPLIT; ++k) {
        v += *(const device float4*)(p + k * M * N + base);
    }
    *(device bfloat4*)((device bfloat*)out + base) = bfloat4(v);
"""

# MLX's own NAX qmm (qmm_t_nax_tgp_impl) with the weight loader swapped for
# the packed layout. The dequantize math stays in float: bfloat arithmetic is
# emulated and costs about 25%.
_GEMM_SOURCE = """
    using namespace mlx::steel;
    constexpr int GROUPS = K / 64;
    constexpr int BK = 64;
    constexpr int BKP = BK + 8;
    constexpr int NT = WM * WN * 32;
    constexpr int SM = BM / WM;
    constexpr int SN = BN / WN;
    constexpr int SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TNF = SN / 16;
    constexpr short TK = SK / 16;
    threadgroup bfloat Ws[BN * BKP];

    const int M = x_shape[0];
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint tid = simd_gid * 32 + thread_index_in_simdgroup;
    const int y_row = threadgroup_position_in_grid.y * BM;
    const int y_col = threadgroup_position_in_grid.x * BN;
    const int ptile = y_col / TILE_N;
    const int pcol0 = y_col % TILE_N;
    const device uchar* wtile = (const device uchar*)w + ulong(ptile) * GROUPS * TILE_N * 32;
    const device bfloat* stile = sc + ulong(ptile) * GROUPS * TILE_N;
    const device bfloat* btile = bi + ulong(ptile) * GROUPS * TILE_N;

    const short tm = SM * (simd_gid / WN);
    const short tn = SN * (simd_gid % WN);
    const short sgp_sm = min(int(SM), M - (y_row + tm));
    const bool aligned_m = sgp_sm == SM;

    NAXTile<float, TM, TNF> Dtile;
    Dtile.clear();
    const device bfloat* xp = x + ulong(max(0, min(y_row + tm, M - 1))) * K;

    for (int g = 0; g < GROUPS; ++g) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // BN columns x 32 bytes = 2 * BN chunks of 16 bytes.
        for (int c = tid; c < BN * 2; c += NT) {
            const int row = c >> 1;
            const int chunk = c & 1;
            const int col = pcol0 + row;
            const ulong at = (ulong(g >> 2) * TILE_N + col) * 4 + (g & 3);
            const float s = float(stile[at]);
            const float b = float(btile[at]);
            threadgroup bfloat* dst = Ws + row * BKP + chunk * 32;
            const uint4 v = *(const device uint4*)(wtile + (ulong(g) * TILE_N + col) * 32 + chunk * 16);
            const uint words[4] = {v.x, v.y, v.z, v.w};
            for (int wi = 0; wi < 4; ++wi) {
                vec<bfloat, 8> o;
                for (int j = 0; j < 8; ++j) {
                    o[j] = bfloat(float((words[wi] >> (4 * j)) & 15u) * s + b);
                }
                *(threadgroup vec<bfloat, 8>*)(dst + wi * 8) = o;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgp_sm > 0) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                NAXTile<bfloat, TM, TK> Atile;
                NAXTile<bfloat, TNF, TK> Btile;
                volatile int compiler_barrier;
                if (aligned_m) {
                    Atile.load(xp + kk1, K);
                } else {
                    Atile.load_safe(xp + kk1, K, short2(SK, sgp_sm));
                }
                Btile.template load<bfloat, BKP, 1>(Ws + tn * BKP + kk1);
                tile_matmad_nax(
                    Dtile, Atile, metal::bool_constant<false>{},
                    Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
        }
        xp += BK;
    }
    if (sgp_sm > 0) {
        device bfloat* yp = y + ulong(y_row + tm) * N + y_col + tn;
        if (aligned_m) {
            Dtile.store(yp, N);
        } else {
            Dtile.store_safe(yp, N, short2(SN, sgp_sm));
        }
    }
"""


def _steel_header() -> str:
    """MLX's steel NAX tile headers, inlined for a JIT kernel."""
    root = os.path.join(
        os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels"
    )
    parts = []
    for rel in (
        "steel/defines.h",
        "steel/utils/type_traits.h",
        "steel/utils/integral_constant.h",
        "steel/gemm/nax.h",
    ):
        with open(os.path.join(root, rel)) as f:
            text = f.read().replace("#pragma once", "")
        parts.append(re.sub(r'#include\s+"mlx/[^"]+"', "", text))
    return "\n".join(parts)


def _kernels():
    if not _KERNELS:
        _KERNELS["small"] = mx.fast.metal_kernel(
            name="omlx_qwen35_packed_small",
            input_names=["x", "w", "sc", "bi"],
            output_names=["y"],
            header=_SMALL_HEADER,
            source=_SMALL_SOURCE,
        )
        _KERNELS["matvec"] = mx.fast.metal_kernel(
            name="omlx_qwen35_packed_matvec",
            input_names=["x", "w", "sc", "bi"],
            output_names=["y"],
            source=_MATVEC_SOURCE,
        )
        _KERNELS["reduce"] = mx.fast.metal_kernel(
            name="omlx_qwen35_packed_reduce",
            input_names=["p"],
            output_names=["out"],
            source=_REDUCE_SOURCE,
        )
        _KERNELS["gemm"] = mx.fast.metal_kernel(
            name="omlx_qwen35_packed_gemm",
            input_names=["x", "w", "sc", "bi"],
            output_names=["y"],
            header=_steel_header(),
            source=_GEMM_SOURCE,
        )
    return _KERNELS


# Target threadgroup count for the small-row kernels. In a dependent layer
# chain one kernel runs at a time, so the grid itself must fill the GPU.
_TARGET_TGS = 1400


def _ksplit(tiles: int, units: int) -> int:
    """Largest divisor of ``units`` (K steps) keeping tiles x ksplit near the target."""
    best = 1
    for d in range(1, units + 1):
        if units % d == 0 and tiles * d <= _TARGET_TGS:
            best = d
    return best


def _small_geometry(K: int, N: int) -> tuple[int, int]:
    """(K splits across threadgroups, partitions inside one) for 2..8 rows."""
    steps = K // 256
    tiles = N // TILE_N
    ksplit = _ksplit(tiles, steps)
    parts = 2 if tiles < 256 and (steps // ksplit) % 2 == 0 else 1
    return ksplit, parts


def _gemm_tile(M: int) -> tuple[int, int, int, int]:
    """(BM, BN, WM, WN) for the GEMM."""
    if M <= 64:
        return 32, 64, 2, 2
    if M <= 512:
        return 64, 64, 2, 2
    return 128, 64, 2, 2


def packed_matmul(x2: mx.array, w, sc, bi, K: int, N: int) -> mx.array:
    """``x2`` (M, K) bf16 times a packed (N, K) weight -> (M, N) bf16."""
    kernels = _kernels()
    M = int(x2.shape[0])
    x2 = mx.contiguous(x2)
    if M > _SMALL_ROWS:
        bm, bn, wm, wn = _gemm_tile(M)
        (y,) = kernels["gemm"](
            inputs=[x2, w, sc, bi],
            template=[
                ("K", K),
                ("N", N),
                ("TILE_N", TILE_N),
                ("BM", bm),
                ("BN", bn),
                ("WM", wm),
                ("WN", wn),
            ],
            grid=(32 * wm * wn * (N // bn), -(-M // bm), 1),
            threadgroup=(32 * wm * wn, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.bfloat16],
        )
        return y
    if M == 1:
        return _matvec(x2, w, sc, bi, K, N)
    ksplit, parts = _small_geometry(K, N)
    sg = TILE_N // 16
    (y,) = kernels["small"](
        inputs=[x2, w, sc, bi],
        template=[
            ("K", K),
            ("N", N),
            ("TILE_N", TILE_N),
            ("SG", sg),
            ("PARTS", parts),
            ("KSPLIT", ksplit),
        ],
        grid=(32 * sg * parts * (N // TILE_N), ksplit, 1),
        threadgroup=(32 * sg * parts, 1, 1),
        output_shapes=[(M, N) if ksplit == 1 else (ksplit * M * N,)],
        output_dtypes=[mx.bfloat16 if ksplit == 1 else mx.float32],
    )
    return _reduce(y, ksplit, M, N)


def _matvec(x2: mx.array, w, sc, bi, K: int, N: int) -> mx.array:
    """SIMD matvec for one row."""
    M = int(x2.shape[0])
    ksplit = _ksplit(N // TILE_N, K // 256)
    (y,) = _kernels()["matvec"](
        inputs=[x2, w, sc, bi],
        template=[
            ("K", K),
            ("N", N),
            ("M", M),
            ("TILE_N", TILE_N),
            ("KSPLIT", ksplit),
        ],
        grid=(N, ksplit, 1),
        threadgroup=(TILE_N, 1, 1),
        output_shapes=[(M, N) if ksplit == 1 else (ksplit * M * N,)],
        output_dtypes=[mx.bfloat16 if ksplit == 1 else mx.float32],
    )
    return _reduce(y, ksplit, M, N)


def _reduce(y: mx.array, ksplit: int, M: int, N: int) -> mx.array:
    """Sum fp32 K-split partials into (M, N) bf16."""
    if ksplit == 1:
        return y
    threads = M * N // 4
    (out,) = _kernels()["reduce"](
        inputs=[y],
        template=[("KSPLIT", ksplit), ("M", M), ("N", N)],
        grid=(threads, 1, 1),
        threadgroup=(min(256, threads), 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


class _PackedStore:
    """Packed buffers shared by projections that read the same input."""

    __slots__ = ("K", "N", "w", "sc", "bi")


class PackedLinear(nn.Module):
    """A 4-bit, group-64, bias-free affine linear in the packed tile layout.

    ``packed_*`` are views into a store shared with sibling projections;
    ``_tile0`` is the first tile of this projection inside the store.
    """

    def __init__(self, store: _PackedStore, tile0: int, output_dims: int):
        super().__init__()
        self.input_dims = store.K
        self.output_dims = output_dims
        self.bits = 4
        self.group_size = 64
        self._store = store
        self._tile0 = tile0
        groups = store.K // 64
        tiles = output_dims // TILE_N
        codes = groups * TILE_N * 32
        affine = groups * TILE_N
        self.packed_weight = store.w[tile0 * codes : (tile0 + tiles) * codes]
        self.packed_scales = store.sc[tile0 * affine : (tile0 + tiles) * affine]
        self.packed_biases = store.bi[tile0 * affine : (tile0 + tiles) * affine]
        self.freeze()

    def _extra_repr(self) -> str:
        return f"input_dims={self.input_dims}, output_dims={self.output_dims}, packed"

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x2 = x.reshape(-1, self.input_dims).astype(mx.bfloat16)
        y = packed_matmul(
            x2,
            self.packed_weight,
            self.packed_scales,
            self.packed_biases,
            self.input_dims,
            self.output_dims,
        )
        return y.reshape(*x.shape[:-1], self.output_dims).astype(dtype)


def project(linears, x: mx.array):
    """Outputs of adjacent packed projections sharing one store, or None.

    One dispatch covers the whole run of tiles, then the output is split.
    """
    if len(linears) < 2 or not all(isinstance(lin, PackedLinear) for lin in linears):
        return None
    store = linears[0]._store
    tile = linears[0]._tile0
    for lin in linears:
        if lin._store is not store or lin._tile0 != tile:
            return None
        tile += lin.output_dims // TILE_N
    first = linears[0]._tile0
    n = (tile - first) * TILE_N
    groups = store.K // 64
    codes = groups * TILE_N * 32
    affine = groups * TILE_N
    if first == 0 and n == store.N:
        w, sc, bi = store.w, store.sc, store.bi
    else:
        w = store.w[first * codes : tile * codes]
        sc = store.sc[first * affine : tile * affine]
        bi = store.bi[first * affine : tile * affine]
    x2 = x.reshape(-1, store.K).astype(mx.bfloat16)
    y = packed_matmul(x2, w, sc, bi, store.K, n)
    y = y.reshape(*x.shape[:-1], n).astype(x.dtype)
    outputs = []
    offset = 0
    for lin in linears:
        outputs.append(y[..., offset : offset + lin.output_dims])
        offset += lin.output_dims
    return tuple(outputs)


def eligible(linear: Any) -> bool:
    return (
        type(linear) is nn.QuantizedLinear
        and getattr(linear, "mode", "affine") == "affine"
        and int(linear.bits) == 4
        and int(linear.group_size) == 64
        and "bias" not in linear
        and linear.get("biases") is not None
        and linear.scales.dtype == mx.bfloat16
        and int(linear.weight.shape[0]) % TILE_N == 0
        and int(linear.weight.shape[1]) * 8 % 256 == 0
    )


def _pack(linears) -> list[PackedLinear]:
    """Pack same-input linears into one store; return their replacements."""
    K = int(linears[0].weight.shape[1]) * 8
    groups = K // 64
    weight = mx.concatenate([lin.weight for lin in linears])
    scales = mx.concatenate([lin.scales for lin in linears])
    biases = mx.concatenate([lin.biases for lin in linears])
    n = int(weight.shape[0])
    tiles = n // TILE_N
    codes = weight.view(mx.uint8).reshape(tiles, TILE_N, groups, 32)

    def affine(a):
        a = a.reshape(tiles, TILE_N, groups // 4, 4).transpose(0, 2, 1, 3)
        return mx.contiguous(a).reshape(-1)

    store = _PackedStore()
    store.K = K
    store.N = n
    store.w = mx.contiguous(codes.transpose(0, 2, 1, 3)).reshape(-1)
    store.sc = affine(scales)
    store.bi = affine(biases)
    mx.eval(store.w, store.sc, store.bi)
    packed = []
    tile0 = 0
    for lin in linears:
        rows = int(lin.weight.shape[0])
        packed.append(PackedLinear(store, tile0, rows))
        tile0 += rows // TILE_N
    mx.eval([(p.packed_weight, p.packed_scales, p.packed_biases) for p in packed])
    return packed


# Projections that read the same input, in the order the verifier asks for them.
_LAYER_GROUPS = (
    ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
    ("linear_attn", ("out_proj",)),
    ("self_attn", ("q_proj", "k_proj", "v_proj")),
    ("self_attn", ("o_proj",)),
    ("mlp", ("gate_proj", "up_proj")),
    ("mlp", ("down_proj",)),
)


def _runs(parent, names):
    """Maximal runs of adjacent eligible projections."""
    run = []
    for name in names:
        if eligible(getattr(parent, name, None)):
            run.append(name)
            continue
        if run:
            yield run
        run = []
    if run:
        yield run


def _pack_layer(layer: Any) -> int:
    count = 0
    for attr, names in _LAYER_GROUPS:
        parent = getattr(layer, attr, None)
        if parent is None:
            continue
        for run in _runs(parent, names):
            packed = _pack([getattr(parent, name) for name in run])
            for name, module in zip(run, packed):
                setattr(parent, name, module)
            count += len(run)
    return count


def pack_drafter(model: Any) -> int:
    """Replace eligible 4-bit projections of a DFlash draft model.

    Its forward runs block x rows through every projection, like a verify.
    """
    count = sum(_pack_layer(layer) for layer in getattr(model, "layers", ()))
    fc = getattr(model, "fc", None)
    if eligible(fc):
        model.fc = _pack([fc])[0]
        count += 1
    if count:
        warmup(model)
    return count


def pack_model(model: Any) -> int:
    """Replace eligible 4-bit projections of a dense Qwen3.5 text model.

    Returns the number of replaced layers. Must run on the MLX executor
    thread after the weights are materialized.
    """
    from ..scheduler import _sync_and_clear_cache

    language = getattr(model, "language_model", model)
    inner = getattr(language, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    count = 0
    for layer in layers:
        count += _pack_layer(layer)
        # Freed originals sit in the MLX buffer pool; drain per layer so the
        # transient stays at one layer's worth.
        _sync_and_clear_cache()
    head = getattr(language, "lm_head", None)
    if eligible(head):
        language.lm_head = _pack([head])[0]
        count += 1
        _sync_and_clear_cache()
    if count:
        start = time.perf_counter()
        warmup(model)
        _sync_and_clear_cache()
        logger.info(
            "Qwen packed projection kernels compiled in %.1fs",
            time.perf_counter() - start,
        )
    return count


# Row counts that select distinct pipelines: matvec, the tensor-unit kernel
# with each K-split reduce, and the three GEMM tiles.
_WARM_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 65, 513)


def warmup(model: Any) -> None:
    """Compile the kernels for every packed shape before the first request.

    Covers each projection alone (decode, prefill) and each multi-projection
    store (verify dispatches through ``project``).
    """
    shapes: dict = {}
    for _, module in model.named_modules():
        if not isinstance(module, PackedLinear):
            continue
        shapes.setdefault(
            (module.input_dims, module.output_dims),
            (module.packed_weight, module.packed_scales, module.packed_biases),
        )
        store = module._store
        if store.N != module.output_dims:
            shapes.setdefault((store.K, store.N), (store.w, store.sc, store.bi))
    for (K, N), arrays in shapes.items():
        for rows in _WARM_ROWS:
            x = mx.zeros((rows, K), dtype=mx.bfloat16)
            mx.eval(packed_matmul(x, *arrays, K, N))


def enabled(model: Any) -> bool:
    """Dense Qwen3.5-family text model on a GPU with the NAX tensor unit."""
    language = getattr(model, "language_model", model)
    module = type(language).__module__ or ""
    if not ("qwen3_5" in module or "qwen35" in module) or "moe" in module:
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


def apply(verifier_cls) -> bool:
    """Serve same-store projection tuples of the verifier in one dispatch."""
    if getattr(verifier_cls, "_omlx_packed_linear", False):
        return True
    original_linears = verifier_cls._linears

    def _linears(self, linears, x):
        linears = tuple(linears)
        out = project(linears, x)
        if out is not None:
            return out
        packed = [i for i, lin in enumerate(linears) if isinstance(lin, PackedLinear)]
        if len(packed) < 2 or packed != list(range(len(packed))):
            return original_linears(self, linears, x)
        head = project(linears[: len(packed)], x)
        if head is None:
            return original_linears(self, linears, x)
        return head + tuple(original_linears(self, linears[len(packed) :], x))

    verifier_cls._linears = _linears
    verifier_cls._omlx_packed_linear = True
    return True
