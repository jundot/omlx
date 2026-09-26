# SPDX-License-Identifier: Apache-2.0
#
# Adapted from MTPLX (mtplx/verify_kernels.py)
#   Copyright 2026 Youssof Altoukhi
#   Licensed under the Apache License, Version 2.0
#   https://github.com/youssofal/mtplx
#   "Powered by MTPLX by Youssof Altoukhi"
#
# The split-K and multi-simdgroup (msg) kernel morphologies and their Metal
# source generation below are ported from MTPLX. The simdgroup-matrix tile
# kernel (``mma``) for 8..32 rows is adapted from dflash-mlx
# (dflash_mlx/verify_qmm.py, Copyright 2026 bstnxbt, Apache-2.0). The oMLX
# integration — thread-local armed routing through ``nn.QuantizedLinear``,
# the hybrid dispatch, and the row/N-floor gating — is original to oMLX.
"""Verify-shape quantized-matmul kernels for native MTP.

Speculative verify multiplies a skinny row batch (M = 1 + draft depth)
against the model's quantized weights. Stock MLX qmm is tuned for M=1
(decode) and large-M (prefill); at M=3..6 it pays a steep per-row penalty —
measured on Qwen3.6-27B/M3 Ultra the lm_head scales linearly (1.2ms at M=1 →
4.3ms at M=4) and the summed MLP/GDN projections add ~15ms per verify call.

Two kernel morphologies:

- split-K: grid y = N/4 column tiles, K reduction split across 2..4
  simdgroups with a threadgroup reduction. Wins in-context (mixed scheduling
  with attention/GDN kernels) — deep occupancy queues + latency hiding.
- msg (multi-simdgroup): one threadgroup carries NSG barrier-free
  simdgroups, each owning a BN=4 column tile with the full-K reduction
  lane-strided over pack-interleaved weight words. Wins on huge-N (lm_head)
  where the split-K tiny-tile grid thrashes the scheduler.

Dispatch: split-K everywhere, msg for N >= 100k, and for 7..24 rows (batched
verify: rows = requests x block) the ``mma`` tile kernel, which dequantizes
16-column weight tiles into threadgroup memory once and multiplies them
against every row with simdgroup matrices, so a batch of rows costs one pass
over the weights. Routing is armed only around the MTP verify forward via a
thread-local flag (set by
``omlx.patches.mlx_lm_mtp.batch_generator._call_backbone``) so nothing else
in the process sees the patched ``nn.QuantizedLinear``.

Numerics: fp32 accumulation in lane-strided K order differs from stock qmm
at bf16 tail-ULP level. Greedy outputs can therefore occasionally diverge
from the unrouted path (the token is still trunk-verified — the divergence
class is the same as any kernel change).

Supported: 4-bit and 8-bit affine, group_size in {32, 64, 128}, bf16/fp16
activations, M in 3..6, K % 64 == 0, N % 4 == 0. The mma path takes 4-bit
and 5-bit affine, M in 7..24, K % 256 == 0, N % 16 == 0. Everything else falls back to
stock.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_KERNEL_CACHE: dict = {}
_ROUTE_ARMED = threading.local()

_MSG_NSG = 8  # simdgroups per threadgroup for the msg (lm_head) kernel

# Only route projections with N >= this. The vk kernels beat stock qmm on GPU
# time at every eligible shape, but each ``mx.fast.metal_kernel`` invocation
# pays more Python/dispatch overhead than the built-in fast path. With ~330
# projections per verify forward, routing the small GDN/attention shapes
# (~1.0x GPU win) costs more on the CPU than it saves; the large-N shapes are
# few calls with real wins (lm_head 2.6-3.3x at M=3-4).
_MIN_ROUTE_N = 16384
# The mma path wins from much smaller N because stock qmm reads the weights
# more than once (qmv_wide tiles) or wastes half a 32-row tile at these M.
_MIN_MMA_ROUTE_N = 4096


def set_verify_qmm_armed(flag: bool) -> None:
    """Arm/disarm verify-qmm routing (MTP verify forwards only)."""
    _ROUTE_ARMED.value = bool(flag)


def _is_armed() -> bool:
    return getattr(_ROUTE_ARMED, "value", False)


# ---------------------------------------------------------------------------
# msg kernel — lm_head geometry.
# ---------------------------------------------------------------------------


def _fma_block(m: int, bits: int) -> str:
    per = 8 if bits == 4 else 4
    mask = "0xFu" if bits == 4 else "0xFFu"
    shift = 4 if bits == 4 else 8
    lines = [f"for (int ki = 0; ki < {per}; ++ki) {{"]
    for j in range(4):
        lines.append(
            f"    float w{j} = float((p{j} >> (ki * {shift})) & {mask}) * s{j} + b{j};"
        )
    for j in range(4):
        for r in range(m):
            lines.append(
                f"    acc[{j} * {m} + {r}] += "
                f"float(v{r}[ki{'' if bits == 4 else ' + koff'}]) * w{j};"
            )
    lines.append("}")
    return "\n            ".join(lines)


def _build_msg_kernel(m: int, bits: int, group_size: int, dtype, nsg: int):
    import mlx.core as mx

    key = ("msg", m, bits, group_size, dtype, nsg)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    xloads = "\n            ".join(
        f"Vec8 v{r} = xv[({r} * K + k_base) / 8];" for r in range(m)
    )
    if bits == 4:
        pack_setup = """
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_p + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_p + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_p + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_p + pack];
        """
        body = f"""
        for (int pack = int(lane); pack < K_by_p; pack += 32) {{
            {pack_setup}
            {xloads}
            float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
            _Pragma("unroll")
            {_fma_block(m, 4)}
        }}
        """
    else:
        body = f"""
        for (int pair = int(lane); pair < K_by_p; pair += 32) {{
            int k_base = pair * 8;
            int gi = k_base / GS;
            {xloads}
            _Pragma("unroll")
            for (int wsel = 0; wsel < 2; ++wsel) {{
                int koff = wsel * 4;
                uint32_t p0 = w_q[(n0 + 0) * (K / 4) + pair * 2 + wsel];
                uint32_t p1 = w_q[(n0 + 1) * (K / 4) + pair * 2 + wsel];
                uint32_t p2 = w_q[(n0 + 2) * (K / 4) + pair * 2 + wsel];
                uint32_t p3 = w_q[(n0 + 3) * (K / 4) + pair * 2 + wsel];
                float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
                float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
                float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
                float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
                float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
                float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
                float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
                float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
                _Pragma("unroll")
                {_fma_block(m, 8)}
            }}
        }}
        """

    n_acc = 4 * m
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};
        constexpr int MROWS = {m};

        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_p = {"K / 8" if bits == 4 else "K / 8"};
        int K_by_gs = K / GS;
        int n0 = (int(tg_n) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {body}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        if (lane < {n_acc}) {{
            int j = int(lane) / MROWS;
            int row = int(lane) - j * MROWS;
            y[row * N + n0 + j] = T(acc[int(lane)]);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_m{m}_q{bits}_nsg{nsg}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# split-K kernel — default.
# ---------------------------------------------------------------------------


def _pack_block(m: int, bits: int, sfx: str) -> str:
    p = f"pack{sfx}"
    lines = [f"int k_base{sfx} = {p} * 8;", f"int gi{sfx} = k_base{sfx} / GS;"]
    for r in range(m):
        lines.append(f"Vec8 v{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8];")
    if bits == 4:
        for j in range(4):
            lines.append(f"uint32_t p{sfx}_{j} = w_q[(n0 + {j}) * K_by_p + {p}];")
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t packed = p{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 8; ++ki) {",
                "        float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wv;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    else:
        for j in range(4):
            lines.append(
                f"uint32_t pa{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2];"
                f" uint32_t pb{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2 + 1];"
            )
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t pa = pa{sfx}_{j};",
                f"    uint32_t pb = pb{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 4; ++ki) {",
                "        float wa = float((pa >> (ki * 8)) & 0xFFu) * s + b;",
                "        float wb = float((pb >> (ki * 8)) & 0xFFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wa;"
                )
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki + 4]) * wb;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    return "\n            ".join(lines)


def _build_ksplit_kernel(m: int, bits: int, group_size: int, dtype, *, k_parts: int):
    import mlx.core as mx

    key = ("ksplit", m, bits, group_size, dtype, k_parts)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    n_acc = 4 * m
    loop = f"""
        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {_pack_block(m, bits, "A")}
        }}
    """

    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int K_by_p = K / 8;
        int K_by_w = K / 4;
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;
        int N = int(N_size);
        int n0 = int(tg_n) * 4;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {loop}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < {n_acc}) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[row * N + n0 + j] = T(total);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_ks_m{m}_q{bits}_kp{k_parts}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# mma kernel — 7..32 rows, simdgroup matrix tiles (adapted from dflash-mlx's
# combo_ktmpl morphology).
# ---------------------------------------------------------------------------

_MMA_BN = 16
_MMA_BK = 32
_MMA_NSG = 8


def _build_mma_kernel(
    row_tiles: int, k_val: int, group_size: int, dtype, bits: int = 4
):
    """Rows padded to ``8 * row_tiles`` x 16 columns per threadgroup.

    Eight simdgroups each own one K chunk: they dequantize their BK x BN
    weight tile into a private threadgroup buffer and multiply it against
    every row tile with simdgroup matrices, so no cross-simdgroup barrier
    sits in the K loop. Partial sums are reduced once at the end. K is a
    template constant so the chunking folds into the code.
    """
    import mlx.core as mx

    key = ("mma", row_tiles, int(k_val), group_size, dtype, bits)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    if bits == 4:
        # One uint32 holds the pack's eight nibbles.
        unpack = """
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }"""
    else:
        # MLX 5-bit affine: eight values per five little-endian bytes.
        unpack = """
                const device uchar* wb = ((const device uchar*)w_q)
                    + (n_global * K_by_8 + (k_base >> 3)) * 5;
                uint64_t packed = uint64_t(wb[0]) | (uint64_t(wb[1]) << 8)
                    | (uint64_t(wb[2]) << 16) | (uint64_t(wb[3]) << 24)
                    | (uint64_t(wb[4]) << 32);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = uint32_t((packed >> (ki * 5)) & 0x1Fu);
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }"""

    decl = "\n        ".join(
        f"simdgroup_matrix<T, 8, 8> a{r};"
        f" simdgroup_matrix<float, 8, 8> c{r}L = simdgroup_matrix<float, 8, 8>(0.0f);"
        f" simdgroup_matrix<float, 8, 8> c{r}R = simdgroup_matrix<float, 8, 8>(0.0f);"
        for r in range(row_tiles)
    )
    loads = "\n                ".join(
        f"simdgroup_load(a{r}, x + {r} * 8 * K + k0 + ks * BK_SUB, K);"
        for r in range(row_tiles)
    )
    macs = "\n                ".join(
        f"simdgroup_multiply_accumulate(c{r}L, a{r}, b_L, c{r}L);"
        f" simdgroup_multiply_accumulate(c{r}R, a{r}, b_R, c{r}R);"
        for r in range(row_tiles)
    )
    stores = "\n        ".join(
        f"simdgroup_store(c{r}L, tg_partials[sg_id] + {r} * 8 * BN, BN);"
        f" simdgroup_store(c{r}R, tg_partials[sg_id] + {r} * 8 * BN + 8, BN);"
        for r in range(row_tiles)
    )
    source = f"""
        using namespace metal;
        constexpr int BM = {8 * row_tiles};
        constexpr int BN = {_MMA_BN};
        constexpr int BK = {_MMA_BK};
        constexpr int BK_SUB = 8;
        constexpr int NSG = {_MMA_NSG};
        constexpr int GS = {group_size};
        constexpr int K = {int(k_val)};
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane = tid % 32;
        uint tg_n = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<T, 8, 8> b_L, b_R;
        {decl}

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                {unpack}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                {loads}
                simdgroup_load(b_L, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                {macs}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {stores}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc = 0.0f;
            _Pragma("unroll")
            for (int g = 0; g < NSG; ++g) {{
                acc += tg_partials[g][off];
            }}
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_mma_rt{row_tiles}_k{int(k_val)}_q{bits}_gs{group_size}_"
        f"{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def mma_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    return (
        int(bits) in (4, 5)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        # Above 24 rows stock qmm fills its 32-row tile and wins.
        and 7 <= int(M) <= 24
        and int(K) % (_MMA_BK * _MMA_NSG) == 0
        and int(N) % _MMA_BN == 0
        and int(N) >= _MIN_MMA_ROUTE_N
    )


def vk_qmm_mma(x2, w_q, scales, biases, *, group_size: int, bits: int = 4):
    """(M, K) x (N, K)^T -> (M, N) for M in 7..24 through the mma tile kernel."""
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    row_tiles = (M + 7) // 8
    bm = 8 * row_tiles
    xm, M0 = _pad_rows(mx, x2, bm)
    kernel = _build_mma_kernel(row_tiles, K, group_size, x2.dtype, bits)
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, N],
        template=[("T", x2.dtype)],
        grid=(32 * _MMA_NSG, N // _MMA_BN, 1),
        threadgroup=(32 * _MMA_NSG, 1, 1),
        output_shapes=[(bm, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M0, :] if M0 < bm else y


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


def _pad_rows(mx, x2, m: int):
    M = int(x2.shape[0])
    if M < m:
        pad = mx.zeros((m - M, x2.shape[1]), dtype=x2.dtype)
        return mx.contiguous(mx.concatenate([x2, pad], axis=0)), M
    return mx.contiguous(x2), M


def vk_qmm(x2, w_q, scales, biases, *, bits: int, group_size: int):
    """Verify-shape qmm: (M, K) x (N, K)^T -> (M, N), M in 2..6.

    Rows are padded up to the kernel's M template. split-K for regular
    shapes, msg tile for huge N (lm_head).
    """
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    m = 4 if M <= 4 else 6

    if N >= 100000 and N % (4 * _MSG_NSG) == 0:
        xm, M0 = _pad_rows(mx, x2, m)
        kernel = _build_msg_kernel(m, bits, group_size, x2.dtype, _MSG_NSG)
        cols = 4 * _MSG_NSG
        (y,) = kernel(
            inputs=[xm, w_q, scales, biases, K, N],
            template=[("T", x2.dtype)],
            grid=(32 * _MSG_NSG, (N + cols - 1) // cols, 1),
            threadgroup=(32 * _MSG_NSG, 1, 1),
            output_shapes=[(m, N)],
            output_dtypes=[x2.dtype],
        )
        return y[:M0, :] if M0 < m else y

    # Small M gets a dedicated template (fewer wasted accumulators).
    m = max(2, min(6, M))
    xm, M0 = _pad_rows(mx, x2, m)
    k_parts = 2 if N >= 4096 else 4
    kernel = _build_ksplit_kernel(m, bits, group_size, x2.dtype, k_parts=k_parts)
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(32 * k_parts, N // 4, 1),
        threadgroup=(32 * k_parts, 1, 1),
        output_shapes=[(m, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M0, :] if M0 < m else y


def vk_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    # M >= 3: at M=2 (depth-1 verify) the dispatch overhead eats the GPU win
    # AND skipping it keeps depth-1 greedy output bit-identical to the
    # unrouted path. Depth >= 2 verifies at M >= 3 where the kernels pay.
    return (
        int(bits) in (4, 8)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 3 <= int(M) <= 6
        and int(K) % 64 == 0
        and int(N) % 4 == 0
        and int(N) >= _MIN_ROUTE_N
    )


# ---------------------------------------------------------------------------
# QuantizedLinear routing patch.
# ---------------------------------------------------------------------------

_QL_PATCHED = False


def apply_verify_qmm_patch() -> bool:
    """Route verify-shaped ``nn.QuantizedLinear`` calls to the vk kernels.

    Strictly gated: only while the MTP verify flag is armed
    (``set_verify_qmm_armed``), only rank-3 inputs whose rows (batch x block)
    fall in 3..6 (split-K) or 7..24 (mma), only affine layouts the kernels
    support. Everything else takes the stock path.
    """
    global _QL_PATCHED
    if _QL_PATCHED:
        return True

    import mlx.core as mx
    import mlx.nn as nn

    cls = nn.QuantizedLinear
    if getattr(cls, "_omlx_verify_qmm_patched", False):
        _QL_PATCHED = True
        return True

    orig_call = cls.__call__

    def patched_call(self, x):
        if (
            not _is_armed()
            or x.ndim != 3
            or getattr(self, "mode", "affine") != "affine"
            or (x.shape[0] == 1 and x.shape[1] < 2)
        ):
            return orig_call(self, x)
        batch, length, K = x.shape
        rows = batch * length
        N = self.scales.shape[0]
        route = None
        if mma_eligible(rows, K, N, self.bits, self.group_size, x.dtype):
            route = "mma"
        elif rows <= 6 and vk_eligible(rows, K, N, self.bits, self.group_size, x.dtype):
            route = "vk"
        if route is None:
            return orig_call(self, x)
        try:
            # Batched verify rows (requests x block) share one weight pass.
            x2 = x[0] if batch == 1 else x.reshape(rows, K)
            if route == "mma":
                y = vk_qmm_mma(
                    x2,
                    self.weight,
                    self.scales,
                    self.biases,
                    group_size=self.group_size,
                    bits=self.bits,
                )
            else:
                y = vk_qmm(
                    x2,
                    self.weight,
                    self.scales,
                    self.biases,
                    bits=self.bits,
                    group_size=self.group_size,
                )
            if hasattr(self, "bias"):
                y = y + self.bias
            return y[None] if batch == 1 else y.reshape(batch, length, N)
        except Exception:
            logger.debug("verify qmm route failed; stock fallback", exc_info=True)
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_verify_qmm_patched = True
    _QL_PATCHED = True
    logger.info("MTP verify qmm patch applied (rows 2..6 split-K, 7..24 mma)")
    return True
