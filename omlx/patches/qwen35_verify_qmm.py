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
#
# The ``sg8`` kernels' bf16 ``0x4300 | q`` weight operand and per-group input
# sum correction are adapted from Splash
# (runtime/metal/kernels/decode/linear_q4_sgmatrix.metal, Apache-2.0,
# https://github.com/incoai/splash).
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

Blocks of four to eight rows (DFlash and Lightning MTP verifies) take
``sg8`` instead: bf16 rows, 4- or 5-bit weights, N >= 1024. Each lane builds its simdgroup-matrix
operands in registers from one packed word per 32 inputs (bf16 ``128 + q``,
exact), so no threadgroup dequant tile is needed; the group fold removes
``128 * sum(x)`` and applies scale and bias. The verifier's projection tuples
share one launch (``vk_group_sg8``, mixed bit widths allowed) and the dense
MLP's gate/up/SiLU is one launch (``vk_swiglu_sg8``).

The armed verifier also fuses each residual add with the next RMSNorm
(bit-exact to the separate ops) and submits the partial graph every few
layers, so the GPU starts while the host still builds the verify.
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
# Draft k/v and conv projections (N 1024..1280) also gain from sg8.
_MIN_SG8_ROUTE_N = 1024
_SG8_MIN_ROWS = 4


def set_verify_qmm_armed(flag: bool) -> None:
    """Arm/disarm verify-qmm routing (MTP verify forwards only)."""
    _ROUTE_ARMED.value = bool(flag)
    _ROUTE_ARMED.layers = 0


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


# ---------------------------------------------------------------------------
# sg8 kernel - up to eight rows on exact bf16 (128 + q) simdgroup operands.
# ---------------------------------------------------------------------------


# One simdgroup computes C^T = W X^T for CT x 8 columns and all rows. A lane
# (fm, fn) owns W[n0 + fm][k + fn, k + fn + 1] (one byte of a 4-bit word), so
# the MLX layout feeds the matrix operand directly. The bf16 bits 0x4300 | q
# encode 128 + q exactly; each group then folds
# s * (acc - 128 * sum(x)) + b * sum(x) into the fp32 output. The two-matrix
# form reads gate and up weights in one pass and writes silu(gate) * up.


def _sg8_source(mats: int) -> str:
    names = [("w_q", "scales", "biases"), ("w_q2", "scales2", "biases2")][:mats]
    decl = "\n".join(
        f"    float2 out{m}[CT];\n"
        f"    for (int c = 0; c < CT; ++c) out{m}[c] = float2(0.0f);"
        for m in range(mats)
    )
    acc_decl = "\n".join(
        f"        simdgroup_matrix<float, 8, 8> acc{m}[CT];\n"
        f"        for (int c = 0; c < CT; ++c) acc{m}[c] = "
        f"simdgroup_matrix<float, 8, 8>(0.0f);"
        for m in range(mats)
    )
    mma = "\n".join(f"""                {{
                    const device uchar* wg =
                        (const device uchar*){w} + long(n) * ROW_BYTES + long(g) * WB;
                    uint wds[2];
                    load_words(wg, qd, wds);
                    for (int j = 0; j < 4; ++j) {{
                        simdgroup_matrix<T, 8, 8> a, b;
                        uint pair = operand_pair(wds, j);
                        a.thread_elements()[0] = as_type<T>(ushort(pair));
                        a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                        b.thread_elements()[0] = xa[j];
                        b.thread_elements()[1] = xb[j];
                        simdgroup_multiply_accumulate(acc{m}[c], a, b, acc{m}[c]);
                    }}
                }}""" for m, (w, _, _) in enumerate(names))
    fold = "\n".join(f"""            {{
                float s = float({s}[n * GROUPS + g]);
                float bias = float({b}[n * GROUPS + g]);
                float2 a2 = float2(acc{m}[c].thread_elements()[0], acc{m}[c].thread_elements()[1]);
                out{m}[c] += s * a2 + (bias - 128.0f * s) * xsum;
            }}""" for m, (_, s, b) in enumerate(names))
    parts = "\n".join(
        f"    for (int c = 0; c < CT; ++c) part[(({m} * CT + c) * NSG + sg) * 32 + lane] = out{m}[c];"
        for m in range(mats)
    )
    if mats == 1:
        epilogue = """
            float2 t = sum_parts(part, 0, c, lane);
            float2 v = t;"""
    else:
        # Round gate and up to T like the separate projections, then swiglu.
        epilogue = """
            float2 g2 = float2(T(sum_parts(part, 0, c, lane).x), T(sum_parts(part, 0, c, lane).y));
            float2 u2 = float2(T(sum_parts(part, 1, c, lane).x), T(sum_parts(part, 1, c, lane).y));
            float2 v = g2 / (1.0f + exp(-g2)) * u2;"""
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    uint qid = lane >> 2;
    uint fm = (qid & 4) | ((lane >> 1) & 3);
    uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    constexpr int GROUPS = K / GS;
    constexpr int PER = GROUPS / NSG;
    constexpr int WB = GS * BITS / 8;
    constexpr long ROW_BYTES = long(K) * BITS / 8;
    int g_begin = int(sg) * PER;
    int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
    bool r0 = int(fn) < M;
    bool r1 = int(fn) + 1 < M;
    // Step j of a 32-wide k block: A columns 2i and 2i + 1 are values j and
    // j + 4 of the lane's eight packed values i * 8 .. i * 8 + 7, so both bf16
    // operands come from one shift and mask. Its x values over j are adjacent.
    uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    // Lanes past the last row read row M - 1 and never store, so the hot
    // loop has no row branches.
    const device T* x0 = x + min(int(fn), M - 1) * K + kx;
    const device T* x1 = x + min(int(fn) + 1, M - 1) * K + kx;
    auto load_words = [&](const device uchar* wg, int qd, thread uint* wds) {{
        if (BITS == 4) {{
            wds[0] = ((const device uint*)(wg + qd * 16))[fn >> 1];
        }} else {{
            // 40 bits from byte 5 i of the 20-byte block, split at value 4.
            const device uint* wp = (const device uint*)(wg + qd * 20);
            uint at = (fn >> 1) * 5;
            ulong v = (ulong(wp[at / 4 + 1]) << 32) | ulong(wp[at / 4]);
            v >>= (at % 4) * 8;
            wds[0] = uint(v);
            wds[1] = uint(v >> 20);
        }}
    }};
    auto operand_pair = [&](thread uint* wds, int j) {{
        if (BITS == 4)
            return ((wds[0] >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
        return ((wds[0] >> (5 * j)) & 0x1Fu) | (((wds[1] >> (5 * j)) & 0x1Fu) << 16)
            | 0x43004300u;
    }};
{decl}
    for (int g = g_begin; g < g_begin + PER; ++g) {{
{acc_decl}
        float2 xsum = float2(0.0f);
        for (int qd = 0; qd < GS / 32; ++qd) {{
            int kq = g * GS + qd * 32;
            T xa[4], xb[4];
            vec<T, 4> va = *((const device vec<T, 4>*)(x0 + kq));
            vec<T, 4> vb = *((const device vec<T, 4>*)(x1 + kq));
            for (int j = 0; j < 4; ++j) {{
                xa[j] = va[j];
                xb[j] = vb[j];
                xsum += float2(float(xa[j]), float(xb[j]));
            }}
            for (int c = 0; c < CT; ++c) {{
                int n = nbase + c * 8 + int(fm);
{mma}
            }}
        }}
        // Lanes sharing fn differ only in the fm bits of the lane id.
        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));
        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));
        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));
        for (int c = 0; c < CT; ++c) {{
            long n = nbase + c * 8 + int(fm);
{fold}
        }}
    }}
    threadgroup float2 part[{mats} * NSG * CT * 32];
{parts}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sum_parts = [&](threadgroup float2* p, int m, int c, uint ln) {{
        float2 t = float2(0.0f);
        for (int i = 0; i < NSG; ++i) t += p[((m * CT + c) * NSG + i) * 32 + ln];
        return t;
    }};
    if (sg == 0) {{
        for (int c = 0; c < CT; ++c) {{
            int n = nbase + c * 8 + int(fm);{epilogue}
            if (r0) y[int(fn) * N + n] = T(v.x);
            if (r1) y[(int(fn) + 1) * N + n] = T(v.y);
        }}
    }}
"""


_XSUM_ACCUM = (
    "                xsum += float2(float(xa[j]), float(xb[j]));\n",
    "                xsum += float2(float(xa), float(xb));\n",
)
_XSUM_REDUCE = (
    "        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));\n"
    "        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));\n"
    "        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));\n"
)


def _with_group_sums(source: str) -> str:
    """Read the per-group input sums from ``xs`` (M, K / GS) instead."""
    for line in _XSUM_ACCUM:
        source = source.replace(line, "")
    assert source.count(_XSUM_REDUCE) == 1
    source = source.replace(_XSUM_REDUCE, "")
    return source.replace(
        "        float2 xsum = float2(0.0f);\n",
        "        float2 xsum = float2(xs[min(int(fn), M - 1) * GROUPS + g],\n"
        "                             xs[min(int(fn) + 1, M - 1) * GROUPS + g]);\n",
    )


def _build_sg8_kernel(mats: int = 1, sums: bool = False):
    import mlx.core as mx

    key = ("sg8", mats, sums)
    if key not in _KERNEL_CACHE:
        inputs = ["x", "w_q", "scales", "biases"]
        if mats == 2:
            inputs += ["w_q2", "scales2", "biases2"]
        source = _sg8_source(mats)
        if sums:
            inputs.append("xs")
            source = _with_group_sums(source)
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"omlx_vk_sg8_m{mats}" + ("_xs" if sums else ""),
            input_names=inputs,
            output_names=["y"],
            source=source,
        )
    return _KERNEL_CACHE[key]


def _sg8_geometry(K: int, N: int) -> tuple[int, int]:
    """Simdgroups per threadgroup (K split) and 8-column tiles per simdgroup.

    Measured with serialized dependent projections on M3 Ultra: vocab-size N
    fills the GPU with four simdgroups and four tiles; everything else wants
    eight K partitions, and fat-K down projections two tiles to reuse x.
    """
    if N >= 65536:
        return 4, 4
    return 8, 2 if K >= 16384 else 1


def sg8_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    nsg, ct = _sg8_geometry(int(K), int(N))
    return (
        int(bits) in (4, 5)
        and int(group_size) in (32, 64, 128)
        # 0x4300 | q is the bf16 encoding of 128 + q.
        and dtype == mx.bfloat16
        # Two or three rows stay on split-K, which is faster there in context.
        and _SG8_MIN_ROWS <= int(M) <= 8
        and int(K) % (int(group_size) * nsg) == 0
        and int(N) % (8 * ct) == 0
        and int(N) >= _MIN_SG8_ROUTE_N
    )


def _sg8_call(x2, mats, weights, *, group_size: int, bits: int, sums=None):
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(weights[0].shape[0])
    nsg, ct = _sg8_geometry(K, N)
    if sums is not None and int(group_size) != _SUM_GROUP:
        sums = None
    extra = [] if sums is None else [sums]
    (y,) = _build_sg8_kernel(mats, sums is not None)(
        inputs=[x2, *weights, *extra],
        template=[
            ("T", x2.dtype),
            ("K", K),
            ("N", N),
            ("M", M),
            ("BITS", int(bits)),
            ("GS", int(group_size)),
            ("NSG", nsg),
            ("CT", ct),
        ],
        grid=(32 * nsg, N // (8 * ct), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x2.dtype],
    )
    return y


def vk_qmm_sg8(x2, w_q, scales, biases, *, group_size: int, bits: int = 4, sums=None):
    """(M, K) x (N, K)^T -> (M, N) for M in 7..8 through the sg8 kernel."""
    return _sg8_call(
        x2, 1, (w_q, scales, biases), group_size=group_size, bits=bits, sums=sums
    )


def vk_swiglu_sg8(x2, gate, up, sums=None):
    """silu(x gate^T) * (x up^T) for M in 7..8 in one pass over both weights."""
    return _sg8_call(
        x2,
        2,
        (gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases),
        group_size=gate.group_size,
        bits=gate.bits,
        sums=sums,
    )


# Inputs whose producer already wrote their per-64 group sums, by identity.
_SUM_GROUP = 64


def register_group_sums(x, sums) -> None:
    table = getattr(_ROUTE_ARMED, "sums", None)
    if table is None:
        table = _ROUTE_ARMED.sums = {}
    table[id(x)] = (x, sums)


def group_sums_for(x):
    entry = (getattr(_ROUTE_ARMED, "sums", None) or {}).get(id(x))
    return entry[1] if entry is not None and entry[0] is x else None


_GROUP_SUMS_SOURCE = """
    uint e = thread_position_in_grid.x;
    const device T* p = x + long(e) * 64;
    float acc = 0.0f;
    for (int i = 0; i < 64; i += 4) {
        vec<T, 4> v = *((const device vec<T, 4>*)(p + i));
        acc += float(v[0]) + float(v[1]) + float(v[2]) + float(v[3]);
    }
    xs[e] = acc;
"""


def group_sums(x):
    """Per-64 sums of a row-major ``(..., K)`` input, as ``(rows, K / 64)``."""
    import mlx.core as mx

    if "group_sums" not in _KERNEL_CACHE:
        _KERNEL_CACHE["group_sums"] = mx.fast.metal_kernel(
            name="omlx_verify_group_sums",
            input_names=["x"],
            output_names=["xs"],
            source=_GROUP_SUMS_SOURCE,
        )
    K = int(x.shape[-1])
    rows = x.size // K
    (sums,) = _KERNEL_CACHE["group_sums"](
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(rows * K // 64, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, K // 64)],
        output_dtypes=[mx.float32],
    )
    return sums


def clear_group_sums() -> None:
    _ROUTE_ARMED.sums = None


def sg8_swiglu_eligible(gate, up, x) -> bool:
    """Gate and up share geometry and quantization, and rows fit sg8."""
    import mlx.nn as nn

    if not (
        _is_armed()
        and isinstance(gate, nn.QuantizedLinear)
        and isinstance(up, nn.QuantizedLinear)
        and x.ndim == 3
        and getattr(gate, "mode", "affine") == "affine"
        and getattr(up, "mode", "affine") == "affine"
        and "bias" not in gate
        and "bias" not in up
        and gate.weight.shape == up.weight.shape
        and gate.bits == up.bits
        and gate.group_size == up.group_size
    ):
        return False
    rows = x.shape[0] * x.shape[1]
    K = x.shape[2]
    N = gate.scales.shape[0]
    return sg8_eligible(rows, K, N, gate.bits, gate.group_size, x.dtype)


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
    # metal_kernel copies non-contiguous inputs itself, so aligned rows pass
    # through without an extra dispatch.
    M = int(x2.shape[0])
    if M < m:
        pad = mx.zeros((m - M, x2.shape[1]), dtype=x2.dtype)
        return mx.concatenate([x2, pad], axis=0), M
    return x2, M


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


def _sg8_group_source(ns: tuple, bits: tuple) -> str:
    """One launch for projections sharing x; matrix i owns tiles [E_{i-1}, E_i).

    Matrices may mix 4- and 5-bit storage; the branch is uniform per
    threadgroup.
    """
    select, end = [], 0
    for i, (n, b) in enumerate(zip(ns, bits)):
        start, end = end, end + n // 8
        head = "if" if i == 0 else "} else if"
        select.append(
            f"    {head} (tile < {end}) {{\n"
            f"        wbase = (const device uchar*)w{i};\n"
            f"        sp = s{i};\n"
            f"        bp = b{i};\n"
            f"        yp = y{i};\n"
            f"        n_out = {n};\n"
            f"        wbits = {b};\n"
            f"        n = (tile - {start}) * 8 + int(fm);"
        )
    select.append("    }")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    uint qid = lane >> 2;
    uint fm = (qid & 4) | ((lane >> 1) & 3);
    uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    constexpr int GROUPS = K / GS;
    constexpr int PER = GROUPS / NSG;
    int tile = int(threadgroup_position_in_grid.y);
    const device uchar* wbase;
    const device T* sp;
    const device T* bp;
    device T* yp;
    int n_out = 0;
    int n = 0;
    int wbits = 4;
{chr(10).join(select)}
    const int WB = GS * wbits / 8;
    const long ROW_BYTES = long(K) * wbits / 8;
    int g_begin = int(sg) * PER;
    bool r0 = int(fn) < M;
    bool r1 = int(fn) + 1 < M;
    // Same operand order as the single-matrix kernel.
    uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    // Lanes past the last row read row M - 1 and never store, so the hot
    // loop has no row branches.
    const device T* x0 = x + min(int(fn), M - 1) * K + kx;
    const device T* x1 = x + min(int(fn) + 1, M - 1) * K + kx;
    float2 out = float2(0.0f);
    for (int g = g_begin; g < g_begin + PER; ++g) {{
        simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);
        float2 xsum = float2(0.0f);
        const device uchar* wg = wbase + long(n) * ROW_BYTES + long(g) * WB;
        for (int qd = 0; qd < GS / 32; ++qd) {{
            int kq = g * GS + qd * 32;
            uint w0, w1 = 0;
            if (wbits == 4) {{
                w0 = ((const device uint*)(wg + qd * 16))[fn >> 1];
            }} else {{
                const device uint* wp = (const device uint*)(wg + qd * 20);
                uint at = (fn >> 1) * 5;
                ulong v = (ulong(wp[at / 4 + 1]) << 32) | ulong(wp[at / 4]);
                v >>= (at % 4) * 8;
                w0 = uint(v);
                w1 = uint(v >> 20);
            }}
            vec<T, 4> va = *((const device vec<T, 4>*)(x0 + kq));
            vec<T, 4> vb = *((const device vec<T, 4>*)(x1 + kq));
            for (int j = 0; j < 4; ++j) {{
                T xa = va[j];
                T xb = vb[j];
                xsum += float2(float(xa), float(xb));
                uint pair = wbits == 4
                    ? ((w0 >> (4 * j)) & 0x000F000Fu) | 0x43004300u
                    : ((w0 >> (5 * j)) & 0x1Fu) | (((w1 >> (5 * j)) & 0x1Fu) << 16) | 0x43004300u;
                simdgroup_matrix<T, 8, 8> a, b;
                a.thread_elements()[0] = as_type<T>(ushort(pair));
                a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                b.thread_elements()[0] = xa;
                b.thread_elements()[1] = xb;
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }}
        }}
        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));
        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));
        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));
        float s = float(sp[long(n) * GROUPS + g]);
        float bias = float(bp[long(n) * GROUPS + g]);
        float2 a2 = float2(acc.thread_elements()[0], acc.thread_elements()[1]);
        out += s * a2 + (bias - 128.0f * s) * xsum;
    }}
    threadgroup float2 part[NSG * 32];
    part[sg * 32 + lane] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        float2 t = float2(0.0f);
        for (int i = 0; i < NSG; ++i) t += part[i * 32 + lane];
        if (r0) yp[int(fn) * n_out + n] = T(t.x);
        if (r1) yp[(int(fn) + 1) * n_out + n] = T(t.y);
    }}
"""


_SG8_GROUP_NSG = 8


def sg8_group_eligible(linears, x) -> bool:
    """Projections of one input with a shared layout that fit one sg8 launch."""
    import mlx.core as mx
    import mlx.nn as nn

    if not (
        _is_armed() and len(linears) > 1 and x.ndim == 3 and x.dtype == mx.bfloat16
    ):
        return False
    rows = x.shape[0] * x.shape[1]
    K = x.shape[2]
    first = linears[0]
    for linear in linears:
        if not (
            isinstance(linear, nn.QuantizedLinear)
            and getattr(linear, "mode", "affine") == "affine"
            and "bias" not in linear
            and int(linear.bits) in (4, 5)
            and linear.group_size == first.group_size
            and linear.weight.shape[1] * 32 // linear.bits == K
            and linear.scales.shape[0] % 8 == 0
            # The kernel binds every matrix's parameters to one pointer type.
            and linear.scales.dtype == x.dtype
            and linear.biases.dtype == x.dtype
        ):
            return False
    return (
        _SG8_MIN_ROWS <= rows <= 8
        and int(first.group_size) in (32, 64, 128)
        and K % (int(first.group_size) * _SG8_GROUP_NSG) == 0
        and max(linear.scales.shape[0] for linear in linears) >= _MIN_MMA_ROUTE_N
    )


def vk_group_sg8(x2, linears, sums=None):
    """Every projection of ``x2`` (M, K) in one launch; returns their outputs."""
    import mlx.core as mx

    ns = tuple(int(linear.scales.shape[0]) for linear in linears)
    bits = tuple(int(linear.bits) for linear in linears)
    if sums is not None and int(linears[0].group_size) != _SUM_GROUP:
        sums = None
    key = ("sg8_group", ns, bits, sums is not None)
    if key not in _KERNEL_CACHE:
        inputs = ["x"]
        for i in range(len(ns)):
            inputs += [f"w{i}", f"s{i}", f"b{i}"]
        source = _sg8_group_source(ns, bits)
        if sums is not None:
            inputs.append("xs")
            source = _with_group_sums(source)
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name="omlx_vk_sg8_group_"
            + "_".join(f"{n}q{b}" for n, b in zip(ns, bits))
            + ("_xs" if sums is not None else ""),
            input_names=inputs,
            output_names=[f"y{i}" for i in range(len(ns))],
            source=source,
        )
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    first = linears[0]
    args = [x2]
    for linear in linears:
        args += [linear.weight, linear.scales, linear.biases]
    if sums is not None:
        args.append(sums)
    return _KERNEL_CACHE[key](
        inputs=args,
        template=[
            ("T", x2.dtype),
            ("K", K),
            ("M", M),
            ("GS", int(first.group_size)),
            ("NSG", _SG8_GROUP_NSG),
        ],
        grid=(32 * _SG8_GROUP_NSG, sum(ns) // 8, 1),
        threadgroup=(32 * _SG8_GROUP_NSG, 1, 1),
        output_shapes=[(M, n) for n in ns],
        output_dtypes=[x2.dtype] * len(ns),
    )


def _patch_verify_grouped_linears() -> None:
    """Serve the verifier's same-input projection tuples in one sg8 launch."""
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._linears
    if getattr(original, "_omlx_sg8_group", False):
        return

    def _linears(self, linears, x):
        linears = tuple(linears)
        if sg8_group_eligible(linears, x):
            batch, length, K = x.shape
            outs = vk_group_sg8(
                x.reshape(batch * length, K), linears, group_sums_for(x)
            )
            return tuple(out.reshape(batch, length, -1) for out in outs)
        return original(self, linears, x)

    _linears._omlx_sg8_group = True
    Qwen3_5BatchInvariantForward._linears = _linears


def _patch_verify_swiglu() -> None:
    """Fuse the dense verify MLP's gate/up projections and swiglu into sg8."""
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._feed_forward
    if getattr(original, "_omlx_sg8_swiglu", False):
        return

    def _feed_forward(self, feed_forward, x):
        gate = getattr(feed_forward, "gate_proj", None)
        up = getattr(feed_forward, "up_proj", None)
        down = getattr(feed_forward, "down_proj", None)
        if (
            down is not None
            and not hasattr(feed_forward, "switch_mlp")
            and sg8_swiglu_eligible(gate, up, x)
        ):
            batch, length, K = x.shape
            hidden = vk_swiglu_sg8(
                x.reshape(batch * length, K), gate, up, group_sums_for(x)
            )
            hidden = hidden.reshape(batch, length, -1)
            if down.group_size == _SUM_GROUP:
                register_group_sums(hidden, group_sums(hidden))
            return self._linear(down, hidden)
        return original(self, feed_forward, x)

    _feed_forward._omlx_sg8_swiglu = True
    Qwen3_5BatchInvariantForward._feed_forward = _feed_forward


_FLUSH_LAYERS = 8

# Residual add then MLX rms_norm in one launch, one threadgroup per row.
# The sum is rounded like the separate add, and the square sums follow the
# per-thread order of MLX's rms kernels (four reads per step, 1024 lanes).
_ADD_RMS_SOURCE = """
    uint lid = thread_position_in_threadgroup.x;
    uint lsize = threads_per_threadgroup.x;
    uint sg = lid / 32;
    uint lane = lid % 32;
    uint row = threadgroup_position_in_grid.x;
    constexpr int STEPS = (D + 4095) / 4096;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    const device T* ap = a + long(row) * D;
    const device T* bp = b + long(row) * D;
    T vals[STEPS * 4];
    float acc = 0.0f;
    for (int st = 0; st < STEPS; ++st) {
        for (int i = 0; i < 4; ++i) {
            int idx = st * int(lsize) * 4 + int(lid) * 4 + i;
            T v = T(0);
            if (idx < D) {
                v = ap[idx] + bp[idx];
                s_out[long(row) * D + idx] = v;
                float xi = v;
                acc += xi * xi;
            }
            vals[st * 4 + i] = v;
        }
    }
    acc = simd_sum(acc);
    if (sg == 0)
        local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0)
            local_inv[0] = metal::precise::rsqrt(acc / D + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = local_inv[0];
    for (int st = 0; st < STEPS; ++st) {
        float part = 0.0f;
        int base = st * int(lsize) * 4 + int(lid) * 4;
        for (int i = 0; i < 4; ++i) {
            int idx = base + i;
            if (idx < D) {
                T nv = w[idx] * static_cast<T>(vals[st * 4 + i] * inv);
                n_out[long(row) * D + idx] = nv;
                part += float(nv);
            }
        }
        // Sixteen lanes cover one 64-wide group of the normed row.
        for (int off = 1; off < 16; off <<= 1)
            part += simd_shuffle_xor(part, ushort(off));
        if ((lid & 15) == 0 && base < D)
            xs_out[long(row) * (D / 64) + base / 64] = part;
    }
"""


def add_rms_norm(a, b, norm):
    """``(a + b, rms_norm(a + b), group sums)``; the first two match the
    separate MLX ops bit for bit, the sums are per 64 normed values."""
    import mlx.core as mx

    if "add_rms" not in _KERNEL_CACHE:
        _KERNEL_CACHE["add_rms"] = mx.fast.metal_kernel(
            name="omlx_verify_add_rms",
            input_names=["a", "b", "w", "eps"],
            output_names=["s_out", "n_out", "xs_out"],
            source=_ADD_RMS_SOURCE,
        )
    D = int(a.shape[-1])
    rows = a.size // D
    return _KERNEL_CACHE["add_rms"](
        inputs=[a, b, norm.weight, mx.array([norm.eps], dtype=mx.float32)],
        template=[("T", a.dtype), ("D", D)],
        grid=(1024 * rows, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[a.shape, a.shape, (rows, D // 64)],
        output_dtypes=[a.dtype, a.dtype, mx.float32],
    )


def _add_rms_eligible(a, b, norm) -> bool:
    import mlx.core as mx
    import mlx.nn as nn

    return (
        type(norm) is nn.RMSNorm
        and a.dtype == mx.bfloat16
        and b.dtype == a.dtype
        and a.shape == b.shape
        and norm.weight.dtype == a.dtype
        and norm.weight.shape == (a.shape[-1],)
        and 4096 < a.shape[-1] <= 16384
        and a.shape[-1] % 64 == 0
    )


def _patch_verify_layer_flush() -> None:
    """Fuse residual adds with the following RMSNorm and flush while building.

    An armed verify layer adds its mixer output and applies the MLP norm in
    one launch, and adds its MLP output together with the next layer's input
    norm. Every few layers the partial graph is submitted so the GPU starts
    while the host is still building the 27B verify.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._layer
    original_model = Qwen3_5BatchInvariantForward._model
    if getattr(original, "_omlx_flush", False):
        return
    import mlx.core as mx

    def _model(self, model, *args, **kwargs):
        layers = model.layers
        _ROUTE_ARMED.next_norm = {
            id(layer): nxt.input_layernorm for layer, nxt in zip(layers, layers[1:])
        }
        _ROUTE_ARMED.normed = None
        clear_group_sums()
        try:
            return original_model(self, model, *args, **kwargs)
        finally:
            _ROUTE_ARMED.next_norm = None
            _ROUTE_ARMED.normed = None
            clear_group_sums()

    def _mixed(self, layer, hidden, mask, cache, position_ids, position_embeddings):
        pending = getattr(_ROUTE_ARMED, "normed", None)
        if pending is not None and pending[0] is hidden:
            normed = pending[1]
        else:
            normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual = self._gated_delta(layer.linear_attn, normed, mask, cache)
        else:
            residual = self._attention(
                layer.self_attn,
                normed,
                mask,
                cache,
                position_ids,
                position_embeddings,
            )
        post = layer.post_attention_layernorm
        if not _add_rms_eligible(hidden, residual, post):
            hidden = hidden + residual
            return hidden + self._feed_forward(layer.mlp, post(hidden))
        hidden, normed, sums = add_rms_norm(hidden, residual, post)
        register_group_sums(normed, sums)
        out = self._feed_forward(layer.mlp, normed)
        nxt = (getattr(_ROUTE_ARMED, "next_norm", None) or {}).get(id(layer))
        if nxt is None or not _add_rms_eligible(hidden, out, nxt):
            return hidden + out
        hidden, normed, sums = add_rms_norm(hidden, out, nxt)
        register_group_sums(normed, sums)
        _ROUTE_ARMED.normed = (hidden, normed)
        return hidden

    def _layer(self, layer, hidden, *args, **kwargs):
        if not _is_armed():
            return original(self, layer, hidden, *args, **kwargs)
        hidden = _mixed(self, layer, hidden, *args, **kwargs)
        count = getattr(_ROUTE_ARMED, "layers", 0) + 1
        _ROUTE_ARMED.layers = count
        if count % _FLUSH_LAYERS == 0:
            mx.async_eval(hidden)
        return hidden

    _model._omlx_flush = True
    _layer._omlx_flush = True
    Qwen3_5BatchInvariantForward._layer = _layer
    Qwen3_5BatchInvariantForward._model = _model


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
        if sg8_eligible(rows, K, N, self.bits, self.group_size, x.dtype):
            route = "sg8"
        elif mma_eligible(rows, K, N, self.bits, self.group_size, x.dtype):
            route = "mma"
        elif rows <= 6 and vk_eligible(rows, K, N, self.bits, self.group_size, x.dtype):
            route = "vk"
        if route is None:
            return orig_call(self, x)
        try:
            # Batched verify rows (requests x block) share one weight pass.
            x2 = x.reshape(rows, K)
            if route == "sg8":
                y = vk_qmm_sg8(
                    x2,
                    self.weight,
                    self.scales,
                    self.biases,
                    group_size=self.group_size,
                    bits=self.bits,
                    sums=group_sums_for(x),
                )
            elif route == "mma":
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
            return y.reshape(batch, length, N)
        except Exception:
            logger.debug("verify qmm route failed; stock fallback", exc_info=True)
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_verify_qmm_patched = True
    _patch_verify_swiglu()
    _patch_verify_grouped_linears()
    _patch_verify_layer_flush()
    _QL_PATCHED = True
    logger.info("MTP verify qmm patch applied (rows 2..6 split-K, 7..8 sg8, 9..24 mma)")
    return True
