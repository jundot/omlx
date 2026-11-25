# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-expert projection for row-exact verify blocks, grouped by expert.

A row-exact verify block keeps ``gather_qmm``'s unsorted per-pair gather, so
every (row, expert) pair streams its expert's weights on its own. Consecutive
tokens share experts (20 to 37% of the pairs of 2 to 4 rows of code), and
those repeats cost almost as much as a new expert.

This runs the same pairs in expert order, like ``moe_verify_gather`` does for
affine experts: one threadgroup per pair and output tile, the grid covering
every pair of a tile before the next tile, so pairs that share an expert run
back to back and read the tile from cache. Per pair the arithmetic is MLX
0.32.4's ``fp_qmv_impl`` (``fp_quantized.h``, the kernel ``gather_qmm`` takes
for one-row NVFP4 pairs whose K is not a multiple of 512), so each output
equals the ``gather_qmm`` output bit for bit. The match depends on the
kernel repeating that arithmetic, so rerun
tests/test_inkling2_verify_experts.py after an MLX pin change.

A one-row decode step takes it too: with no shared experts it still beats
``gather_qmm``'s launch for the same pairs, and the bits do not change. So does
any block below ``gather_qmm``'s switch to its batched sorted kernel, such as
a short prompt suffix after a prefix cache hit: there ``gather_qmm`` also runs
each pair on its own. With float16 activations the NVFP4 tensor scale is
applied in the epilogue, as ``gather_qmm`` does when it is passed
``global_scale``.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

# Lightning MTP verifies up to 16 rows.
MAX_ROWS = 16
# MLX 0.32.4's gather_qmm takes its batched kernel for sorted one-row pairs
# from four pairs per expert on; below that it runs each pair on its own.
PAIRS_PER_EXPERT = 4

_HEADER = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int GS = 16;
constant constexpr int VALUES_PER_THREAD = 8;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int BN = 8;
// 1 when the NVFP4 tensor scale is applied in the kernel (float16).
constant constexpr int GLOBAL = __GLOBAL__;
// 1 to decode the codes through a threadgroup byte table (M1/M2), 0 through
// a constant nibble table (M3 and later).
constant constexpr int TG_TABLE = __TG_TABLE__;
// 1 when every pair has its own expert (a decode step): no expert order.
constant constexpr int DIRECT = __DIRECT__;

// The fp4 values, as fp4_value decodes them.
constant float E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// fp4_e2m1 -> float, as fp4.h converts it (through half).
inline float fp4_value(uint8_t bits) {
  half converted = as_type<half>(ushort((bits & 7) << 9));
  converted *= 16384.0;
  return static_cast<float>(bits & 8 ? -converted : converted);
}

// fp8_e4m3 -> float, as fp8.h converts it (through half).
inline float fp8_value(uint8_t bits) {
  uint16_t v = bits & 127;
  uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted * 256.0);
}

// qdot / qdot_safe of fp_quantized.h at bits == 4 over the first n values.
// The codes are read through a table of their fp4 values, decoded as
// fp4_value does: every fp4 value is exact in float, so each product and the
// sum order are unchanged. With TG_TABLE, ``lut[b]`` holds the two values of
// byte b (low nibble first).
inline float qdot_n(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    int n,
    threadgroup const float2* lut) {
  float accum = 0;
  if (TG_TABLE) {
    for (int i = 0; i < (n / 4); i++) {
      float2 lo = lut[w[2 * i]];
      float2 hi = lut[w[2 * i + 1]];
      accum +=
          (x_thread[4 * i] * lo.x + x_thread[4 * i + 1] * lo.y +
           x_thread[4 * i + 2] * hi.x + x_thread[4 * i + 3] * hi.y);
    }
  } else {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (n / 4); i++) {
      uint16_t v = ws[i];
      accum +=
          (x_thread[4 * i] * E2M1[v & 15] +
           x_thread[4 * i + 1] * E2M1[(v >> 4) & 15] +
           x_thread[4 * i + 2] * E2M1[(v >> 8) & 15] +
           x_thread[4 * i + 3] * E2M1[(v >> 12) & 15]);
    }
  }
  return scale * accum;
}
"""

# Each threadgroup takes the pair at sorted position ``y`` and output tile
# ``z``. Per pair this is the full-tile branch of fp_qmv_impl (N is a
# multiple of 8, so no tile is moved back), with its ``has_global_scale``
# epilogue when GLOBAL is set.
_SOURCE = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    // Byte -> two fp4 values, shared by the threadgroup's 64 threads. On M1
    // the per-value decode costs more than the weight reads.
    threadgroup float2 lut[TG_TABLE ? 256 : 1];
    if (TG_TABLE) {
      for (uint b = simd_gid * SIMD_SIZE + simd_lid; b < 256;
           b += 2 * SIMD_SIZE) {
        lut[b] = float2(fp4_value(uint8_t(b & 15)), fp4_value(uint8_t(b >> 4)));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int pair = DIRECT ? int(threadgroup_position_in_grid.y)
                            : int(order[threadgroup_position_in_grid.y]);
    // An expert stack can pass 2 GiB, so its offsets need 64 bits.
    const size_t expert = size_t(rhs[pair]);

    const int in_vec_size_w = K_SIZE / 2;
    const int in_vec_size_g = K_SIZE / GS;
    const int out_row = int(threadgroup_position_in_grid.z) * BN +
        int(simd_gid) * RESULTS_PER_SIMDGROUP;

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * (size_t(N_SIZE) * in_vec_size_w) + out_row * in_vec_size_w +
        int(simd_lid) * 4;
    const device uint8_t* sc = scales +
        expert * (size_t(N_SIZE) * in_vec_size_g) + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + (pair / LHS_DIV) * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RESULTS_PER_SIMDGROUP] = {0};

    int k = 0;
    for (; k < K_SIZE - BLOCK_SIZE; k += BLOCK_SIZE) {
      for (int i = 0; i < VALUES_PER_THREAD; i++) {
        x_thread[i] = xp[i];
      }
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = fp8_value(sc[row * in_vec_size_g]);
        result[row] += qdot_n(wl, x_thread, s, VALUES_PER_THREAD, lut);
      }
      ws += BLOCK_SIZE / 2;
      sc += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    const int remaining = clamp(
        int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD),
        0,
        VALUES_PER_THREAD);
    if (remaining > 0) {
      for (int i = 0; i < remaining; i++) {
        x_thread[i] = xp[i];
      }
      for (int i = remaining; i < VALUES_PER_THREAD; i++) {
        x_thread[i] = 0;
      }
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = fp8_value(sc[row * in_vec_size_g]);
        result[row] += qdot_n(wl, x_thread, s, remaining, lut);
      }
    }

    float inv_scale_enc = 1.0f;
    if (GLOBAL) {
      inv_scale_enc = gscale[expert] / (448.0f * 6.0f);
    }
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      float v = simd_sum(result[row]);
      if (simd_lid == 0) {
        if (GLOBAL) {
          y[pair * N_SIZE + out_row + row] = static_cast<T>(v * inv_scale_enc);
        } else {
          y[pair * N_SIZE + out_row + row] = static_cast<T>(v);
        }
      }
    }
"""


@cache
def _threadgroup_table() -> bool:
    """M1 and M2 GPUs (applegpu_g13*, g14*) decode faster through a
    threadgroup byte table; on M3 and later its 2 KB and barrier per
    threadgroup cost more than a constant nibble table."""
    arch = mx.device_info().get("architecture", "")
    digits = "".join(c for c in arch.rsplit("_g", 1)[-1] if c.isdigit())
    return bool(digits) and int(digits) <= 14


@cache
def _kernel(global_scale: bool, direct: bool):
    tg = _threadgroup_table()
    header = (
        _HEADER.replace("__GLOBAL__", str(int(global_scale)))
        .replace("__TG_TABLE__", str(int(tg)))
        .replace("__DIRECT__", str(int(direct)))
    )
    return mx.fast.metal_kernel(
        name=(
            f"omlx_inkling2_verify_experts_nvfp4_g{int(global_scale)}"
            f"_t{int(tg)}_d{int(direct)}"
        ),
        input_names=["x", "w", "scales", "gscale", "rhs", "order"],
        output_names=["y"],
        header=header,
        source=_SOURCE,
    )


def supported(proj, dtype) -> bool:
    """True when ``proj`` is an NVFP4 expert stack this kernel matches.

    BF16 or FP16 activations (FP16 applies the tensor scale in the kernel, as
    ``gather_qmm`` does with ``global_scale``), whole 8-column output tiles,
    and a K that MLX sends to ``fp_qmv`` and not to ``fp_qmv_fast``
    (multiples of 512).
    """
    if mx.default_device() != mx.gpu or proj.mode != "nvfp4":
        return False
    n, k = proj.weight.shape[-2], proj.weight.shape[-1] * 8
    return dtype in (mx.bfloat16, mx.float16) and n % 8 == 0 and k % 512 != 0


def gather(
    proj, x: mx.array, rhs: mx.array, pairs_per_row: int, order: mx.array | None
):
    """``gather_qmm`` over P (row, expert) pairs, pairs grouped by expert.

    ``rhs`` holds the expert of each pair; pair ``p`` reads row
    ``p // pairs_per_row`` of ``x`` (``[R, K]``). ``order`` is
    ``pair_order(rhs)``, or None when no two pairs share an expert (one
    decode row), which runs the pairs as they come. Returns ``[P, N]``.
    """
    k = int(x.shape[-1])
    n = int(proj.weight.shape[-2])
    pairs = int(rhs.shape[0])
    in_kernel = proj.scales_in_kernel(x.dtype)
    # Unused without the in-kernel tensor scale.
    gscale = proj.global_scale if in_kernel else proj.scales
    direct = order is None
    return _kernel(in_kernel, direct)(
        inputs=[x, proj.weight, proj.scales, gscale, rhs, rhs if direct else order],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("LHS_DIV", int(pairs_per_row)),
        ],
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]
