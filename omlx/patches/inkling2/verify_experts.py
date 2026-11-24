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
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

# Lightning MTP verifies up to 16 rows.
MAX_ROWS = 16

_HEADER = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int GS = 16;
constant constexpr int VALUES_PER_THREAD = 8;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int BN = 8;

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
inline float qdot_n(
    const device uint8_t* w, const thread float* x_thread, float scale, int n) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (n / 4); i++) {
    accum +=
        (x_thread[4 * i] * fp4_value(ws[i]) +
         x_thread[4 * i + 1] * fp4_value(ws[i] >> 4) +
         x_thread[4 * i + 2] * fp4_value(ws[i] >> 8) +
         x_thread[4 * i + 3] * fp4_value(ws[i] >> 12));
  }
  return scale * accum;
}
"""

# Each threadgroup takes the pair at sorted position ``y`` and output tile
# ``z``. Per pair this is the full-tile branch of fp_qmv_impl without a
# global scale (N is a multiple of 8, so no tile is moved back).
_SOURCE = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int pair = int(order[threadgroup_position_in_grid.y]);
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
        result[row] += qdot_n(wl, x_thread, s, VALUES_PER_THREAD);
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
        result[row] += qdot_n(wl, x_thread, s, remaining);
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      float v = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[pair * N_SIZE + out_row + row] = static_cast<T>(v);
      }
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_experts_nvfp4",
        input_names=["x", "w", "scales", "rhs", "order"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
    )


def supported(proj, dtype) -> bool:
    """True when ``proj`` is an NVFP4 expert stack this kernel matches.

    BF16 activations only (FP16 takes the tensor scale inside ``gather_qmm``),
    whole 8-column output tiles, and a K that MLX sends to ``fp_qmv`` and not
    to ``fp_qmv_fast`` (multiples of 512).
    """
    if mx.default_device() != mx.gpu or proj.mode != "nvfp4":
        return False
    n, k = proj.weight.shape[-2], proj.weight.shape[-1] * 8
    return dtype == mx.bfloat16 and n % 8 == 0 and k % 512 != 0


def gather(proj, x: mx.array, rhs: mx.array, pairs_per_row: int, order: mx.array):
    """``gather_qmm`` over P (row, expert) pairs, pairs grouped by expert.

    ``rhs`` holds the expert of each pair; pair ``p`` reads row
    ``p // pairs_per_row`` of ``x`` (``[R, K]``). ``order`` is
    ``pair_order(rhs)``. Returns ``[P, N]``.
    """
    k = int(x.shape[-1])
    n = int(proj.weight.shape[-2])
    pairs = int(rhs.shape[0])
    return _kernel()(
        inputs=[x, proj.weight, proj.scales, rhs, order],
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
