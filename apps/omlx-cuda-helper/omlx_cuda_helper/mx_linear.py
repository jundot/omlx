# SPDX-License-Identifier: Apache-2.0
"""MXFP8 linear layers on the block-scaled mma of sm_120 and sm_121.

The E4M3 codes and E8M0 scales (one per 32 values) of an MXFP8 weight multiply
the FP8 codes and scales of the activations as stored: the mma applies both
scales, so no product needs a correction on the CUDA cores. The sums are
float32, the output bf16.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_HEADER = r"""
__device__ __forceinline__ uint32_t mxl_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// A copy of `valid ? bytes : 0` bytes; the rest of the destination is zeroed.
__device__ __forceinline__ void mxl_cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void mxl_cp4(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void mxl_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void mxl_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void mxl_ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
"""

# y [M, N] of codes xa [M, K] and wb [N, K]. A block holds a 128 x 128 tile in
# 8 warps of 64 x 32; each stage brings 128 values of K, swizzled in 16-byte
# chunks so ldmatrix reads 8 rows without bank conflicts.
_MATMUL = r"""
extern __shared__ __align__(128) uint8_t smem[];
constexpr int BM = 128, BN = 128, KS = 128, KT = K / KS, G = KS / 32, STG = 2;
constexpr int A_ST = BM * KS, B_ST = BN * KS, STAGE = A_ST + B_ST + (BM + BN) * G;
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int wm = warp >> 2, wn = warp & 3;
const int M = xa_shape[0];
// Groups of 8 row blocks sweep the column blocks together, so each weight
// tile is read from memory once per group while the group's rows stay in L2.
const int nb = gridDim.x, mb = gridDim.y, id = blockIdx.y * nb + blockIdx.x;
const int first = id / (8 * nb) * 8, span = mb - first < 8 ? mb - first : 8;
const long m0 = (long)(first + id % (8 * nb) % span) * BM;
const long n0 = (long)(id % (8 * nb) / span) * BN;
const int mv = M - m0 < BM ? (int)(M - m0) : BM;

auto load = [&](int t) {
  uint8_t* s = smem + (t % STG) * STAGE;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = tid + i * 256, row = q >> 3, c = q & 7;
    const int sw = (c ^ (row & 7)) << 4;
    const bool ok = row < mv;
    mxl_cp16(mxl_smem(s + row * KS + sw), xa + (m0 + (ok ? row : 0)) * K + t * KS + c * 16, ok);
    mxl_cp16(mxl_smem(s + A_ST + row * KS + sw), wb + (n0 + row) * K + t * KS + c * 16, true);
  }
  // The four scales of a row in a stage are one 4-byte copy.
  if (tid < BM) {
    const bool ok = tid < mv;
    mxl_cp4(mxl_smem(s + A_ST + B_ST + tid * G), xs + (m0 + (ok ? tid : 0)) * (K / 32) + t * G, ok);
  } else {
    const int row = tid - BM;
    mxl_cp4(mxl_smem(s + A_ST + B_ST + (BM + row) * G), ws + (n0 + row) * (K / 32) + t * G, true);
  }
};

float acc[4][4][4];
#pragma unroll
for (int a = 0; a < 4; ++a)
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    #pragma unroll
    for (int v = 0; v < 4; ++v) acc[a][j][v] = 0.f;

load(0);
mxl_commit();
for (int t = 0; t < KT; ++t) {
  mxl_wait<0>();
  __syncthreads();
  if (t + 1 < KT) load(t + 1);
  mxl_commit();
  const uint8_t* s = smem + (t % STG) * STAGE;
  const uint8_t* sb = s + A_ST;
  const uint8_t* sas = sb + B_ST;
  const uint8_t* sbs = sas + BM * G;
  #pragma unroll
  for (int kb = 0; kb < G; ++kb) {
    uint32_t af[4][4], bf[4][2], sfa[4], sfb[4];
    const int idx = lane >> 3;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
      const int row = wm * 64 + a * 16 + (idx & 1) * 8 + (lane & 7);
      const int c = kb * 2 + (idx >> 1);
      mxl_ldsm4(mxl_smem(s + row * KS + ((c ^ (row & 7)) << 4)), af[a][0], af[a][1], af[a][2], af[a][3]);
      // Lane l supplies the scale of row 8 * (l % 2) + l / 4 of the 16.
      sfa[a] = sas[(wm * 64 + a * 16 + 8 * (lane & 1) + (lane >> 2)) * G + kb];
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = wn * 32 + p * 16 + (idx >> 1) * 8 + (lane & 7);
      const int c = kb * 2 + (idx & 1);
      mxl_ldsm4(mxl_smem(sb + n * KS + ((c ^ (n & 7)) << 4)), bf[2 * p][0], bf[2 * p][1], bf[2 * p + 1][0], bf[2 * p + 1][1]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) sfb[j] = sbs[(wn * 32 + j * 8 + (lane >> 2)) * G + kb];
// CUDA 13 names an sm_12xa target by __CUDA_ARCH_SPECIFIC__, CUDA 12 by the FEAT macros.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ / 100 == 12) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ / 100 == 12)
    const uint16_t z = 0;
    #pragma unroll
    for (int a = 0; a < 4; ++a)
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
            : "+f"(acc[a][j][0]), "+f"(acc[a][j][1]), "+f"(acc[a][j][2]), "+f"(acc[a][j][3])
            : "r"(af[a][0]), "r"(af[a][1]), "r"(af[a][2]), "r"(af[a][3]), "r"(bf[j][0]), "r"(bf[j][1]),
              "r"(sfa[a]), "h"(z), "h"(z), "r"(sfb[j]), "h"(z), "h"(z));
#else
    __trap();
#endif
  }
}
#pragma unroll
for (int a = 0; a < 4; ++a)
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = wm * 64 + a * 16 + h * 8 + (lane >> 2);
    if (row < mv) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const long col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
        *reinterpret_cast<__nv_bfloat162*>(y + (m0 + row) * N + col) =
            __floats2bfloat162_rn(acc[a][j][2 * h], acc[a][j][2 * h + 1]);
      }
    }
  }
"""
_SHARED = 2 * (128 * 128 * 2 + 256 * 4)

_kernels: dict[str, Any] = {}


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str], **extra):
    if name not in _kernels:
        _kernels[name] = mx.fast.cuda_kernel(
            name="mx_linear_" + name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_HEADER,
            **extra,
        )
    return _kernels[name]


def available() -> bool:
    """Whether this GPU has the block-scaled FP8 mma (sm_120, sm_121)."""
    if not mx.cuda.is_available():
        return False
    return mx.device_info().get("compute_capability_major") == 12


def fits(spec: dict | None, out_dim: int, in_dim: int) -> bool:
    """Whether a weight of this quantization and shape runs on the kernel."""
    return (
        bool(spec)
        and spec.get("mode") == "mxfp8"
        and spec.get("group_size", 32) == 32
        and not out_dim % 128
        and not in_dim % 128
    )


def codes(weight: mx.array) -> mx.array:
    """The E4M3 codes [N, K] of an MXFP8 weight [N, K / 4] (uint32)."""
    return weight.view(mx.uint8)


def matmul(xa: mx.array, xs: mx.array, wb: mx.array, ws: mx.array) -> mx.array:
    """bf16 [M, N] of activation codes xa [M, K] and scales xs [M, K / 32]
    with weight codes wb [N, K] and scales ws [N, K / 32]."""
    rows, depth = xa.shape
    out = wb.shape[0]
    return _kernel(
        "matmul", _MATMUL, ["xa", "xs", "wb", "ws"], ["y"], shared_memory=_SHARED
    )(
        inputs=[xa, xs, wb, ws],
        template=[("K", depth), ("N", out)],
        grid=(256 * (out // 128), -(-rows // 128), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, out)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]
