// Adapted from TensorFold zig/kernels/cuda/minimax_m3.cuh (Apache-2.0,
// Copyright 2026 TensorFold contributors). Changes: only the two helpers the
// attention kernels use are kept, renamed with an mm3_ prefix.
#include <cuda_bf16.h>
#include <math_constants.h>

typedef __nv_bfloat16 mm3_bf16;

__device__ __forceinline__ void mm3_mma16816(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t mm3_pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
