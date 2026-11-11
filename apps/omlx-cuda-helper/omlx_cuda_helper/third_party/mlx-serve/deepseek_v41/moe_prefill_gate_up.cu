// Adapted from mlx-serve src/kernels/moe/moe_prefill_cuda_gu.cu (MIT,
// Copyright (c) 2026 David Dalcu). Changed: affine banks with biases, and the
// epilogue is DeepSeek V4.1's SwiGLU: gate and up as bf16, clamped by LIMIT,
// silu(gate) * up times the route weight in float, rounded to bf16, then the
// FP8 round trip of each group of 32 values when QBITS is 8.
extern __shared__ __align__(128) uint8_t smem[];
__shared__ int range[2];
const uint32_t e = blockIdx.y;
msv_expert_range(inds, inds_shape[0], e, range);
const int begin = range[0], end = range[1];
const long n0 = (long)e * I + blockIdx.x * 64;
const uint8_t* gcodes = reinterpret_cast<const uint8_t*>(gw) + n0 * (K * BITS / 8);
const uint8_t* ucodes = reinterpret_cast<const uint8_t*>(uw) + n0 * (K * BITS / 8);
const T* gscales = gs + n0 * (K / 64);
const T* gbiases = gb + n0 * (K / 64);
const T* uscales = us + n0 * (K / 64);
const T* ubiases = ub + n0 * (K / 64);
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
const float limit = __uint_as_float(LIMIT);
T* tile = reinterpret_cast<T*>(smem);
for (int r0 = begin + (int)blockIdx.z * 64; r0 < end; r0 += (int)gridDim.z * 64) {
  const int mv = end - r0 < 64 ? end - r0 : 64;
  float acc[2][4][4];
  msv_expert_tile<BITS, K, BDEP, STG>(smem, mv,
      [&](int row) { return x + (long)rows[r0 + row] * K; },
      [&](int r) { return (r < 64 ? gcodes : ucodes) + (long)(r & 63) * (K * BITS / 8); },
      [&](int r) { return (r < 64 ? gscales : uscales) + (long)(r & 63) * (K / 64); },
      [&](int r) { return (r < 64 ? gbiases : ubiases) + (long)(r & 63) * (K / 64); }, acc);
  // Stage both projections as T: columns 0-63 gate, 64-127 up.
  #pragma unroll
  for (int a = 0; a < 2; ++a)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        const int row = wm * 32 + a * 16 + hlf * 8 + (lane >> 2);
        const int col = wn * 32 + j * 8 + (lane & 3) * 2;
        *reinterpret_cast<__nv_bfloat162*>(tile + row * 128 + col) = __floats2bfloat162_rn(acc[a][j][2 * hlf], acc[a][j][2 * hlf + 1]);
      }
  __syncthreads();
  // One thread per row and group of 32 output values.
  const int row = tid >> 1, c0 = (tid & 1) * 32;
  if (tid < 128 && row < mv) {
    const float weight = rw[r0 + row];
    float v[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float g = v41_float(tile[row * 128 + c0 + i]);
      float u = v41_float(tile[row * 128 + 64 + c0 + i]);
      if (LIMIT) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
      }
      const float ex = 1.0f / (1.0f + expf(fabsf(g)));
      const float value = __fmul_rn(__fmul_rn(g, g < 0.0f ? ex : 1.0f - ex), u);
      v[i] = v41_round(__fmul_rn(value, weight), T());
    }
    if (QBITS) v41_quantize<8, FLOOR, INV_LIMIT>(v);
    T* dst = act + (long)(r0 + row) * I + blockIdx.x * 64 + c0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) v41_store(v[i], dst + i);
  }
  __syncthreads();
}
