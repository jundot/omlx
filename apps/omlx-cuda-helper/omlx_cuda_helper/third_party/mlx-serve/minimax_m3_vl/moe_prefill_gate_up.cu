// Adapted from mlx-serve src/kernels/moe/moe_prefill_cuda_gu.cu (MIT,
// Copyright (c) 2026 David Dalcu). Changes: the epilogue is MiniMax's clipped
// SwiGLU in float32 with a float32 scale per expert for gate and up, in place
// of silu(gate) * up.
//
// Inputs: x [T, K] bf16 token rows, rows [R] uint32 (the token of each routed
// row), inds [R] uint32 sorted experts, gw/uw [E, I, K * bits / 32] uint32 and
// gs/us [E, I, K / group] uint8 (the gate and up banks), gscale/uscale [E]
// float32, act_params float32 [alpha, limit, beta].
// Output: act [R, I] bf16.
// Grid: one 256-thread block per (64-column tile of I, expert, split).
constexpr int BITS = FMT == 1 ? 8 : 4;
constexpr int GROUP = FMT == 3 ? 16 : 32;
extern __shared__ __align__(128) uint8_t smem[];
const uint32_t* lut = msv_lut<FMT, STG>(smem);
__shared__ int range[2];
const uint32_t e = blockIdx.y;
msv_expert_range(inds, inds_shape[0], e, range);
const int begin = range[0], end = range[1];
const long n0 = (long)e * I + blockIdx.x * 64;
const uint8_t* gb = reinterpret_cast<const uint8_t*>(gw) + n0 * (K * BITS / 8);
const uint8_t* ub = reinterpret_cast<const uint8_t*>(uw) + n0 * (K * BITS / 8);
const uint8_t* gsb = gs + n0 * (K / GROUP);
const uint8_t* usb = us + n0 * (K / GROUP);
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
T* tile = reinterpret_cast<T*>(smem);
for (int r0 = begin + (int)blockIdx.z * 64; r0 < end; r0 += (int)gridDim.z * 64) {
  const int mv = end - r0 < 64 ? end - r0 : 64;
  float acc[2][4][4];
  msv_expert_tile<FMT, K, BDEP, STG, T>(smem, lut, mv,
      [&](int row) { return x + (long)rows[r0 + row] * K; },
      [&](int r) { return (r < 64 ? gb : ub) + (long)(r & 63) * (K * BITS / 8); },
      [&](int r) { return (r < 64 ? gsb : usb) + (long)(r & 63) * (K / GROUP); }, acc);
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
  const int row = tid >> 2, c0 = (tid & 3) * 16;
  if (row < mv) {
    T* dst = act + (long)(r0 + row) * I + blockIdx.x * 64 + c0;
    // MiniMax's clipped SwiGLU in float32 on the rounded projections, each
    // restored by its expert's global scale, rounded once.
    const float gate_s = gscale[e], up_s = uscale[e];
    const float alpha = act_params[0], limit = act_params[1], beta = act_params[2];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float glu = fminf(__bfloat162float(tile[row * 128 + c0 + i]) * gate_s, limit);
      const float lin = fminf(fmaxf(__bfloat162float(tile[row * 128 + 64 + c0 + i]) * up_s, -limit), limit);
      // MLX's sigmoid: 1 / (1 + exp(|z|)), mirrored for z >= 0.
      const float z = alpha * glu;
      const float low = 1.f / (1.f + expf(fabsf(z)));
      dst[i] = __float2bfloat16_rn(glu * (z < 0.f ? low : 1.f - low) * (lin + beta));
    }
  }
  __syncthreads();
}
