// From mlx-serve src/kernels/moe/moe_prefill_cuda.cu (MIT, Copyright (c) 2026
// David Dalcu), unchanged below this comment.
//
// Inputs: x [R, K] bf16 rows sorted by expert, w [E, N, K * bits / 32] uint32
// and s [E, N, K / group] uint8 (one expert bank), inds [R] uint32 sorted.
// Output: y [R, N] bf16 = each row times its expert's weights.
// Grid: one 256-thread block per (128-column tile of N, expert, split).
constexpr int BITS = FMT == 1 ? 8 : 4;
constexpr int GROUP = FMT == 3 ? 16 : 32;
extern __shared__ __align__(128) uint8_t smem[];
const uint32_t* lut = msv_lut<FMT, STG>(smem);
__shared__ int range[2];
const uint32_t e = blockIdx.y;
msv_expert_range(inds, inds_shape[0], e, range);
const int begin = range[0], end = range[1];
const uint8_t* wb = reinterpret_cast<const uint8_t*>(w) + ((long)e * N + blockIdx.x * 128) * (K * BITS / 8);
const uint8_t* sb = s + ((long)e * N + blockIdx.x * 128) * (K / GROUP);
const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp >> 2, wn = warp & 3;
for (int r0 = begin + (int)blockIdx.z * 64; r0 < end; r0 += (int)gridDim.z * 64) {
  const int mv = end - r0 < 64 ? end - r0 : 64;
  float acc[2][4][4];
  msv_expert_tile<FMT, K, BDEP, STG, T>(smem, lut, mv,
      [&](int row) { return x + (long)(r0 + row) * K; },
      [&](int r) { return wb + (long)r * (K * BITS / 8); },
      [&](int r) { return sb + (long)r * (K / GROUP); }, acc);
  #pragma unroll
  for (int a = 0; a < 2; ++a)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        const int row = wm * 32 + a * 16 + hlf * 8 + (lane >> 2);
        if (row < mv) {
          const int col = blockIdx.x * 128 + wn * 32 + j * 8 + (lane & 3) * 2;
          *reinterpret_cast<__nv_bfloat162*>(y + (long)(r0 + row) * N + col) = __floats2bfloat162_rn(acc[a][j][2 * hlf], acc[a][j][2 * hlf + 1]);
        }
      }
}
