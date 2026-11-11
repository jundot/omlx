// Adapted from mlx-serve src/kernels/moe/moe_prefill_cuda.cu (MIT, Copyright
// (c) 2026 David Dalcu). Changed: an affine bank with biases, or an mxfp4 bank.
extern __shared__ __align__(128) uint8_t smem[];
__shared__ int range[2];
const uint32_t e = blockIdx.y;
msv_expert_range(inds, inds_shape[0], e, range);
const int begin = range[0], end = range[1];
const long n0 = (long)e * N + blockIdx.x * 128;
const uint8_t* codes = reinterpret_cast<const uint8_t*>(w) + n0 * (K * BITS / 8);
// Scales (and biases) are bf16 for affine codes, uint8 E8M0 with MX.
const auto* scales = s + n0 * (K / GROUP);
const auto* biases = b + n0 * (K / GROUP);
const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp >> 2, wn = warp & 3;
for (int r0 = begin + (int)blockIdx.z * 64; r0 < end; r0 += (int)gridDim.z * 64) {
  const int mv = end - r0 < 64 ? end - r0 : 64;
  float acc[2][4][4];
  msv_expert_tile<BITS, K, BDEP, STG, MX>(smem, mv,
      [&](int row) { return x + (long)(r0 + row) * K; },
      [&](int r) { return codes + (long)r * (K * BITS / 8); },
      [&](int r) { return scales + (long)r * (K / GROUP); },
      [&](int r) { return biases + (long)r * (K / GROUP); }, acc);
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
