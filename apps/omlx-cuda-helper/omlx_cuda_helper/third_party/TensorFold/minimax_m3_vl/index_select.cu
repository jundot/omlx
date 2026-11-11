// Adapted from TensorFold zig/kernels/cuda/minimax_m3_attn.cu,
// tf_mm3_index_select (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: rewritten as an mx.fast.cuda_kernel body; the sizes come from the
// input shapes and TOPK is a template value.
//
// Inputs: scores [rows, heads, blocks] float32, pos int32 [1].
// Outputs: sel [rows, heads, TOPK] int32, a (row, head)'s kept blocks in
// ascending order (-1 past the last): every visible block up to TOPK, else
// its own block and the best others (the lower block on a tie); bits
// [rows, heads, words] uint32, the same blocks as a bitmap.
// Grid: one 32-thread block per (head, row).
const int rows = scores_shape[0], heads = scores_shape[1], blocks = scores_shape[2];
const int words = (blocks + 31) / 32;
const int row = blockIdx.y, h = blockIdx.x;
const int lane = threadIdx.x;
if (row >= rows) return;
const int nb = (pos[0] + row) / 128 + 1;
const float* s = scores + ((size_t)row * heads + h) * blocks;
int* out = sel + ((size_t)row * heads + h) * TOPK;
int list[32];
int count = 0;
if (nb <= TOPK) {
  for (int b = 0; b < nb; ++b) list[count++] = b;
} else {
  list[count++] = nb - 1;
  float last_best = CUDART_INF_F;
  int last_id = nb - 1;
  for (int k = 1; k < TOPK; ++k) {
    float best = -CUDART_INF_F;
    int bid = 0x7fffffff;
    for (int b = lane; b < nb - 1; b += 32) {
      const float v = s[b];
      const bool after = v < last_best || (v == last_best && b > last_id);
      if (after && (v > best || (v == best && b < bid))) {
        best = v;
        bid = b;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bid, off);
      if (ob > best || (ob == best && oi < bid)) {
        best = ob;
        bid = oi;
      }
    }
    if (bid == 0x7fffffff || best == -CUDART_INF_F) break;
    list[count++] = bid;
    last_best = best;
    last_id = bid;
  }
}
for (int i = 1; i < count; ++i)
  for (int m = i; m > 0 && list[m] < list[m - 1]; --m) {
    const int tmp = list[m];
    list[m] = list[m - 1];
    list[m - 1] = tmp;
  }
if (lane == 0)
  for (int i = 0; i < TOPK; ++i) out[i] = i < count ? list[i] : -1;
unsigned* row_bits = bits + ((size_t)row * heads + h) * words;
for (int wd = lane; wd < words; wd += 32) {
  unsigned m = 0;
  for (int i = 0; i < count; ++i)
    if (list[i] / 32 == wd) m |= 1u << (list[i] % 32);
  row_bits[wd] = m;
}
