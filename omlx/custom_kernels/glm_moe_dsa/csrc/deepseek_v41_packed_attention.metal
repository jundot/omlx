#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
#include "kernels/steel_deepseek_v41_packed_attention.h"

#define instantiate_deepseek_v41_packed_attention(tname, dtype, bk, dc, h, d, wm) \
  instantiate_kernel(                                                            \
      "deepseek_v41_packed_attention_" #tname "_bk" #bk "_dc" #dc "_h" #h       \
      "_d" #d "_wm" #wm,                                                        \
      deepseek_v41_packed_attention,                                              \
      dtype,                                                                     \
      bk,                                                                        \
      dc,                                                                        \
      h,                                                                         \
      d,                                                                         \
      wm,                                                                        \
      uint,                                                                      \
      float)

instantiate_deepseek_v41_packed_attention(bfloat16, bfloat16_t, 64, 32, 64, 512, 8);

// Prefill chunks; short blocks keep the staged kernel above.
instantiate_kernel(
    "deepseek_v41_packed_attention_bfloat16_bk64_dc32_h64_d512_wm8_directq",
    deepseek_v41_packed_attention,
    bfloat16_t, 64, 32, 64, 512, 8, uint, float, true);

// Same arithmetic, reading cache rows already converted to BF16.
instantiate_kernel(
    "deepseek_v41_packed_attention_bfloat16_bk64_dc32_h64_d512_wm8_directq_unpacked",
    deepseek_v41_packed_attention,
    bfloat16_t, 64, 32, 64, 512, 8, uint, float, true, true);

// Convert each packed cache row to BF16 once, with the per-value conversion
// the attention kernel uses.
kernel void deepseek_v41_prefill_unpack_kv(
    const device uchar* local [[buffer(0)]],
    const device uchar* pooled [[buffer(1)]],
    device bfloat16_t* local_dense [[buffer(2)]],
    device bfloat16_t* pooled_dense [[buffer(3)]],
    constant DeepseekV4SparseAttentionParams* params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint rows = params->localL + params->pooledL;
  if (index >= uint(params->B) * rows * 512) return;
  const uint d = index % 512, row = (index / 512) % rows, b = index / (rows * 512);
  if (row < uint(params->localL)) {
    const device uchar* src = local + size_t(b) * params->Local_strides[0] +
        size_t(row) * params->Local_strides[2];
    local_dense[(size_t(b) * params->localL + row) * 512 + d] =
        bfloat16_t(deepseek_v41_packed_value(src, d, false));
  } else {
    const uint r = row - params->localL;
    const device uchar* src = pooled + size_t(b) * params->Pooled_strides[0] +
        size_t(r) * params->Pooled_strides[1];
    pooled_dense[(size_t(b) * params->pooledL + r) * 512 + d] =
        bfloat16_t(deepseek_v41_packed_value(src, d, true));
  }
}
