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
