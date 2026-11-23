# Provenance: mlx-serve kernels for MiniMax M3

Source: mlx-serve, commit 9191fbdf697df333cac19ec7a93eaa1098ce3ad5, MIT
(`../LICENSE`), Copyright (c) 2026 David Dalcu.

| File here | Source file | Changes |
|-----------|-------------|---------|
| `moe_prefill_header.cu` | `src/kernels/moe/moe_prefill_cuda_header.cu` | None. |
| `moe_prefill_gather.cu` | `src/kernels/moe/moe_prefill_cuda.cu` | None. |
| `moe_prefill_gate_up.cu` | `src/kernels/moe/moe_prefill_cuda_gu.cu` | The epilogue is MiniMax's clipped SwiGLU in float32 with a float32 scale per expert for gate and up, in place of `silu(gate) * up`; the sigmoid is MLX's formula. |
| `moe_prefill_sum.cu` | `src/kernels/moe/moe_prefill_cuda_reduce.cu` | The routing weights are float32; each weighted row rounds in float32 and the rows add in expert order, rounded once to bfloat16. |

The launch geometry in `omlx_cuda_helper/families/minimax_m3_vl/kernels.py`
(tile sizes, shared memory, the split of a popular expert's rows) follows
`src/sdk/blocks/moe/moe_prefill_cuda.zig` of the same commit.
