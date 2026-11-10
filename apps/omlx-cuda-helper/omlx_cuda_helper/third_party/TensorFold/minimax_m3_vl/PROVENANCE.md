# Provenance: TensorFold kernels for MiniMax M3

Source: https://github.com/ashhart/TensorFold, commit
33ada77c3122fca844a6825316ce8bbabaca05be, Apache-2.0 (`../LICENSE`,
`../NOTICE`), Copyright 2026 TensorFold contributors.

| File here | Source file | Changes |
|-----------|-------------|---------|
| `msa_header.cuh` | `zig/kernels/cuda/minimax_m3.cuh` | Only `mma16816` and `pack_bf16` are kept, renamed with an `mm3_` prefix. |
| `index_scores.cu` | `zig/kernels/cuda/minimax_m3_attn.cu`, `tf_mm3_index_scores_tc` | Rewritten as an `mx.fast.cuda_kernel` body. Index queries are read as [heads, rows, 128]. Sizes come from the input shapes. A key block that no row of the tile can select is skipped. |
| `index_select.cu` | `zig/kernels/cuda/minimax_m3_attn.cu`, `tf_mm3_index_select` | Rewritten as an `mx.fast.cuda_kernel` body. Sizes come from the input shapes and `TOPK` is a template value. The bitmap is always written. |
| `attention.cu` | `zig/kernels/cuda/minimax_m3_attn.cu`, `tf_mm3_attn_prompt_tc` | Rewritten as an `mx.fast.cuda_kernel` body. Queries and outputs are [heads, rows, 128], caches [kv heads, keys, 128]. The fixed draft window is removed. The key and value tiles live in dynamic shared memory. A block of 128 keys is scored whole before its softmax weights are formed. Each softmax weight enters the value product as two bfloat16 parts (its rounding and the remainder), so the product keeps float32 precision. With the `PARTIAL` template value each block's normalized output rounds to bfloat16 and the blocks combine by their log-sum-exp, as oMLX's MSA prefill kernels do on Metal; without it the softmax runs online across the kept blocks. |

Loaded by `omlx_cuda_helper/families/minimax_m3_vl/kernels.py`.
