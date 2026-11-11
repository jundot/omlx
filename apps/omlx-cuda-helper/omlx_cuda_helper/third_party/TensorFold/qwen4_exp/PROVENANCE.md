# Provenance: TensorFold kernels for Qwen3.8 Flash-Next

Source: https://github.com/ashhart/TensorFold, commit
33ada77c3122fca844a6825316ce8bbabaca05be, Apache-2.0 (`../LICENSE`,
`../NOTICE`), Copyright 2026 TensorFold contributors.

| File here | Source file | Changes |
|-----------|-------------|---------|
| `qsa_header.cuh` | `zig/kernels/cuda/common/qmm_frag.cuh`, `common/cp_async.cuh` | Only cp.async with zero fill, ldmatrix x4 (plain and transposed) and the bf16 m16n8k16 mma are kept, renamed with a `qsa_` prefix. `qsa_mma_rn` (each 16-product step from zero, added with a correctly rounded float add) and `qsa_split` (a float32 pair as two bf16 parts) are added. |
| `qsa_attention.cu` | `zig/kernels/cuda/flashnext_attn.cu`, `attn_chunks` as `fn_attn_fused` runs it | Rewritten as an `mx.fast.cuda_kernel` body. Queries are [heads, rows, D], caches [kv heads, keys, D], the output [rows, heads, D]. A row's keys come from its QSA block picks and causal tail, not a gathered id list. Keys and values have their own tiles, so the values load while the scores are taken. Score steps use `qsa_mma_rn`. Each softmax weight enters the value product as two bf16 parts (its rounding and the remainder), so that product keeps float32 precision. The output is o / l in float32, rounded once. The 32-key tiles, the 16-row query tile, the warp split (8 keys each for the scores, a quarter of the dims each for the values) and the online softmax are TensorFold's. |
| `gdn_header.cuh` | `zig/kernels/cuda/gdn_prefill.cu`, `Prefetch` | bf16 queries, keys and update rates only; renamed with a `gdn_` prefix. |
| `gdn_chain.cu` | `zig/kernels/cuda/gdn_prefill.cu`, `chain_kernel` | Rewritten as an `mx.fast.cuda_kernel` body with the sizes from the input shapes; bf16 queries, keys and update rates only. The recurrence, the two threads a value row and the staged steps are TensorFold's. |

Loaded by `omlx_cuda_helper/families/qwen4_exp/kernels.py`.
