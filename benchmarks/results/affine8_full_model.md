# Affine8 full-model validation

Apple M5 Pro, 48 GiB; measured 2026-09-11 with MLX 0.32.2, mlx-lm 0.31.3, and installed mlx-vlm 0.6.3. Both models use local oQ4e weights. Native attention KV is BF16; model-weight quantization is unchanged. Server result records do not embed a source commit, so these are point-in-time operational measurements rather than commit-pinned comparisons.

Each row is one unpaired server run. Prompts use the deterministic `code_python` profile and contain N+1 input tokens for N prefill rows. MTP, VLM MTP, DFlash, SpecPrefill, ANE prefill, and Qwen INT8-activation prefill are disabled. Affine8 retains the final full-attention layer in native precision. Every completed request reports zero reused prefix tokens and finishes by the fixed generation length.

## Qwen3.8-27B capacity and endurance

| Prefill rows | Generated tokens | Prefill tok/s | Decode tok/s | Peak active MLX GiB | Sampled physical peak GiB | Result |
|---:|---:|---:|---:|---:|---:|---|
| 50,000 | 128 | 410.3 | 14.3 | 18.98 | 21.71 | Completed |
| 150,000 | 128 | 295.3 | 10.7 | 22.66 | 28.37 | Completed |
| 200,000 | 128 | 259.5 | 9.3 | 24.46 | 31.98 | Completed |
| 200,000 | 4,096 | 255.3 | 9.4 | 24.46 | 36.13 | Completed |
| 250,000 | 128 | — | — | — | — | Rejected by memory guard |

The 250K request was rejected before generation because estimated peak memory was 35.33 GB while the dynamic ceiling was 34.79 GB. This is graceful admission behavior under memory conditions recorded for that run, not a fixed Affine8 context limit.

## Qwen3.6-35B-A3B sustained decode

| Prefill rows | Generated tokens | Prefill tok/s | Decode tok/s | Peak active MLX GiB | Sampled physical peak GiB |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 4,096 | 2,433.2 | 82.8 | 20.75 | 22.09 |
| 200,000 | 4,096 | 885.4 | 37.7 | 22.94 | 26.67 |

Decode throughput retained 45.5% of the 8K result at 200K in these two Affine8 runs. This within-format observation is not a repeated estimate.

For historical context, the earlier repeated Qwen3.6 200K experiment measured Affine4 at 44.8–49.0 tok/s, TurboQuant4 at 21.5–23.5 tok/s, and native BF16 at 26.3–32.5 tok/s. Those results came from a different experiment and are not paired controls for the Affine8 runs above.

## Interpretation

These single-run measurements validate Affine8 prefill, memory admission, and sustained generation on two full models. They do not establish a speedup over another cache format, confidence intervals, semantic quality, or long-context retrieval accuracy. Affine8 compression is lossy, and incremental prefill can change generated text.

`peak_memory_bytes` in the raw records measures active MLX allocations. Sampled physical footprint is a separate process-memory metric; the two values must not be added.

Raw results:

- [Qwen3.8 50K](affine8_qwen38_capacity_50k.json)
- [Qwen3.8 150K](affine8_qwen38_capacity_150k.json)
- [Qwen3.8 200K](affine8_qwen38_capacity_200k.json)
- [Qwen3.8 250K guard rejection](affine8_qwen38_capacity_250k.json)
- [Qwen3.8 sustained 200K](affine8_qwen38_sustained_200k.json)
- [Qwen3.6 sustained 8K](affine8_qwen36_sustained_8k.json)
- [Qwen3.6 sustained 200K](affine8_qwen36_sustained_200k.json)
