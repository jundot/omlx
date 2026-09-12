**Affine4 performance investigation — Apple M5 Pro, 48 GiB, 2026-09-07**

The current branch has measurable attention regressions. Some of the apparent end-to-end gap also comes from different benchmark settings. The native M5 route is active for both Qwen models.

The implementation under test is the uncommitted `codex/generic-affine4-kv` branch on upstream `aa8db734`. Its `omlx/affine4.py` SHA-256 is `a93c081aa17f5962a8255d57637d75f6b2fb650584faa402e70e909f11b6bf81`. The original oMLX implementation is `1ee0d24a`; the VLM PR checkout is `10f9030d`. Product code was unchanged during this investigation. Faster alternatives below are isolated diagnostic variants.

**What changed between implementations**

| Property | Original oMLX patch | VLM PR | Current oMLX branch |
|---|---|---|---|
| Native implementation | Compiled C++/Metal, Qwen3.8 geometry | Generic JIT Metal MPP | Generic JIT Metal MPP |
| Per-vector scale | FP16 | FP16 | FP32 |
| Native attention value accumulation | FP16 | FP16 | FP32, with range normalization |
| Native masks/layouts | Narrow Qwen integration | Causal/left padding; contiguous kernel inputs | Boolean masks and strided inputs also supported |
| Long-query quantized-cache prefill | Existing TQ fallback; historical server test disabled prefix cache | Dequantize, cast to query dtype, MLX SDPA | Rotated FP32 SDPA, explicitly evaluated query tiles |
| Cold server prefill | Full precision, convert before decode | Benchmark quantized during chunked prefill | Full precision, convert before decode |
| Configuration | Environment toggle/private TQ convention | Explicit cache format | Explicit format, dashboard, persistence and validation |

FP32 scales add two bytes per vector: 132 rather than 130 bytes for a packed D=256 vector, about 1.5% extra. This is much smaller than the measured execution-time difference.

The old oMLX server table used MTP, `world ` prompts, no prefix cache, and up to 128 generated tokens. Its Affine4 results were 29.83 tok/s at 8,052 prompt tokens and 25.34 tok/s at 32,052. The recent oMLX matrix disabled MTP, used code prompts and fixed 1,024-token decoding, and measured 16.8 and 15.5 tok/s at roughly 8K and 32K. Those are different workloads. The VLM PR's 128-token direct-generation benchmark did disable MTP, but it used a different backend and random-token prompts.

**Native attention: controlled comparison**

All implementations consumed the same BF16 random input tensors in one process. Each case had ten warmups and 41 randomly interleaved timing rounds. These are median milliseconds per attention call, including query/output rotation. They are not model throughput.

| Geometry | Context | Query rows | Original oMLX | VLM PR | Current | Current, FP16 accumulator probe |
|---|---:|---:|---:|---:|---:|---:|
| 24q / 4kv / D256 | 32,768 | 1 | 0.554 | 0.482 | 0.692 | 0.518 |
| 24q / 4kv / D256 | 200,000 | 1 | 1.865 | 1.827 | 3.206 | 1.823 |
| 24q / 4kv / D256 | 200,000 | 4 | 6.016 | 5.527 | 7.454 | 6.301 |
| 16q / 2kv / D256 | 32,768 | 1 | — | 0.370 | 0.484 | 0.400 |
| 16q / 2kv / D256 | 200,000 | 1 | — | 0.996 | 1.636 | 1.080 |
| 16q / 2kv / D256 | 200,000 | 4 | — | 4.719 | 5.749 | 5.486 |

Changing only the AV accumulator type recovers most of the single-token regression while retaining FP32 scales and the current generic implementation. Multi-row attention retains a gap; the remaining normalization, mask and stride handling costs have not been individually isolated.

A blanket FP16 revert would be incorrect. A supported synthetic case with 64 KV heads, D=32 and 32,768 tokens assigns 8,192 tokens to each partition. Constant -8 value codes produce an unnormalized partial sum of -65,536, outside FP16's finite range. Both the actual VLM PR and the FP16-accumulator probe produced non-finite outputs; current FP32 produced the exact expected output. A fast path therefore needs a conservative partition bound and an FP32 fallback.

**Quantized-cache prefill: controlled comparison**

Same BF16 inputs, causal attention, 32,768 KV tokens, three warmups and eleven interleaved rounds. The BF16 probe keeps the current codec and fused unpacking, computes SDPA in rotated coordinates, and rotates the result back. It dequantizes each layer's KV for SDPA.

| Geometry | Query rows | Current FP32 | VLM PR | Current codec + BF16 fused SDPA |
|---|---:|---:|---:|---:|
| 24q / 4kv / D256 | 256 | 75.00 ms | 34.46 ms | 17.24 ms |
| 24q / 4kv / D256 | 2,048 | 564.02 ms | 110.19 ms | 101.84 ms |
| 16q / 2kv / D256 | 256 | 32.59 ms | 19.60 ms | 11.11 ms |
| 16q / 2kv / D256 | 2,048 | 249.43 ms | 69.83 ms | 66.06 ms |

For dense Qwen geometry, the bounded BF16 probe is 5.5 times faster than current attention at 2,048 query rows and slightly faster than the VLM port. The separately measured default-dispatch BF16 probe had output cosine 0.99998–0.99999 against current FP32. This is a numerical smoke test, not a model-quality evaluation.

Cold oMLX prefill does not use this quantized-cache fallback: conversion happens afterward. The recent warm-prefix server test did exercise it, and Qwen3.8's 2K extension took 13.09 s with current Affine4, 10.92 s with TQ4 and 6.19 s with native KV.

**Large-context server checks**

Same installed oQ4e model weights, Affine4, last full-attention layer unquantized, MTP/ANE/DFlash/SpecPrefill disabled, greedy decoding with termination suppression, 128 generated tokens, and zero prefix-cache hits. Prompt sizes are N+1 tokens for N requested prefill rows. The native full-precision cache dtype is BF16.

| Model | Requested context | Prefill tok/s | Decode tok/s | MLX process peak | Outcome |
|---|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B | 100,000 | 1,120.2 | 43.2 | 22.61 GiB | 128 tokens, length finish |
| Qwen3.6-35B-A3B | 150,000 | 882.3 | 37.6 | 24.77 GiB | 128 tokens, length finish |
| Qwen3.6-35B-A3B | 200,000 | 731.1 | 32.6 | 24.70 GiB | 128 tokens, length finish |
| Qwen3.8-27B | 100,000 | 280.5 | 10.4 | 23.74 GiB | 128 tokens, length finish |
| Qwen3.8-27B | 150,000 | — | — | — | Memory guard rejected fixed-chunk and adaptive attempts |
| Qwen3.8-27B | 200,000 | — | — | — | Memory guard rejected adaptive attempts, including no-prefix-cache control |

The admin benchmark normally pins a 2,048-token minimum chunk. Qwen3.8's fixed-chunk 150K attempt reached about 117K processed tokens before rejection; the guard projected 34.83 GB against its 33.70 GB safety cap. Adaptive retries retained the guard and Apple's 37.44 GB Metal cap, and also failed. Disabling prefix caching did not make 200K fit. Resume counters can restart in the traces; the final error's counter is not a standalone maximum-context measurement.

The 200K cache-enabled request switched to adaptive policy after 59,392 processed tokens, before pressure required shrinking. That switch is recorded separately. The adaptive 150K retry and no-prefix-cache 200K control used separate result files. Failed attempts have no valid throughput rows and are not counted as successful context tests.

The guard also forced memory-bounded full-precision prefill around 72K in the Qwen3.6 run. This policy differs from the old direct VLM benchmark. Historical VLM Affine4 throughput at 100K/150K/200K was 60.71/54.35/49.53 tok/s, but those absolute numbers do not isolate the cache implementation.

Qwen3.8's full-precision KV storage is 64 KiB/token, about 12.2 GiB at 200K, before model weights, recurrent state, allocator retention and prefill temporaries. An older memory note doubled that KV estimate. Affine4's final compressed size does not remove the current server's cold-prefill peak.

**Shared-model control**

The harness used one 200,000-token native-BF16-prefilled Qwen3.6 state, identical rotation seeds and a fixed native-greedy continuation for every cache. All cases used a synchronous direct forward loop, with eight untimed warmup tokens followed by restoration of the original cache before timing 128 steps. Prefill and conversion are outside these timings. This loop is intended for paired comparison; its absolute throughput differs from the historical asynchronous `generate_step` benchmark.

| Cache | Decode tok/s | Next-token agreement with native |
|---|---:|---:|
| Native BF16 | 24.99 | 128/128 |
| TQ4 | 18.52 | 126/128 |
| VLM PR Affine4 | 37.37 | 128/128 |
| Current Affine4 | 29.90 | 127/128 |
| Current with FP16 accumulator probe | 36.00 | 127/128 |

Current Affine4 is 20.0% slower than the VLM PR in this controlled test. The accumulator-only probe improves current throughput by 20.4%. Current and VLM were each repeated in reverse order: current 29.900/29.910 tok/s and VLM 37.250/37.491 tok/s. Both repetitions retained their respective next-token agreement counts. This confirms a cache implementation regression independent of MTP, server scheduling and prompt differences. The agreement counts are a small teacher-forced check, not a retrieval or model-quality benchmark.

The raw result is `affine4_same_model_200000.json`. The separate 512-token harness check replayed the native continuation exactly (128/128).

**Implementation direction**

Restore FP16 AV accumulation where a conservative per-partition range bound permits it, retaining FP32 accumulation elsewhere. Keep FP32 cache scales and the current serialized format. Use optimized BF16 SDPA for eligible prefill inputs with a bounded fused route and preserve the generic fallback for other cases. These changes can recover performance without discarding the broader format, mask and layout support.

Supporting the dense model at 150K–200K on this server also needs cold-prefill memory work. The VLM port's incremental cache compression is a relevant design difference, but transferring it into oMLX requires checking hybrid recurrent state, batching and prefix restoration. The kernel probes alone do not establish that integration's correctness.

Raw evidence is in the adjacent `affine4_history_attention.json`, `affine4_history_prefill.json`, `affine4_accumulator_range.json`, `affine4_large_context_server.json`, `affine4_large_context_adaptive_switch.json`, `affine4_large_context_adaptive_retry.json`, and `affine4_large_context_no_prefix_cache.json` files. No public benchmark upload, commit, push or PR was performed.

`affine4_regression_reproduction.zip` contains the scripts, exact tested Affine4 source, and relevant raw logs. The scripts retain this machine's local checkout/model paths; adjust those paths before replaying elsewhere.
