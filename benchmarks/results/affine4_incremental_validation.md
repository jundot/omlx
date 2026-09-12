# Affine4 incremental-cache validation

Apple M5 Pro, 48 GiB, 2026-09-07. Baseline: `423c2d35`; final kernel and integration: `99d314fd` on `codex/generic-affine4-kv`. The baseline was committed before optimization. Earlier implementations are oMLX `1ee0d24a` and mlx-vlm `10f9030d`.

Affine4 now compresses full-attention KV during each layer's prefill update. Qwen3.8 completes 150K and 200K cold prompts with the unchanged memory guard. The shared-model decode deficit against the VLM port falls from 20.0% in the earlier paired control to 3.1%; complete performance parity is not established.

## Changes and validation

| Commit | Completed change |
|---|---|
| `423c2d35` | Generic cache, explicit dashboard format, persistence and baseline evidence |
| `6805ce1f` | FP16 value partials where bounded; FP32 for larger partitions |
| `41262a64` | Direct BF16 unpacking and fused prefill SDPA |
| `cfb485be` | Incremental Affine4 conversion in both scheduler prefill routes |
| `d123d200` | Prefill residency estimates, workspace reservation and absolute resumed context lengths |
| `2f487029` | Actual cache/allocation/pool/physical traces and explicit MLX peak provenance |
| `8617fa3f` | Correct availability reporting for an empty prefix cache |
| `99d314fd` | Mask specialization and measured long-context partition tuning |

The final focused suite passed **805 tests**, with one skipped and seven deselected by the repository's default slow/integration marker filter. It covers Affine4, TurboQuant persistence and batching, scheduler prefill/eviction/resume, memory monitors, MTP chunking, benchmarks and admin i18n. Native reference checks include 150K contexts for both tuned geometries and FP16/BF16 queries. Uniform extreme-code tests cover the FP16-to-FP32 accumulator boundary. Real small Llama, Qwen2 and Qwen3.5 hybrid models validate incremental replay.

FP16 AV partials are limited to 96 tiles (6,144 tokens), bounding exact magnitude at 49,152 before rounding. Query/value range normalization, softmax, reductions and persisted scales remain FP32. This preserves the finite-range protection missing from a blanket FP16 revert. BF16 fused prefill retires each layer's unpacked workspace; explicit masks, sinks and unsupported shapes retain bounded portable attention. Recurrent/sliding caches and the default uncompressed final full-attention layer are preserved.

## Paired performance

The [shared-model result](affine4_final_same_model_200000.json) uses one Qwen3.6 200K native-BF16-prefilled snapshot, fixed teacher-forced native continuation, eight warmup tokens, and 128 timed steps per run. Current/VLM/VLM/current order uses the same model backend, weights, state and rotation seed; conversion and prefill are outside timings. This isolates decode and does not validate incremental-prefill quality.

| Implementation | Mean decode tok/s | Repetitions | Next-token agreement |
|---|---:|---:|---:|
| Final Affine4 | 42.19 | 2 | 127/128 in both |
| VLM PR Affine4 | 43.54 | 2 | 128/128 in both |

The earlier paired control measured 29.90 versus 37.37 tok/s. Absolute VLM throughput also changed between sessions, so the 20.0% to 3.1% relative deficit is the useful comparison; the entire absolute throughput increase cannot be attributed to the patch.

[Accumulator measurements](affine4_decode_optimized.json) at 200K reduced dense one-row attention from 2.897 to 1.784 ms and MoE from 1.610 to 1.061 ms. [Mask specialization](affine4_mask_tuning.json) further reduced these to 1.729/1.044 ms; VLM measured 1.584/0.995 ms in that run. Original oMLX dense one-row attention measured 1.816 ms, and dense four-row attention 5.298 ms versus final 5.407 ms and VLM 4.878 ms. Dense decode therefore retains a measurable kernel gap.

[Partition tuning](affine4_partition_crossover.json) reduced 200K, 16-query/2-KV-head, four-row attention from 5.155 to 4.057 ms, versus VLM 4.711 ms. A 32-query/1-KV-head, one-row control improved from 2.795 to 2.409 ms, versus VLM 2.599 ms. The dispatch is limited to batch one, D256, 32 grouped rows, at most two KV heads and at least 150K tokens. [Threshold checks](affine4_partition_threshold.json) showed the one-KV-head case regressed at 131,072, so shorter contexts keep the previous partition count. Cooperative query staging was measured and rejected because it did not improve timings.

[Fused prefill measurements](affine4_prefill_optimized.json), 32K context and 2,048 query rows, measured 85.75 ms for dense geometry and 55.94 ms for MoE, versus FP32 baseline 486.05/211.40 ms and VLM 95.39/60.18 ms. Attention-output cosine against FP32 was approximately 0.99999. Kernel timings use identical tensors and randomized interleaving; they are not whole-model throughput.

## Real server comparison

MLX 0.32.2, mlx-lm 0.31.3 and installed mlx-vlm 0.6.3; identical local oQ4e weights in every format. Native KV is **BF16**, despite the historical `native16`/FP16 naming. These are cache-format comparisons, not full-precision model-weight comparisons. MTP, VLM MTP, DFlash, SpecPrefill and ANE prefill are disabled. Prefix caching is enabled but cleared before each cold trial. Aligned code prompts contain N+1 tokens for N prefill rows. Benchmarks suppress termination tokens to enforce the requested generation length; ordinary requests retain normal stopping. No public upload occurs.

The [focused 8K comparison](affine4_incremental_server_comparison.json) is a single pass per format, not a full repeated matrix. Each comparison generates 1,024 tokens; each Affine4 endurance run generates 4,096. Incremental quantization can change generated continuations, so these greedy server timings complement the teacher-forced control.

| Model | Cache | Generated tokens | Prefill tok/s | Decode tok/s | Peak active MLX GiB |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | native16 | 1,024 | 2,098.2 | 71.8 | 20.92 |
| Qwen3.6-35B-A3B | tq4 | 1,024 | 2,094.6 | 63.9 | 20.89 |
| Qwen3.6-35B-A3B | affine4 | 1,024 | 2,045.6 | 68.2 | 20.79 |
| Qwen3.6-35B-A3B | affine4 | 4,096 | 2,050.2 | 67.9 | 20.79 |
| Qwen3.8-27B | native16 | 1,024 | 381.6 | 13.4 | 18.11 |
| Qwen3.8-27B | tq4 | 1,024 | 373.1 | 13.0 | 18.01 |
| Qwen3.8-27B | affine4 | 1,024 | 365.0 | 13.5 | 17.67 |
| Qwen3.8-27B | affine4 | 4,096 | 374.4 | 15.6 | 17.67 |

All eight requests completed their full token budgets with zero prefix hits. Peak active MLX includes the resident model and temporary allocations; it is not KV bytes or total process memory.

Qwen3.8 native BF16 measured 13.4 tok/s here versus 16.3 in the earlier forward matrix, and TQ4 measured 13.0 versus 15.8. The common slowdown was not isolated to a cause. Desktop activity continued during measurement; a read-only snapshot showed no `pmset` thermal/performance warning. Treat these single-pass server results as measurements under the recorded conditions, not a controlled claim about changes between sessions. The interleaved kernel tests and same-session shared-model comparison isolate the implementation more directly.

## Large context and actual memory

The [large-context runs](affine4_incremental_large_context.json) used integration commit `2f487029`, before the later mask/partition tuning. Each completed 128 generated tokens, with no cache hit, eviction or adaptive throttling. The balanced guard stayed enabled: Apple Metal cap 37.44 GiB and prefill safety cap 33.70 GiB. No system memory-limit change was made.

| Model | Context rows | Prefill tok/s | Decode tok/s | Peak active MLX GiB | Periodically sampled physical peak GiB |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 200,000 | 250.0 | 11.8 | 21.81 | 26.13 |
| Qwen3.8-27B | 150,000 | 291.9 | 12.3 | 20.71 | 24.31 |
| Qwen3.6-35B-A3B | 200,000 | 869.8 | 47.8 | 22.16 | 24.40 |

The earlier Qwen3.8 150K/200K attempts were rejected by the guard during native cold prefill. Its native KV grows by 65,536 bytes/token: 12.21 GiB at 200K before recurrent state and workspaces. With 15 compressed and one retained full-attention layer, Affine4 grows by 19,936 bytes/token: 3.71 GiB at 200K. Qwen3.6 grows by 6,800 versus 20,480 bytes/token. TQ4 still prefills natively, and its guard estimate now reflects that fact.

The [memory trace summary](affine4_incremental_memory.json) distinguishes logical attention storage, cache-object `nbytes` including recurrent state/capacity, MLX active allocations, free allocator pool and physical footprint. The trace maxima and periodic sampler can catch different instants; their peaks must not be added.

| Model | Context | Actual cache objects at end of prefill GiB | Maximum physical footprint in chunk traces GiB |
|---|---:|---:|---:|
| Qwen3.8-27B | 200,000 | 3.86 | 26.32 |
| Qwen3.8-27B | 150,000 | 2.93 | 24.36 |
| Qwen3.6-35B-A3B | 200,000 | 1.33 | 24.54 |

The trace can observe the MLX pool counter reaching zero before macOS physical accounting drops. One read-only `vmmap` snapshot during the first Qwen3.8 200K prefill recorded 19.4 GiB physical footprint and 18.3 GiB IOAccelerator graphics mappings at that instant; its diagnostic overhead is included in the run. This does not establish a complete attribution of every physical-memory gap. Admission therefore still uses the larger of active MLX memory and physical footprint after the existing CPU hot-cache adjustment.

Qwen3.6's earlier 200K server run measured 731.1 prefill / 32.6 decode tok/s; incremental prefill measured 869.8 / 47.8, increases of 19.0% and 46.6%. Qwen3.8's previously failed 150K/200K runs now finish. Existing 100K baseline results remain in the earlier investigation; a new full 100K/150K/200K format matrix was not run.

## Prefix persistence and MTP

The [real Qwen3.8 smoke test](affine4_incremental_smoke.json) checks a cold prompt, warm reuse, SSD reuse after a server restart, then Lightning MTP after incremental prefill. The prompt stores verification code 42 before a long reference prefix; every response must retrieve it. The MTP request also counts from 1 to 20.

| Case | Prompt tokens | Cached tokens | Generated tokens |
|---|---:|---:|---:|
| cold | 7,098 | 0 | 2 |
| warm | 7,098 | 6,144 | 2 |
| restart | 7,098 | 6,144 | 2 |
| mtp | 7,107 | 6,144 | 53 |

The MTP smoke returned `42 1 2 ... 19 20`. The server recorded 36/36 accepted drafts across 16 cycles, confirming actual speculation rather than only an enabled setting.

A separate [short MTP control](affine4_incremental_mtp_comparison.json) uses the same 8K code prompt, Affine4, cold cache and 128 generated tokens, first without and then with Lightning MTP:

| Lightning MTP | Prefill tok/s | Decode tok/s | Generated tokens |
|---|---:|---:|---:|
| Disabled | 483.6 | 16.9 | 128 |
| Enabled | 472.9 | 36.1 | 128 |

The enabled timing run accepted 84/97 drafts (86.6%) across 42 cycles. This is a short, single-pair workload check. Historical original-oMLX MTP timings used a different prompt; they remain unsuitable for a direct absolute speed comparison.

These are execution, persistence and retrieval smoke checks. They do not establish general long-context quality, equivalence to native KV, or quality across architectures. Incremental compression changes prefill hidden states, and chunk size can affect outputs. The shared-model test deliberately starts from native prefill to isolate decode; it cannot substitute for an incremental quality evaluation.

The remaining measured performance deficit is approximately 3.1% in shared-model Qwen3.6 decode, with a larger gap for dense VLM attention kernels. The implementation preserves generic mask/stride support, FP32 scales and range normalization; those costs have not all been individually attributed. Some prefill and long-context verification cases exceed the earlier VLM implementation, while native BF16 remains competitive at short context.

`affine4_incremental_reproduction.zip` contains the scripts, exact source snapshots, source manifest and raw logs. Paths are local to this machine and require adjustment elsewhere. GPU tests and benchmark runs were sequential. No push or PR was created.
