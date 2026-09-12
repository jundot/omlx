# Qwen3.6-35B sustained decode: native BF16, TurboQuant4 and Affine4

Apple M5 Pro, 48 GiB; measured 2026-09-07 on implementation commit `2e993f3bd1171e52b1a2eefe4724d97c4006fa59`. MLX 0.32.2, mlx-lm 0.31.3, installed mlx-vlm 0.6.3. All cases use the same local Qwen3.6-35B-A3B-oQ4e-mtp weights. Native KV is BF16; weight quantization is unchanged.

## Sustained server throughput

Every request generates 4,096 tokens after a cold prompt. Rates below are arithmetic means of the server-reported rates; ranges show individual repetitions, not confidence intervals. Memory columns report the largest peak among repetitions. Native-relative throughput compares formats at the same context length.

| Context rows | KV format | Runs | Prefill tok/s | Decode tok/s (range) | Decode / native | Peak active MLX GiB | Sampled physical peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 8,192 | Native BF16 | 1 | 2,467.7 | 82.7 | 100.0% | 20.92 | 21.88 |
| 8,192 | TurboQuant4 | 1 | 2,467.6 | 76.8 | 92.9% | 20.89 | 21.85 |
| 8,192 | Affine4 | 1 | 2,406.2 | 83.2 | 100.6% | 20.79 | 21.77 |
| 200,000 | Native BF16 | 2 | 774.9 | 29.4 (26.3–32.5) | 100.0% | 24.72 | 31.59 |
| 200,000 | TurboQuant4 | 2 | 847.8 | 22.5 (21.5–23.5) | 76.5% | 25.17 | 29.29 |
| 200,000 | Affine4 | 2 | 766.0 | 46.9 (44.8–49.0) | 159.5% | 22.16 | 24.22 |

At 200K, Affine4 is **2.08× TurboQuant4** (108.4% higher decode throughput). Relative to native BF16 at that context, Affine4 throughput is **59.5% higher**, while TurboQuant4 is **23.5% lower**.

From 8K to 200K, decode throughput falls 64.4% for native BF16, 70.7% for TurboQuant4, and 43.6% for Affine4. These are distinct from the same-context compression penalties: long-context native attention also slows down.

Mean 200K prefill throughput is 9.6% lower with Affine4 than TurboQuant4. Peak active MLX memory is 22.16 versus 25.17 GiB, and the sampled physical peak is 24.22 versus 29.29 GiB.

## Individual runs and generation intervals

First/last windows contain approximately 512 tokens; actual token counts and monotonic generation timestamps are in the raw JSON. Stream output can deliver multiple tokens at a time, so boundaries are not exact multiples of 512. Window rates use their actual counts and elapsed times. No extra GPU synchronization is added.

| Run | Prefill tok/s | Decode tok/s | First window tok/s | Last window tok/s | Peak active MLX GiB | Physical peak GiB | Maximum thermal state | System swap-out delta MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192-native16-r1 | 2467.7 | 82.7 | 82.9 | 81.2 | 20.92 | 21.88 | 0 | 0.0 |
| 8192-tq4-r1 | 2467.6 | 76.8 | 77.8 | 75.1 | 20.89 | 21.85 | 0 | 0.0 |
| 8192-affine4-r1 | 2406.2 | 83.2 | 83.1 | 82.4 | 20.79 | 21.77 | 0 | 0.0 |
| 200000-affine4-r1 | 834.5 | 49.0 | 48.8 | 48.8 | 22.16 | 24.22 | 0 | 0.0 |
| 200000-tq4-r1 | 842.9 | 21.5 | 23.0 | 18.6 | 24.89 | 29.29 | 1 | 450.3 |
| 200000-native16-r1 | 715.0 | 26.3 | 25.1 | 31.9 | 24.72 | 31.59 | 1 | 0.0 |
| 200000-native16-r2 | 834.8 | 32.5 | 32.1 | 32.4 | 24.72 | 31.58 | 1 | 0.0 |
| 200000-tq4-r2 | 852.6 | 23.5 | 23.5 | 23.2 | 25.17 | 29.18 | 1 | 0.0 |
| 200000-affine4-r2 | 697.5 | 44.8 | 39.1 | 39.5 | 22.16 | 24.16 | 1 | 0.0 |

## Conditions and limits

- One GPU model workload at a time, with a fresh server process and empty base/cache directory for every case. The idle model on the separate manual server was unloaded for the experiment and restored afterward.
- Order: 8K native/TurboQuant4/Affine4; 200K Affine4/TurboQuant4/native; then 200K native/TurboQuant4/Affine4. Each server warms up with 2,048 prefill rows and up to eight generated tokens, excluded from the table.
- MTP, external MTP, DFlash, SpecPrefill and ANE prefill are disabled. Quantized formats retain the final full-attention layer in native precision. Maximum context is 262,144 tokens.
- Deterministic `code_python` prompts contain N+1 tokens for N prefill rows. Prompt-token SHA-256 matches across formats and repetitions at each length. Output-text hashes also match across both 200K repetitions within each format. All nine requests finish by length with exactly 4,096 generated tokens and zero prefix-cache hits.
- The benchmark harness uses greedy sampling and suppresses termination tokens only for timed benchmark requests. Lossy formats can produce different continuations, including different MoE expert routing. This is a complete-server throughput comparison, not teacher-forced timing or a quality evaluation.
- The balanced memory guard and Apple's 37.44 GiB Metal cap remain enabled. Native/TurboQuant prefill can enter the existing memory-bounded SDPA path as guard headroom decreases; Affine4 compresses incrementally. These routing effects are included in server measurements.
- Peak active MLX includes model weights, cache and temporary allocations. Physical footprint is sampled separately at 250 ms intervals. The two measurements must not be added and neither is a measurement of KV bytes alone.
- Thermal state 0 is nominal and 1 is fair. This is a desktop-machine run, not a controlled thermal chamber: system-wide swap activity and thermal observations are retained in each result. They cannot be attributed solely to the benchmark process. Reversed order and two long-context repetitions show the observed spread but do not isolate every source of variability. The 8K controls are single-pass.

## Evidence and reproduction

[Raw server results and protocol](affine4_35b_long_decode.json) include model settings, request bodies, source hashes, prompt/output hashes, generation intervals and environment snapshots. [Derived summary](affine4_35b_long_decode_summary.json) contains the means, ratios and individual window rates.

`affine4_35b_long_decode_reproduction.zip` contains the exact local runner, the interval-timing wrapper, the analysis script, model config, server-settings template, per-case progress and server logs. Local paths and the separate manual-server connection in the runner require adjustment on another machine. The server-settings template omits authentication and unrelated local directories. Logs redact the disposable benchmark login key. The implementation and shared benchmark harness are available at the commit above; source SHA-256 values identify the files used.

For each case, start the archived timing wrapper against that checkout with a new base directory and the recorded model settings; submit the recorded request to `/admin/api/bench/start` and poll its results to completion. Keep other model workloads unloaded. The original runner automates the recorded nine-case order. The `public_upload: false` protocol field means automatic dashboard benchmark-service upload was disabled; these artifacts are explicitly published with the PR.
