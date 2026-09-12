# Qwen server KV cache comparison

Apple M5 Pro, 48 GiB unified memory. MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.3. Both models use the same installed oQ4e weights in every format. Their native KV dtype is BF16, not FP16.

Affine4's clearest benefit is long-context decode. At 64K, it reached 61.8 tokens/s on Qwen3.6 and 13.9 tokens/s on Qwen3.8, compared with TurboQuant's 46.1 and 11.6. Its advantage over uncompressed KV is smaller and depends on context length.

Warm-prefix prefill remains a weakness. For a 2K suffix after restoring 32K tokens, Affine4 took 3.76 seconds on Qwen3.6 versus 1.96 seconds uncompressed. On Qwen3.8 it took 13.09 seconds, versus 10.92 with TurboQuant and 6.19 uncompressed. Cold-prefill peak memory remained almost identical across formats because cache conversion follows cold prefill.

42 controlled cases completed, generating 61,440 tokens. All six 4,096-token endurance cases finished at the requested length. Every measured cold trial had zero cached tokens. All 12 warm trials restored 32,768 tokens. Tested source hashes match the current files.

The server benchmark uses deterministic Python-code corpus prompts, greedy sampling, and benchmark-only termination-token suppression. It reserves one extra prompt token, so 4K/8K/32K/64K below mean 4,097/8,193/32,769/65,537 actual prompt tokens. MTP, VLM MTP, DFlash, SpecPrefill and ANE prefill are disabled consistently. The final full-attention layer remains uncompressed: 9 of 10 attention layers are compressed on Qwen3.6, and 15 of 16 on Qwen3.8.

8K and 32K table values are medians of two runs, with reversed format order for the second pass. 4K, 64K and endurance values are single runs. Small percentage differences should not be overinterpreted. The first small prefill measurement can include compilation not covered by the 2K warmup. These are performance and stability checks; output hashes and excerpts are retained, but model quality was not benchmarked.

## Qwen3.6-35B-A3B

**Decode throughput (tokens/s)**

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 86.2 | 81.7 | 84.2 |
| 8K | 84.4 | 77.3 | 82.2 |
| 32K | 70.2 | 59.8 | 71.3 |
| 64K | 56.2 | 46.1 | 61.8 |

**Cold prefill throughput (tokens/s)**

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 2350 | 2526 | 2551 |
| 8K | 2530 | 2487 | 2507 |
| 32K | 2064 | 2003 | 2028 |
| 64K | 1604 | 1599 | 1602 |

**Largest measured MLX process peak (GiB)**

These peaks include the model and cold prefill; they do not measure KV-only or decode-only residency.

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 20.84 | 20.81 | 20.81 |
| 8K | 20.92 | 20.89 | 20.89 |
| 32K | 21.28 | 21.29 | 21.29 |
| 64K | 21.91 | 21.91 | 21.91 |

**4,096-token endurance from an 8K prompt**

| Format | Generated tokens | Decode tokens/s | Total seconds |
| --- | ---: | ---: | ---: |
| Native BF16 KV | 4096 | 82.4 | 53.02 |
| TurboQuant 4-bit | 4096 | 76.5 | 56.82 |
| Affine4 | 4096 | 81.6 | 53.50 |

**Warm-prefix prefill: time to first token (seconds)**

Normal streaming completions, with a 32,769-token cold seed and then fresh suffixes. Each warm request reuses 32,768 tokens and generates one token. Latency includes prefix restoration and uncached prefill. The reported API prompt-TPS field includes cached tokens and is not used here.

| Format | Cold 32K seed | +256 suffix tokens | +2,048 suffix tokens |
| --- | ---: | ---: | ---: |
| Native BF16 KV | 15.19 | 0.72 | 1.96 |
| TurboQuant 4-bit | 15.20 | 0.80 | 3.81 |
| Affine4 | 15.19 | 0.83 | 3.76 |

## Qwen3.8-27B

**Decode throughput (tokens/s)**

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 16.6 | 16.3 | 17.1 |
| 8K | 16.6 | 16.1 | 16.8 |
| 32K | 14.8 | 13.9 | 15.5 |
| 64K | 12.9 | 11.6 | 13.9 |

**Cold prefill throughput (tokens/s)**

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 486 | 486 | 498 |
| 8K | 488 | 487 | 493 |
| 32K | 442 | 441 | 448 |
| 64K | 382 | 381 | 396 |

**Largest measured MLX process peak (GiB)**

These peaks include the model and cold prefill; they do not measure KV-only or decode-only residency.

| Prefill rows | Native BF16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 4K | 17.86 | 17.76 | 17.76 |
| 8K | 18.11 | 18.01 | 18.01 |
| 32K | 19.32 | 19.32 | 19.32 |
| 64K | 21.42 | 21.42 | 21.42 |

**4,096-token endurance from an 8K prompt**

| Format | Generated tokens | Decode tokens/s | Total seconds |
| --- | ---: | ---: | ---: |
| Native BF16 KV | 4096 | 16.2 | 270.50 |
| TurboQuant 4-bit | 4096 | 16.0 | 272.42 |
| Affine4 | 4096 | 16.7 | 261.44 |

**Warm-prefix prefill: time to first token (seconds)**

Normal streaming completions, with a 32,769-token cold seed and then fresh suffixes. Each warm request reuses 32,768 tokens and generates one token. Latency includes prefix restoration and uncached prefill. The reported API prompt-TPS field includes cached tokens and is not used here.

| Format | Cold 32K seed | +256 suffix tokens | +2,048 suffix tokens |
| --- | ---: | ---: | ---: |
| Native BF16 KV | 73.30 | 1.11 | 6.19 |
| TurboQuant 4-bit | 71.58 | 1.57 | 10.92 |
| Affine4 | 71.42 | 1.98 | 13.09 |

## Validation details

One Qwen3.6 reload was refused when the dynamic memory ceiling was 19.84 GB for an estimated 20.26 GB model. A fresh server process restored headroom, and the missing 8K/32K repeat passed with the balanced memory guard unchanged. The failed attempt is retained in the original results; recovery measurements are separate. A 4K/16-token priming run preceded recovery and is excluded from the tables.

[Full matrix and environment](affine4_qwen_server_comparison.json), [flat measurements](affine4_qwen_server_comparison.csv), [reload retry](affine4_qwen_reload_retry.json), [warm-prefix measurements](affine4_qwen_warm_prefix.json).

Reproduction uses [the isolated benchmark launcher](../serve_affine4_benchmark.py) and [the HTTP runner](../bench_affine4_server.py). Aligned diagnostic requests disable public leaderboard upload. The warm-prefix measurements use ordinary `/v1/completions` requests and normal EOS handling.
