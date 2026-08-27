<!-- markdownlint-disable MD013 -->

# Benchmark hardware and result provenance

Performance numbers published for this fork must identify the source commit,
physical topology, workload, cache state, concurrency, and statistic. A bare
tokens-per-second value is not reproducible and must not be presented as a
general oMLX result.

## Beta 1 qualification topology

The Qwen qualification snapshot below was captured from clean source commit
`6ebc22a822e3b50fe3c3d59acf1da62e8694f5dc`. The current lossless DS4 snapshot
was captured from clean source commit
`5575294f13fad2a2985ac2746ea5fcf6223c911b`. They record accepted end-to-end
observations; neither is a statistical claim beyond the identified runs.

For this snapshot, **cold prefix** means `prefix-miss` on an already loaded
deployment. It does not mean `cold-process`, does not include model-load time,
and must not be relabeled that way in public comparisons.

### Qwen text-model qualification

| Component | Reference configuration |
| --- | --- |
| Rank 0 | Apple M3 Ultra, 256 GB unified memory |
| Rank 1 | Apple M5 Max, 128 GB unified memory |
| Interconnect | Direct Thunderbolt 5 link |
| Collective backend | JACCL Thunderbolt RDMA |
| Model | `mlx-community/Qwen3-30B-A3B-4bit` |
| Tensor-parallel split | Equal `1:1` |
| Speculative verification | Not qualified; no Qwen MTP claim for Beta 1 |

Qwen3.8 models containing a vision tower are VLMs and are excluded from the
distributed Beta 1 qualification. The Qwen result demonstrates a second
text-only model family; it is not a universal-model compatibility claim.

### DS4 optimization reference

The current DeepSeek-V4-Flash optimization campaign uses this heterogeneous
Apple Silicon setup:

| Component | Reference configuration |
| --- | --- |
| Rank 0 | Apple M3 Ultra, 256 GB unified memory |
| Rank 1 | Apple M5 Max, 128 GB unified memory |
| Interconnect | Direct Thunderbolt 5 link |
| Collective backend | JACCL Thunderbolt RDMA |
| Model | DeepSeek-V4-Flash-0731-MXFP4-MLX (DS4 MXFP4) |
| Tensor-parallel split | Equal `4:4` (M3 Ultra:M5 Max) |
| Speculative verification | MTP fixed depth 5, when explicitly enabled |

The equal split is deliberate. A `3:5` candidate was faster (about 826 tok/s
average at 30K) but changed a deterministic greedy completion on the parity
corpus, so its exact runtime key is persisted as rejected and Automatic falls
back to `4:4`. Results from a single-rank microbenchmark, a modeled JACCL
transfer, or only one of the two machines must say so; those results are not
end-to-end two-host measurements. Record whether the direct TB5/JACCL path was
physically active and measured rather than inferred from a bandwidth model.

### Accepted Beta 1 measurements

API and rank-marker rates are separate measurements. `Aggregate` always means
the sum across the identified concurrent requests, never the rate of one
request.

| Model / topology | Scenario | Cache / load label | API tok/s | Rank-marker tok/s | Qualification detail |
| --- | --- | --- | ---: | ---: | --- |
| Qwen TP2 `1:1` | 30K prefill | cold-prefix (`prefix-miss`), single-stream | 1,533.73 | 1,537.15 | Direct completion |
| Qwen TP2 `1:1` | Decode | `prefix-miss`, single-stream, non-MTP | 56.83 | 56.46 | Same 30K request |
| Qwen TP2 `1:1` | Exact-prompt repeat | `hot-prefix-hit` (rank prompt LRU), fixed prompt | not applicable | not applicable | 30,004/30,005 tokens cached; TTFT 19.92 s -> 0.67 s |
| Qwen TP2 `1:1` | Concurrent decode | `concurrent-4`, independent prompts, non-MTP | 216.3 aggregate | not recorded here | 4 x 512 output tokens; 2,048 tokens in 9.469 s |
| Qwen TP2 `1:1` | In-flight cancellation | cold-prefix prefill | not applicable | not applicable | Both ranks stopped at 26,624 processed tokens and returned ready |
| DS4 TP2 `4:4` | 30K prefill (`metal`) | cold-prefix (`prefix-miss`), single-stream | 742.17 | not recorded here | Non-MTP, promoted equal-M3 output-chain + routed-MoE tail kernels |
| DS4 TP2 `4:4` | 30K prefill (`hello`) | cold-prefix (`prefix-miss`), single-stream | 722.80 | not recorded here | Same gate and identical completion hash; two-prompt average 732.49 |
| DS4 TP2 `4:4` | 100K prefill (`world`) | cold-prefix (`prefix-miss`), single-stream | 684.66 | not recorded here | Identical completion hash; 6.5% taper from the matched 30K prompt |
| DS4 TP2 `4:4` | Decode | `prefix-miss`, single-stream, non-MTP | approximately 29-31 | approximately 30 | B1; exact M=1 HC continuation remained off after a flat physical A/B |
| DS4 TP2 `4:4` | Decode | `prefix-miss`, single-stream, MTP fixed depth 5 | 79.8-80.6 raw | approximately 77-80 | High-acceptance prompts; acceptance-dependent chat cases are lower |
| DS4 TP2 `4:4` | Concurrent decode | `concurrent-2`, short-prefix reuse, non-MTP | 47.35 aggregate | not recorded here | Two distinct live rows/equal hashes; 23.69 tok/s per request |
| DS4 TP2 `4:4` | Concurrent decode | `concurrent-4`, short-prefix reuse, non-MTP | 75.22 aggregate | not recorded here | Four distinct live rows/equal hashes; about 18.82 tok/s per request |
| DS4 single M3 Ultra | 30K prefill | cold-prefix (`prefix-miss`), single-stream | 481.02 | not applicable | Accepted single-node kernel stack; no distributed transport |

The Qwen parity set used deterministic greedy requests with thinking disabled:
four of six cases were token-exact, and two of six were semantically
equivalent. This supports functional parity only. It is not evidence of
bit-exact or universal token-exact distributed execution.

## Required run metadata

Every published benchmark table or release note must include:

- exact git commit and whether the worktree was clean;
- oMLX, Python, MLX, MLX-LM, macOS, Xcode/Metal, and JACCL versions;
- model repository/revision plus a stable artifact or manifest hash;
- both device names, unified-memory capacities, power modes, and thermal state;
- physical link, negotiated Thunderbolt topology, RDMA readiness, and backend;
- tensor-parallel split and rank assignment;
- prompt and generated-token counts, sampling settings, and stop conditions;
- DFlash/MTP mode and depth, including acceptance rate when speculation is on;
- per-request decode throughput and aggregate throughput as separate values;
- warmup count, measured repetitions, statistic (median/p50/p95), and raw log;
- active feature/rollback gates and any non-default environment variables.

Avoid embedding local usernames or absolute model paths in published logs.

## Cache, temperature, and concurrency labels

Use all applicable labels. Do not shorten them to simply `cached` or `warm`.

| Dimension | Required label | Meaning |
| --- | --- | --- |
| Process/model | `cold-process` | New process and model load; no prior Metal pipeline or allocator warmup |
| Process/model | `warm-process` | Same process/model after declared warmup passes |
| Prompt reuse | `prefix-miss` | Zero reusable prompt tokens; report `cached_tokens=0` |
| Prompt reuse | `hot-prefix-hit` | Reused from the in-memory prompt/KV tier; report hit tokens and hit rate |
| Prompt reuse | `ssd-prefix-hit` | Restored from the SSD tier; report hit tokens, bytes, and restore time |
| Prompt reuse | `mixed-prefix-hit` | Hot and SSD tiers both contributed; report each tier separately |
| Request load | `single-stream` | Exactly one active generation request |
| Request load | `concurrent-N` | N active requests; report per-request and aggregate tok/s |
| Input identity | `fixed-prompt` | Repeated identical prompt, useful for cache-hit tests |
| Input identity | `independent-prompts` | Distinct prompts, suitable for cache-miss and sustained-load tests |

For tiered-cache claims, a qualifying run includes an initial miss, an
in-memory reuse, an eviction/spill into the SSD tier, a subsequent SSD restore,
and reuse after process restart when persistence is being claimed. Report cache
hit tokens and cache-source telemetry alongside timing. A warm Metal pipeline
with a new KV cache is `warm-process / prefix-miss`, not a fully cold run.

For a distributed deployment, `hot-prefix-hit` means the rank-local MLX-LM
prompt LRU and `ssd-prefix-hit` means the durable, rank-unanimous prompt-snapshot
tier. Those are not the local scheduler's paged hot/SSD KV-cache implementation;
public results must name which engine and cache implementation produced the hit.

For cancellation or steering measurements, also record time from client action
to stopped token production, whether the request had entered prefill or decode,
whether a replacement/steered request reused the valid prefix, and whether
other concurrent requests continued without interruption.

## Release benchmark matrix

At minimum, a performance-bearing prerelease should link raw evidence for:

1. cold-process, prefix-miss, single-stream;
2. warm-process, prefix-miss, single-stream;
3. warm-process, hot-prefix-hit, single-stream;
4. warm-process, SSD-prefix-hit, single-stream;
5. warm-process, prefix-miss, concurrent-N;
6. warm-process, prefix-hit, concurrent-N;
7. the same declared DS4/MTP depth-5 and lossless `4:4` TP configuration used for a DS4
   headline comparison;
8. the same declared Qwen non-MTP and 1:1 TP configuration used for a Qwen
   headline comparison, with Qwen MTP marked `not qualified` until separately
   measured.

If a cell was not run, mark it `not measured`; do not silently substitute a
projection, kernel-only measurement, or a different topology.
