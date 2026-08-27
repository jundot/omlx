# Disaggregated prefill/decode over JACCL RDMA

Status: beta-grade persistent serving mode, default off and selectable as
**Phase split** in Cluster v2. Physical public-API, tool-call, cache-reuse,
cancellation, B4 overlap, unload/reload and telemetry gates pass on the two-Mac
JACCL/RDMA reference cluster.

## Persistent serving integration (2026-08-25)

The signed deployment schema now carries `serving_mode=disaggregated` plus
explicit prefill/decode rank ownership. The dashboard plans two complete
replicas, proves that weights plus the requested KV budget fit independently,
labels each Mac's phase, and launches a dedicated persistent worker. Rank 1
prefills request N+1 while rank 0 decodes request N; rank 0 remains the public
OpenAI-compatible API owner.

Production behavior now includes:

- streaming and non-streaming chat/completions, reasoning and structured tool
  calls without protocol-marker leakage;
- chunk-boundary prefill cancellation, decode cancellation, queue overlap and
  clean communicator reuse after a disconnected client;
- per-request compute-boundary prefill/decode rates, queue depth and measured
  cache-handoff bytes/time/bandwidth in the Cluster dashboard;
- hot exact/nearest-prefix reuse on the prefill owner, including exact final
  logits, plus optional persistent SSD snapshots in the existing
  `~/.omlx/cache/cluster-prompt-snapshots/<deployment>/<plan>/rank-1` tree;
- the existing all-rank hot/SSD clear protocol and quiescence-gated
  unload/reload/model-change workflow; and
- serving-profile replans that preserve phase ownership instead of silently
  reverting to sharded mode.

Latest integrated public-API evidence after batching eight ordered JACCL cache
operations per Metal evaluation:

| gate | result |
|---|---:|
| 9,410-token cold prefill compute average | **991.36 tok/s** |
| same request headline cold prefill | **966.26 tok/s** |
| decode | **29.59 tok/s** |
| 770.64 MB / 128-array cache handoff | **104.97 ms / 7.34 GB/s** |
| pre-window handoff baseline | ~6.1 GB/s |
| exact 9,410-token prefix repeat | **12.31 s → 1.40 s** |
| B4 2,219-token + 32 overlap | **10.86 s wall / ~1.29x** vs sequential |
| direct prefill disconnect | retired cleanly; following request HTTP 200 |
| registered `get_weather` tool | `finish_reason=tool_calls`, no leaked tag |

Persistent Phase SSD qualification uses the same root as ordinary oMLX cache
settings: `GlobalSettings.cache.get_ssd_cache_dir(base_path) /
cluster-prompt-snapshots`. The server places that resolved value in the signed
rank hostfile, so a configured custom SSD cache directory is inherited by peer
ranks too.

The physical 18,259-token persistence gate passed end to end:

- cold request: 0 cached tokens, 21.30 s;
- immediate hot repeat: 18,259 cached tokens, 1.42 s;
- four 4,096-boundary snapshots committed on rank 1 (1.69 GB, zero failures);
- unload/reload: 16,384 tokens restored from SSD, only the 1,875-token suffix
  prefetched, 3.66 s wall;
- normal **Clear SSD Cache** while loaded deleted all four snapshots and
  acknowledged both ranks;
- unload/reload after clear returned to 0 cached tokens and 20.87 s cold;
- normal **Clear SSD Cache** while the cluster was unloaded used the enrolled
  SSH path to remove the peer snapshot root as well, reporting both configured
  ranks. The remote root was absent afterward.

The live cluster was returned to loaded/ready with persistent Phase caching
enabled and the synthetic test cache cleared.

This is a throughput mode, not a claim that two full replicas accelerate one
request's prefill compute. The M5 Max supplies the roughly 1K tok/s prefill;
the second Mac overlaps decode and API work. DeepSeek-V4-Flash still cannot use
this exact two-full-replica topology because it does not fit on the 128 GB Mac;
its future phase topology must make each logical phase a shard group.

## Contract

Two workers load identical full model replicas. The selected prefill worker
processes every prompt token and samples the first output token. It then sends:

1. a bounded manifest containing model identity, prompt length, MLX-LM cache
   class names, tensor tree paths, shapes, dtypes and `meta_state` strings;
2. every cache tensor directly with `mx.distributed.send`; and
3. the first sampled token.

The decode worker verifies the model identity, receives each tensor with
`mx.distributed.recv`, reconstructs the original MLX-LM cache classes through
their `from_state` contract, and continues generation from the first token.
No cache tensor is copied through Python bytes, HTTP, SSD or the CPU.

Readiness and flow control use the existing `RankControlPlane` TCP/system-
Python proxy. JACCL is idle throughout arbitrarily long prefill and is entered
only after both workers are ready to transfer cache tensors. This separation is
load-bearing: an early prototype used periodic RDMA heartbeats and eventually
hit the 30-second progress guard during a 30K prefill.

## Capability rule

The current topology requires the complete model and its admitted cache budget
to fit independently on both workers. Qwen3.8-27B-4bit is about 15 GB and fits
comfortably on both test Macs. DeepSeek-V4-Flash does not fit as a full replica
on the 128 GB M5 Max (its two current tensor shards together measure about
171 GB), so DS4 needs a future form where each logical prefill/decode worker is
itself a shard group.

The cache codec is model-independent. A model is admitted when all cache state
leaves are MLX arrays and every cache class is an installed MLX-LM class with a
`from_state` method. Unknown cache classes, non-array leaves, model-identity
mismatches, malformed shapes/dtypes and unbalanced byte ledgers fail before
decode.

## Physical evidence

Hardware:

- rank 0: Apple M3 Ultra, 256 GB;
- rank 1: Apple M5 Max, 128 GB;
- JACCL over direct Thunderbolt RDMA;
- patched MLX `0.32.2.dev20260825+ceab91938`;
- model: `Qwen3.8-27B-4bit`, identity
  `f4ad23f9019f77fcc8e494ff76423e577cdd48fac923f93fb83c6b5f3872b022`.

| prompt | prefill role | decode role | prefill | cache bytes | handoff | wire rate | decode | parity |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 512 | M3 | M5 | 302.33 tok/s | 187.50 MB | 42.58 ms | 4.40 GB/s | 30.15 tok/s | 32/32 exact |
| 512 | M5 | M3 | 874.77 tok/s | 187.50 MB | 39.28 ms | 4.77 GB/s | 29.37 tok/s | 32/32 exact |
| 4,096 | M5 | M3 | 1,005.74 tok/s | 422.38 MB | 64.79 ms | 6.52 GB/s | 31.21 tok/s | 64/64 exact |
| 30,000 | M5 | M3 | 871.82 tok/s | 2.120 GB | 323.98 ms | 6.54 GB/s | 27.37 tok/s | 64/64 exact |

At 30K, the complete cache handoff costs 0.94% of prefill time. At 4K it costs
about 1.0%. Both cache directions and both role assignments passed.

The physical two-request 4K+64 pipeline measured:

- two independent prompt/decode hashes, both exact;
- 12.073 s when the M5 source performed the same two prefills and two control
  decodes serially;
- 10.040 s through the disaggregated fill/overlap/drain pipeline;
- **1.2025x measured request-throughput speedup** including two handoffs and
  pipeline fill/drain;
- a 3.965 s overlap window in which the complete 2.019 s request-0 decode was
  hidden beneath request-1 prefill.

The same stage rates imply a steady filled-pipeline interval of roughly
`max(4.0 s prefill, 2.0 s decode) + 0.065 s handoff = 4.07 s`, versus about
6.0 s serially. The deeper physical gates approached that ceiling:

| queue | exact requests | serial source | pipeline | per request | speedup |
|---:|---:|---:|---:|---:|---:|
| B2 | 2/2 | 12.073 s | 10.040 s | 5.020 s | **1.2025x** |
| B4 | 4/4 | 24.730 s | 18.822 s | 4.706 s | **1.3139x** |
| B8 | 8/8 | 50.545 s | 36.699 s | 4.587 s | **1.3773x** |

B8 moved eight 422.38 MB cache images at 5.8-6.5 GB/s each while decode held
31.3-32.6 tok/s. Per-request wall fell from 6.318 s serial to 4.587 s through
the measured pipeline. The first request still pays fill and handoff, so this
mode improves sustained throughput rather than single-request latency.

## Implemented artifacts

- `omlx/cluster/cache_transfer.py`: bounded universal cache manifest,
  reconstruction and direct point-to-point tensor transport.
- `omlx/cluster/disaggregated_worker.py`: two-rank full-replica parity worker,
  role reversal, two-request overlap and control/data-plane separation.
- `omlx/cluster/disaggregated.py`: fail-closed full-replica fit check and role
  planner. It evaluates both orientations and recommends disaggregation only
  when the estimated steady interval clears the configured gain threshold.
- `benchmarks/bench_disaggregated_prefill_decode.py`: configured-fabric launcher
  and durable JSON report.
- `tests/test_cluster_cache_transfer.py` and
  `tests/test_disaggregated_worker.py`: schema/class/tensor guards and scheduler
  frontier helpers.

## Remaining beta work

1. Reduce worst-case cancellation latency by adding a smaller interrupt quantum
   without regressing the 4,096-token prefill shape (current observed retirement
   was about 8.7 seconds after a 4,096-token progress boundary at long context).
2. Soak mixed prompt/decode lengths, 100K transfer, forced rank loss and repeated
   reloads; preserve the current fail-closed communicator replacement behavior.
3. Profile ordered handoff windows 4/8/16 at 30K and 100K. Window 8 is the first
   promoted physical win; multi-communicator/multi-rail transfer remains off.
4. Add VLM input transfer and grammar-constrained generation only after each
   has an exact cross-phase ownership contract.
5. Keep MTP/speculative state fail-closed. DS4 cannot enter this two-replica
   topology on the current 128 GB node, and no speculative cache handoff has
   yet passed parity.
6. Complete a packaged-app visual click-through. Source/UI tests pass, but the
   final browser surface was unavailable during this maintenance run.
