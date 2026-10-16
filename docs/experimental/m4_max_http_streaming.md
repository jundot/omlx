# Fixed-token HTTP streaming measurements on M4 Max

This historical case study shows how to report client-observed streaming latency
and throughput separately from oMLX's admin benchmark PP/TG rates. It records
one short-prompt workload, not a hardware ceiling, a current-main benchmark, or
a before/after performance change.

## Tested setup

- Apple M4 Max, 40 GPU cores, 48 GiB unified memory; AC power; macOS 27.0.1.
- Python 3.13.14, MLX 0.32.3, oMLX 0.7.1.dev1 from the
  [released CPython 3.13 wheel](https://github.com/jundot/omlx/releases/download/v0.7.1.dev1/omlx-0.7.1.dev1-cp313-cp313-macosx_15_0_universal2.whl).
  The installed wheel metadata does not identify a source Git commit. The
  archived source/native manifest has 2,688 files and tree SHA-256
  `291516fcd115066a1c01b23e5753d2827cc95a010cca6ac78b2f6da6b27a9f0e`;
  this is a manifest digest, not a Git revision. These results do not test the
  current `main` branch. Recovering the same Python/native dependencies is
  required to reproduce the original environment.
- `mlx-community/Qwen3.8-27B-nvfp4`, snapshot
  `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`; NVFP4 group size 16,
  BF16 activations; ordinary VLM batched engine, no speculative decoding.
- Balanced burst mode: 64 maximum steps, 0.1 s single-request budget and
  default 0.03 s multi-request budget. Prefill batch size 1; completion batch
  limit equal to client concurrency; prefix caching and chunked prefill off.
- The benchmark child processes explicitly used `MLX_MAX_OPS_PER_BUFFER=50`,
  `MLX_MAX_MB_PER_BUFFER=50`, `MLX_BFS_MAX_WIDTH=20`, and
  `MLX_METAL_FAST_SYNCH=0`. Report these custom runtime controls when comparing
  results; this is not a measurement of unspecified stock runtime settings.

## Workload and timing

Use the fixed 19-token prompt in the linked reference package, greedy sampling,
normal EOS behavior and a 32-token output cap. Every retained request actually
produced 32 tokens with zero cached prompt tokens. Send groups of C concurrent
requests with `POST /v1/completions` at C = 1, 2 and 4. Decode the fixed
19 prompt IDs to the request's text `prompt` using the checkpoint tokenizer
with `skip_special_tokens=False`; do not apply a chat template.
Record monotonic client POST, first nonempty text and SSE `[DONE]` timestamps;
exclude response cleanup from `[DONE]` timing.

Report the following quantities separately:

- **Stream proxy x:** reconstructed tokens after the first text chunk divided
  by `[DONE] - first text`, per request. Reconstruct with the checkpoint's
  tokenizer and account for the first chunk's token count. This is a client
  delivery proxy, not producer/GPU decode throughput.
- **Group throughput y:** `C * 32 / (last DONE - first POST)`. This includes
  HTTP, queuing, prefill and delivery. Do not divide by a single request's
  duration or omit the concurrency factor.
- **First-text latency:** first nonempty text minus that request's POST.
- **Complete-response latency:** `[DONE]` minus that request's POST.

Take request medians within each group for x and both latencies; y has one
value per group. Discard three warmup groups, then measure three groups per
fresh server. Take the median of those three measured groups per server and
then the median of three fresh servers per concurrency. The oMLX subset has
9 fresh processes, 126 requests / 4,032 output tokens including warmups, and
63 measured requests / 2,016 measured output tokens. Its arms were interleaved
with other runtimes; the reference plan preserves the complete execution order.

## Observed results

All rates are tokens/s and all latencies are seconds. Each row is a median of
three fresh-process medians, with the aggregation above.

| C | Stream proxy x | Group throughput y | First text | Complete response |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29.646924 | 25.401468 | 0.214130 | 1.259770 |
| 2 | 13.156233 | 20.303770 | 0.762054 | 3.115390 |
| 4 | 11.623697 | 34.445485 | 1.027729 | 3.640560 |

Keep every fresh-process median visible: C1's aggregate hides a slower third
process. These are the nine oMLX processes in their original within-C order.

| Process | Stream proxy x | Group throughput y | First text | Complete response |
| --- | ---: | ---: | ---: | ---: |
| C1-block1-omlx | 29.646924 | 25.401468 | 0.214130 | 1.259770 |
| C1-block2-omlx | 29.670230 | 25.427213 | 0.213676 | 1.258494 |
| C1-block3-omlx | 17.690135 | 15.016193 | 0.378644 | 2.131033 |
| C2-pair1-omlx | 10.888296 | 17.308253 | 0.763974 | 3.662384 |
| C2-pair2-omlx | 13.156233 | 20.303770 | 0.762054 | 3.115390 |
| C2-pair3-omlx | 15.643899 | 23.786802 | 0.677555 | 2.659172 |
| C4-pair1-omlx | 11.490359 | 34.445485 | 0.936699 | 3.635532 |
| C4-pair2-omlx | 11.879737 | 34.450933 | 1.045944 | 3.640560 |
| C4-pair3-omlx | 11.623697 | 33.683407 | 1.027729 | 3.720573 |

The first two oMLX C1 processes sampled nominal thermal state (0); the third
and all C2/C4 processes sampled fair (1), which the original guard allowed.
The wider campaign also slowed in another runtime before the first sampled
fair state. All 359 archived resource snapshots in that campaign had AC power,
memory-pressure level 1 and no increase in Swapouts. Resource checks ran
outside measured windows. There is no clock, temperature or power telemetry
to establish a cause; this is not evidence of stable thermal performance.

Earlier sequential screens on the same Mac had about twice the C2 group
throughput. Their actual MLX buffer/synchronization controls were not recorded;
probe cadence and thermal observations also differed. A bounded source/raw
arithmetic audit found no concrete denominator, barrier, scheduler, cache or
pipeline bug. The cause of the absolute difference remains unknown. Do not
pool those campaigns or present either as the hardware's maximum capability.

## Evidence and reuse

The [published evidence directory](https://github.com/logitshq/minivllm/tree/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final) contains
[all process medians and metric definitions](https://github.com/logitshq/minivllm/blob/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final/INDEPENDENT-STATS.json),
[the independent protocol/resource/lifecycle review](https://github.com/logitshq/minivllm/blob/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final/INDEPENDENT-REVIEW.md),
and [the cross-campaign limitations](https://github.com/logitshq/minivllm/blob/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final/CROSS-CAMPAIGN-SANITY.md).
Only the oMLX rows are reproduced here; no relative-engine speedup is claimed.

For a new run, start from the [static reference package](https://github.com/logitshq/minivllm/blob/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final/m4-final-static-reference.tar.gz)
and its [manifest and limitations](https://github.com/logitshq/minivllm/blob/6b50184714a3a7f4daef85ae3ff8beab498b3b65/docs/benchmarks/m4-max-final/REFERENCE-PACKAGE.json).
Read its README, adapt paths/checkpoint/environment, rebuild the relevant
runtime and native dependencies, and create a new source/environment manifest.
Historical machine paths and authorization/pin files are not portable launch
commands. The package supplies the request/timing code and fixed workload;
it is not a self-contained installation or the complete original raw archive.
Record the new versions, process order, resource observations and all per-process
results. Revalidate token counts, cache state and server teardown before making
a new comparison. Changing the prompt, output length, caching or speculation
creates a different workload.
