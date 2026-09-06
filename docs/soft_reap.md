# SSD expert streaming

SSD expert streaming reduces the resident memory of supported mixture-of-experts
models by keeping a bounded expert bank in memory and loading exact routed experts
on demand. The feature is experimental and disabled by default. When disabled, no
model classes are patched, no checkpoint reader is opened, and normal model loading
and evaluation are unchanged.

## Residency modes

- `soft_reap` pins experts listed in a Soft-REAP manifest and uses the cache for
  remaining routes.
- `cache_only` starts with no pinned routed experts and learns the working set at
  runtime.

Both modes execute exact router choices. They do not substitute, predict, or
preselect experts. Shared experts and the non-expert model trunk remain resident.
Qwen3.8 Flash Next checkpoints identify their architecture as `qwen4_exp`. For
this family, the 48 backbone MoE layers are streamed while shared experts and the
separate MTP expert layer remain resident. Enabling expert streaming also forces
the large PLE tensors onto their existing mmap/SSD path instead of materializing
them as resident arrays.

The implementation discovers layouts from safetensor metadata. It supports stacked
and per-expert tensors, fused or separate gate/up projections, and quantized affine
weights with scales and biases. Execution is selected from shape and dtype rather
than a hard-coded quant name.

## Experimental settings

| Setting | Default | Purpose |
| --- | ---: | --- |
| `expert_streaming_enabled` | `false` | Opt in to SSD expert streaming. |
| `expert_streaming_mode` | `soft_reap` | Manifest pinning or cache-only residency. |
| `expert_streaming_manifest` | unset | Soft-REAP JSON manifest; required by `soft_reap`. |
| `expert_streaming_cache_experts` | `32` | Dynamic resident expert slots per layer. |
| `expert_streaming_scratch_experts` | `32` | Cold execution slots per layer for prompt batches. |
| `expert_streaming_cache_policy` | `route_frequency` | `route_frequency` or `lru` eviction. |
| `expert_streaming_fast_resource_loading` | `true` | Use the Metal I/O loader. |
| `expert_streaming_direct_io` | `true` | Permit direct loads into idle Metal buffers. |
| `expert_streaming_native_demand` | `true` | Resolve exact decode demand at native evaluation. |
| `expert_streaming_decode_scratch_as_cache` | `true` | Lend scratch slots to the decode cache. |
| `expert_streaming_io_coalescing_kib` | `64` | Maximum gap merged into one storage read. |

The Admin dashboard and macOS model editor expose these settings under
Experimental. Profiles include all of them. Native demand and direct I/O require
Fast Resource Loading.

## I/O and weight format

Fast Resource Loading can issue staged Metal I/O reads for any indexed layout.
Direct I/O is used only when the destination is known idle and the on-disk tensor
representation exactly matches the runtime bank. Otherwise the loader safely
stages the read and copies or converts it.

For the DeepSeek V4 Flash mixed 2.4-bit checkpoint, keep packed quantized weights
packed and store routed-expert affine metadata (`scales` and `biases`) as FP16.
This matches the FP16 expert bank and permits the fastest direct path. BF16 affine
metadata remains supported through staged BF16-to-FP16 conversion, but cannot use
that direct path. The conversion utility is:

```bash
python scripts/convert_deepseek_v4_streaming_layout.py /path/to/model
python scripts/convert_deepseek_v4_streaming_layout.py /path/to/model --apply
```

The first command is a dry run. Use a checkpoint copy unless in-place conversion
is intentional. The utility validates tensor metadata and writes a conversion
report.

## Runtime behavior

Prompt processing groups routes into a preallocated execution bank so cold experts
do not evict the hot cache for every prompt group. During token generation, native
demand releases the Python GIL while the Metal graph is evaluated, loads exact
missing rows, and publishes them before dispatch. At the prefill/decode boundary,
the scratch bank can be deterministically reclassified as extra cache and restored
before the next prefill.

Runtime statistics include cache hits and misses, just-in-time loads, SSD bytes and
operations, native-demand callbacks, direct-load counts, I/O timing, and elastic
cache transitions. Admin benchmark snapshots include these counters.

## Runtime notes

- While at least one native-demand runtime is installed, `mlx.core.synchronize`
  is replaced by a variant that releases the GIL. A native-demand miss is
  resolved inside a Metal completion handler that needs the GIL, and the
  stock call waits with the GIL held, which deadlocks the process. The
  original is restored when the last runtime closes.
- The remap kernel binds the pool's route maps directly and the pool updates
  them in place, so a decode graph built one step ahead still reads current
  residency when it executes. Resident routes therefore release the MoE from
  the completion handler without entering Python.
- Fast Resource Loading waits on Metal IO tickets without the GIL, on a
  semaphore signalled by the ticket's completion handler, and fails a ticket
  that does not complete within 120 s instead of blocking the server. Staged
  tickets draw their staging buffers from a small pool instead of allocating
  one per load. `OMLX_FRL_POLL_WAIT=1` restores the earlier polling wait.
- Streamed prefill keeps several scratch groups of I/O in flight ahead of
  the group being computed (`OMLX_SCRATCH_PREFETCH_DEPTH`, default 3). The
  runtime snapshot and the benchmark delta report the scratch path's submit,
  wait, load, gather, compute and scatter seconds.
- For Qwen4-Exp checkpoints the PLE gather prefaults the rows a prompt
  touches through the file descriptor on a thread pool before indexing the
  MADV_RANDOM mapping (`OMLX_QWEN4_PLE_PREFAULT`, default on;
  `OMLX_QWEN4_PLE_PREFAULT_WORKERS`, default 16). Without it a 1024-token
  prompt faulted about 16k random rows one at a time while the GPU idled,
  and the same prompt took between 3.6 s and 10 s depending on what the page
  cache still held.
- The runtime drains the stream and clears the MLX buffer cache after every
  non-decode pass, since a streamed prefill chunk otherwise leaves several
  GiB of short-lived buffers in the pool and the prefill memory guard prices
  them as the next chunk's transient.
- For Qwen4-Exp checkpoints the PLE always takes its mmap path under
  streaming and is excluded from the residency estimate.

## Sizing guidance

Measured on a 64 GB M5 Pro (Qwen3.8 Flash Next oQ2, 48 layers of 512
experts, top-10; 1024-token prompt, 256 generated, MTP depth 3):

| Cache experts per layer | Resident | Hit rate | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 288 | 32.6 GB | 0.65 | 25 |
| 352 | 37.6 GB | 0.97 | 26 |
| 416 | 42.9 GB | 0.995 | 29 to 35 |

Above roughly 400 experts the SSD stops mattering and decode is bounded by
the per-layer host round trip that resolves misses; GPU utilization sits
near 65 percent while an eager load of the same model reaches 98 percent.
That round trip measures about 150 us per layer with every expert resident
(47 us from the GPU finishing to the CPU observing it, 106 us from the CPU
signal to the next command buffer starting), independent of whether a
completion handler or a polling thread does the signalling. A GPU-side wait
is not an option on Apple silicon: a running kernel observes a CPU store
only when its cache line happens to be evicted, milliseconds to seconds
later. Fewer route resolutions per token (batching, MTP) are the only lever.
When every expert is resident the pool switches to its resident mode and
matches eager speed, but that needs the whole expert set plus, for Qwen4-Exp,
page cache for the mmapped PLE. Larger scratch banks slowed prompt
processing; route-frequency eviction beat LRU on DeepSeek V4 Flash.

## Benchmarking

`benchmarks/soft_reap_streaming.py` reports projected residency, validates direct
row reads against MLX checkpoint loading, and can run an end-to-end generation:

```bash
python benchmarks/soft_reap_streaming.py \
  --model /path/to/model \
  --streaming-mode cache_only \
  --cache-experts 74 \
  --scratch-experts 12 \
  --full-load
```

Cache and scratch sizes are deployment budgets, not model constants. Tune them for
expert-row size, available memory, prompt/decode mix, and storage latency. Compare
warmed repeated runs and record just-in-time loads alongside PP and TG throughput;
a smaller resident budget is not automatically faster.
