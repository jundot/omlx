# Soft-REAP expert streaming

Soft-REAP keeps manifest-selected MoE experts and the rest of the model resident,
then loads non-pinned experts from the original safetensor shards into a fixed
expert bank. The bank is allocated once at model load, so its shape does not change
as experts are promoted or evicted. Dynamic slots use decayed route-frequency
scoring rather than plain LRU.

The setting **Hot experts per layer** is an expert count, not a byte budget. oMLX
reserves at least the model's top-k working set. The model settings page shows the
projected resident size before load.

## Residency modes

**Soft-REAP pinning** keeps every expert in the uploaded manifest permanently
resident. Non-pinned experts share the additional hot cache.

**Analytical cache only** requires no manifest and pins no routed experts. Every
expert slot is evictable. The cache scores actual router selections, including
misses, halves the scores every 16 routed tokens, and evicts the lowest score with
recency as a tie-break. Experts used repeatedly by the current workload therefore
survive one-off routes without becoming permanent. The minimum cache remains the
model's top-k so one decode route can always execute exactly.

## Manifest

Official REAP maps are accepted directly:

```json
{
  "0": [1, 4, 9],
  "1": [0, 3, 8]
}
```

The wrapped keys `layers`, `pinned_experts`, and `kept_experts` are also accepted.
Every routed layer must be present, and IDs must be unique and valid for the model.

## Resident substitution

The optional substitution threshold is the maximum relative router-weight loss
allowed when replacing a cold expert with a resident pinned or hot expert. For
example, at 5%, a resident expert with at least 95% of the cold expert's router
weight may displace it. Original router weights are retained and renormalized after
selection.

- `0%` preserves exact routing.
- Small non-zero values are approximate and require task-specific evaluation.
- `100%` restricts routing to the resident set.

## Execution modes

**Checked** is the default. It reads router IDs at each layer and loads any
misses before that layer runs. Although this introduces per-layer synchronization,
it avoids wasted whole-model passes when any SSD miss is possible.

**Speculative** is an opt-in resident-hit mode. It maps logical expert IDs to the
fixed resident bank entirely inside the MLX graph, avoiding a CPU/GPU
synchronization at every layer. The completed forward pass is checked once. If a
cold expert was selected, oMLX restores the pre-pass KV/SSM cache state, promotes
the first missing layer from SSD, and retries once. If that retry misses, it falls
back to the checked path. Rejected passes are never returned, so routing remains
exact at a `0%` substitution threshold.

Prompt and multi-token prefill always use checked mode. This warms the hot cache
and prevents a new request from cascading through many speculative misses. MTP
hidden-state forwards use the same whole-pass cache transaction as ordinary
decode.

## Performance status

On the 98.995 GiB Qwen3.8-Flash-Next oQ4e MTP baseline with the 288-expert REAP
manifest:

| Hot experts/layer | MLX active after load | Load time |
| ---: | ---: | ---: |
| 10 | 72.565 GiB | 19.8-20.3 s |
| 32 | 75.281 GiB | 20.1 s |

For the same checkpoint, cache-only mode with 32 experts per layer removes the
35.596 GiB pinned tier; its header-based projected residency is approximately
41.65 GiB including the estimator's safety overhead. The measured MLX active
memory was 39.684 GiB and load time was 6.794 s. A warm-SSD two-token smoke test
completed in 1.918 s with the same `", what"` output as the pinned baseline. It
loaded 933 experts (2.402 GiB) dynamically and had a 35.2% route hit rate. This is
still a capacity-oriented mode: cold-storage and longer-run results must be
measured separately, and low hit rates increase SSD traffic substantially.

Warm-SSD, two-token generation produced identical text in both modes. Checked
mode completed in 14.709 s. Speculative mode completed in 166.965 s because both
speculative calls still missed after two promotions and fell back; it is therefore
not the default. In an isolated identical-forward replay with all 48 layers hot,
speculative output was bit-identical (`max_abs=0`) and took 0.579 s versus 0.556 s
for checked mode. This baseline shows no current throughput benefit from the
larger speculative graph, even at a 100% resident hit rate.

The streamed expert computation is bit-exact at a `0%` threshold. Before the
graph-native speculative path, exact on-demand loading required reading router IDs
back to the host at every MoE layer: a one-token cold run measured 127.9-132.8
seconds, while a repeated warm-cache run measured 78.4 seconds. The controlled
execution-mode results above supersede those earlier timings. The 5% substitution
experiment remains disabled by default pending end-to-end quality evaluation.

MLX `export_function` can persist graph IR, but imported functions still need
`mx.compile` in the new process. It does not persist a ready-to-run Metal executable,
and host-side SSD decisions cannot be captured inside the compiled graph.

## DwarfStar-derived optimizations

The cache now adopts DwarfStar's decayed route-hotness eviction and recency
tie-break, protects experts in the current route from eviction, reuses fixed bank
slots, and submits all dynamic weight/scale/bias reads through one persistent
parallel `pread` queue. These are compatible with MLX's fixed-shape banks.

Further promising work, in priority order:

1. An expert-major sidecar that stores all nine quantized components contiguously,
   reducing random reads and safetensor-header fragmentation.
2. An optional learned hotlist profile that preloads experts as *evictable* cache
   entries, unlike Soft-REAP pins.
3. Per-layer read/bind/wait timings and adaptive cache sizing from real miss cost.
4. Cross-layer predictive prefetch. Exact next-layer routing is data-dependent, so
   this requires either a predictor or deeper graph/runtime changes and carries a
   higher regression risk.
