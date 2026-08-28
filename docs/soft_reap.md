# Soft-REAP expert streaming

Soft-REAP keeps manifest-selected MoE experts and the rest of the model resident,
then loads non-pinned experts from the original safetensor shards into a fixed
expert bank. The bank is allocated once at model load, so its shape does not change
as experts are promoted or evicted. Dynamic slots use decayed route-frequency
scoring rather than plain LRU.

The settings **Hot experts per layer** and **Cold execution bank** are expert
counts, not byte budgets. oMLX reserves at least the model's top-k hot working
set. The cold execution bank is a separate fixed allocation used only by wide
miss-heavy batches; it never reduces or evicts the configured hot cache. The
model settings page includes both banks in the projected resident size before
load.

## Residency modes

**Soft-REAP pinning** keeps every expert in the uploaded manifest permanently
resident. Non-pinned experts share the additional hot cache.

**Analytical cache only** requires no manifest and pins no routed experts. Every
expert slot is evictable. The cache scores actual router selections, including
misses, halves the scores every 16 routed tokens, and evicts the lowest score with
recency as a tie-break. Experts used repeatedly by the current workload therefore
survive one-off routes without becoming permanent. The minimum cache remains the
model's top-k so one decode route can always execute exactly.

When the scheduler has an SSD cache directory, both modes also maintain a compact
learned hotlist there. On clean shutdown, oMLX stores per-layer router selection
counts. A later load prioritizes the most-used experts, then optimistically fills
every remaining configured slot with unobserved experts. Both learned and
optimistic entries remain fully evictable; optimistic entries start at zero
hotness and are therefore the first eviction candidates when real routing data
arrives. The configured RAM is fully populated without changing the expert count,
pinning fallback experts, or altering the Soft-REAP manifest. Profiles are
checkpoint-fingerprinted and written atomically. A missing, stale, or malformed
profile falls back to a fully populated optimistic cache.

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
Models with dense and sparse layers use their original model-layer IDs; dense layers
must not appear in the manifest.

## Supported MoE layouts

SSD Expert Streaming discovers routed layers under both `mlp` and `ffn` and skips
dense layers. It supports stacked and per-expert safetensor layouts, separate and
fused gate/up projections, affine, MXFP4, and NVFP4 metadata, optional quantization
biases, and routed layers that request an internal weighted sum. This covers the
SwitchGLU implementations used by Qwen 3.5/3.8/4-Exp, DeepSeek V3.2/V4, GLM
MoE/GLM5, MiMo-v2, Hy-v3, Laguna, Bailing Hybrid, and compatible MiniMax variants.

Resident substitution remains Qwen-only because each family applies its router
correction, normalization, and shared-expert path differently. Exact `0%` routing,
Soft-REAP pinning, and analytical cache-only mode are shared across the supported
families.

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
and prevents a new request from cascading through many speculative misses.
Single-token decoding also remains checked until every layer's evictable cache
has reached its configured capacity. Readiness is latched after the cache fills,
so steady-state decoding does not scan every layer on each token. A speculative
miss binds the promoted bank rows lazily; the retry's QMM materializes those writes
with their first consumer instead of forcing a redundant full-bank evaluation.
MTP hidden-state forwards use the same whole-pass cache transaction as ordinary
decode. Runtime execution stats expose how many otherwise-speculative passes were
held back by the cache-fill gate.

When prefill selects more cold experts than the hot tier can hold, oMLX executes
them in groups through the preallocated cold execution bank. The first cold group
is requested while resident experts execute, and each following group is requested
while the current group runs. Only one bounded group is in flight. Prefetch workers
perform CPU-only `pread` and NumPy decoding; MLX array construction remains on the
inference thread so thread-local Metal streams are never transferred between
threads. Small nearby
safetensor reads are coalesced with a 64 KiB maximum gap; pinned preload remains
strictly component-at-a-time to cap transient memory. This keeps one-shot prefill
experts from replacing analytically hot experts and avoids unbounded read-ahead.

Resident prefill routes are sorted by their physical bank slot once a group reaches
64 routed rows, restoring MLX's standard sorted gather-QMM path even when manifest
order and cache placement differ from logical expert IDs. Gate and up projections
share one preallocated quantized bank and one gather-QMM; decode therefore uses two
expert QMM launches per layer instead of three, without increasing bank memory.

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
not the default. That measurement predates the cache-fill gate; the current path
keeps such a partially filled cache in checked mode and avoids those rejected
whole-model passes. In an isolated identical-forward replay with all 48 layers hot,
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

Bounded synthetic measurements at 512×128 and 1024×256 expert geometry validated
the current optimizations without loading the production checkpoint. The cold
execution bank preserved exact output, reduced hot-cache evictions from 42 to 0,
and made the miss-heavy prefill-to-decode fixture 2.12-2.19× faster. At 1024×256,
physical-slot sorting improved 256-token prefill throughput by 26.6%; its output
was bit-identical to MLX's standard sorted path. Gate/up fusion reduced decode-shape
QMM calls from three to two and improved the isolated single-token fixture by 8.6%;
it was neutral within noise for prefill. Peak MLX memory for that isolated run was
391 MiB.

## DwarfStar-derived optimizations

The cache now adopts DwarfStar's decayed route-hotness eviction and recency
tie-break, protects experts in the current route from eviction, reuses fixed bank
slots, and submits all dynamic weight/scale/bias reads through one persistent
parallel `pread` queue. It also applies Darwin read-ahead hints to cacheable cold
reads and uses a learned popularity hotlist to warm ordinary evictable slots on
future loads. Runtime stats split SSD I/O, payload decode, bank binding, and bank
materialization time and count QMM projection invocations. These are compatible
with MLX's fixed-shape banks. The separate cold execution bank, bounded overlapped
group loading, 64 KiB read coalescing, gate/up bank fusion, and physical-slot
sorting are oMLX-specific extensions around that cache foundation.

Local throughput benchmarks store per-trial streaming counter deltas alongside
active and peak Metal memory. The benchmark log identifies the configured cache
and execution-bank width, hotlist preload count, hit rate, misses, evictions,
expert loads, SSD traffic, I/O and decode time, bank update time, expert-major
calls, scratch prefetch wait, sorted groups and routes, QMM calls, resident fill,
and hotlist warm-start coverage. Analytical-cache
runs with incomplete warm-start coverage are marked and should only be compared
with a run having the same learned coverage. Optimistic preload count and total
startup-fill coverage are reported separately; total startup fill should equal the
user-configured capacity even when no learned profile exists.

Further promising work, in priority order:

1. An expert-major sidecar that stores all nine quantized components contiguously,
   reducing random reads and safetensor-header fragmentation.
2. Adaptive per-layer cache sizing driven by measured miss and bank-update cost.
3. Cross-layer predictive prefetch. Exact next-layer routing is data-dependent, so
   this requires either a predictor or deeper graph/runtime changes and carries a
   higher regression risk.
