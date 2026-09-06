# Qwen4-Exp (Qwen3.8-Next) PLE N-gram residency

The Qwen3.8-Next / Qwen4-Exp architecture carries a **PLE N-gram table** —
~30% of total parameters on Qwen3.8-Next-Flash (51B of 176B) — that exports
lookupable facts out of the transformer weights. This doc covers how oMLX
stores and serves that table.

## What the table is

`Qwen4ExpNGramEmbedding`
(`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`)
hashes the last `ngram_size - 1` tokens per position with splitmix64-derived
multipliers (seeded per PLE layer), maps each hash into per-head prime-sized
vocabularies, and gathers embedding rows from a sharded table. History resets
at EOS so hashes never cross segment boundaries. `Qwen4ExpPLELayer` gates the
gathered embeddings against the hidden state (key/value projections plus a
dilated short conv) and adds the result at the start of every layer listed in
`config.ple_layer_ids`.

Because a lookup is a pure row gather — **no matmuls** — the table can live on
SSD and be paged on demand with no throughput cost.

## Residency modes

| Mode | Behavior |
| --- | --- |
| `mmap` (default) | `DiskBackedShardedEmbedding` memory-maps the safetensors shards (`MADV_RANDOM`) and gathers only the requested rows, in dense BF16/F16/F32/F8_E4M3 or oQ-affine (U32 + scales + biases) layouts. |
| `resident` | `ShardedEmbedding` keeps the whole table in memory. Opt-in only. |

Resolution order (`configure_ple_runtime` → `resolve_ple_runtime_mode`):

1. Explicit mode from the loader (per-model setting, see below).
2. `OMLX_QWEN4_PLE_MODE` env var (`auto` / `resident` / `mmap`).
3. `qwen4_exp_artifact.ple_residency` in the checkpoint's `config.json`.
4. `auto` → **mmap**.

The PLE table may live in a separate directory referenced by
`qwen4_exp_artifact.ple_artifact` (relative path, confined to the artifact
root).

## Why mmap is the default

- Identical throughput: lookups are gathers, and hot rows stay in the OS page
  cache. llama.cpp measures exactly the same tokens/s with `--mmap` on the
  same checkpoint family, while wired (pinned) memory drops ~26GB on a 128GB
  Mac (~25% of total) — see the r/LocalLLM writeup "Got Qwen3.8-Next-Flash
  ngram SSD offload working in llama.cpp".
- The backbone fits in ~75% of the quant's disk size, which lets larger
  quants fit a given memory ceiling.
- Measured on oMLX (M4 Pro 48GB, Qwen3.8-Flash-Next-oQ4e-mtp on an external
  Thunderbolt NVMe, `bench/bench_ple_residency.py`). The PLE gather is the
  only code that differs between modes:
  - **Page-cache warm**: mmap ≈ resident. Per decode step (16 rows, one PLE
    layer): 0.47 ms mmap vs 0.49 ms resident; per 4096-token prefill (65,536
    rows): 55 ms vs 31 ms — noise against multi-second prefill.
  - **Truly cold** (pages evicted): 6.6 ms per decode step / ~0.5 s per
    512-token prefill. Each gather touches up to 16 shard files × 3 tensors
    (weight/scales/biases are separate tensors) ≈ 48 independent 4 KB page
    reads at ~130 µs each on this enclosure. In the end-to-end bench the
    decode step was ~760 ms (expert streaming, 32% hit rate), so even fully
    cold PLE was ~0.9% of the step.
  - Cache retention makes cold a first-touch tax: touched PLE pages stay
    cached until memory pressure evicts them, and identical rows never cost
    twice. Future layout optimization if cold ever matters: interleave
    scales/biases rows with weight rows at conversion time to cut 48 reads
    per step to 16.

The per-model setting `qwen4_ple_ssd_offload` (default **on**, "SSD N-gram
Offload" in both the macOS app and the admin dashboard) opts out of mmap and
pins the table resident. The runtime still forces mmap when the resident load
exceeds the configured model-memory ceiling but the mmap load fits
(`engine_pool._qwen4_ple_offload_status`). On 48GB machines resident is
moot for the Flash-Next checkpoint anyway: the full table is 29.8 GiB and
the backbone 69.2 GiB.

### Settings-file migration

Settings files at version 1 persisted `qwen4_ple_ssd_offload: false` because
that was the old field default, not a deliberate opt-out. Loading a v1 file
coerces those blobs to the new default once (logged); an opt-out saved on
version 2+ is respected.

## FP8 checkpoints

The FP8 Qwen3.8-Flash-Next variant stores each PLE shard as raw `F8_E4M3`
rows with one shared `weight_scale` per sharded embedding.
`virtual_ple.py` exposes those shards as unscaled BF16 virtual tensors so
streaming oQ quantization can convert them without materializing the full
51.2B-row table.
