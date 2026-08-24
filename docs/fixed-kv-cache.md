# Fixed KV-cache memory

oMLX reserves a model's serving KV cache when the model launches. The reservation covers the configured context window for every session slot in the model's cache pool. This makes the main inference memory cost visible before launch instead of letting it grow as requests get longer.

The model launch dialog reports:

- model weights;
- KV memory for one session;
- requested and reserved session slots;
- the full fixed KV-cache pool;
- other known fixed allocations;
- the estimated total;
- detected unified memory, available memory, and projected remaining memory when the platform supplies reliable readings.

Values marked **Estimated before launch** come from model metadata and a point-in-time system memory reading. Values marked **Fixed KV committed at launch** describe the cache pool that the loaded engine materialized. After launch, oMLX refreshes available memory and headroom from a post-materialization snapshot. Those readings can still change because macOS unified memory is shared with other applications.

## Allocation lifecycle

1. The estimator resolves the model's effective context window and validates it against the model limit.
2. It builds a config-derived cache tensor manifest. The manifest records each tensor's shape, dtype, byte count, token capacity, cache type, and source.
3. The launch check combines model weights, the cache pool, and other known fixed allocations. Launch fails with a suggested remedy if no session slot fits.
4. During model load, oMLX runs a minimal cache probe against the loaded model, creates the fixed pool from those concrete cache classes, shapes, and dtypes, then forces MLX to evaluate its backing arrays. The probe is normally one token; bounded token-history caches such as LongCat N-gram are primed to their declared history length. Launch aborts if the live pool byte count differs from the config-derived plan or MLX reports a material deficit after evaluation. This matters because creating a lazy MLX expression alone does not commit the memory.
5. Each active request leases one reserved slot and reuses its arrays within the configured capacity. A warm-up or generation request does not create another full cache. Multi-session pools also commit one shared row-compaction workspace at launch. MLX uses this workspace when a finished row leaves a hole or a newly admitted request must move into that hole. DeepSeek pooled caches additionally commit a small remainder rollback buffer per slot and layer so rejected speculative tokens can be undone without allocating or corrupting cache state.
6. When every slot is busy, later requests wait in the scheduler queue. They do not grow the pool.
7. Unloading the model releases its slot pool with the model engine.

Requested concurrency and reserved slots can differ. If the requested number of slots would cross the current memory limit but at least one slot fits, oMLX reserves the feasible count and reports the cap before launch. For example, a server configured for eight concurrent sequences may reserve four slots for a large model. The fifth simultaneous request waits until a slot is returned.

The context window is the total capacity of one slot. A request's prompt tokens plus its maximum output tokens must fit within that value. oMLX rejects an oversized request with the required and configured token counts instead of extending the slot.

Fixed mode currently bypasses tiered prefix-cache reads and writes for that model. mlx-lm compacts batch rows after a request finishes, so an extracted row aliases storage that the next batch step may overwrite. Preserving it would require a second full per-request KV allocation and break the fixed-memory promise. Models launched without fixed mode keep the existing hot and SSD prefix cache. Direct prefix sharing inside the fixed pool needs a paged attention backend and is future work.

## Per-model opt-out

Fixed allocation is on by default for supported local LLM and VLM engines. Open the model's settings and turn off **Fixed KV cache** to restore the original growing cache for that model. The context window still limits each request, but oMLX no longer commits that capacity at launch. KV memory grows as inference proceeds, so the prelaunch total cannot include its eventual size.

Changing this setting changes the loaded engine. If the model is running, oMLX reloads or unloads it through the normal settings-update path before the new mode takes effect. The setting is stored as `fixed_kv_cache_enabled: false`. Older settings files omit the field and continue to use fixed allocation.

## How the estimate is calculated

The estimator sums physical tensor descriptors. It does not apply one generic `layers × heads × context` formula.

- MHA and GQA descriptors use the model's actual KV head count and head dimension.
- Sliding-window, rotating, and chunked attention layers use their own physical token limits in the preflight manifest. Full-attention layers in the same model retain the full configured context.
- DeepSeek MLA descriptors store the compressed latent and RoPE parts used by that architecture, rather than expanded conventional keys and values.
- DSA and pooled layouts include their extra index or pool tensors when the config exposes enough information to estimate them.
- Hybrid and recurrent models are planned layer by layer, including fixed convolution and SSM state that does not scale with context. Unknown custom state layouts fail closed instead of being treated as ordinary attention.
- The physical token capacity may be rounded up to the allocator step. The UI shows the requested context, while descriptor details retain the physical capacity used for the byte total.

The validated layout adapters cover the cache classes used by OMLX's current text and VLM launch paths:

- ordinary `KVCache` rows for MHA and GQA, including different key and value widths;
- `RotatingKVCache` and sliding-window/full-attention hybrids;
- Llama 4 `ChunkedKVCache` (currently serialized to one session because mlx-lm's chunk-trimming ownership is not batch-safe);
- nested `CacheList` and recurrent `ArraysCache` state, with config-specific manifests for Qwen GatedDeltaNet, Nemotron-H/Nemotron-NAS, Mamba/Mamba2, Jamba, Falcon-H1, Granite hybrid, PLaMo2, RWKV7, Griffin/Recurrent Gemma, Kimi Linear, LFM2, Ling/Bailing, Inkling, Baichuan M1, and LongCat N-gram;
- multi-pass and architecture-specific attention layouts used by LongCat MLA, AFM7, MiMo V2, and IQuest LoopCoder;
- DeepSeek V2 expanded MLA, DeepSeek V3 compressed MLA, DeepSeek V3.2/GLM DSA index caches, and DeepSeek V4 pooled DSA state;
- built-in native MTP head caches, including Qwen/GLM full-attention heads, DeepSeek Lightning head copies, and DeepSeek V4 DSpark context rings;
- MiniMax M3's sparse index cache and Unlimited OCR's prompt-plus-ring cache.

Together these adapters cover the cache trees produced by the repository's pinned mlx-lm revision. This is not a promise about arbitrary future mlx-lm modules or third-party remote code: a new cache tree must gain a descriptor and runtime adapter before oMLX will launch it in fixed mode.

This list describes cache layouts, not weight encodings. Affine 4/8-bit, NVFP4, and MXFP4 change model-weight storage and kernels. They do not by themselves change the runtime KV tensor layout, which normally remains FP16/BF16 unless the model explicitly declares a different KV-cache dtype. A converted DeepSeek checkpoint therefore uses the same DeepSeek cache adapter when its architecture metadata is unchanged. oMLX reads the actual checkpoint config and verifies the materialized tensors; it does not infer cache geometry from the weight quantization label.

The reported totals are:

```text
fixed KV cache = per-session KV bytes × reserved session slots
estimated total = model weights + fixed KV cache + other known fixed allocations
projected remaining = binding available-memory reading or ceiling - estimated total
```

These equations combine byte totals already produced by the descriptor planner. They are not a substitute for its architecture-specific tensor shapes.

For a pool with more than one session, `other_fixed_bytes` also includes a materialized row-compaction workspace. Layers with the same role, dtype, and non-batch shape share that workspace because mlx-lm filters cache layers sequentially. Chunked caches and DSpark rings also reserve the particular workspace they need with a one-session pool. DeepSeek pooled layouts reserve per-slot, per-layer remainder undo tensors; these are not shared because every layer may need its rollback state at the same time. The API reports the exact total as `pool_scratch_bytes` and as a named component.

## DeepSeek V3 at a 200k context

This deterministic example uses the DeepSeek V3 MLA layout with 61 layers, KV latent rank 512, RoPE head dimension 64, and an FP16 cache. A requested 200,000-token context rounds to 200,192 physical tokens at the 256-token allocation step.

For each layer, the cache descriptor contains:

```text
compressed latent: [1, 1, 200192, 512] × 2 bytes
RoPE component:    [1, 1, 200192,  64] × 2 bytes
```

Across 61 layers, one session uses exactly 14,067,892,224 bytes, or **13.102 GiB**. The pool then scales with the number of reserved slots:

The concrete allocator also commits one shared MLA row workspace of 230,621,184 bytes, or **0.215 GiB**, when the pool has more than one slot. It is shared across the 61 identical layer layouts and is shown under other fixed allocations.

| Reserved slots | Session KV rows | Shared scratch | Committed cache allocation |
|---:|---:|---:|---:|
| 1 | 13.102 GiB | 0 GiB | 13.102 GiB |
| 2 | 26.203 GiB | 0.215 GiB | 26.418 GiB |
| 4 | 52.407 GiB | 0.215 GiB | 52.622 GiB |
| 8 | 104.814 GiB | 0.215 GiB | 105.029 GiB |

Model weights and any other fixed allocations are added to those values. On a machine that can fit the weights plus four slots but not eight, the launch preview reports eight requested and four reserved. Up to four sessions run at once; further requests queue.

The example is specific to this descriptor set. A DeepSeek checkpoint with a different layer count, latent rank, cache dtype, DSA index cache, or architecture revision produces a different result.

## Admin API

The dashboard uses the following additive admin APIs:

```http
GET /admin/api/models/{model_id}/memory-estimate?max_context_window=200000
PUT /admin/api/models/{model_id}/settings
POST /admin/api/models/{model_id}/load
POST /admin/api/models/{model_id}/unload
```

The estimate response includes the context and model limit, weight bytes, per-session KV bytes, requested and reserved slots, fixed pool bytes, other fixed bytes, estimated total, system memory readings, fit result, lifecycle state, and component descriptors. Byte values are integers. Clients should format them for display but must not recalculate KV shapes.

The existing body-less load endpoint remains valid. It performs the same launch validation, so older clients receive a clear error if the configured context cannot reserve one session slot. The dashboard first saves a changed `max_context_window`, then calls that endpoint.

`PUT /admin/api/models/{model_id}/settings` accepts the additive `fixed_kv_cache_enabled` boolean. Missing or `null` means enabled. When it is false, the memory endpoint reports `lifecycle: "disabled"`, zero fixed KV bytes, and a weight-only known total. Its fit reason states that dynamic KV growth is excluded.

## Limits and interpretation

- Weight size is an estimate before load unless an exact loaded measurement is available. Quantization metadata, allocator bookkeeping, and temporary load work can make peak launch memory higher than the steady fixed total.
- Available memory can change between preview and launch. oMLX validates again at launch.
- macOS may compress or swap unified memory after commitment. Committed means MLX evaluated and retained the backing arrays, not that the operating system promises they will remain resident on a particular physical page.
- The estimator reports unavailable system readings as unavailable. It does not turn a failed probe into zero bytes.
- Every model still passes a live cache probe at launch. A future or third-party model with an unverified custom/recurrent layout is rejected instead of silently receiving an MHA estimate. Supporting a new layout requires both a config descriptor adapter and a matching fixed runtime adapter.
- Cache quantization changes the descriptors and must be supported by the active allocator. oMLX does not assume a bit width from a UI toggle when it cannot verify the backing layout.
- The runtime allocator supports the validated layouts listed above, including chunked, sparse-index, recurrent, ring, and pooled caches. A cache class outside that set is rejected and never falls back to a growing cache.
- Fixed-pool chunked prefills are admitted one at a time so a later short prompt cannot overwrite a row owned by an earlier long prompt. Decode remains concurrent up to the reserved slot count, and additional sessions wait in the normal scheduler queue.
- This is a dense fixed-row pool, not a paged-attention implementation. Aligning requests of different lengths can copy materialized cache rows through the committed scratch workspace. At million-token contexts that preserves the memory ceiling but can move tens of gigabytes and increase time to first token when sessions join or leave a batch.
- Fixed mode currently skips tiered prefix-cache reuse for the launched model. This avoids retaining extracted aliases or allocating a second full request cache while batch rows are compacted.
- Built-in native MTP is part of the fixed plan and pool. Fixed allocation still rejects distributed engines, DFlash, TurboQuant KV, SpecPrefill, and external VLM MTP. Distributed support needs rank-local reservations. Those optional paths allocate auxiliary weights or caches outside the target model's cache tree, so they need their own descriptors and fixed-pool adapters before their memory can be promised at launch.
- VLM media tensors, request-specific logits workspaces, and third-party custom code can add transient memory. Measurable fixed allocations are included in `other_fixed_bytes`; the named component list is present only when the server can attribute them. Unknown transient peaks are not included.
- The fit result applies to one model launch on the current machine. Concurrent model loads and other applications can consume the reported headroom.
