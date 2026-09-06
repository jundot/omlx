# Expert Streaming (SSD)

Run Mixture-of-Experts (MoE) models that are larger than your Mac's RAM by keeping only the dense weights resident and streaming routed expert banks from SSD.

Inspired by [slipstream](https://github.com/dwijenpatel/slipstream) (Swift/Metal expert LRU from SSD) and [colibri](https://github.com/JustVugg/colibri) (learned pin store + multi-tier memory), ported to oMLX's Python/MLX stack.

## When to use

- **You have a large MoE** (e.g. `glm_moe_dsa` / Qwen3.6-35B-A3B) and a **16–24 GB Mac**. Without streaming the model needs `resident_bytes` (checkpoint × 1.05) — often above the wired limit. With streaming it needs `dense_bytes × 1.05` (page-cache-only default), so a 35B MoE that needs ~21 GB resident fits in ~5–8 GB.
- You care about **fitting, not single-stream speed**. Streaming is slower than fully resident and disables continuous batching for that model (one request at a time). Use the default resident mode when the model already fits.
- Streaming auto-enables when the resident model would not fit: `resident × 1.10 > ceiling ≥ streaming × 1.10 + one-layer-bank` (10% margin + streaming transient, so an exactly-at-ceiling load is refused). The admin payload reports the same rule so the UI can show an amber "auto-enabled" hint — the same pattern as `qwen4_ple_ssd_offload`.

## How it works

- **Dense stays resident**: attention, shared experts, embeddings, LM head — always in unified memory.
- **Experts live on SSD**: the stacked `switch_mlp.(gate|up|down)_proj` banks (`(E, O, I)` tensors) stay memory-mapped from the original safetensors files (MADV_RANDOM, like the Qwen4 PLE `DiskBackedShardedEmbedding`). No duplicate copy is made.
- **Per-layer demand loads**: on a batch, the union of routed experts is resolved per layer; hits in the (optional) LRU run immediately, misses read the expert slice with coalesced `preadv` runs on the process-wide `_EXPERT_IO_POOL` (16 workers, QD16 — QD32 measured slower; never per-call pools) plus a separate run-read pool; the mini-bank is assembled on the inference thread. Quantized scales/biases ride along.
- **Page-cache-only is the default** (no app-level LRU): expert reuse rides the OS file cache (clean, evictable pages, never swapped). `expert_streaming_budget_gib > 0` re-enables a bounded LRU; `budget_auto` scales it to free RAM. Eviction is LRU by default, S3-FIFO behind `expert_streaming_cache_policy = "s3fifo"`.
- **Fused `gate_up` models budget 2 projections/expert** (split: 3), reconciled to the majority layout at convert time. With an active cold tier the estimate reports tier-aware effective bytes (`expert_bytes_effective`, `uniform`/`hobbit`/`none`) in the admin payload.
- **Per-request health**: one `expert_streaming req` log line per request (LRU hit rate, effective policy, governor state, read latencies) and the same numbers in `expert_streaming_health` on `GET /admin/models` and the Active Models card.
- **Per-model eval boundaries**: GLM/DeepSeek decoders evaluate + clear the allocator per layer natively; qwen4_exp gets it from a decoder-class patch (see [Per-layer eval boundary](#per-layer-eval-boundary)). This is what keeps long prefills from accumulating one mini-bank per layer in the lazy graph.
- **0.6.4 optimized paths stay engaged**: streaming only replaces `switch_mlp`, never attention — Qwen4's exact QSA prefill/decode (`qsa_fast.py`), resident-PLE/`hc_projection` opts and sparse-native kernel, and GLM-5.3's affine prefill tile all keep working under streaming. The scheduler gates *wide* Qwen4 prefill chunks on the sparse native kernel being built; PLE speculative-rollback keeps its simplified single-site capture and `ValueError` validation, fail-closed on `estimate.supported`. The MoE weighted-sum routes to `glm_moe_weighted_sum` (native ext, `mx.fast` fallback) with scatter-unsort as the last resort.

## Supported models

| model_type | example | MoE | per-expert (oQ4e) |
|---|---|---|---|
| `glm5_next` / `glm5_next_text` | GLM-5.3-Flash-oQ4e (190G) | 42 layers × 288 routed | ~13 MB weight + scales |
| `qwen4_exp` / `qwen4_exp_text` | Qwen3.8-Flash-Next-oQ4e (99G) | 48 layers × 512 routed + PLE | ~2.7 MB weight + scales |
| `glm_moe_dsa` | GLM-5.2 MoE DSA | — | ~1.7 MB (BF16) / quantized packed |
| `deepseek_v32` | DeepSeek-V3.2 family | — | quantized packed |
| `deepseek_v4` / `deepseek_v4_mtp` | DeepSeek-V4-Flash-0731-oQ4e-mtp (166G) | 43 layers + 3 MTP stages × 256 routed | ~12.6 MB (mxfp4 gs32) |
| `qwen3_moe` | Qwen3-MoE family (128 experts, top-8) | sparse-step pattern¹ | stacked `switch_mlp` |
| `qwen2_moe` | Qwen2-MoE family (60 experts, top-4) | every layer is MoE² | stacked `switch_mlp` |
| `deepseek_v3` | DeepSeek-V3 (256 routed, top-8) | dense-first layers skipped³ | stacked `switch_mlp` |
| `glm4_moe` | GLM-4-MoE (128 routed, top-8) | dense-first layers skipped³ | stacked `switch_mlp` |

The allowlist (`SUPPORTED_TYPES` in `residency.py`, the single source of truth) is a hard gate: an unlisted `switch_mlp` family reports `supported = False` (fail closed) instead of forcing streaming without the lazy load — that combination used to materialize multi-hundred-GB banks before conversion (OOM/SIGKILL).

¹ `qwen3_moe`: a layer is MoE iff `(layer_idx + 1) % decoder_sparse_step == 0` and it is not in `mlp_only_layers` (mirrors mlx_lm; defaults resolve to all-MoE). ² `qwen2_moe`: the mlx decoder is unconditionally all-MoE. ³ `deepseek_v3` / `glm4_moe`: resolved from `first_k_dense_replace` (generic fallback). All four carry `moe_intermediate_size`, which the estimate reads instead of the 1407 fallback. Runtime conversion of the widened four is covered by fake-checkpoint walk tests; bit-exact validation on real weights is a recorded follow-up.

Loading a glm5_next / qwen4_exp checkpoint with `expert_streaming_enabled` uses the lazy loader (`lazy=True`) and converts to streaming **before** `materialize_lazy_state` — the multi-hundred-GB MoE banks are dropped as lazy arrays instead of ever being materialized. GLM decoders additionally get `compile_ffn` disabled and a per-layer `mx.eval(out)` + `mx.clear_cache()` so the per-layer expert mini-banks (~3.4 GB at prefill) do not accumulate in the lazy graph / allocator and swap the machine. Text-engine loads (BatchedEngine) apply the same lazy + convert-before-materialize order for streaming-supported model types — this is what makes `deepseek_v4` viable on 16 GB Macs.

## Configuration

All expert-streaming knobs are per-model settings (`ModelSettings` → `PUT /admin/models/{id}/settings` → runtime signature → WebUI → macOS app), with env vars as fallback when the setting is `None`. They are load-time settings: toggling unloads (and re-loads if pinned) the model. They are excluded from cross-model profile templates (hardware-specific, like the other streaming fields).

| Setting | Values | Default (env fallback) | UI |
|---|---|---|---|
| `expert_streaming_enabled` | bool | off | Toggle, WebUI Advanced card + macOS Model Settings → Advanced |
| `expert_streaming_budget_gib` | float? | `null` | `0` = **page-cache only** (no app-level LRU); `> 0` = fixed LRU heap, clamp 0–64. Any explicit value (including `0`) wins over `budget_auto` | "Cache budget (GiB)" input |
| `expert_streaming_budget_auto` | bool? | on | RAM-scaled cache: `min((ceiling − streaming − 2 GiB) × 0.5, min(8 GiB, knee))`, whole MiB. `null` follows the default (on); `false` opts back to page-cache only | "Auto RAM-scaled cache" toggle |
| `expert_streaming_io_depth` | int? | QD16 | IO queue depth of the shared pool | — (autotune) |
| `expert_streaming_coalesce` | bool? | on | Coalesce consecutive expert ids into one `pread` per bank key (`expert_run`) | — (autotune) |
| `expert_streaming_readahead` | bool? | on | `F_RDADVISE` kernel readahead on predicted next-layer runs | — (autotune) |
| `expert_streaming_seed` | bool? | on | Prefill-hotness page-cache seed at the prefill→decode boundary | — (autotune) |
| `expert_streaming_pins` | bool? | off | mlock observed hot experts | Toggle + rows |
| `expert_streaming_pin_gib` | float? | 1.25 | Pin budget | "Expert LRU budget" row |
| `expert_streaming_pin_sync` | bool? | off | Apply learned pins synchronously at load | Toggle |
| `expert_streaming_pin_regime` | str? | `decode` | Which routing sample pins read: `decode` or `prefill` | Field |
| `expert_streaming_topk_threshold` | float? | `null` | `null` / >= 1.0 = exact routing (**bit-exact**); 0.05–0.95 = adaptive top-k mass truncation (**approximate, changes outputs**) | Field, warning hint |
| `expert_streaming_cache_prior` | float? | 0.0 | Logit-space bonus for LRU-resident experts before top-k (`0.0` = exact, bit-identical) | Field |
| `expert_streaming_cold_tier` | str? | `null` | "2"–"8": requantized low-precision tier under `<model>/expert_cold/` | Field (gated on tier presence) |
| `expert_streaming_hot_fraction` | float? | `null` | HOBBIT hot/cold split fraction (0–1) under a cold tier + learned profile | Field |
| `expert_streaming_cache_policy` | `lru` / `s3fifo` | `lru` | Cache eviction policy | "Cache & memory" select + app picker |
| `expert_streaming_dynamic` | tri-state | off | Dynamic residency governor | Toggle + Default/On/Off menu |
| `expert_streaming_dynamic_max_gib` | 0–64 | 6 | Governor growth ceiling | Numeric field (governor on) |
| `expert_streaming_per_layer_eval` | bool? | on | Per-layer eval boundary (qwen4_exp patch) | Toggle in both UIs |

The transition table (k+1 overfetch, `OMLX_EXPERT_STREAMING_TRANSITION`) is default-on power-user tuning without UI.

**Policy: defaults are bit-exact.** Every quality lever (topk, cache-prior, cold tier) is opt-in, documented, and gated by its own perplexity measurement; the exact path is verified token-identical by construction and by bench token gates.

Edge cases: `dynamic=true` with `budget=0` does not arm the governor (page-cache-only is an operational decision; the UI hints this); any policy change reloads the model via the runtime signature; the cold tier accepts 2–8 bits in the PUT.

## Cache policy: page-cache-only by default

Warm-page-cache baseline (48 GiB Mac, external ~3.7 GB/s NVMe, decode 48 after a short prefill):

| model | budget | hit rate | tok/s | notes |
|---|---|---|---|---|
| Qwen 99G | 0.5 GiB (8/layer) | 0 % | ~1.0 | flat — bottleneck is per-call serial load, not hits |
| Qwen 99G | 1–2 GiB (8–16/layer) | 0 % | ~0.85 | same |
| Qwen 99G | 4 GiB (32/layer) | 23 % | ~0.9 | knee — more memory buys nothing |
| Qwen 99G | 8 GiB (64/layer) | 32 % | 0.34 | **negative**: big cache evicts OS page cache → misses re-read SSD |
| GLM 190G | 1/2/4 GiB | 0 % | 0.065–0.072 | 13 MB/expert; ~120 ms/call serialized copies dominate |

Cold A/B of the default change (16-token decode):

| config | tok/s | RSS (decode avg) |
|---|---|---|
| Qwen LRU 4G | 0.834 | 7.3 GB |
| Qwen **page-cache only** | **1.007** (warm 1.133) | 5.6–5.8 GB |
| GLM LRU 8G | 0.363 — 0% hit rate, 8 GB of bundles pinned in RSS | 11.7 GB |
| GLM **page-cache only** | **0.381** | 4.8 GB |

Even at a 14.5% LRU hit rate (Qwen) the heap loses: it pins RSS the OS could have used for page cache (the same finding slipstream and FlashNext report). A large LRU goes sharply net-negative past the knee — Qwen 8k prompt, LRU budget 6 GiB: hit 4.95% with 130k evictions and **0.649 tok/s vs 2.86 page-cache only**; swap writes spiked 519 MB/s. Operational default for these models is budget 0.

`budget_auto` (on by default) sizes the LRU from free RAM: `min((ceiling − streaming − 2 GiB) × 0.5, min(8 GiB, knee))` — resolved identically in the loader and the admission path so pre-load accounting matches. The autotuner emits the measured knee (`budget_knee_gib`) into its recommendation and, on `--apply`, into the model's `.omlx/expert_budget_knee.json`.

S3-FIFO (`expert_streaming_cache_policy = "s3fifo"`) is a scan-resistant alternative to plain LRU. LRU **admission** filtering (insert only experts seen ≥2 times) was measured and closed: at the pin-equivalent budget the Belady oracle bound is ~33% — the top-frequency set that pins already capture, and pins measured that the approximation does not improve decode on this SSD class.

## I/O pipeline: pread, coalescing, queue depth

The access method dominates everything else. Cold micro-benchmark on a GLM oQ4e shard (40 × 4 MB expert slices):

| method | cold GB/s |
|---|---|
| mmap + MADV_RANDOM (page faults) | **0.35** |
| `os.pread` (single contiguous read) | **10.6** |

GLM 190G oQ4e, 8 GiB budget, decode 16, cold cache:

| load path | tok/s | TTFT | disk (decode) | load ms/call | GPU |
|---|---|---|---|---|---|
| mmap faults (first implementation) | 0.063 | ~200 s | 330 MB/s | ~190 ms | 11 % |
| pread, serial | 0.249 | ~40 s | 1.4 GB/s | 22.5 ms | 29 % |
| pread, 4 workers | 0.313 | 22.9 s | 1.85 GB/s | 15.4 ms | 35 % |
| pread, 8 workers | **0.336** | 22.8 s | 1.85–2.3 GB/s | 14.7 ms | 35 % |

Queue-depth sweep on Qwen (48-token decode, cold): QD8 1.538 → **QD16 2.061 (+34%)** → QD32 2.097 (plateau). `_EXPERT_IO_POOL` runs at QD16; `OMLX_EXPERT_STREAMING_QD` overrides. Disk read peaks ~2.5 GB/s of the 3.7 GB/s sequential spec; random-read ceiling is what decode sees.

**Run coalescing** (consecutive expert ids → one `pread` per bank key, `_ShardReader.expert_run`, verified bit-exact): +4% decode, ~2% at 8k prefill. Free, kept. `expert_streaming_coalesce = false` disables.

**Fixed-cost cuts (bit-exact)**: BF16 promotion via `mx.array(v).view(mx.bfloat16)` bit-reinterpret (~9× on 4 MB slices, and it matches `mx.load` exactly — the old `shift → f32 → astype` roundtrip flushed bf16 subnormals, so the *old* path was the inexact one); one shared `_RemapPlan` per MoE layer (eval + unique + compact remap) reused by gate/up/down; demand reads sorted by ascending expert id (= ascending file offset). Wall/call dropped 21.8 → 9.2 ms.

**Hybrid decode fast path**: decode-shaped calls (≤ 64 routed rows) resolve through the UNION mode — all projections in flight at once — while prefill keeps the rolling window (bounded RSS). Interleaved A/B, 2k prompt: decode **3.002 vs 2.651 tok/s (+13.2%)**, spread 10.2% vs 27.1%; 8k TTFT −2.7% with decode +5.9%; tokens identical in every arm. `ctx_fallback_to_legacy` counters (per reason: bank_too_large, read_failure, tier_mismatch, dict_backing) make the fast path observable; decode measures zero fallbacks.

**Rolling layer context**: the rolling path resolves one projection through the layer context while prefetching the next, so banks are not all resident at once; single-promotion promotes one bank per (key, tier) instead of building U per-expert arrays. The run-gap bridge (merging small gaps into one read) measured a net TTFT **loss** in both regimes (2k: 55.0 vs 47.5 s; 8k: 107.2 vs 98.9 s) — the knob stays for slower backends, default off.

**Prefill regime pool**: prefill-shaped calls can use a separate, deeper pool (`OMLX_EXPERT_STREAMING_PREFILL_QD`, default 24) so a long chunk's dense demand does not contend with decode's latency-sensitive reads.

**F_RDADVISE kernel readahead** (Darwin `fcntl` radvisory, best-effort): predicted next-layer expert runs (from the previous token's routing) are advised as one kernel hint per run — zero userspace copies. Measured: decode +0.8%, TTFT −1.4%, demand p50 −8.5% / p95 −3.5% (interleaved 3-rep A/B, tokens identical). Default on.

Net pipeline effect (8k prompt, single request, budget 0): TTFT ~74 s with Metal peaks 8.3–10.4 GiB vs the pre-port ~208 s / ~34.5 GiB IOAccelerator — decode runs disk and GPU concurrently instead of serializing per call.

## Per-layer eval boundary

The chunk forward is lazy: without an explicit boundary, every MoE layer's assembled mini-bank stays referenced until the chunk-end eval, so one ~1.5k-token chunk commits ~32 GB of Metal transients on qwen4_exp (48 layers × uniq × ~2.5 MB).

The boundary applies `mx.eval(out)` + `mx.clear_cache()` after each decoder layer during prefill-shaped calls — the DeepSeek/GLM pattern, ported to qwen4_exp via a decoder-class patch that covers both `Qwen3_5MoeDecoderLayer` (installed mlx_vlm) and the vendored `Qwen4ExpDecoderLayer`, with candidate resolution at apply time and idempotent double-wrap. Decode and MTP verify stay lazy (48 forced syncs/token would erode the QD16 win).

- Knob: `expert_streaming_per_layer_eval` (default **on**), a runtime-signature knob like the IO overrides; toggles in both UIs.
- Bit-exact by construction (`mx.eval` materializes what the next layer reads anyway).
- Measured (Qwen 2k prompt, 96-token decode, warm, interleaved OFF-ON-OFF to control drift): TTFT 119.2 s (off) vs 120.9 s (on) — a tie; decode **0.405 → 0.580 tok/s (+43%)**.
- With the boundary installed, the guard can optionally price the collapsed live set (`OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT`, default off): ~1.1 GB instead of ~26 GB on qwen4_exp.

## Learned pins and cache seeding

- **mlock pins** (`expert_streaming_pins`): observe routing for the first decode calls, then `mlock` the page-aligned file ranges of the most frequent experts per layer within `expert_streaming_pin_gib` (default 1.25). Zero-copy — the locked pages ARE the file cache pages — but they become wired memory. Qwen 48-token decode: **1.538 → 1.764 tok/s (+15%)**.
- **Learned pin store**: the profile is saved on engine stop and reloaded at convert, wiring the hot set from token 1. Measured: TTFT 8.3–8.9 → **6.9 s (−22%)**, decode **+6–10%** with a same-prompt profile. Deterministic uncorrelated warmup measured ~0/negative — only routing-correlated warmup pays.
- **v2 regime profiles**: version, model fingerprint (config sha + source/cold packing + hot fraction + profile format), and separate `decode`/`prefill` frequency tables. The fingerprint gates every load — a mismatch logs and ignores the profile, never applies silently; profiles do not transfer between packings (4S `oQ4e3b` vs 4M `oQ4e4b` reject by design). v1 profiles migrate to the decode regime; the HOBBIT hot set reads the decode regime.
- **Honest negative**: the pin-knee matrix (16 slots/layer — the SCH knee; 4S 1.5 GiB / 4M 2.0 GiB, 3 interleaved reps, own v2 profiles) measured **null (−0.3%)** on both. On this box the page cache already serves the decode working set up to the oracle ceiling; mlock only trades evictable for wired. Pins stay opt-in.
- **Prefill-hotness seed** (`expert_streaming_seed`, default on): a `PrefillHotnessRecorder` accumulates per-layer expert frequency during prefill-sized calls; at the first decode-sized call the cache swaps to the prompt-wide hot set — budget > 0: retain-hot; budget 0 (default): a bounded discarded-read burst into the page cache (`OMLX_EXPERT_STREAMING_SEED_GIB`, 2 GiB cap), async on the warm pool. Measured: 720 slices seeded at the prefill→decode boundary on a 512-token prompt; TTFT 9–11 s for short prompts with ~1008 slices seeded.
- **Profile caveats (measured)**: regimes diverge — a decode-learned hot set shares only 35% of its top-72 experts with a prefill-learned profile; domains diverge more (book×code top-72 overlap 22.4%; a book-learned profile on code costs +2.22% ppl vs +0.39% matched, ~5× the penalty). Learn the profile on the workload you serve. Pins cost ~10% decode in the bench's cold-cache regime (mlock pass + seeding traffic) — the benefit is on the *warm next load*, not the current run.

## Dynamic residency governor

A fixed budget is decided at load and never revisited, but free memory keeps moving. The governor revisits LRU capacity at request boundaries from available memory (`psutil.available`, `vm_stat` fallback):

- free < 10% of RAM → `clear()` the cache (last resort; pages re-read from SSD)
- free < 20% → halve capacity (floor `min_cap`)
- free > 40% → double capacity (ceiling `max_budget`)
- 20–40% → stable (hysteresis; zero resize churn)

30 s cooldown between actions. Opt-in: `expert_streaming_dynamic = true` with budget > 0 (a budget-0 model is page-cache-only by operational choice and stays untouched); growth ceiling `expert_streaming_dynamic_max_gib` (default 6). Runs on the inference thread at the per-request summary point — no race with put/get. State in `expert_streaming_summary` (`governor.actions/last_action/last_free_gib/capacity`); actions log one `expert_streaming governor: ...` line.

Calibration note: under a 4-request bench with a dirty page cache, `available` sits in the 20–40% band and the governor deliberately does nothing — hysteresis working as designed. On an idle machine it grows (3077 → 6162 slots at the first boundary). Thresholds are relative to total RAM, not absolute.

## Cold precision tier and HOBBIT hot/cold split (opt-in)

`tools/requant_cold_tier.py` writes `<model>/expert_cold/`: the full `switch_mlp` expert set requantized at `--bits 3` (or 2) with the source group size, same shard/key names, packing recorded in shard `__metadata__`. Only affine banks with a `.biases` key convert (the bias must ride along). Runtime: `expert_streaming_cold_tier` ("2"–"8") makes the backing resolve expert-bank keys from `expert_cold/` first — slices, coalesced runs, pins and F_RDADVISE all funnel through the same reader — and the converter overrides the streaming linears' bits/group size with the recorded packing. Bytes per token drop 25% (3-bit) or ~50% (2-bit): the direct lever on the I/O floor. Partial tiers are refused (`cold_tier_status`); the UI input is gated on `expert_streaming_cold_tier_present`.

**HOBBIT hot/cold split** (`expert_streaming_hot_fraction`, 0–1): with a cold tier active and a learned profile present, hot experts keep the SOURCE packing (4-bit), everyone else reads the tier; the linear builds one mini-bank per tier with a masked dual `gather_qmm`. Tier-suffixed LRU keys (`#c`) keep the two packings from aliasing; coalesced runs break at tier boundaries.

Quality gates (perplexity harness, fixed corpus):

GLM (24 × 1000-token windows, 23,976 tokens, budget 2 GiB):

| arm | ppl | Δ ppl vs base |
|---|---|---|
| base (oQ4e 4-bit) | **2.2225** | — |
| cold-uniform (3-bit) | 2.5552 | **+14.96%** |
| HOBBIT 0.25 | **2.2505** | **+1.26%** |

Fraction sweep: 0.125 → +3.81%, **0.25 → +1.26%** (91.6% of the penalty recovered — the knee), 0.5 → +0.42%.

Qwen3.8-Flash-Next (12 × 2048-token windows, 24,564 tokens):

| arm | ppl | Δ ppl | tok/s (48-tok decode) | TTFT |
|---|---|---|---|---|
| base (oQ4e 4-bit) | 1.3119 | — | 1.622 | 10.7 s |
| cold3-uniform | 1.3718 | +4.57% | 2.068 (+27%) | 8.5 s |
| **HOBBIT 0.25 @ 3-bit** | **1.3147** | **+0.21%** | **1.804 (+11%)** | 8.4 s |
| cold2-uniform | 1.6998 | +29.6% | 2.322 (+43%) | — |
| **HOBBIT 0.25 @ 2-bit** | **1.3246** | **+0.97%** | **1.955 (+21%)** | — |

Speed A/Bs (short prompt, 48-token decode, idle window): GLM base 0.582/0.575 → cold-uniform 0.708/0.748 (**+22–30%**), HOBBIT 0.25 0.647/0.625 (**+9–11%**, ~84–88% of the uniform tier's speed); GLM TTFT 18.7 → 15.9 s under the uniform tier. Qwen: base 1.436 → cold-3bit 1.575 (**+9.7%**), TTFT 10.8 → 9.5 s.

Decision guidance: the quality/speed Pareto point is model-dependent — HOBBIT 0.25 gives GLM users the near-lossless option (+1.3% ppl for +9–11% decode); on Qwen the 3-bit split is near-lossless (+0.21%) and the 2-bit tier under HOBBIT turns a catastrophic uniform tier (+29.6% ppl) into a usable extreme-speed point (+0.97% ppl for +21%). All tiers stay opt-in; the cold tier was rejected as a default line of development (the near-lossless claim only holds for the split, and only on the smaller-per-expert, wider-routing models).

Domain note: the split is even kinder on code corpora (+0.39% ppl on the project's own sources vs +1.26% on a book), consistent with routing being more skewed on code — but hot sets are domain-specific, so learn the profile on the served workload.

## Approximate routing: top-k truncation and cache-prior

Both levers change outputs by design; both are opt-in per-model settings with quality gates, never defaults.

**Adaptive top-k truncation** (`expert_streaming_topk_threshold`, hooks exist for `qwen4_exp` and `glm5_next`(+`_text`); other types warn and stay exact via `is_topk_applicable`): after top-k selection, keep the smallest score-descending prefix whose relative mass reaches the threshold; dropped slots reuse the top expert (duplicates collapse — no extra I/O); kept scores renormalize to the original mass. `null`/≥ 1.0 bypasses everything — bit-exact by construction.

| threshold | Qwen (cold, 16-tok) | GLM |
|---|---|---|
| exact | ~1.0 | 0.381 |
| 0.85 | 1.091 | **0.485 (+27%)** |
| 0.70 | 1.197 | — |

JANG 4S interleaved A/B (v2 protocol, warmup discarded): base median 2.817 → 0.85 **3.142 tok/s (+11.5%)**; ppl cost +3.9% (4S), +3.0% (4M). GLM-JANG: **+52.2%** at 0.85; 0.70 is a cliff (+16.5% ppl).

**Cache-prior rerank** (`expert_streaming_cache_prior`): a bonus in logit space for experts resident in the LRU before top-k (Qualcomm 2412.00099). The GLM rerank applies the bonus to raw router logits before `group_expert_select`. `0.0`/`null` = exact routing, bit-identical.

Calibration (Qwen-JANG_4M, budget 1 GiB):

| Prior bonus | short tok/s | short hit | 2k tok/s | 2k hit | Output quality |
|---|---|---|---|---|---|
| 0.0 (exact) | 1.88 (×2) | 9.2% | 1.43 | 1.0% | reference |
| 0.5 | 1.76 | 14.4% | — | — | coherent |
| 1.0 | 2.08 (×3) | 19.3% | — | — | coherent |
| 2.0 | **2.78 (×3)** | 38.0% | **1.76** | 17.6% | coherent (short + 2k) |
| 4.0 | 1.01 | 17.7% | — | — | **degenerate output** |

Fidelity cliff between 2.0 and 4.0; recommended 2.0 where quality headroom exists, 1.0 conservative. TTFT is unchanged (the hook only fires in decode); prefill ppl is identical (3.3144 exact and prior, canary windows) — the rerank only bites in decode, where the LRU has something to say. **Budget precondition**: with budget 0 the resident set is empty and the rerank is pure overhead — the converter refuses the prior without an LRU with a warning (autotune re-measures the baseline in that case).

GLM-5.3-Flash-JANG-MTP A/B (after the loader fix that made the raw-transformers export loadable at all): prior 2.0 **1.141 tok/s (×3) vs 1.008 exact (×2) = +13.2%**, hit 8.5% → 25.8%, TTFT ~11 s unchanged. GLM canary ppl identical (5.7644 both arms, nll bit-identical) — the prior reorders I/O, it does not degrade fidelity. Sweep: 1.0 → 1.063 / **2.0 → 1.277** / 3.0 → 1.233 — 2.0 is the GLM optimum too.

**Combinations are additive** (GLM-JANG, budget 1 GiB): topk 0.85 + prior 2.0 = **+81.4%** (1.826 ×3) for the ppl cost of the topk alone (+3.3%) — the truncation cuts bytes/token, the prior reorders the I/O of what remains. With MTP d3 on top: **+99.8%**.

## Multi-token prediction (MTP)

Under streaming, MTP pays only when the draft/verify cycle's extra expert reads cost less than the accepted tokens save.

**When it wins** — Qwen4 Lightning MTP (`qwen4_exp` with `mtp.*` weights; the draft layer carries its own `switch_mlp` bank and streams like any other layer): at QD8 the cycle runs 3.4 target forwards per token but wall/call had dropped to ~4 ms, so **MTP = +27–37% decode** (1.538 → 1.958/2.110; with pins 2.058). At QD16 the edge shrinks to +4–5% (the pool resize cut the per-forward cost MTP amortizes). Draft block sweep (no-MTP 2.135): block 2 → 2.097, 3 → 2.165, 4 → 2.213, 8 → 2.248 — but pins no longer stack with deep drafting (block 8 + pins: 2.012/2.069, reproduced). Guidance: MTP default-off; when enabled, `draft_block_size` 4–8 with pins off — or QD16 alone, which beats MTP alone. **Enable by default only for streaming qwen4_exp checkpoints that ship `mtp.*` weights.**

**GLM-5.3-Flash-JANG-MTP native port** (the checkpoint stores the draft as one extra trunk layer, `model.layers.45.*`, nextn convention):

- sanitize remaps `model.layers.45.*` → `language_model.mtp.0.*`, fuses the draft `kv_b_proj` → `embed_q`/`unembed_out`, and propagates quant overrides (else the 2-bit gs64 draft bank falls to global 8-bit and fails strict load);
- HC identity for the draft (no `attn_hc`/`ffn_hc` weights: parameter-free residual, bit-exact when streams broadcast identically);
- streaming discovers `language_model.mtp` even when `layers` resolves via the root, accepts `mlp` alongside `ffn`, and disables `compile_ffn` on the inner DecoderLayer (`mx.eval` of the plan is illegal under `mx.compile`) — 43 banks converted (42 trunk + 1 draft);
- `--mtp --mtp-depth N` in the bench; with MTP inactive the patch is pass-through.

Measured (GLM-JANG, budget 1 GiB, 3 reps): d1 **+45.7%** (accept 80.8%, 1.85 tok/cycle, draft = 3% of backbone, 32/32 tokens bit-identical to the non-MTP run — sampling-exact by construction); d2 +37.7%, d3 +28.6% (d1 > d2 > d3 on prose: the adaptive controller explores deep drafts that fail past position 1). Combos: d3 + topk 0.85 + prior 2.0 = **+99.8%**.

**Structural negative** — single-stage MTP checkpoints (the upstream `mtp.layers.[0]` + single proposal head layout): depth 2/3 have no stage, and the d1 verify forward (2 positions, distinct expert sets) costs **2.3× the base step's SSD reads** — accepted tokens pay back 1.79× at best. **MTP off is the recommendation under streaming for this layout**; without streaming (resident) the verify is ~1× and MTP would win.

**Depth controller**: the adaptive controller caps draft depth per request; `--mtp-depth` fixes the ceiling in the bench.

**VLM MTP adapter fallback** (`tests/test_vlm_mtp_adapter.py`): absent `mtp_clamp_accept` now means no clamp and absent rollback falls back to the non-MTP step (the chain already provides it) — previously a partial draft rejection crashed the request (`LanguageModel has no mtp_clamp_accept`). No change for models with the hook.

## Reference Mac performance

One box, one protocol, one number per cell. MacBook Pro M4 Pro, 48 GB unified memory, checkpoints on an external ~3.7 GB/s NVMe. Protocol: `bench/bench_expert_streaming.py --budget 0 --single-request --prompt-len short --decode 64`, everything else at the shipped default (QD16, hybrid context, kernel readahead on), measured on build `eaa24ea6`. Without streaming the loader estimates 75–101 GiB resident for these checkpoints — none of the four loads on a 48 GB Mac, so only checkpoints present on the test machine are listed.

| model | RAM after load | session peak | TTFT, short prompt | decode |
|---|---|---|---|---|
| GLM-5.3-Flash-JANG-MTP | 10.37 GiB | 14.58 GiB | 12.72 s | **1.52 tok/s** |
| Qwen3.8-JANG 4S | 6.93 GiB | 11.03 GiB | 10.21 s | **2.60 tok/s** |
| Qwen3.8-JANG 4M | 7.02 GiB | 10.99 GiB | 12.05 s | **2.39 tok/s** |
| DeepSeek-V4-Flash-0731-JANG | 8.18 GiB | 16.86 GiB | 7.99 s | **2.05 tok/s** |

Decode runs to 64 tokens and stops at EOS (4S stopped at 43, 4M at 42); outputs are captured for cross-run comparison. The pattern is the same one the experiment sections measure in detail: GLM decode sits near the SSD random-read ceiling (13 MB experts, 288 routed per layer), while Qwen’s smaller experts ride the page cache — 2.4–2.6 tok/s against GLM’s 1.5 on the same disk.

## Model-family notes

- **GLM-5.3-Flash**: large experts (13 MB) put decode on the I/O floor; QD16 lifted it 0.381 → **0.697 tok/s (+83%)** and TTFT 23 → 18.5 s. Pins are too sparse to pay (1.25 GiB ≈ 4–5 experts/layer of 288); the levers that pay are QD16, topk 0.85 (+27%), the cold tier (+26–33% at +24% ppl) and HOBBIT 0.25 (+9–11% at +1.3% ppl).
- **Qwen3.8-Flash-Next**: PLE N-gram residency is a separate feature ([qwen4-exp-ple.md](qwen4-exp-ple.md)); inter-token reuse is high (SCH 34–74% within a document) so pins/seed pay; HOBBIT 0.25 @ 3-bit is near-lossless.
- **DeepSeek V4 Flash** (`deepseek_v4`): nests the MoE under `layer.ffn` and keeps one routed bank per MTP/DSpark stage under `mtp.<stage>[.block].ffn.switch_mlp` (43 layers + 3 draft stages share the LRU; the stage banks count as expert bytes in the estimate so the admin flags stay accurate). Layers 0..2 are hash-routed (`tid2eid[input_ids]`) — routing untouched, only the banks swap. The GLU activation (`LimitedSwiGLU`) is copied as-is so output stays bit-exact.
  **Spill-stacking load**: the JANG checkpoint stores 256 experts/layer ungrouped (`layers.N.ffn.experts.{i}.w{1,2,3}.*`) — stacking all of them at once peaks ~64 GiB and SIGKILLs a 48 GiB box. The loader stacks **one layer at a time (~2 GiB transients)** into `.omlx_spill/<model>/spill_layer_*.safetensors` beside the checkpoint and reloads via mmap; a manifest (sizes+mtimes) makes second loads spill-hits. Byte-identity of spill vs in-RAM stacking verified (9/9 layer-0 banks identical). Kill-switch `OMLX_DSV4_SPILL=0`; details in `omlx/patches/deepseek_v4/spill.py`. Measured battery (48 GiB box): load **3.4–3.5 s**, **8.1–8.2 GiB** resident (lifetime peak ~17 GiB), TTFT 5.8–6.8 s, decode **1.48–2.61 tok/s**. A resident bit-exact gate is unavailable by construction (the resident model does not fit); the guarantees are spill identity + output sanity.

## Prefill memory safety

- **Streaming transient accounting**: the chunk forward is lazy; every MoE layer's mini-bank stays live until the chunk-end eval. The guard charges the streaming term as `uniq experts/layer ≈ 0.2 × chunk tokens` (measured 0.145 on qwen4_exp) via `backing.streaming_guard_info`, so admission does not green-light 400-token chunks whose real peak is ~26 GB.
- **Live Metal only** for streaming models: evictable file pages of streamed experts are not commitment; the per-chunk transient tracker probes `mx.get_active_memory()` instead of phys footprint (which counted page-cache poisoning at 61 MB/token).
- **Size-invariant floors discounted**: streaming bank transients scale ~linearly with chunk size, so a large chunk's measured max must not floor-limit smaller chunks (the guard would reject every shrunken chunk and the prefill dies).
- **Static-estimate reconciliation**: the static SDPA+KV estimate may only raise the prediction up to **3× the measured** rate once real samples exist (a generic dense formula over-predicts a 4-bit MoE by ~40× and would permanently crush chunks to the floor); with no samples the static is the conservative first-chunk fallback. A restored prior is NOT measurement (samples clamp to 0, raw deltas zero) — the first chunk of a changed regime prices the static, not a stale prior.
- **Allocator-pool release**: releasing the MLX cache pool at chunk tails engages on the multi-request server path; the bench's external prefill path sees the pool near zero at tails because the peaks are intra-chunk allocator high-water. The intra-step bound is the per-layer eval boundary.
- **HOBBIT 8k plateau**: the dual-tier prefill peak (14.23 GiB Metal) is a plateau across the whole lazy chunk, not a step spike; per-tier banks are ~0.2% of the graph; tier-lifetime variants (eval-between-tiers, mask-free reassembly, small-tier-first) all measured within jitter — the only decomposers are the eval boundary and chunk sizing. Within the 28 GiB ceiling on the reference box, the guard throttles 8k chunks to 1024 rows and the run stays swap-free.
- **KV is not the envelope**: qwen4_exp is a hybrid (36×GDN O(1) state; only 12 full-attention layers keep KV ≈ 0.17 GiB at 8k); dsv4 MLA ≈ 0.3 GiB at 8k. The long-context memory that matters is (a) expert mini-bank transients (charged by the guard) and (b) the unreleased MLX allocator pool (bounded by the per-layer eval boundary).

## Analysis and measurement tooling

- `bench/bench_expert_streaming.py` — the single-request memory-aware protocol (`--single-request`: one `stream_chat`, TTFT to first token, decode after; token-ID gate `--gate-tokens` requires non-empty token lists, `bit_exact_kind=tokens`). Arms: model (qwen|glm|dsv4), budget, cold tier, hot fraction, topk, prior, MTP depth, pins. 3 interleaved reps per arm, warm page cache; results carry an immutable `effective_config` block (git sha, model fingerprint, chunk schedule, every knob) and `bench/compare_results.py` refuses any comparison whose critical fields mismatch.
- `bench/bench_expert_batch.py` (`--concurrency K`) and `bench/bench_multi_request.py` (concurrency × budget matrix) — batched decode.
- `bench/cache_cool.py` (`--gb N`) evicts the page cache between runs without root; `bench/resource_sampler.py` samples GPU/CPU/disk/RSS per phase.
- `OMLX_EXPERT_STREAMING_TRACE=<path>` + `bench/lrc_analysis.py` — routing trace and the LRC metrics (SCH = Belady oracle hit rate per cache size; SRP = fixed-group coverage). Measured (Qwen3.8-JANG 4S/4M, both quants within 0.3pp — routing is a model property, not a quant property): decode SCH ceiling **77.5–77.8%** (knee S=16 at 65%, saturation by 64); adjacent-call repeat ~38.5%; SRP(G=64) demand coverage ~90%; prefill SCH ceiling 50% (near-broadcast union — caching prefill is structurally capped, the seeded burst is the right mechanism). The SCH curve is the sizing answer for any future device-side residency scheme — not a page-pin budget (the pin-knee matrix refuted that).
- `bench/ppl_expert_streaming.py` — token NLL/perplexity over a local corpus via disjoint context windows; `--streaming` loads through the streaming engine; `--cold-tier`, `--topk`, `--prior` select arms. Streaming compute is bit-exact vs resident (test-pinned), so a resident measurement represents the streamed path.
  **Affine-tile race at T ≥ 1024** (found by this harness): the GLM q8 affine prefill tile intermittently produced garbage at T ≥ 1024 (first symptom: streaming ppl ~28k, uniform logits). The scheduler's bank guard steps prefill chunks to ≤ 512 tokens so the production path never routes it; a guard in the vendored `linear.py` blocks the tile at T ≥ 1024 (falls back to `mx.quantized_matmul`); q8 indexer behavior below 1024 is test-pinned. The underlying race stays open; the guard is containment.
- Read-stat vocabulary (`read_stats` stage buckets, profiled only): `worker_start_delay_us` (submit → worker start), `read_duration_us` (inside the read syscall), `window_wait_us` (caller blocked on window futures), `last_future_wait_us` (final-run tail), `component_e2e_us` (what the model feels). Renames are caught by snapshot tests.
- Per-backing phase telemetry (`ReadTelemetry`), run-pool concurrency (`RunPoolTelemetry`, owner-tagged), and `MemTracer` (phase/request context, per-(layer, proj) event ordering) are bounded and profile-gated — zero cost when off.
- Discipline: A/B only in idle windows; size ceilings to *available* RAM, not capacity; never compare TTFT across PROFILE on/off arms (the `effective_config` block fails such pairs loudly).

## Autotuner

`bench/autotune_expert_streaming.py` turns the hand-run A/Bs into an automated, safety-railed search of the knobs that measured meaningful deltas: budget, IO depth, coalescing, readahead, seed (all bit-exact) — plus `topk`/`prior`/`cold tier` only with explicit flags (`--sweep-topk`, `--sweep-prior`, `--sweep-cold-tier`, the latter needs `expert_cold/` present), because they trade output fidelity.

A session (~1.5–2 h for the default shape):

1. **Probe** (no model load): RAM/available/swap; the enforcer's static + Metal ceilings; sequential and random-expert-size read bandwidth on the model's own shard (a near-saturated random probe prunes the QD sweep).
2. **Calibration** (discarded): default config at the screening context (2k/32 tokens) — warms the page cache and measures the loaded footprint used by every preflight.
3. **Screening**: one-factor-at-a-time trials scored against the calibration reference (TTFT 50% + decode 50%, minus a penalty for observed swap growth); budget candidates filtered by memory actually left after load + reserve.
4. **Head-to-head + 8k validation**: the screening winner vs the default at the long context — a winner that regresses there is not recommended.
5. **Recommendation**: writes `recommendation.json` (every trial row, the winning config, machine probe numbers) under `bench/results/autotune/<model>_<stamp>/` on the machine that ran it.

Memory safety is the design constraint: per-trial ceiling = `min(static/metal cap, available − reserve)` sized to *available*; a watchdog samples every 2 s (swap growth > 2 GiB or available < 5 GiB twice → SIGKILL the trial, record a safe failure, raise the reserve); trials are skipped (not failed) when memory cannot hold the loaded runtime + reserve; the session aborts after two consecutive kills. Nothing runs on its own — a human launches it; `--dry-run` prints the probe and trial plan.

`--apply` persists the winning knobs into the model's per-model settings (the same store the UI edits) and the measured LRU knee. Apply with the server stopped (a running server keeps its own in-memory settings manager).

```bash
.venv/bin/python bench/autotune_expert_streaming.py --model qwen --dry-run
.venv/bin/python bench/autotune_expert_streaming.py --model qwen
.venv/bin/python bench/autotune_expert_streaming.py --model qwen --apply
```

## What does not work

Measured negative — kept as documentation, the code is not shipped:

- **Router-lookahead prefetch (PILOT-style staging)**: strictly negative on a saturated SSD — 0.037 → 0.011 tok/s in the original A/B (94% of staged bundles dropped, ~380 GB wasted reads), and −35% with the cold tier active. The kernel-side `F_RDADVISE` readahead keeps the useful half of the idea (zero-copy, mildly positive).
- **Warm-only next-layer prefetch**: neutral to negative (1.531 vs 1.538 alone; drags pins 1.764 → 1.624) — the demand path already saturates the NVMe.
- **Speculative stash ring**: decode 1.92 vs ~3.0 tok/s on the same window — speculative reads saturate the shared IO pool even from a warm page cache.
- **Slot-bank (device-side expert residency)**: +0.4% (null) on the frozen A/B at S=16; the full-hit fast path fired on only 4.9% of calls (27% at S=128, ~6.4 GiB wired) because partial hits still pay promote+stack, and Belady's oracle (65% at S=16 vs LRU's 49%) is unreachable by eviction policy.
- **Full-bank external mmap wrap** (ported with credit from jundot/omlx PR #3437 by @alytaphoenix): serving every prefill `gather_qmm` from mmap'd repacked banks measured **~12× worse TTFT** (420.8 vs 34.9 s) and −65% decode — the GPU faults 46.5 GiB of artifact pages at the sparse-fault rate (~0.35 GB/s) instead of the pread rate (2.3–10.6 GB/s). Reverted, not parked. One durable salvage: the external-wired accounting contract (discount once at the sample site, active term only, caches store pre-discounted) is the right wiring for any future external-mmap surface.
- **Batched decode with distinct prompts**: per-step working set scales ~linearly with N while the SSD stays saturated — 2 requests halve per-request throughput; batch pays only when requests share routing.
- **SpecPrefill + streaming**: anti-synergistic — sparse-selected tokens spread routing 4× across the bank (2k TTFT 25.7 → 93.8 s cold). SpecPrefill is a resident-model lever.
- **ANE prefill under streaming**: prefill is GPU-bound (81–85%); ANE touches dense MLPs/GDN projections, not expert QMM — ~3% at best. Keep ANE for resident models.
- **Dual-SSD striping with mismatched disks**: supported mechanically (`bench/stripe_model.py` + `OMLX_EXPERT_STREAMING_EXTRA_ROOTS`), but pays only with a second *fast* disk; striping a 3.7 GB/s NVMe with an 875 MB/s one measured ~1.85× slower than the primary alone.
- **Gap-bridged coalesced reads**: merging small gaps into one pread loses TTFT in both regimes (see I/O pipeline).

## Recommendations

| hardware | budget | pins | topk | MTP |
|---|---|---|---|---|
| 48 GB+, model >> RAM | default (0 / auto) | optional (+15% Qwen) | 1.0 exact | on for qwen4_exp with `mtp.*` |
| 48 GB+, quality flexible | default | optional | 0.85 | on (qwen4_exp with `mtp.*`) |
| 16 GB-class | default / auto | `pins=on` | 0.85 | on (qwen4_exp with `mtp.*`) |

GLM on the I/O floor: uniform 3-bit tier (+26–33% decode, +24% ppl) or HOBBIT 0.25 (+9–11%, +1.3% ppl) — both opt-in. Qwen: HOBBIT 0.25 @ 3-bit is near-lossless (+11% decode, +0.21% ppl); add cache-prior 2.0 for +10.8% where quality headroom exists.

## Limitations

- Streaming models serve one request at a time (continuous batching is disabled for the streaming model).
- TTFT is high on GLM-class models (prefill faults ~all experts once); repeated prompts are covered by the existing paged SSD KV cache.
- Pins pay only on Qwen-like routing (high inter-token reuse) and not when the page cache already saturates the working set; profiles are fingerprint-gated and do not transfer across packings, regimes, or domains.
- The widened four (`qwen3_moe`, `qwen2_moe`, `deepseek_v3`, `glm4_moe`) are conversion-tested on fake checkpoints; bit-exact validation on real multi-GB weights is pending.
- The GLM affine-tile race at T ≥ 1024 is guarded, not fixed.
- macOS has no multi-range read syscall with holes — singleton read coalescing beyond adjacent runs is not available (io_uring-style batched preadv does not exist there).

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25% hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
- Paper survey & gap analysis (2024–26 MoE-offloading literature vs this implementation): [expert-streaming-papers.md](expert-streaming-papers.md).
