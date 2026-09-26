# MoE Expert Offload: run MoE models larger than your memory

For Mixture-of-Experts models, most parameters sit idle on any given token —
only the routed experts do work. Expert offload keeps a configurable fraction
of each layer's experts resident in a fixed slot cache and streams the rest
**from the checkpoint's own safetensors** on demand (mmap slab reads — no
converted copy, no extra disk). Routing is computed exactly as shipped: a
cache miss changes *when* an expert's weights are read, never *which* expert
runs, so accuracy is preserved by construction and the entire cost is
latency.

Measured on `gemma-4-26b-a4b-it-4bit` (30 MoE layers × 128 experts), loaded
through the batched engine's own path:

| | fully resident | offload @ 25% |
|---|---|---|
| load peak memory | 14.28 GB | **4.69 GB** |
| steady memory after generation | 14.20 GB | **4.57 GB** |
| greedy outputs vs resident | — | **bit-identical** (test prompts) |

The load peak is the important number: the load stays lazy and the stock
expert modules are dropped **before** anything materializes them, so the full
expert set is never in memory at any point — which is what lets a model
larger than physical memory load at all.

## Enabling it

Per model, in the admin dashboard: **Model Settings → MoE Expert Offload**,
with a resident-fraction field accepting 5% to 95%, including fractional percentages such as 12.5%. The API accepts any fraction in (0, 1]. Values outside the UI range are preserved until the field is edited. The following example uses the settings API.

```json
{"moe_expert_offload_enabled": true, "moe_expert_offload_resident_fraction": 0.25}
```

Toggling triggers an engine reload (it is a load-time transform). The env
kill switch `OMLX_MOE_EXPERT_OFFLOAD=0` disables it regardless of settings.

Two env vars tune the reader, and neither changes what is computed:

| variable | default | effect |
|---|---|---|
| `OMLX_MOE_OFFLOAD_IO_WORKERS` | 12 | threads reading missing experts. `1` or less (or an unparseable value) keeps the serial path and starts no threads |
| `OMLX_MOE_OFFLOAD_IO_BATCH` | `4 x workers` | experts whose reads may be in flight at once — the bound on the host memory the pipeline holds ahead of the slot writes |

### Sizing a residency to your memory

Under the residency field the dashboard offers the largest residency that
admission accepts within the memory ceiling as **Fit to memory** (for
example `41.4% resident · ~99.2 GB`); clicking it fills the field. It is a
whole number of experts per layer, so it round-trips through the setting
exactly. `GET /api/models` carries it per model, along with the admission
size of the 12.5%, 25%, 50% and 75% residencies and whether each fits:

```json
{
  "moe_expert_offload_presets": [
    {"fraction": 0.125, "bytes": 3906250000, "fits": true},
    {"fraction": 0.25, "bytes": 5312500000, "fits": true}
  ],
  "moe_expert_offload_fit_fraction": 0.4140625,
  "moe_expert_offload_fit_bytes": 106500000000
}
```

`moe_expert_offload_fit_fraction` is `null` when the ceiling is unknown or
when even the routing floor exceeds it. Programmatically, the engine pool's
`moe_offload_admission_bytes(entry, settings, fraction)` and
`fit_moe_offload_fraction(entry, settings, budget_bytes)` run the same
arithmetic admission runs, for every adapter; the common adapter's
`fit_resident_fraction(model_path, full_size, budget_bytes)` in
`omlx.patches.moe_expert_offload` answers the same question from the shard
headers alone. The budget doubles as the ceiling the SSD fallbacks (Qwen4
PLE, DeepSeek V4.1 Engram) decide against, so a table that stops fitting at
some residency is priced at its mmap size from there on, as admission would.

## Performance

`gemma-4-26b-a4b-it-4bit`, 585-token prompt, 256 generated tokens, warm
cache (second request; the cold first request additionally pays the initial
fill):

| residency | memory after generation | decode tok/s | TTFT | per-request hit rate |
|---|---|---|---|---|
| 100% (resident) | 14.20 GB | 122.5 | 0.30 s | — |
| 50% | 7.78 GB | 59.4 | 2.9 s | 0.89 |
| 25% | 4.57 GB | 40.1 | 9.6 s | 0.67 |
| 12.5% | 2.96 GB | 29.3 | 18.1 s | 0.45 |

Decode throughput degrades gracefully. TTFT was the pain point at low
residency in the first version: a long prefill routes to most experts per
layer, and chunking the prefill by tokens re-fetched an expert in every
chunk that touched it, evicting on the way. Over-capacity prefill is now
chunked on expert boundaries instead — the routes are sorted by expert and
each chunk holds every route of up to `capacity` distinct experts, the same
shape as the DeepSeek V4.1 adapter's sorted prefill — so each expert is read
at most once per layer per model call. Measured with
`benchmarks/moe_offload_prefill_bench.py` on the same model (585-token
prompt, 32 decode tokens, single runs; warm = second identical request, cold
= first request after load; filesystem page-cache state is not controlled).
Fetch counts are sampled at the first yielded token and include the decode
step mlx-lm runs ahead of that yield, rather than measuring pure prefill:

| residency | expert fetches through first token, before → after | TTFT warm, before → after | TTFT cold, after | decode tok/s |
|---|---|---|---|---|
| 50% | 6,073 → 1,719 | 2.38 s → 0.69 s | 1.64 s | 60.6 |
| 25% | 30,302 → 2,675 | 8.87 s → 0.85 s | 1.51 s | 43.5 |
| 12.5% | 64,369 → 2,913 | 16.60 s → 0.97 s | 11.92 s | 31.7 |

Decode is untouched by the change (it takes the no-sync fast path). The
remaining follow-up is decode prefetch (layer L+1's fetches during layer
L's compute), which has measured LRU→optimal headroom of +17pp hit rate at
low residency.

A call's misses are read in parallel with `os.pread` on a shared thread pool. `ensure()` schedules missing experts before the serial install loop. Slot writes, LRU updates, and hit/miss counters stay on the calling thread.

## Supported models

The experimental toggle is available for `deepseek_v41`, `deepseek_v4`, `qwen4_exp`, `qwen3_5_moe` (Qwen3.5/3.6), `gemma4` MoE, `olmoe`, `glm_moe_dsa`, and `glm5_next` checkpoints whose expert tensor layout passes validation. Dense Gemma models and other model types do not show the toggle. The settings API and model loader use the same eligibility check.

Qwen3.5/3.6 supports stacked expert projections under `language_model.model.layers.*.mlp.switch_mlp` and `model.layers.*.mlp.switch_mlp`. The offload adapter resolves the original checkpoint path after the loader normalizes the text-only naming layout.

The common adapter supports stacked `[num_experts, ...]` quantized
`SwitchGLU` projections and the per-expert layout used by OLMoE conversions.
Both are read from either an MXFP checkpoint or a community `mlx_lm` affine
conversion, whose projections carry a U32 weight with float scales and biases
and declare their format in the checkpoint's own `quantization` dict.
All backbone layers must have the expected tensor names, shapes, storage
dtypes, and quantization metadata. Fused or renamed projections, missing
experts, unquantized weights, and per-expert linear bias are rejected.
DeepSeek V4.1, the GLM-5.x flagship (`glm_moe_dsa`), and the DeepSeek V4 /
GLM-5.3-Flash block (`deepseek_v4`, `glm5_next`) have their own adapters,
described below. Their eligibility validation checks the stacked
`ffn.switch_mlp` / `mlp.switch_mlp` slabs the adapters read positionally.

When offload wraps layers, the Qwen gate/up fusion is skipped automatically:
fusion rewrites stock expert weights in RAM, which cannot apply to experts
that are never materialized.

## Why not pin the "hot" experts instead?

Pinning a fixed expert subset looks like a cheaper version of the same idea
and is the design to avoid: measured with usage-calibrated pins (the strong
form), zeroing the experts outside the pinned half costs ~91% of gsm8k
accuracy, because multi-step generation compounds per-token errors — the top
half of experts carries ~80% of routing decisions, and losing the other 20%
of decisions is catastrophic, not proportional. Fetch-on-miss keeps the
computation exact and pays in latency; pinning silently changes what the
model computes. (Full measurement record:
[clausius FINDINGS](https://github.com/beatakouchnir/clausius/blob/main/FINDINGS.md).)

## Verifying behavior (and how not to)

Do not acceptance-test offload by diffing outputs across residency settings.
Cache capacity changes gather/reduction order, so greedy outputs can fork at
marginal token choices mid-generation even though the computation is
semantically exact — deterministic at any fixed setting, paraphrase-level,
never at token 0. The valid comparison is behavioral: labeled accuracy at
sufficient n, or a paired per-token-entropy comparison on unlabeled prompts.
Measured at 25% residency on gemma-4-26b (60 mixed prompts, greedy,
1536-token cap): 57/60 generations bit-identical to resident, 3
paraphrase-level forks, paired-entropy verdict clean, and labeled gsm8k
(n=200) statistically indistinguishable (McNemar p = 0.61). The test suite
(`tests/test_moe_expert_offload*.py`) encodes exactly this policy: bit-exact
where the kernel path is identical, rounding-bounded where it is not.


## DeepSeek V4.1

DeepSeek V4.1 uses its own expert adapter and loader. The original checkpoint,
oMLX converted MXFP/oQ checkpoints, and community `mlx_lm` affine conversions
are supported. Expert weights stay in the existing safetensors files; an
affine source reports its format in the checkpoint's `quantization` dict and
counts its packed weight plus BF16 scales and biases toward the resident set.
The resident fraction applies to the routed experts in each backbone layer,
with capacity floored at the number selected by one token. Shared experts,
attention, and other backbone weights remain resident.

Non-resident experts are read with positional `pread` calls on a small
reader pool of their own, not through the Engram row-gather mapping: an
expert is megabytes of contiguous bytes, and a faulting `MADV_RANDOM` gather
reads it one page at a time. A residency update starts the misses' reads
ahead of the installs, at most 512 MiB of payload in flight, and installs
them serially in the order the misses were seen, so eviction victims, hit
and miss counts, and resident bytes are identical to a serial fetch. Sorted
prefill routes are chunked on expert boundaries (every route of up to
`capacity` distinct experts per chunk), so a prefill reads each expert once
per layer and runs one kernel per chunk.

Measured on a synthetic checkpoint with the oQ3e expert geometry (384
experts, 3-bit affine, 14.8 MiB per expert, 4 layers, random weights),
cold reads from the internal SSD of an M5 Max, 12.5% residency:

| | mmap gather (before) | positional reads |
|---|---:|---:|
| decode, one token, per MoE layer | 362 ms | 9.1 ms |
| expert fetch throughput | 0.23 GB/s | 9.3 GB/s |
| sorted prefill, 256 tokens | 33 token-layers/s | 568 token-layers/s |

These are single runs of adapter-level calls on synthetic weights; they
exclude attention, Engram, and the rest of the forward.

### Measured on a 128 GB Mac

`Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp` on an M5 Max with 128 GB and the
internal SSD (`iogpu.wired_limit_mb` unset), Engram on SSD, native kernels
built, run with `benchmarks/deepseek_v41_offload_bench.py` on a 433-token
prose prompt in one prefill chunk followed by 64 greedy tokens. Single runs:

| residency | experts per layer | load | Metal active | peak footprint | prefill | decode | decode hit rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.5% | 48 | 3.9 s | 38.0 GiB | 49.6 GiB | 28 tok/s | 5.6 tok/s | 0.69 |
| 25% | 96 | 4.5 s | 65.7 GiB | 77.4 GiB | 25 tok/s | 4.2 tok/s | 0.78 |

Both settings produce coherent, on-topic continuations. Served through
`omlx serve` with the same settings (discovered from the HF cache, Engram
forced to SSD by admission, 12.5% residency, engine load 4.4 s), two
64-token chat completions ran at 2.7 tok/s on cold expert slots and
4.1 tok/s after. Prefill reads every
expert the prompt routes to once per layer (about 226 of 384 per layer for
this prompt, 130 GiB in total) at 8 to 9 GB/s. Decode is bound by miss
latency at one to two misses per layer per token. The lower residency
decodes faster: the RAM the resident slots do not take is used by the page
cache, which serves repeated misses far faster than the SSD (6.5 GB/s
effective at 12.5% against 3.4 GB/s at 25%). On a 128 GB machine 12.5% is
the better default. Expect the page cache to take all remaining RAM during
a run; it is reclaimable and is not part of the Metal working-set limit.
Higher residencies fit the limit on paper (`fit_resident_fraction` reports
41% at 107.5 GiB) but leave no headroom for the KV cache and prefill
transients, and were not measured.

### Sizing on a 128 GB Mac

For `Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp`, the shard headers give 221.5 GiB
of routed experts, 91.9 GiB of Engram tables, 7.5 GiB of DSpark draft
weights (not loaded under offload), and 10.1 GiB of everything else. With
Engram on SSD the resident set is:

| resident fraction | experts per layer | resident weights |
|---:|---:|---:|
| 12.5% | 48 | 38 GiB |
| 25% | 96 | 65 GiB |
| 33.3% | 128 | 84 GiB |
| 37.5% | 144 | 93 GiB |

The Metal working-set limit on a 128 GB machine with `iogpu.wired_limit_mb`
unset is about 107 GiB, and KV cache, prefill transients, and the Engram
page cache share it. `admission_bytes(path, fraction)` and
`fit_resident_fraction(path, budget_bytes)` in
`omlx.patches.deepseek_v41.moe_offload` give the engine pool's admission
estimate for a fraction and the largest fraction whose estimate fits a byte
budget; the dashboard's residency field offers the fit (see *Sizing a
residency to your memory* above).

For a 384-expert checkpoint, 12.5% keeps 48 experts per layer. The adapter
preserves V4.1's activation quantization, clamped SwiGLU, and application of
routing weights before the down projection. Large routed batches are split
into bounded chunks; kernel rounding may differ from a fully resident run.

Enable `moe_expert_offload_enabled` and set
`moe_expert_offload_resident_fraction` to `0.125` in model settings. Engram
storage is independent: `deepseek_v41_engram_ssd_offload` can be enabled at
the same time. Memory admission, loaded-model accounting, and unload targets
include the expert savings and the selected Engram storage mode.

MoE offload cannot be combined with Lightning MTP (including DSpark), VLM
MTP, or DFlash. Disable these before enabling offload. Settings and runtime
validation reject conflicting combinations. V4.1 offload skips retained
DSpark tensors even when the checkpoint includes them; the checkpoint is
not modified. Disable offload and reload to use MTP again.

GLM-5.3-Flash (`glm5_next`) and the custom DeepSeek V4 expert kernels are not
supported by this adapter. Unmatched modules remain resident, and unsupported
custom model families receive no offload admission discount.

## GLM-5.x flagship (glm_moe_dsa)

GLM-5.2 and GLM-5.3 (`model_type: glm_moe_dsa`: 78 layers, the first 3 dense,
256 routed experts, top-8, one shared expert) run their routed experts
through oMLX's own GLM SwitchGLU, which fuses the gate and up projections at
load and, on sorted calls, returns the routes already weighted and summed by
the native `glm_moe_weighted_sum` kernel. The adapter in
`omlx/patches/glm_moe_dsa/moe_offload.py` keeps that module and its kernels:
each projection's parameters are replaced by resident slots before lazy
weights materialize, expert ids are translated to slot ids, and the module's
own forward runs unchanged on them. Every kernel decision inside it is a
function of the routes and the slot tensors' shapes, and every use of an
index is a gather, so a route computes the same numbers against its slot as
against its expert; the test suite asserts bit-exact output against the
resident module for decode, for sorted prefill through the native weighted
sum, and for sorted calls that fit the cache, and a rounding-scale bound for
over-capacity prefill, which is chunked on expert boundaries like the other
adapters. A miss reads the expert's split `gate_proj`, `up_proj` and
`down_proj` slabs from the checkpoint, stacked or one tensor per expert (the
layout the loader stacks at load), and writes the two halves of the fused
row in place.

Sizing, from the shard headers of `mlx-community/GLM-5.2-4bit` (409 GiB
estimated, about 5.2 GiB of routed experts per layer, 20 MiB per expert):

| resident fraction | experts per layer | admission estimate |
|---:|---:|---:|
| 12.5% | 32 | 76.8 GiB |
| 20% | 51 | 105.0 GiB |
| 25% | 64 | 124.3 GiB |

On a 128 GB machine (about 107 GiB working-set limit) 20% is the largest
residency admission accepts; the routing floor is 8.

Measured on an M5 Max 128 GB (internal SSD), `mlx-community/GLM-5.2-4bit`
loaded through the engine's own sequence, a 36-token chat prompt, 24 greedy
tokens, single runs:

| residency | experts per layer | footprint | load | cold TTFT | decode | decode hit rate | misses per token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | 51 | 86.6 GiB | 3.3 s | 20.4 s | 1.8 tok/s | 0.63 | 213 |
| 12.5% | 32 | 58.4 GiB | 2.6 s | 18.8 s | 1.5 tok/s | 0.52 | 275 |

Decode is bound by expert reads: a miss is a 20 MiB expert, and 213 of them
per token is 4.2 GiB, which the positional reads move at about 8 GiB/s. The
same 20% configuration fetching through the common store's memmap path
measured 646 s to first token and 0.09 tok/s, because a slab faulted in
16 KiB pages on the compute thread reads at 0.4 GB/s; that is why this
adapter reads misses through the store's positional reads on the shared
reader pool, as the V4.1 adapter does. Continuations were coherent and identical between the cold and warm
runs at each residency; between residencies the two texts forked at one
marginal token (a capitalization), which is the documented behavior of
over-capacity prefill under a different chunking.

`omlx serve` end to end on the same machine (balanced memory guard, which
puts the dynamic ceiling at 102.5 GB): the checkpoint is discovered from the
HF cache; at 20% the request is refused with 507 because the 105.0 GB
estimate does not fit that ceiling (the aggressive tier, or raising
`iogpu.wired_limit_mb`, would admit it); at 12.5% it is admitted at a
76.9 GB estimate (57.6 GB actual), the engine loads in 3.7 s with 75 layers
wrapped and gate/up fusion skipped, and two 24-token chat completions took
35.7 s and 34.0 s including prefill.

Set `moe_expert_offload_enabled: true` and
`moe_expert_offload_resident_fraction` in the model's experimental settings.
Lightning MTP, VLM MTP, and DFlash must be disabled. Loaded-model memory
accounting includes the expert savings.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next checkpoints with `model_type: qwen4_exp` and stacked
quantized `switch_mlp` projections use the common expert adapter. At 12.5%
residency, a 512-expert checkpoint keeps 64 experts per layer; routing still
selects the checkpoint's original top 10 experts per token. Shared experts
remain resident, and large routed batches are processed in bounded chunks.

Set `moe_expert_offload_enabled: true` and
`moe_expert_offload_resident_fraction: 0.125` in the model's experimental
settings. PLE SSD offload (`qwen4_ple_ssd_offload`) is independent and can be
enabled alongside expert offload. The PLE automatic fallback decision and
loaded-model memory accounting include expert savings without counting them
twice. Lightning MTP, VLM MTP, and DFlash must be disabled. Checkpoints that
include MTP weights can still be used; inactive MTP weights are omitted by
the existing Qwen loader.
