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
with a resident-fraction selector (12.5% – 75%). Or via the settings API:

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

Decode throughput degrades gracefully; TTFT is the pain point at low
residency, because a long prefill routes to most experts per layer and pays
the fetch churn up front. A call's misses are therefore read *in parallel*:
`ensure()` first classifies every expert the call needs without touching the
cache, starts the missing experts' reads on a shared pool (`os.pread`, so a
read needs nothing but a descriptor and an offset), and only then runs the
serial install loop, which takes bytes from the pipeline instead of reading
them itself. The slot writes, the LRU order, the eviction victims and the
hit/miss counters all stay on the calling thread in the serial order, so the
pipeline changes nothing but *when* the bytes arrive.

The remaining headroom is scheduling, not IO width: prefill's full
expert-access schedule is computable *before* any fetch (run the router over
the whole prompt — no prediction needed), and decode prefetch (layer L+1's
fetches during layer L's compute) has measured LRU→optimal headroom of
+17pp hit rate at low residency.

## Supported models

The experimental toggle is available for `deepseek_v41`, `qwen4_exp`,
`gemma4` MoE, and `olmoe` checkpoints whose expert tensor layout passes
validation. Dense Gemma models and other model types do not show the toggle.
The settings API and model loader use the same eligibility check.

The common adapter supports stacked `[num_experts, ...]` quantized
`SwitchGLU` projections and the per-expert layout used by OLMoE conversions.
All backbone layers must have the expected tensor names, shapes, storage
dtypes, and quantization metadata. Fused or renamed projections, missing
experts, unquantized weights, and per-expert linear bias are rejected.
DeepSeek V4.1 has a separate adapter described below. DeepSeek V4 and
GLM-5.3 are outside the current support list.

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

DeepSeek V4.1 uses its own expert adapter and loader. Both the original
checkpoint and oMLX converted MXFP/oQ checkpoints are supported. Expert
weights stay in the existing safetensors files. The resident fraction applies
to the routed experts in each backbone layer, with capacity floored at the
number selected by one token. Shared experts, attention, and other backbone
weights remain resident.

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

GLM-5.3-Flash and the custom DeepSeek V4 expert kernels are not supported by
this adapter. Unmatched modules remain resident, and unsupported custom
model families receive no offload admission discount.

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
