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
the fetch churn up front. That is also the clearest follow-up: v1 fetches
synchronously on miss, while prefill's full expert-access schedule is
computable *before* any fetch (run the router over the whole prompt — no
prediction needed), and decode prefetch (layer L+1's fetches during layer
L's compute) has measured LRU→optimal headroom of +17pp hit rate at low
residency.

## Supported models

Targets any model whose MoE layers use upstream mlx-lm `SwitchGLU` with
quantized expert projections (~40 model files: Qwen MoE families, Gemma
MoE, Mixtral, Llama-4, Kimi, OLMoE, gpt-oss, and more), in both checkpoint
layouts found in the wild — stacked `[num_experts, ...]` tensors, and the
older one-tensor-per-expert layout that mlx-lm's `sanitize()` stacks at
load. End-to-end verified on `gemma-4-26b-a4b-it-4bit` (stacked) and
`OLMoE-1B-7B-0125-Instruct-4bit` (per-expert; 3.89 GB → 1.17 GB at 25%
residency, generations bit-identical). Every layer is verified against the
checkpoint (tensor names — all experts individually in the per-expert
layout — shapes, storage dtypes, quantized projections) before wrapping;
anything unmatched — non-quantized projections, fused `gate_up_proj`,
per-expert `bias`, renamed projections such as Mixtral's `w1/w2/w3`,
unknown quantization formats — is skipped with a logged reason and runs
resident as before, so the failure mode for an unverified family is "no
offload", never "wrong outputs". DeepSeek-V4 and GLM-5.2 use oMLX's native
switch kernels and are a planned follow-up on the same store.

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
