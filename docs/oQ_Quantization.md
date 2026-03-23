# oQ: oMLX Universal Dynamic Quantization

Quantization should not be exclusive to any particular inference server. oQ produces standard mlx-lm compatible models that work everywhere — oMLX, mlx-lm, and any app that supports MLX safetensors. No custom loader required.

**oQ is a data-driven mixed-precision quantization system for Apple Silicon.** Instead of assigning bits by fixed rules or tensor type, oQ measures each layer's actual quantization sensitivity through calibration and allocates bits where the data says they matter most.

### Benchmarks (Qwen3.5-35B-A3B)

| Benchmark | Samples | oQ2 | mlx-lm 2-bit | oQ3 | mlx-lm 3-bit | oQ4 | mlx-lm 4-bit |
|-----------|---------|-----|-------------|-----|-------------|-----|-------------|
| MMLU | 300 | **64.0%** | 14.0% | **85.0%** | 76.3% | **83.3%** | 79.7% |
| TRUTHFULQA | 300 | **80.0%** | 17.0% | **86.7%** | 81.7% | **88.0%** | 87.7% |
| HUMANEVAL | 164 (full) | **78.0%** | 0.0% | **86.6%** | 84.8% | 85.4% | **87.2%** |
| MBPP | 300 | **63.3%** | 0.3% | **72.0%** | 69.0% | **74.3%** | 71.7% |

## Quantization Levels

| Level | Base Bits | Target bpw | Description |
|-------|-----------|------------|-------------|
| oQ2 | 2 | ~2.9 | Extreme compression |
| oQ3 | 3 | ~3.5 | Balanced |
| oQ3.5 | 3 | ~3.8 | Quality balanced |
| oQ4 | 4 | ~4.6 | Recommended |
| oQ5 | 5 | ~5.5 | High quality |
| oQ6 | 6 | ~6.5 | Near-lossless |
| oQ8 | 8 | ~8.6 | Near-lossless |

Base format is affine quantization (group_size=64) for all levels except 8-bit, which uses mxfp8 (group_size=32).

oQ and oQ+ share the same levels. oQ+ adds AWQ weight equalization before quantization.

## Pipeline

### oQ+ (Enhanced)

```
1. Load model
2. Measure per-layer sensitivity (relative MSE on original weights)
3. Build budget plan (sensitivity-driven bit allocation)
4. AWQ weight equalization (uses actual target bits from plan)
5. Quantize with mixed-precision predicate
6. Save
```

Sensitivity is measured before AWQ so the budget plan knows the true per-layer importance. AWQ then uses the plan's assigned bits for each tensor during its grid search, ensuring the equalization is optimized for the actual quantization configuration.

### oQ (Streaming)

```
1. Load tensors via mmap
2. Apply model sanitize
3. Measure per-layer sensitivity (temporary model load)
4. Build budget plan
5. Per-tensor quantize + shard flush
6. Save config + tokenizer
```

## Bit Allocation

### Mandatory Protection (Always Applied)

| Tensor | Treatment |
|--------|-----------|
| lm_head | 8-bit (within budget) |
| MoE router | fp16 |
| shared_expert_gate | fp16 |
| Vision encoder | fp16 |
| SSM state params | fp32 |

### Sensitivity-Driven Allocation (oQ2-oQ6)

This is the core differentiator of oQ. Instead of fixed tier systems that assign bits by tensor type, oQ runs actual calibration inference through the model and measures where quantization error hurts the most:

```
sensitivity = MSE(float_output, quantized_output) / mean(float_output²)
```

Normalizing by output magnitude prevents later layers from appearing artificially sensitive due to residual accumulation.

The sensitivity score determines the boost tier:

| Sensitivity Ratio | Boost | Example (oQ4) |
|-------------------|-------|---------------|
| Top (≥50% of max) | base+4 | 4 → 8 bit |
| High (≥20% of max) | base+2 | 4 → 6 bit |
| Moderate (<20%) | base+1 | 4 → 5 bit |

Boosts apply only to non-expert tensors. Routed experts (93-98% of MoE params) stay at base bits — not by rule, but because their byte cost relative to quality gain makes them poor candidates in the budget optimization.

The budget plan ensures total bpw stays within the target and hard cap for each level. The result is that every model gets a different bit allocation tailored to its specific layer sensitivities, rather than a one-size-fits-all profile.

### Minimal Protection (oQ8)

No budget plan. Position-based heuristics only:

- lm_head: 6-bit
- SSM output: 8-bit
- Embedding: base+2
- Sensitive layers (first/last 12.5%): base+1
- Everything else: base

## AWQ Weight Equalization

AWQ inserts per-channel scaling factors between adjacent layers. The math cancels out so the model output is unchanged:

```
Y = (LayerNorm(X) / s) @ (W * s) = LayerNorm(X) @ W
```

Scale is computed using the duo_scaling formula with grid search over 20 ratios:

```
scales = x_mean^r / (w_mean^(1-r) + 1e-4)
scales = scales / sqrt(max(scales) * min(scales))
```

Scaling is only applied when it measurably reduces quantization error.

### AWQ Pairs

Each smooth layer appears in exactly one pair to prevent double-scaling:

| Pair | Smooth Layer | Balance Layers |
|------|-------------|---------------|
| 1 | input_layernorm | q/k/v_proj (self_attn) OR in_proj_* (linear_attn) |
| 2 | v_proj | o_proj (skipped for GQA mismatch) |
| 3 | post_attention_layernorm | All gate/up_proj (shared + routed, single pair) |
| 4 | up_proj | down_proj (per component: shared expert, routed expert, dense) |

For hybrid attention models (Qwen3.5), `is_linear` determines which attention pair to generate per block. Norm is never scaled twice.

### Dtype Preservation

AWQ scales are computed in float32 but applied back in the original weight dtype (typically bfloat16). This prevents float32 promotion that would double the scale/bias overhead in the output.

## Streaming Quantization

For large models (70B+), the streaming path processes tensors one at a time via safetensors mmap.

- No full model instantiation.
- Shards flushed at 5 GB boundary.
- All tensors saved in original dtype.
- Sensitivity measurement requires temporary model load (peak memory ≈ model size).

## Calibration Data

Built-in calibration dataset shipped with oQ. No download required.

600 samples across 7 categories, ~726 KB total:

| Category | Samples | Composition |
|----------|---------|-------------|
| code | 200 | Python classes, imports, JS snippets (avg 26 lines) |
| en | 150 | Wikipedia + C4 web text + OpenOrca conversations |
| ko | 60 | Wikipedia |
| zh | 50 | Wikipedia |
| ja | 60 | Wikipedia |
| tool_calling | 40 | Function call patterns |
| reasoning | 40 | GSM8K, chain-of-thought |

Code samples include real-world patterns (class definitions, import chains, multi-language) rather than benchmark-only code. Reasoning category covers mathematical and step-by-step inference, which is absent from typical calibration sets.

## Supported Models

### Enhanced Path (oQ+)

| Architecture | AWQ Equalization | Notes |
|-------------|-----------------|-------|
| Qwen3.5 MoE (hybrid attn) | Full | is_linear-aware pair generation |
| Qwen3.5 dense (hybrid attn) | Full | Same hybrid handling |
| MiniMax-M2.5 MoE | Full | block_sparse_moe support |
| DeepSeek V3.2 MoE (MLA) | MLP only | MLA attention pairs not yet implemented |
| GLM MoE (MLA) | MLP only | Same MLA limitation |
| Llama, Mistral, dense models | Full | Standard pair structure |
| Devstral-2 (Ministral3) | Full | Standard dense transformer |
| VLM models | Full (text) | Vision weights kept fp16 |

### Streaming Path (oQ)

All models supported by mlx-lm/mlx-vlm. No architecture restrictions.

### AWQ Not Yet Supported

These models can be quantized with oQ (streaming) but AWQ equalization is not available:

| Architecture | Reason |
|-------------|--------|
| Nemotron-H | Non-standard structure (single `mixer` module, no layernorm split) |

## Acknowledgments

oQ's AWQ weight equalization is based on the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) duo_scaling formula. The MoE pair structure and single-smooth-layer constraint were validated against [llm-compressor](https://github.com/vllm-project/llm-compressor) by the vLLM project.
