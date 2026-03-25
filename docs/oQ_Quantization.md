# oQ: oMLX Universal Dynamic Quantization

Quantization should not be exclusive to any particular inference server. oQ produces standard mlx-lm compatible models that work everywhere — oMLX, mlx-lm, and any app that supports MLX safetensors. No custom loader required.

**oQ is a data-driven mixed-precision quantization system for Apple Silicon.** Instead of assigning bits by fixed rules or tensor type, oQ measures each layer's actual quantization sensitivity through calibration and allocates bits where the data says they matter most.

### Benchmarks (Qwen3.5-35B-A3B)

<table>
<tr>
<th rowspan="2">Benchmark</th><th rowspan="2">Samples</th>
<th colspan="2" align="center">2-bit</th>
<th colspan="2" align="center">3-bit</th>
<th colspan="2" align="center">4-bit</th>
</tr>
<tr>
<th>mlx-lm</th><th>oQ</th>
<th>mlx-lm</th><th>oQ</th>
<th>mlx-lm</th><th>oQ</th>
</tr>
<tr><td>MMLU</td><td>300</td><td>14.0%</td><td><b>64.0%</b></td><td>76.3%</td><td><b>85.0%</b></td><td>79.7%</td><td><b>83.3%</b></td></tr>
<tr><td>TRUTHFULQA</td><td>300</td><td>17.0%</td><td><b>80.0%</b></td><td>81.7%</td><td><b>86.7%</b></td><td>87.7%</td><td><b>88.0%</b></td></tr>
<tr><td>HUMANEVAL</td><td>164 (full)</td><td>0.0%</td><td><b>78.0%</b></td><td>84.8%</td><td><b>86.6%</b></td><td><b>87.2%</b></td><td>85.4%</td></tr>
<tr><td>MBPP</td><td>300</td><td>0.3%</td><td><b>63.3%</b></td><td>69.0%</td><td><b>72.0%</b></td><td>71.7%</td><td><b>74.3%</b></td></tr>
</table>

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

oQ and oQ+ share the same levels. oQ+ adds GPTQ-based weight optimization before quantization.

## Pipeline

### oQ+ (Enhanced)

```
1. Load model (full)
2. Measure per-layer sensitivity (relative MSE)
3. Build budget plan (sensitivity-driven bit allocation)
4. GPTQ weight optimization (all quantizable weights)
5. Quantize with mixed-precision predicate
6. Save
```

The GPTQ step uses Hessian-based error compensation to optimize rounding decisions for every quantizable weight in the model. For MoE models, this includes all routed expert weights (typically 90%+ of total parameters), which are processed using a batched algorithm that handles all experts in a layer simultaneously.

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
| MoE router | 8-bit |
| shared_expert_gate | 8-bit |
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

## GPTQ Weight Optimization

oQ+ uses an optimized implementation of GPTQ (Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) to improve quantization quality without changing the output format or inference speed.

### How It Works

Standard quantization rounds each weight to the nearest quantization grid point. GPTQ takes a smarter approach: it processes weights column by column, and when rounding one column introduces error, it adjusts the remaining columns to compensate. The adjustment direction is guided by the inverse Hessian of the calibration inputs, which captures how each weight column affects the layer's output.

```
For each column i:
    q[i] = round_to_grid(w[i])
    error = (w[i] - q[i]) / H_inv[i, i]
    w[i+1:] -= error * H_inv[i, i+1:]    # compensate remaining columns
```

The result is the same 4-bit quantized format — identical structure, identical inference speed — but with rounding decisions that minimize actual output error rather than per-element error.

### MoE Batched Processing

In MoE models, routed experts make up 90%+ of all parameters. Processing them one at a time would take hours. oQ solves this with batched expert GPTQ: all experts in a layer share the same Hessian (since they receive the same input hidden states), so the column-by-column optimization can run on all experts simultaneously as a single batched operation.

For Qwen3.5-35B-A3B (256 experts × 40 layers):
- Per-expert sequential: ~90 minutes
- Batched: **~6 minutes** (15x speedup, identical results)

### Calibration-Aware Bits

The GPTQ optimization uses the actual target bits assigned by the sensitivity budget plan. If a tensor is boosted to 6-bit, the error compensation optimizes for 6-bit quantization boundaries — not the base 4-bit. This eliminates the mismatch between optimization and final quantization.

### Weight Integrity

Unlike smoothing-based methods that modify normalization weights to redistribute quantization difficulty, oQ's GPTQ implementation only adjusts the rounding of weights that will be quantized. Non-quantized weights (norms, biases) remain untouched, preserving the model's original computation graph.

## Streaming Quantization

For large models (70B+), the streaming path processes tensors one at a time via safetensors mmap.

- No full model instantiation.
- Shards flushed at 5 GB boundary.
- Non-quantized float32 weights cast to bfloat16 for inference parity.
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

| Architecture | GPTQ Optimization | Notes |
|-------------|-------------------|-------|
| Qwen3.5 MoE (hybrid attn) | Full (batched experts) | Validated with benchmarks |
| Qwen3.5 dense (hybrid attn) | Full | Same hybrid handling |
| MiniMax-M2.5 MoE | Full | Per-expert dense GPTQ |
| GLM MoE | Full | Fused expert support |
| Step-3.5 MoE | Full | `moe.*_proj` fused support |
| Nemotron-Cascade MoE | Full | Per-expert dense GPTQ |
| Llama, Mistral, dense models | Full | Standard layer structure |
| VLM models | Full (text) | Vision weights kept fp16 |

### Streaming Path (oQ)

All models supported by mlx-lm/mlx-vlm. No architecture restrictions.

## Acknowledgments

oQ's weight optimization is based on the GPTQ algorithm by Frantar et al. The batched expert processing and MoE-aware Hessian sharing are oQ-specific optimizations. Sensitivity-driven budget allocation was inspired by approaches in [llm-compressor](https://github.com/vllm-project/llm-compressor) and [GGUF K-quants](https://github.com/ggml-org/llama.cpp).
