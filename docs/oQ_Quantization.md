# oQ: oMLX Universal Dynamic Quantization

**Quantization should not be exclusive to any particular inference server.** oQ produces standard mlx-lm compatible models that work everywhere, no custom loader required. oMLX, mlx-lm, and any app that supports MLX safetensors format.

## Table of Contents

- [Overview](#overview)
- [Quantization Levels](#quantization-levels)
- [Pipeline Architecture](#pipeline-architecture)
- [Mixed-Precision Bit Allocation](#mixed-precision-bit-allocation)
- [AWQ Weight Equalization](#awq-weight-equalization)
- [Clip Optimization](#clip-optimization)
- [Streaming Quantization](#streaming-quantization)
- [Calibration Data](#calibration-data)
- [Supported Models](#supported-models)
- [oQ vs AutoAWQ](#oq-vs-autoawq)

## Overview

oQ is a mixed-precision post-training quantization system for Apple Silicon. Instead of applying the same bit width to every layer, oQ analyzes each tensor's role and sensitivity to decide the optimal bit allocation automatically.

Four techniques work together:

1. **Mixed-precision predicate.** Per-tensor bit assignment based on 34+ rules covering tensor role, layer position, and measured sensitivity
2. **AWQ weight equalization.** Per-channel scaling between adjacent layers to make weight distributions more quantization-friendly
3. **Per-expert activation-aware scaling.** MoE expert-specific optimization using routed token activations
4. **Clip optimization.** MSE-based per-group weight clipping range search

## Quantization Levels

| Level | Base Bits | Target bpw | Description |
|-------|-----------|------------|-------------|
| oQ2 | 2 | ~2.3 | Extreme compression |
| oQ3 | 3 | ~3.5 | Balanced |
| oQ3.5 | 3 | ~3.9 | Quality balanced (expert down_proj 4-bit) |
| oQ4 | 4 | ~4.5 | Recommended |
| oQ5 | 5 | ~5.5 | High quality |
| oQ6 | 6 | ~6.5 | Near-lossless |
| oQ8 | 8 | ~8.0 | Near-lossless |

Target bpw is higher than the base bits because sensitive layers, embeddings, and special components receive higher bit allocations automatically.

### Quantization Modes

oQ selects the optimal format per tensor:

| Bits | Format | Group Size | Notes |
|------|--------|-----------|-------|
| 4 | mxfp4 | 32 | Microscaling FP4 with uint8 scales |
| 8 | mxfp8 | 32 | Microscaling FP8 with fp16 scales |
| 2, 3, 5, 6 | affine | 64 | Zero-point quantization with fp16 scale + bias |

## Pipeline Architecture

oQ has two execution paths depending on configuration.

<details>
<summary><b>Enhanced path</b> (oQ4+ or clip enabled)</summary>

```
Phase 1    Load model
           ├─ VLM: mlx-vlm.load_model(lazy=True) + mlx-lm tokenizer
           └─ LLM: mlx-lm.load()

Phase 1.5  Weight equalization + Sensitivity measurement
           ├─ Load calibration data (built-in code_multilingual)
           ├─ Per-layer loop:
           │   ├─ AWQ grid search (Norm→Linear pairs)
           │   │   └─ duo_scaling: scales = x_mean^r / (w_mean^(1-r) + 1e-4)
           │   │   └─ 10 ratios tested, apply only if MSE improves
           │   ├─ Per-expert activation-aware scaling (MoE only)
           │   │   └─ Router forward → token routing → per-expert intermediate activation
           │   │   └─ Grid search per expert's up_proj→down_proj
           │   └─ Sensitivity measurement (relative MSE)
           │       └─ Quantize→dequantize→forward→MSE / output_magnitude
           └─ Output: sensitivity_map + equalized weights (in-place)

Phase 2    Clip optimization
           ├─ Per-layer forward pass with calibration data
           └─ Per-group weight clipping search (MSE minimization)

Phase 3    Quantize (mlx-lm.quantize_model)
           └─ universal_quant_predicate with sensitivity_map

Phase 4    Save
           ├─ VLM: mlx-vlm.save_weights() (preserves vision)
           └─ LLM: mlx-lm.save()
```

</details>

<details>
<summary><b>Streaming path</b> (oQ4, no clip)</summary>

```
Phase 1    Load tensors from safetensors (lazy/mmap)
           └─ Apply sanitize chain (name transforms, fused weight splits)

Phase 1.5  Sensitivity measurement (lazy model load)
           ├─ Load model lazily → layer-by-layer calibration forward
           ├─ Per-layer relative MSE measurement (quantize→dequantize→compare)
           ├─ Build sensitivity_map → inject into config for predicate
           └─ Free model (del + mx.clear_cache)

Phase 2    Per-tensor quantization
           ├─ Dtype normalization (config.torch_dtype)
           ├─ For each tensor:
           │   ├─ _get_predicate_bits() → (bits, gs, mode) ← uses sensitivity_map
           │   ├─ mx.quantize() in original dtype (no bf16→fp16 conversion)
           │   └─ mx.save_safetensors() (preserves all dtypes)
           └─ Flush shards at 5GB boundary

Phase 3    Write config.json + index + copy tokenizer files
```

</details>

## Mixed-Precision Bit Allocation

oQ protects critical components at higher bit widths while compressing less sensitive parts aggressively.

### Always Preserved (fp16/fp32)

| Component | Reason |
|-----------|--------|
| MoE router weights (`mlp.gate`, `.router`) | Routing precision |
| Vision encoder (VLM) | Image quality |
| SSM state parameters (Mamba/RWKV) | State precision |

### Higher Bits Than Base Level

| Component | Bits | Condition |
|-----------|------|-----------|
| `lm_head` | 6 | Always |
| `v_proj` | 6 | Sensitive layers |
| `o_proj` | 5 | Dense models |
| `q_proj`, `k_proj` | 5 | Sensitive layers |
| Embeddings | base+2 | Always (error propagates to all layers) |
| MLA projections (DeepSeek) | 6 | Always |
| Shared expert gate/up | 6 | Always |

### Sensitivity Detection

oQ measures each layer's quantization sensitivity using relative MSE:

```
relative_mse = mse(float_output, quantized_output) / mean(float_output²)
```

Normalizing by output magnitude prevents later layers from appearing artificially sensitive due to residual stream accumulation. The top 25% most sensitive layers receive higher bit protection.

This is data-driven. For example, on Qwen3.5-35B-A3B, layer 10 ranks as the most sensitive, something position-based heuristics (first/last N layers) would never catch.

## AWQ Weight Equalization

Weight equalization inserts per-channel scaling factors between adjacent layers to make weight distributions more quantization-friendly. The scaling cancels out mathematically so the model output is identical:

```
Original:   Y = LayerNorm(X) @ W
Equalized:  Y = (LayerNorm(X) / s) @ (W * s)  =  same output
```

oQ uses the AutoAWQ duo_scaling formula with grid search over 10 ratios. Scaling is only applied when it measurably reduces quantization error.

### Scale Pairs

| Pair | Source | Target | Channel Dim |
|------|--------|--------|-------------|
| 1 | input_layernorm | q_proj, k_proj, v_proj | hidden_size |
| 2 | v_proj | o_proj | head_dim x n_heads |
| 3 | post_attention_layernorm | gate_proj, up_proj | hidden_size |
| 4 | up_proj | down_proj | intermediate_size |

### Per-Expert Scaling (MoE)

For MoE models, oQ additionally performs per-expert activation-aware scaling. It routes calibration tokens through the MoE router, gathers per-expert activations, and optimizes scaling factors for each expert independently on the fused tensor without unfusing.

## Clip Optimization

After equalization, clip optimization finds the optimal per-group weight clipping range by grid searching clip ratios from 0.5 to 1.0, minimizing the MSE between original and quantized outputs.

Available for oQ4 and below (4-bit or less). Multiple sublayers with the same shape are batched together for GPU efficiency.

## Streaming Quantization

For large models (70B+), oQ offers a streaming path that processes tensors one at a time via safetensors mmap.

- Memory usage stays at ~3-4 GB regardless of model size
- No full model load required
- All tensors saved in original dtype (no bf16-to-fp16 conversion)
- Shards flushed at 5 GB boundary

## Calibration Data

oQ ships with built-in calibration datasets. No download required.

| Dataset | Composition | Samples |
|---------|-------------|---------|
| `code_multilingual` (default) | Code + English + Korean + Chinese + Japanese + tool calling | 512 |
| `code` | Code + English | ~300 |
| `multilingual` | English + Korean + Chinese + Japanese | ~285 |

## Supported Models

| Architecture | Equalization | Per-Expert Scaling |
|-------------|-------------|-------------------|
| Llama, Qwen, Mistral (dense) | Full | N/A |
| Qwen MoE, MiniMax MoE | Norm-to-Linear | Yes |
| DeepSeek V2/V3 (MLA) | Partial | Partial |
| GatedDeltaNet (hybrid SSM) | Norm-to-attention | Yes |
| VLM (Qwen-VL, etc.) | Full | Model-dependent |

## oQ vs AutoAWQ

oQ borrows the equalization math (duo_scaling formula) from AutoAWQ, but the overall system is different.

| | AutoAWQ | oQ |
|---|---|---|
| **Bit allocation** | Uniform 4-bit | Mixed-precision (34+ rules, sensitivity-driven) |
| **Sensitivity** | None | Relative MSE, top 25% get higher bits |
| **Quantization mode** | Affine only | mxfp4 + mxfp8 + affine auto-selected |
| **Per-expert scaling** | Unfused tensors (PyTorch) | Fused tensor indexing (MLX) |
| **Output format** | AutoAWQ-specific | Standard mlx-lm |
| **Runtime** | PyTorch + CUDA | MLX + Apple Silicon |
