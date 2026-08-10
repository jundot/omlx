# DFlash-MLX Integration Report

Date: 2026-07-28

## Overview

DFlash is a block diffusion speculative decoding technique (arXiv:2602.06036) that accelerates LLM token generation by having a small draft model propose multiple tokens simultaneously, which the target model verifies in a single forward pass. The MLX implementation ([bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx)) has been integrated into oMLX as an experimental engine option.

---

## Architecture

### How DFlash works

```
1. PREFILL: target model processes entire prompt, captures hidden states
2. DRAFT:   draft model generates block of 16 tokens in parallel (block diffusion)
3. VERIFY:  target model verifies all 16 in one forward pass
4. ACCEPT:  greedy prefix match — longest matching prefix is committed
5. REPLAY:  cache rollback via tape replay for hybrid (GatedDeltaNet) models
6. REPEAT:  until max_tokens or EOS
```

Key distinction from traditional speculative decoding: the draft model uses **block diffusion** (parallel denoising) rather than autoregressive token-by-token drafting, allowing all 16 tokens to be proposed simultaneously.

### oMLX integration

```
API Request → server.py → engine_pool.py
                              │
                              ├─ dflash_max_ctx unset, or prompt below limit
                              │     └─ DFlashEngine
                              │           └─ stream_dflash_generate()  [dflash-mlx]
                              │                 └─ draft/verify loop (internal)
                              │
                              └─ configured limit reached
                                    └─ fallback engine (BatchedEngine / VLMBatchedEngine)
                                          └─ BatchGenerator + paged cache + SSD cache
```

DFlashEngine is a `BaseEngine` implementation that:
- Loads target + draft models via `dflash_mlx.runtime.load_target_bundle()` / `load_draft_bundle()`
- Consumes structured events from `stream_dflash_generate()` (prefill, token, summary)
- Bridges sync generation to async streaming via `asyncio.Queue`
- Lazily replaces DFlash with a fallback engine when a configured context
  limit is reached

---

## Current implementation

### Files

| File | Role |
|------|------|
| `omlx/engine/dflash.py` | DFlashEngine class — BaseEngine impl, event consumer, fallback routing |
| `omlx/patches/dflash_laguna.py` | Laguna target adapter, gated drafter, fused-QKV loader, and mixed-cache rollback |
| `omlx/engine/__init__.py` | DFlashEngine export (required dependency) |
| `omlx/engine_pool.py` | DFlash routing: checks `dflash_enabled` before engine type switch |
| `omlx/model_settings.py` | Per-model settings: `dflash_enabled`, `dflash_draft_model`, `dflash_draft_quant_bits` |
| `omlx/admin/routes.py` | Admin API: settings CRUD + `requires_reload` on dflash changes |
| `omlx/admin/templates/dashboard/_modal_model_settings.html` | UI: toggle, draft model dropdown, quantization selector |
| `omlx/admin/static/js/dashboard.js` | Frontend settings binding |
| `omlx/admin/benchmark.py` | Batch test skip guard for DFlashEngine |
| `tests/test_dflash_engine.py` | DFlash engine and routing tests |
| `tests/test_dflash_laguna.py` | Laguna adapter parity, cache rollback, config, and checkpoint-layout tests |

### Dependency

- `dflash-mlx` pinned to `jundot/dflash-mlx` (v0.1.10+omlx.4)
- Listed as required dependency in `pyproject.toml`; the mac-app release
  lockfiles are regenerated from it by the packaging pipeline

### Supported models

DFlash registers `QwenGdnTargetOps`, `Gemma4TargetOps`, and `MuseGlimmerTargetOps`. oMLX also registers a Laguna backend and the `DFlashLagunaForCausalLM` drafter used by Poolside's official checkpoints:

| Target model | Draft checkpoint |
|--------------|-----------------|
| Qwen/Qwen3-4B | z-lab/Qwen3-4B-DFlash-b16 |
| Qwen/Qwen3-8B | z-lab/Qwen3-8B-DFlash-b16 |
| Qwen/Qwen3.5-4B | z-lab/Qwen3.5-4B-DFlash |
| Qwen/Qwen3.5-9B | z-lab/Qwen3.5-9B-DFlash |
| Qwen/Qwen3.5-27B | z-lab/Qwen3.5-27B-DFlash |
| mlx-community/Qwen3.5-27B-8bit | z-lab/Qwen3.5-27B-DFlash |
| mlx-community/Qwen3.5-27B-4bit | z-lab/Qwen3.5-27B-DFlash |
| Qwen/Qwen3.5-35B-A3B | z-lab/Qwen3.5-35B-A3B-DFlash |
| mlx-community/Qwen3.5-35B-A3B-4bit | z-lab/Qwen3.5-35B-A3B-DFlash |
| Qwen/Qwen3.6-27B | z-lab/Qwen3.6-27B-DFlash |
| Qwen/Qwen3.6-35B-A3B | z-lab/Qwen3.6-35B-A3B-DFlash |
| google/gemma-4-31b-it | z-lab/gemma-4-31B-it-DFlash |
| google/gemma-4-26b-a4b-it | z-lab/gemma-4-26B-A4B-it-DFlash |
| poolside/Laguna-XS-2.1 | poolside/Laguna-XS-2.1-DFlash |
| poolside/Laguna-XS-2.1-NVFP4-mlx | poolside/Laguna-XS-2.1-DFlash-NVFP4 |
| poolside/Laguna-S-2.1 | poolside/Laguna-S-2.1-DFlash |
| poolside/Laguna-S-2.1-NVFP4-mlx | poolside/Laguna-S-2.1-DFlash-NVFP4 |
| meta-models/Muse-Glimmer-30B | meta-models/Muse-Glimmer-30B-assistant |

Other model families (Llama, Gemma3, etc.) are not supported — they require both a trained DFlash draft checkpoint and a compatible target adapter in dflash-mlx.

Laguna target and draft checkpoints must be from the same size family and should
use Poolside's quantization-matched draft when one is published (for example,
`Laguna-S-2.1-DFlash-NVFP4` with the NVFP4 target). A checkpoint that explicitly
declares a different precision from the target may reduce acceptance; issue
#2398 motivates checking this, but does not isolate pairing as the sole cause.
The engine warns only when both target and draft expose contradictory precision
metadata, and shows the warning in the dashboard together with acceptance and
separate accepted-draft/output tokens-per-cycle counters. A generic `-DFlash`
suffix is not treated as proof of a BF16-only draft. Poolside also publishes
INT4/FP8 drafters; their vLLM-format
targets are not yet validated in oMLX. The adapter validates target
depth, hidden size, and capture-layer IDs at load time. It implements Laguna's
per-head/per-element softplus attention gating, partial RoPE,
per-captured-layer RMS normalization, Poolside's fused `qkv_proj` checkpoint
layout, mixed full/sliding target caches, and rejection rollback. DDTree
verification, target KV quantization, and the specialized verify-linear path
are deliberately disabled for Laguna until they have dedicated
numerical-parity coverage; ordinary adaptive DFlash verification remains
available.

Note: the `-DFlash` suffix is specific to DFlash draft checkpoints. Gemma4 also ships an `-assistant` variant (e.g. `gemma-4-26B-A4B-it-assistant`) that targets MTP speculative decoding via mlx-vlm — do not mix these in the DFlash toggle. Meta breaks this naming convention: `Muse-Glimmer-30B-assistant` IS a DFlash drafter (block-diffusion, `block_size` 16), not an MTP checkpoint. Drafter routing therefore keys on `config_model_type` (`muse_glimmer_assistant` is in the dashboard's DFlash drafter set), not on the checkpoint name. The Muse Glimmer target is a VLM: DFlash drives its text backbone through dflash-mlx's bundled text-only mlx-lm module, and image requests divert to the VLM fallback engine as usual.

### Per-model settings

| Setting | Type | Description |
|---------|------|-------------|
| `dflash_enabled` | bool | Enable/disable DFlash for this model |
| `dflash_draft_model` | str | Path or HuggingFace repo for draft checkpoint |
| `dflash_draft_quant_enabled` | bool | Draft model quantization enabled |
| `dflash_draft_quant_weight_bits` | int | Draft model quantization weight bits |
| `dflash_draft_quant_activation_bits` | int | Draft model quantization activation bits |
| `dflash_draft_quant_group_size` | int | Draft model quantization group size |
| `dflash_max_ctx` | int or null | Optional prompt-token threshold for batched fallback (`null` = unlimited) |
| `dflash_in_memory_cache` | bool | Enable DFlash L1 prefix snapshots |
| `dflash_ssd_cache` | bool | Enable DFlash L2 snapshot spill |

Configured via web admin UI → Model Settings → Experimental Features → DFlash.

---

## Generation flow

### DFlash path

1. `DFlashEngine.stream_generate()` tokenizes prompt
2. Submits to MLX executor thread via `_run_generate_streaming()`
3. Calls `stream_dflash_generate()` from dflash-mlx
4. dflash-mlx internally handles:
   - Target model prefill + hidden state capture
   - Draft model block diffusion (16 tokens per cycle)
   - Target model verification (single forward pass)
   - Greedy/temperature acceptance matching
   - Tape-based cache rollback for hybrid models (RecurrentRollbackCache)
5. omlx consumes structured events:
   - `"event": "token"` → decode with `NaiveStreamingDetokenizer` → SSE chunk
   - `"event": "summary"` → log metrics (tok/s, acceptance ratio, cycles)
6. EOS tokens filtered from output

### Configured context fallback

1. `DFlashEngine.stream_generate()` detects prompt length exceeds threshold
2. Delegates entire request to `_fallback_engine.stream_generate()`
3. DFlash weights are evicted and BatchedEngine or VLMBatchedEngine starts lazily
4. Full omlx features available: paged cache, SSD cache, prefix cache, continuous batching

### Non-streaming

`DFlashEngine.generate()` uses `generate_dflash_once()` for non-streaming requests with the same fallback logic.

---

## Temperature sampling

### Implementation (fork patch)

The original dflash-mlx uses greedy argmax only. Our fork (`jundot/dflash-mlx@8e1df22`) adds `sample_with_temperature()`:

```python
def sample_with_temperature(logits, temperature, suppress_token_mask=None):
    if temperature < 1e-5:
        return greedy_tokens_with_mask(logits, suppress_token_mask)  # greedy
    scaled = logits / temperature
    return mx.random.categorical(scaled).astype(mx.uint32)           # stochastic
```

Applied to all three sampling points: prefill first token, draft block, and verify posterior.

### Behavior

- **temp=0**: identical to original greedy behavior. Every emitted token = target model's argmax. Lossless, bit-for-bit reproducible.
- **temp>0**: both draft and verify use temperature sampling. Acceptance is still prefix-match based, so acceptance rate drops (draft and target are less likely to agree on stochastic samples). Speed benefit is reduced but diversity is achieved.

### Paper reference

The DFlash paper (arXiv:2602.06036) evaluates both temperature=0 (4.9x speedup) and temperature=1 (4.1x speedup) on H200 GPUs, confirming the algorithm supports non-greedy sampling.

---

## Constraints and limitations

### 1. Single-request engine

DFlashEngine processes one request at a time. No continuous batching — the entire GPU is dedicated to a single draft/verify loop. For concurrent users, requests are serialized on the MLX executor thread.

**Trade-off**: on Apple Silicon with low concurrency, speculation can outweigh
batching when acceptance is high; measure the actual target/draft pair and
workload rather than assuming a fixed speedup.

### 2. Context length limit

DFlash effectiveness degrades with long contexts:
- Verify pass attention cost grows with KV cache size
- `dflash_max_ctx` defaults to unlimited
- Setting a threshold enables automatic fallback to BatchedEngine/VLMBatchedEngine

### 3. Model support

Qwen, Gemma4, and Laguna have compatible target adapters and published draft checkpoints. Each additional model family still requires:
- A trained DFlash draft checkpoint (block diffusion model matching target hidden dimensions)
- Support in dflash-mlx's target model handling (hidden state extraction, cache rollback)

### 4. Memory overhead

DFlashEngine loads both target and draft models simultaneously:
- Draft model: typically ~1B parameters (small relative to target)
- Draft int4 quantization available to reduce footprint
- The fallback engine is loaded only after DFlash weights are evicted

### 5. Separate prefix cache

DFlashEngine does not use omlx's paged KV block cache. It has a separate
dflash-mlx snapshot cache: optional L1 memory entries and L2 SSD spill.

When context fallback is configured, the batched engine provides oMLX's paged
and SSD block cache after the switch.

### 6. No batch benchmark

Admin panel benchmark's batch throughput test is skipped for DFlashEngine since it requires scheduler core access (`engine._engine`) that DFlashEngine doesn't expose. Single-request benchmark tests work normally.

### 7. Greedy verification with temperature

When temperature > 0, acceptance rate drops because draft and target independently sample from the logit distribution. The accepted tokens are always valid target model samples at the given temperature, but fewer draft tokens get accepted per cycle, reducing the speed benefit.

---

## Fallback mechanism

```
DFlashEngine.start()
  ├── load target model (dflash-mlx)
  └── load draft model (dflash-mlx)

Request arrives:
  ├── no configured limit, or prompt below it → DFlash path
  └── configured limit reached
        ├── evict DFlash target + draft
        └── lazily start VLMBatchedEngine or BatchedEngine

DFlashEngine.start() fails:
  └── engine_pool catches exception → creates VLMBatchedEngine or BatchedEngine directly
```

### Engine pool priority

DFlash check runs **before** engine type routing in `_load_engine()`. If `dflash_enabled=True` and `dflash_draft_model` is set, DFlashEngine is created regardless of whether the model would normally be VLM or LLM. On failure, falls back to the model's natural engine type.

---

## Configuration reference

### Environment variables (dflash-mlx)

| Variable | Default | Description |
|----------|---------|-------------|
| `DFLASH_VERIFY_LEN` | block_size | Cap on verify block length |
| `DFLASH_DRAFT_SINK` | 64 | Draft KV cache sink size |
| `DFLASH_DRAFT_WINDOW` | 1024 | Draft KV cache window size |
| `DFLASH_QUANTIZE_DRAFT` | false | Enable draft int4 quantization |

### Admin UI settings

Located in Model Settings → Advanced Settings → Experimental Features → DFlash:
- **Toggle**: enable/disable DFlash
- **Draft Model**: dropdown of available models
- **Draft Quantization**: Disabled (default)
  - **Weight Bits**: 2-bit / 4-bit (default) / 8-bit
  - **Activation Bits**: 16-bit (default) / 32-bit
  - **Group Size**: 32 / 64 (default) / 128

### Logging

DFlash generation completion is logged at INFO level:
```
DFlash generation complete: 502 tokens, 45.3 tok/s, acceptance=87.2%, cycles=38
```

Context fallback is logged:
```
DFlash context fallback: 5120 >= 4096, evicting dflash models and switching to vlm engine
```

---

## Testing

### Unit tests

- ModelSettings: default values, serialization roundtrip, removed field handling
- DFlashEngine: properties, stats, cache stats
- EnginePool routing: disabled/enabled/draft model checks
- Laguna: native-forward parity, hidden-state capture, full/rotating-cache rollback, gated draft forward, target binding, and fused-QKV checkpoint loading

### Manual testing

1. Enable DFlash in admin UI for a supported model
2. Set the matching draft model path (for example, `poolside/Laguna-S-2.1-DFlash-NVFP4` for an NVFP4 Laguna S target)
3. Reload model
4. Send short prompt → verify DFlash logs (acceptance ratio, tok/s)
5. Configure `dflash_max_ctx`, then send a prompt at or above it → verify fallback logs

---

## Future work

- **Upstream sync**: merge temperature patch to bstnxbt/dflash-mlx, update pin
- **Broader model support**: as dflash-mlx adds new model families, omlx gets support automatically
- **Adaptive fallback**: evaluate switching to plain decoding when measured speculation is consistently unprofitable
- **Performance coverage**: add real Laguna matched-pair tests across short and long contexts
