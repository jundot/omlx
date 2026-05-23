# DFlash-MLX Integration Report

Date: 2026-04-14

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
                              ├─ prompt < DFLASH_MAX_CTX (4096)
                              │     └─ DFlashEngine
                              │           └─ stream_dflash_generate()  [dflash-mlx]
                              │                 └─ draft/verify loop (internal)
                              │
                              └─ prompt >= DFLASH_MAX_CTX
                                    └─ fallback engine (BatchedEngine / VLMBatchedEngine)
                                          └─ BatchGenerator + paged cache + SSD cache
```

DFlashEngine is a `BaseEngine` implementation that:
- Loads target + draft models via `dflash_mlx.runtime.load_target_bundle()` / `load_draft_bundle()`
- Consumes structured events from `stream_dflash_generate()` (prefill, token, summary)
- Bridges sync generation to async streaming via `asyncio.Queue`
- Holds an internal fallback engine for long-context requests

---

## Current implementation

### Files

| File | Role |
|------|------|
| `omlx/engine/dflash.py` | DFlashEngine class — BaseEngine impl, event consumer, fallback routing |
| `omlx/engine/__init__.py` | DFlashEngine export (required dependency) |
| `omlx/engine_pool.py` | DFlash routing: checks `dflash_enabled` before engine type switch |
| `omlx/model_settings.py` | Per-model settings: `dflash_enabled`, `dflash_draft_model`, `dflash_draft_quant_bits` |
| `omlx/admin/routes.py` | Admin API: settings CRUD + `requires_reload` on dflash changes |
| `omlx/admin/templates/dashboard/_modal_model_settings.html` | UI: toggle, draft model dropdown, quantization selector |
| `omlx/admin/static/js/dashboard.js` | Frontend settings binding |
| `omlx/admin/benchmark.py` | Batch test skip guard for DFlashEngine |
| `tests/test_dflash_engine.py` | Unit tests (14 tests) |

### Dependency

- `dflash-mlx` pinned to `bstnxbt/dflash-mlx@20d68db` (original repo, v0.1.5.1, includes Gemma4 backend + runtime package refactor)
- Listed as required dependency in `pyproject.toml` and `packaging/venvstacks.toml`

### Supported models

DFlash registers `QwenGdnTargetOps` and `Gemma4TargetOps`. Draft checkpoints currently exist for the Qwen3.x and Gemma4 families:

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

Other model families (Llama, Gemma3, etc.) are not supported — they require both a trained DFlash draft checkpoint and a compatible target adapter in dflash-mlx.

Note: the `-DFlash` suffix is specific to DFlash draft checkpoints. Gemma4 also ships an `-assistant` variant (e.g. `gemma-4-26B-A4B-it-assistant`) that targets MTP speculative decoding via mlx-vlm — do not mix these in the DFlash toggle.

### Per-model settings

| Setting | Type | Description |
|---------|------|-------------|
| `dflash_enabled` | bool | Enable/disable DFlash for this model |
| `dflash_draft_model` | str | Path or HuggingFace repo for draft checkpoint |
| `dflash_draft_quant_enabled` | bool | Draft model quantization enabled |
| `dflash_draft_quant_weight_bits` | int | Draft model quantization weight bits |
| `dflash_draft_quant_activation_bits` | int | Draft model quantization activation bits |
| `dflash_draft_quant_group_size` | int | Draft model quantization group size |

Configured via web admin UI → Model Settings → Experimental Features → DFlash.

---

## Generation flow

### Short context (< DFLASH_MAX_CTX)

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

### Long context (>= DFLASH_MAX_CTX)

1. `DFlashEngine.stream_generate()` detects prompt length exceeds threshold
2. Delegates entire request to `_fallback_engine.stream_generate()`
3. Fallback engine is BatchedEngine or VLMBatchedEngine (started during `DFlashEngine.start()`)
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

**Mitigation**: for Apple Silicon personal use (1-2 concurrent users), single-request 3-4x speed improvement outweighs batching benefits.

### 2. Context length limit

DFlash effectiveness degrades with long contexts:
- Draft model trained on max sequence length 3072-4096 tokens
- Verify pass attention cost grows with KV cache size
- Default `DFLASH_MAX_CTX = 4096` (configurable via environment variable)
- Beyond threshold, automatic fallback to BatchedEngine/VLMBatchedEngine

### 3. Model support

Only Qwen3.5 family has draft checkpoints. Each new model family requires:
- A trained DFlash draft checkpoint (block diffusion model matching target hidden dimensions)
- Support in dflash-mlx's target model handling (hidden state extraction, cache rollback)

### 4. Memory overhead

DFlashEngine loads both target and draft models simultaneously:
- Draft model: typically ~1B parameters (small relative to target)
- Draft int4 quantization available to reduce footprint
- Fallback engine also loaded (for long-context requests) — shares the same model weights via mlx-lm if model path matches

### 5. No paged/SSD cache

DFlashEngine does not use omlx's paged KV cache or SSD cache system. Cache is managed entirely within dflash-mlx's generation loop. Each request does full prefill from scratch (no prefix cache reuse across requests).

**Mitigation**: long-context requests (where prefix caching matters most) fall back to BatchedEngine which has full paged + SSD cache support.

### 6. No batch benchmark

Admin panel benchmark's batch throughput test is skipped for DFlashEngine since it requires scheduler core access (`engine._engine`) that DFlashEngine doesn't expose. Single-request benchmark tests work normally.

### 7. Greedy verification with temperature

When temperature > 0, acceptance rate drops because draft and target independently sample from the logit distribution. The accepted tokens are always valid target model samples at the given temperature, but fewer draft tokens get accepted per cycle, reducing the speed benefit.

---

## Fallback mechanism

```
DFlashEngine.start()
  ├── load target model (dflash-mlx)
  ├── load draft model (dflash-mlx)
  └── start fallback engine
        ├── VLMBatchedEngine (if model detected as VLM)
        └── BatchedEngine (otherwise)

Request arrives:
  ├── len(prompt_tokens) < DFLASH_MAX_CTX → DFlash path
  └── len(prompt_tokens) >= DFLASH_MAX_CTX → fallback engine path

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
| `DFLASH_MAX_CTX` | 4096 | Max prompt tokens before fallback to AR |
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
DFlash context fallback: 5120 >= 4096, using vlm engine
```

---

## Testing

### Unit tests (`tests/test_dflash_engine.py` — 14 tests)

- ModelSettings: default values, serialization roundtrip, removed field handling
- DFlashEngine: properties, stats, cache stats
- EnginePool routing: disabled/enabled/draft model checks

### Manual testing

1. Enable DFlash in admin UI for a supported model
2. Set draft model path (e.g., `z-lab/Qwen3.5-4B-DFlash`)
3. Reload model
4. Send short prompt → verify DFlash logs (acceptance ratio, tok/s)
5. Send long prompt (>4K tokens) → verify fallback to BatchedEngine logs

---

## Future work

- **Upstream sync**: merge temperature patch to bstnxbt/dflash-mlx, update pin
- **Broader model support**: as dflash-mlx adds new model families, omlx gets support automatically
- **Prefix caching**: explore chunked prefill with block-level KV save/restore for multi-turn conversation acceleration (requires dflash-mlx exposing cache manipulation APIs)
- **Benchmark integration**: add DFlash-specific benchmark metrics (acceptance ratio, draft/verify timing breakdown) to admin panel
