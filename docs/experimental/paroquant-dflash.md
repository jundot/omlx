# ParoQuant with DFlash

DFlash text generation supports the dense Qwen3.8-27B ParoQuant layout
(4-bit weights, group size 128, eight rotation stages). A matching draft is
`z-lab/Qwen3.8-27B-DFlash2`.

Install oMLX with the `paroquant` extra, register the target and draft, then
select the draft under the target's DFlash settings. Start with block size 5;
block sizes 3 and 8 are also covered by the correctness checks. A Hugging Face
cache reference must resolve to the snapshot containing the checkpoint files,
not the cache's `models--...` container directory.

ParoQuant targets load through ParoQuant's text loader. Its rotation-aware
projection modules remain intact; target verify-linear replacements are skipped.
The existing Qwen hybrid-attention adapter supplies hidden-state capture,
verification, and recurrent/attention cache rollback. Ordinary MLX targets retain
their original loading and verification paths.

Eligibility is checked consistently by the dashboard, settings API, and target
loader. Other ParoQuant architectures and quantization layouts remain unsupported.
Draft hidden size, vocabulary size, target-layer count, and capture-layer indices
must match the target. Matching dimensions do not establish semantic compatibility:
choose a draft trained for the same base model.

This enables text speculation. Image requests continue through oMLX's existing
VLM fallback. It does not enable ParoQuant's separately gated MTP, SpecPrefill,
IndexCache, or TurboQuant settings.

## Validation

Focused tests:

```sh
python -m pytest tests/test_dflash_paroquant.py tests/test_dflash_engine.py \
  tests/test_dflash_lifecycle.py tests/test_dflash_multimodal_fallback.py \
  tests/test_model_loading.py tests/test_admin_api_key.py
```

Tests involving real rotation modules require the optional ParoQuant package and
Metal. The small numerical fixture includes both recurrent and full-attention
layers, nonzero rotations, and all possible rejection positions in a five-token
verification block. The first position is a confirmed root; accepted draft tokens
are counted separately.

To audit a real checkpoint, compare captured forward logits, force rejection at
block sizes 3/5/8, and compare 64 greedy output tokens on prose, code, and counting:

```sh
python -m benchmarks.paroquant_dflash.validate \
  --target /path/to/Qwen3.8-27B-PARO \
  --draft /path/to/Qwen3.8-27B-DFlash2/snapshot \
  --output /tmp/paroquant-validation.json
```

The loader audit checks missing, unexpected, and mismatched text tensors after
ParoQuant conversion/sanitization. The baseline and speculative runs share the
same loaded weights and tokenizer. The harness explicitly re-arms instance hook
flags when switching from ordinary decoding back to DFlash.

To exercise the oMLX engine's streaming, cold/warm long-context requests, seeded
sampling, cancellation, and reload:

```sh
python -m benchmarks.paroquant_dflash.engine_smoke \
  --target /path/to/Qwen3.8-27B-PARO \
  --draft /path/to/Qwen3.8-27B-DFlash2/snapshot \
  --output /tmp/paroquant-engine.json \
  --contexts 4096 16384 32768
```

These scripts are correctness and integration probes. Their timing fields are
not a controlled comparison against the regular oMLX batched engine. Performance
claims require matched cache state, runtime/memory policy, repeated alternating
runs, and equal correct output. DFlash phase timers can include asynchronous work;
they are not exclusive GPU kernel timings.

## Recorded validation (2026-09-08)

On an M3 Max with 64 GiB unified memory, MLX 0.32.2, the pinned mlx-lm
0.31.3 revision, dflash-mlx 0.1.10+omlx.7, and ParoQuant 0.1.16:

- The real target's sanitized text-weight audit had no missing, extra, or
  mismatched tensors. Hidden-capture logits matched ordinary forward exactly.
- All nine 64-token greedy comparisons matched token for token (three prompts
  at block sizes 3, 5, and 8).
- All 16 rejection cases preserved the next-token argmax. Maximum logit
  difference was 0.033203125 across FP16 block/sequential evaluation paths.
- oMLX cold/warm requests at 4K, 16K, and 32K returned identical text, and
  repeated requests reused the entire prompt prefix.
- Seeded sampling, post-cancellation generation, and unload/reload passed.
- The related regression run passed 309 tests; the final focused suite passed
  24 tests after narrowing eligibility to the loader-supported top-level type.

Machine-readable evidence is in
[`validation_summary.json`](../../benchmarks/paroquant_dflash/validation_summary.json).
Live image generation and a controlled comparison against the batched engine
were not part of this validation.
