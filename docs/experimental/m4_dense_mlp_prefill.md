# M4 Max dense MLP prefill QMM tiles

`OMLX_M4_DENSE_MLP_PREFILL=1` opts loaded Gemma 4 and Llama MLP projections into the existing native BM64/BK32/BN64 QMM tile on Apple M4 Max. It is disabled by default. No checkpoint conversion, weight copies, activation replacement, new Metal kernel, or additional dependency is required.

## Scope and fallback

The instance transform is called from the text engine's post-load transforms and before VLM adapter construction. It rebinds only stock `QuantizedLinear` instances in `mlx_lm.models.llama.MLP`, `mlx_lm.models.gemma4_text.MLP`, and `mlx_vlm.models.gemma4.language.MLP`. It leaves the MLP forward body intact, including Gemma's GeGLU and Llama's SwiGLU. Vision encoders, attention projections, custom linear subclasses and other model families are untouched.

The measured `(input_dim, output_dim)` allowlist is:

- Gemma 4 12B: `(3840, 15360)` and `(15360, 3840)`.
- Llama-style llm-jp 4.1 8B: `(4096, 14336)` and `(14336, 4096)`.

The native route requires one batch row, at least 512 query tokens in the current forward, FP16/BF16 activations with matching quantization metadata, affine Q4/Q8 group64 weights, and no dense projection bias. Other chips, unmeasured shapes, missing native symbols, dtype changes, decode, short cached suffixes and small speculative verify windows keep stock MLX. The threshold concerns the current chunk, not the total context length.

Inputs and quantized arrays pass through `mx.contiguous`: already contiguous buffers are shared; lazy or materialized strided views are normalized at evaluation time. This matters because the native kernel reads dense rows, while lazy views can initially expose provisional contiguous metadata. Synchronous native admission errors fall back to stock QMM; asynchronous Metal/memory errors are not suppressed.

Set the environment variable before loading the model:

```bash
OMLX_M4_DENSE_MLP_PREFILL=1 omlx serve --model-dir /path/to/models
```

Unset the variable and reload the models to restore the stock projection classes. There is no live settings/API toggle. Do not expect this route to improve M5 NAX inference, decode throughput, or requests whose prompt is already cached.

## Validation

Run the targeted suite on a Metal-capable Mac:

```bash
python -m pytest tests/test_m4_dense_mlp_prefill.py tests/test_model_loading.py tests/test_qwen35_q4_mlp.py tests/test_vlm_engine.py tests/test_batched_engine.py -q
```

The new tests cover disabled/default behavior, hardware exclusion, loaded-instance isolation, idempotency, unchanged weight references and MLP class bodies, missing symbols, custom subclasses, Q4/Q8 and FP16/BF16 output parity, short/decode/verify/multi-batch fallback, malformed or changed quantization, owning-stream selection, and lazy/materialized strided inputs and quantized weights. Native parity cases skip when the extension is not built.

Use the paired harness against the same local checkpoint on `main` and the branch:

```bash
HF_HUB_OFFLINE=1 python benchmarks/bench_m4_dense_mlp_prefill.py /path/to/llm-jp-4.1-8b-thinking-mlx-4bit --memory-gib 10 --output llmjp.json
HF_HUB_OFFLINE=1 python benchmarks/bench_m4_dense_mlp_prefill.py /path/to/gemma-4-12B-it-qat-OptiQ-4bit --vlm --memory-gib 14 --output gemma.json
```

The harness holds the loaded arrays constant and switches only the affected instance classes between arms. It warms both routes, creates a fresh KV cache per request, uses 4096 input tokens/1024-token chunks, alternates OFF/ON then ON/OFF, and waits 15 seconds between requests. Prefill timing includes final prompt logits and excludes loading, tokenization and 16-token greedy decode. It records all-logit equality, greedy IDs, routed projection count, memory, dependency versions and commit. Keep unrelated GPU activity idle and measure clocks; small speedups are not guaranteed under contention.

On Apple M4 Max/64 GiB, MLX 0.32.2, mlx-lm `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`, mlx-vlm `ea79808ce1e9a19fcb915a96b0c70e37ad393a99`, compared against `main` `87460f4d50de79aef9b67e99e215c31f0a89b445`:

| Model | Pair order | Stock prefill | Patched prefill | Shorter |
|---|---|---:|---:|---:|
| Gemma 4 12B QAT OptiQ Q4/Q8 | OFF → ON | 7.547 s | 7.180 s | 4.9% |
| Gemma 4 12B QAT OptiQ Q4/Q8 | ON → OFF | 7.433 s | 7.171 s | 3.5% |
| llm-jp 4.1 8B affine Q4 | OFF → ON | 4.528 s | 4.455 s | 1.6% |
| llm-jp 4.1 8B affine Q4 | ON → OFF | 4.546 s | 4.416 s | 2.9% |

All four pairs had zero differing final logits and identical 16-token greedy IDs. MLX allocation peaks were approximately 10.63 GiB for the full Gemma VLM and 5.62 GiB for llm-jp, with no material increase. Gemma checkpoint revision was `dbd61bc3146229d0d6f34a2cb02345ef349541a8` from `rariruluis/gemma-4-12B-it-qat-OptiQ-4bit-MTP`; llm-jp revision was `851832aed3787f408fc76f1e103952ebf08b0d10` from `rariruluis/llm-jp-4.1-8b-thinking-mlx-4bit`. MTP decode was not enabled by the harness. The original production server and UTM remained resident; this is a limited hardware measurement, not a general throughput guarantee.

[Raw paired results](https://gist.github.com/rluisr/7cdcd60f7c3355f0dfaaa2b2c2bb9a32) include input geometry, ordering, dependency versions, projection counts and equality checks.

Native sources are unchanged. The hardware check used the already-built 0.7.0rc1 QMM extension with the current-main Python wrapper and the pinned dependency versions above. Native source rebuild/CI on a clean development install is still recommended. Prefix-reuse, concurrent/streamed serving, images/tools and longer-context server benchmarks remain outside this small performance claim.
