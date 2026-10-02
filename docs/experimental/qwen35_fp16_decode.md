# Qwen GDN decode prework

Fused Gated DeltaNet prework runs automatically for compatible FP16 and BF16 Qwen models loaded through the VLM engine on Apple GPUs. There is no user setting, environment variable, chip allowlist or model-size allowlist. The route uses the model's hidden size and key/value head counts.

This reuses the existing Metal convolution/SiLU/QK-normalization kernel at
single-token decode. It does not requantize weights or change projections,
forget/update gates, recurrent arithmetic, final normalization or output gating.
Prefill is unchanged. The existing Qwen4 BF16 decode and BF16 speculative
verification paths remain separate from ordinary single-token decode.

FP16/BF16 refers to the activations and convolution tensors, not to quantized weight bit width. This optimization does not convert model precision or download a different checkpoint.

The route requires the `Qwen3_5GatedDeltaNet` implementation in inference mode, batch 1, one token, key/value head dimensions of 128, and a four-tap convolution. Inputs, convolution weights and convolution state must share the same FP16 or BF16 dtype; recurrent state must be FP32. A normal, populated `ArraysCache` is required. Missing state, cache subclasses, masks, padding/length metadata, speculation, unsupported head dimensions or precision, CPU execution and extended call keywords retain the original implementation. Projections are still called normally, including quantized projection subclasses.

The log entry `Qwen B1/T1 fused GDN prework engaged` confirms that an
eligible call actually reached the kernel. A successfully installed patch alone
does not demonstrate engagement.

## Validation and performance comparisons

```bash
python -m pytest -q tests/test_qwen35_fp16_decode.py tests/test_qwen35_gdn_prework.py
```

These tests exercise the current mlx-vlm decode convolution and normalization,
FP16/BF16 rounding over several input scales and strides, successive recurrent
states, restored caches, fallback routing and speculative commit. The synthetic
whole-layer fixture uses cheap deterministic projections, so it does not replace
validation with a real quantized checkpoint.

For whole-server comparisons, use the same checkpoint and settings in separate
processes running unchanged main and this branch. Keep sampling, command-buffer settings,
concurrency and prefix-cache state fixed. Warm both paths, alternate run order,
and report decode throughput alongside complete response time, TTFT, prefill
time and peak memory. Check the engagement log, repeated-prefix responses and
concurrent requests (which must fall back when batched). A kernel microbenchmark
alone is insufficient to claim a whole-request speedup.
