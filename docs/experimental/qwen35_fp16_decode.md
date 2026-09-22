# Qwen3.5/3.6 FP16 decode prework

`OMLX_QWEN35_FP16_GDN_DECODE=1` opts into fused Gated DeltaNet prework for
FP16 Qwen3.5/3.6 35B-A3B models loaded through the VLM engine. It is off by
default. Set the environment variable before starting oMLX; the patch reads it
once when installed. Unset it or set it to `0` and restart to use the previous
decode path.

```bash
OMLX_QWEN35_FP16_GDN_DECODE=1 omlx serve --model-dir /path/to/models
```

This reuses the existing Metal convolution/SiLU/QK-normalization kernel at
single-token decode. It does not requantize weights or change projections,
forget/update gates, recurrent arithmetic, final normalization or output gating.
Prefill is unchanged. The existing Qwen4 BF16 decode and BF16 speculative
verification paths are independent of this switch.

The route is restricted to inference with batch 1, one token, hidden size 2048,
16 key heads, 32 value heads, head dimensions 128, and a four-tap convolution.
Inputs, convolution weights and convolution state must be FP16; the recurrent
state must be FP32. A normal, populated `ArraysCache` is required. Missing
state, cache subclasses, masks, padding/length metadata, speculation, different
geometry or precision, CPU execution and extended call keywords retain the
original implementation. Projections are still called normally, including
quantized projection subclasses.

The log entry `Qwen3.5 FP16 B1/T1 fused GDN prework engaged` confirms that an
eligible call actually reached the kernel. A successfully installed patch alone
does not demonstrate engagement.

## Validation and performance comparisons

```bash
python -m pytest -q tests/test_qwen35_fp16_decode.py tests/test_qwen35_gdn_prework.py
```

These tests exercise the current mlx-vlm decode convolution and normalization,
FP16 rounding over several input scales and strides, successive recurrent
states, restored caches, fallback routing and speculative commit. The synthetic
whole-layer fixture uses cheap deterministic projections, so it does not replace
validation with a real quantized checkpoint.

For whole-server comparisons, use the same checkpoint and settings in separate
processes with this flag off/on. Keep sampling, command-buffer settings,
concurrency and prefix-cache state fixed. Warm both paths, alternate run order,
and report decode throughput alongside complete response time, TTFT, prefill
time and peak memory. Check the engagement log, repeated-prefix responses and
concurrent requests (which must fall back when batched). A kernel microbenchmark
alone is insufficient to claim a whole-request speedup.
