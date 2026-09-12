# Signed-affine KV cache

Affine4 and Affine8 are optional signed-integer cache formats for standard
full-attention layers. They use native packed TensorOps on Apple M5 GPUs and a
portable MLX attention path for other hardware and attention shapes. No custom
extension build or environment variable is required. Affine4 maximizes context
capacity; Affine8 keeps roughly twice its KV storage for lower quantization error.

In **Model settings → Advanced settings → Experimental features**, enable
**KV cache compression** and select **Affine4 (4-bit)** or **Affine8 (8-bit)**
under **Format**. The setting takes effect when the engine reloads. Existing
settings continue to select TurboQuant unless the format is explicitly changed.

The corresponding model settings are:

```json
{
  "turboquant_kv_enabled": true,
  "turboquant_kv_scheme": "affine8",
  "turboquant_kv_bits": 8
}
```

The existing setting names are retained for configuration compatibility.
Affine4, Affine8, and TurboQuant are separate formats: TurboQuant indices select
a nonlinear codebook; affine formats store signed integer values. Changing the
format reloads the engine and invalidates incompatible cached prefixes.

## Model and hardware coverage

Cache conversion depends on the attention/cache interface, with no model-name
allowlist or fixed query/KV head counts. Standard MHA, MQA, and GQA layers use
either affine format; recurrent and sliding-window layers in hybrid models retain
their native caches. MLA and composite caches that expose tensors directly remain
ineligible, matching the existing quantized-cache restrictions.

The native M5 path covers short decode/verification queries of one to four rows,
head dimensions divisible by 32 through 512, and up to 32 grouped query rows per
KV head. Other dimensions, longer queries, and attention sinks use the portable
path. Boolean and additive attention masks retain their meaning.

Eligible BF16 prefill uses fused MLX attention with BF16 unpacking in rotated
coordinates. It evaluates each layer's attention before releasing its unpacked
KV workspace. Explicit array masks, sinks and unsupported shapes retain the
portable float32 path. Long portable queries run in bounded blocks, retiring
each block before the next one to avoid allocating a full query-by-context
score matrix. The score budget is
64 MiB per block, or at least one query row; total temporary memory also includes
the unpacked keys, values, and attention intermediates.

Both affine formats compress new full-attention KV during each layer's prefill
update, so the complete uncompressed history never has to be resident. Attention
unpacks one layer at a time as needed. The final full-attention layer remains
uncompressed by default under the existing `turboquant_skip_last` quality guard.
Recurrent and sliding-window state retains its native representation. Unlike
TurboQuant's cold-prefill policy, affine quantization also affects prefill hidden
states, so cache format and prefill chunk size can change generated text.

Admission distinguishes storage size from prefill residency: TurboQuant cold
prefill is priced at the model's native KV width, while affine formats use
packed words plus four-byte vector scales and the retained native layers. Both
formats also reserve one unpacked layer and bounded attention workspace,
including when the prefill chunk is small. Memory-pressure resumes retain absolute
context lengths even with prefix caching disabled.

Benchmark `peak_memory_bytes` measures active MLX allocations; it is not total
process memory. `system_metrics.memory` separately samples physical footprint,
active allocations and the free allocator pool in GiB. Benchmark memory traces
record these components and actual cache bytes before and after chunk reclaim.
The guard continues to use the larger of active MLX memory and physical
footprint after the existing shared CPU hot-cache adjustment.

## Storage and correctness

Keys and values have independent deterministic orthogonal rotations. Power-of-two
head dimensions use a randomized Hadamard transform; other dimensions use a
deterministic orthogonal matrix. Signed 4-bit or 8-bit codes are packed into
`uint32` words. Each token/head vector has its own float32 scale. Keeping scales
in float32 preserves small and large finite values across portable and native
execution. Native attention uses float16 value accumulation only when the
partition length and maximum signed code keep its conservative bound below the
float16 finite range; larger partitions use float32. Value-scale normalization
keeps this bound independent of input magnitude. Score normalization, softmax
statistics, and cross-partition reduction remain float32. Unmasked and causal
decode specialize away array-mask reads;
explicit Boolean masks retain their own kernel specialization. Long-context
partition tuning depends on tensor geometry, not model names.

For a head dimension of 256, an Affine4 K or V vector occupies 132 bytes and an
Affine8 vector occupies 260 bytes, compared with 512 bytes in float16; the
retained full-precision layers reduce the whole-model compression ratio.

Snapshots record the format, seed, and original key/value dimensions. Prefix and
SSD restores preserve the packed state without requantization. Affine4, Affine8,
and TurboQuant blocks remain mutually incompatible even when packed widths match.
Malformed metadata or incompatible block chains cause a cache miss.

Native kernels are compiled lazily. Their first evaluation is checked before
the specialization is reused; unavailable kernels fall back to portable MLX
attention. This is lossy KV compression, so generated text may differ from
full-precision KV or TurboQuant.

## Verification

```sh
python -m pytest tests/test_affine4.py tests/test_affine4_integration.py tests/test_affine4_prefix_cache.py
```

These cover native/reference attention, overflow bounds, unsupported shapes and
masks, snapshot metadata, SSD round trips, continuous batching, scheduler
signatures, and incremental real Llama/Qwen2/Qwen3.5 hybrid forward passes.

The [incremental-cache validation](../benchmarks/results/affine4_incremental_validation.md)
records paired comparisons with the earlier implementations, full-model server
throughput, prefix persistence, and memory traces. Qwen3.8 completed both 150K
and 200K cold prompts on a 48 GiB M5 Pro with the existing memory guard.
These checks validate execution and cache lifecycle, not general model quality.

The earlier [server matrix](../benchmarks/results/affine4_qwen_server_comparison.md)
and [regression investigation](../benchmarks/results/affine4_regression_comparison.md)
describe the FP32-accumulator implementation before incremental prefill and the
current kernel optimizations. Their throughput and memory failures are
historical baselines.

For attention-only comparisons with TurboQuant and float16 KV:

```sh
python benchmarks/bench_affine4.py --fallbacks > attention.json
```

Stop other GPU workloads before measuring. Cache construction is excluded from
timings, and attention speedups do not directly predict whole-model throughput.

The [Affine8 attention comparison](../benchmarks/results/affine8_attention_comparison.md)
reports native, TurboQuant, Affine4, and Affine8 attention at 8K through 250K.
The [Affine8 full-model validation](../benchmarks/results/affine8_full_model.md)
records single-run Qwen3.8 capacity/endurance and Qwen3.6 sustained decode.

The result files retain source hashes and measurement protocols. Microbenchmarks
and fixed-length server trials do not establish a speedup for every architecture
or context length. Finite synthetic attention checks and lower sampled error do
not establish semantic quality or long-context retrieval accuracy. Incremental
lossy prefill also changes hidden states, so teacher-forced decode agreement from
a shared native cache does not establish the quality of an incrementally
compressed prompt.
