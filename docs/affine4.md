# Affine4 KV cache

Affine4 is an optional four-bit cache format for standard full-attention layers.
It uses native signed-int4 TensorOps on Apple M5 GPUs and a portable MLX attention
path for other hardware and attention shapes. No custom extension build or
environment variable is required.

In **Model settings → Advanced settings → Experimental features**, enable
**KV cache compression** and select **Affine4 (4-bit)** under **Format**.
The setting takes effect when the engine reloads. Existing settings continue to
select TurboQuant unless the format is explicitly changed.

The corresponding model settings are:

```json
{
  "turboquant_kv_enabled": true,
  "turboquant_kv_scheme": "affine4",
  "turboquant_kv_bits": 4
}
```

The existing setting names are retained for configuration compatibility.
Affine4 and TurboQuant are separate formats: TurboQuant indices select a
nonlinear codebook; affine4 stores signed integer values. Changing the format
reloads the engine and invalidates incompatible cached prefixes.

## Model and hardware coverage

Cache conversion depends on the attention/cache interface, with no model-name
allowlist or fixed query/KV head counts. Standard MHA, MQA, and GQA layers use
affine4; recurrent and sliding-window layers in hybrid models retain their
native caches. MLA and composite caches that expose tensors directly remain
ineligible, matching the existing quantized-cache restrictions.

The native M5 path covers short decode/verification queries of one to four rows,
head dimensions divisible by 32 through 512, and up to 32 grouped query rows per
KV head. Other dimensions, longer queries, and attention sinks use the portable
path. Boolean and additive attention masks retain their meaning.

Eligible BF16 prefill uses fused MLX attention with BF16 unpacking in rotated
coordinates. It evaluates each layer's attention before releasing its unpacked
KV workspace. Explicit array masks, sinks and unsupported shapes retain the
portable float32 path. Long portable queries run in bounded blocks, retiring
each block before the next one to
avoid allocating a full query-by-context score matrix. The score budget is
64 MiB per block, or at least one query row; total temporary memory also includes
the unpacked keys, values, and attention intermediates.

Affine4 compresses new full-attention KV during each layer's prefill update,
so the complete uncompressed history never has to be resident. Attention
unpacks one layer at a time as needed. The final full-attention layer remains
uncompressed by default under the existing `turboquant_skip_last` quality guard.
Recurrent and sliding-window state retains its native representation. Unlike
TurboQuant's cold-prefill policy, Affine4 quantization also affects prefill
hidden states, so cache format and prefill chunk size can change generated text.

Admission distinguishes storage size from prefill residency: TurboQuant cold
prefill is priced at the model's native KV width, while Affine4 uses packed
words plus four-byte vector scales and the retained native layers. Affine4
also reserves one unpacked layer and bounded attention workspace, including
when the prefill chunk is small. Memory-pressure resumes retain absolute
context lengths even with prefix caching disabled.

## Storage and correctness

Keys and values have independent deterministic orthogonal rotations. Power-of-two
head dimensions use a randomized Hadamard transform; other dimensions use a
deterministic orthogonal matrix. Signed nibbles are packed into `uint32` words.
Each token/head vector has its own float32 scale. Keeping scales in float32
preserves small and large finite values across portable and native execution.
Native attention uses float16 value accumulation for bounded partitions and
float32 for larger partitions. Value-scale normalization keeps the bound
independent of input magnitude; softmax statistics and partition reduction
remain float32.
The [paired accumulator measurements](../benchmarks/results/affine4_decode_optimized.json)
compare this path with the FP32 baseline and both earlier implementations.
At 200K tokens, dense single-row attention fell from 2.90 to 1.78 ms and
MoE single-row attention from 1.61 to 1.06 ms. The earlier VLM kernels measured
1.58 and 0.99 ms respectively in the same run.
The [paired fused-prefill measurements](../benchmarks/results/affine4_prefill_optimized.json)
at 32K context and 2,048 query rows measured 85.75 ms for dense geometry and
55.94 ms for MoE geometry, versus 486.05/211.40 ms for the FP32 baseline and
95.39/60.18 ms for the VLM port. Output cosine against the FP32 path was
approximately 0.99999; this is an attention-level numerical check.
For a head dimension of 256, a compressed K or V vector occupies 132 bytes,
compared with 512 bytes in float16; the retained full-precision layers reduce
the whole-model compression ratio.

Snapshots record the format, seed, and original key/value dimensions. Prefix and
SSD restores preserve the packed state without requantization. Affine4 blocks
cannot be restored as TurboQuant blocks even when their packed widths match.
Malformed metadata or incompatible block chains cause a cache miss.

Native kernels are compiled lazily. Their first evaluation is checked before
the specialization is reused; unavailable kernels fall back to portable MLX
attention. This is lossy KV compression, so generated text may differ from
full-precision KV or TurboQuant.

## Verification

```sh
python -m pytest tests/test_affine4.py tests/test_affine4_integration.py tests/test_affine4_prefix_cache.py
```

These cover native/reference attention, unsupported shapes and masks, snapshot
metadata, SSD round trips, continuous batching, scheduler signatures, and real
Llama/Qwen2 model forward passes.

The [real-server smoke results](../benchmarks/results/affine4_server_smoke.json)
record Qwen3.8 generation, warm-prefix reuse, reuse after restart, and Lightning
MTP verification on an M5 Pro. These validate the execution path and cache
lifecycle; they are not a model-quality benchmark.

The [Qwen server comparison](../benchmarks/results/affine4_qwen_server_comparison.md)
covers Qwen3.6-35B-A3B and Qwen3.8-27B at 4K through 64K context, 4,096-token
endurance runs, and prefill after restoring a 32K prefix. Affine4 improved
long-context decode in these measurements, while restored-prefix prefill
remained slower than uncompressed KV and was slower than TurboQuant on Qwen3.8.

For attention-only comparisons with TurboQuant and float16 KV:

```sh
python benchmarks/bench_affine4.py --fallbacks > attention.json
```

Stop other GPU workloads before measuring. Cache construction is excluded from
timings, and attention speedups do not directly predict whole-model throughput.

The [M5 Pro measurements](../benchmarks/results/affine4_m5_pro.json) include
forward and reverse benchmark ordering, 20 warmups and 50 samples per case, and
source hashes. At 32,768 cached tokens, head dimension 256, 24 query heads and
four KV heads, the median attention times across the two orders were:

| Query rows | Float16 KV | TurboQuant 4-bit | Affine4 |
| --- | ---: | ---: | ---: |
| 1 | 0.79–0.80 ms | 0.88–0.90 ms | 0.72–0.75 ms |
| 4 | 1.71–1.72 ms | 4.19–4.20 ms | 1.32–1.35 ms |

With the native path disabled, fused unpacking reduced one-row portable
attention from 9.97–10.10 ms to 3.46 ms at that geometry. For a 256-row warm
prefill, query blocking reduced measured temporary memory from 1,047 MiB to
332 MiB; a 2,048-row query used 417 MiB. Blocking introduces synchronization to
bound memory. These measurements do not establish model quality or a speedup
for every architecture and context length.
