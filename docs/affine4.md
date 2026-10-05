# Affine4 KV cache

Affine4 is an optional signed 4-bit cache format for standard full-attention
layers. It maximizes context capacity while retaining native cache types for
recurrent and sliding-window layers. Affine4 uses packed TensorOps on Apple M5
GPUs and a portable MLX attention path on other hardware and unsupported shapes.
No custom extension build or environment variable is required.

Enable **KV cache compression** under **Model settings → Advanced settings →
Experimental features**, then select **Affine4 (4-bit)** under **Format**. The
setting takes effect after engine reload:

```json
{
  "turboquant_kv_enabled": true,
  "turboquant_kv_scheme": "affine4",
  "turboquant_kv_bits": 4
}
```

Existing setting names are retained for configuration compatibility. Affine4,
[Affine8](affine8.md), and TurboQuant are distinct formats; changing format
reloads the engine and invalidates incompatible cached prefixes.

## Coverage and lifecycle

Cache conversion depends on attention and cache interfaces, without a model-name
allowlist or fixed query/KV head counts. Standard MHA, MQA, and GQA layers use
Affine4. Recurrent and sliding-window layers in hybrid models retain native
caches. MLA and composite caches that expose tensors directly remain ineligible,
matching existing quantized-cache restrictions.

Native M5 attention covers decode and verification queries of one to four rows,
head dimensions divisible by 32 through 512, and up to 32 grouped query rows per
KV head. Other dimensions, longer queries, and attention sinks use portable MLX
attention. Boolean and additive masks retain their meaning.

New full-attention KV is compressed during each layer's prefill update, so cold
prefill never requires complete native KV history. Portable attention evaluates
and copies each layer output into a graph leaf before returning, bounding unpacked
KV lifetime across layers. Long portable queries also run in bounded query blocks.
The last full-attention layer remains native by default under
`turboquant_skip_last`.

Snapshots record format, seed, and original K/V dimensions. Prefix and SSD
restores preserve packed state without requantization. Restored prefixes, normal
prefill, SpecPrefill sparse continuation, and decode all retain selected format.
Malformed metadata or incompatible block chains cause a cache miss.

Keys and values use independent deterministic orthogonal rotations. Signed 4-bit
codes are packed into `uint32` words with one float32 scale per token/head vector.
For head dimension 256, one Affine4 K or V vector occupies 132 bytes, compared
with 512 bytes in float16. Retained native layers reduce whole-model compression.
Affine4 is lossy: generated text can differ from native KV, TurboQuant, or
Affine8, and prefill chunk size can affect output.

## Format comparison

Apple M5 Pro, 48 GiB; Qwen3.8-27B-oQ4e-mtp; 8,192 prompt tokens; 128
generated tokens; 2,048-token cold-prefill chunks; greedy decode with termination
tokens suppressed; speculation and ANE disabled. Identical model weights and
prompts were used. Each format ran twice in forward/reverse order; throughput is
arithmetic mean and memory is maximum active MLX allocation.

| KV format | Prefill tok/s | Decode tok/s | Peak active MLX GiB | K/V storage at head dim 256 |
|---|---:|---:|---:|---:|
| Native BF16 | 467.0 | 16.0 | 18.692 | 512 B/vector |
| TurboQuant4 | 465.8 | 15.5 | 18.692 | format-dependent packed codebook |
| Affine4 | 456.1 | 16.1 | 18.345 | 132 B/vector |
| Affine8 | 459.8 | 15.9 | 18.462 | 260 B/vector |

Peak memory includes model weights and prefill transients, not only KV storage.
These focused local measurements show representative tradeoffs, not confidence
intervals or universal speedups.

## Long-context quality check

A paired 13,235-token passkey retrieval check used same Qwen3.8 weights, prompt,
greedy decoding, and cold incremental prefill for every format. Chunk sizes 512
and 2,048 were tested in forward and reverse format order. Native BF16,
TurboQuant4, Affine4, and Affine8 all returned `ZXQJ-4827` exactly: 8/8 successful
runs. This focused check exercises information near prompt start across long
filler; it does not establish general semantic quality or long-context accuracy.

## Tests

```sh
python -m pytest tests/test_affine4.py tests/test_affine4_integration.py tests/test_affine4_prefix_cache.py
```

Coverage includes native/reference attention, portable multi-layer workspace
bounds, masks, unsupported shapes, overflow bounds, snapshot metadata, SSD
round trips, continuous batching, scheduler admission, incremental real-model
prefill, restored prefixes, and SpecPrefill cache lifecycle for both affine
formats.
