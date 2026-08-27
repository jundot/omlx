# DS4 BF16 attention finalizer, M=1024

This experiment fuses the exact Q-head RMSNorm+RoPE and weighted KV
RMSNorm+RoPE chains into two isolated Metal dispatches. It supports local head
counts 24, 32, and 40 and has no production model dispatch.

## Numerical contract

The kernels reproduce MLX v0.32.0's actual Metal boundaries rather than only
the mathematical formulas:

- BF16 input, four scalar reads per one of 128 threads;
- one `simd_sum` per 32-lane group, followed by a second `simd_sum` over four
  group totals and 28 zero lanes;
- `metal::precise::rsqrt(acc / 512 + eps)`;
- normalization rounded to BF16 before RoPE;
- KV norm weight multiplied after that BF16 rounding, matching MLX;
- supplied FP32 frequencies, including 224 infinity-valued no-RoPE pairs;
- `1.0 / freq`, `metal::fast::cos/sin`, traditional adjacent pairs, and a
  final BF16 rotated store.

The Q kernel groups four heads at one token so their identical trigonometric
values are computed once. Each head still has its own independent 128-thread
RMS reduction, and output is written directly from stock pre-transpose layout
`[1,1024,H,512]` to contiguous `[1,H,1024,512]`. KV maps
`[1,1024,512]` directly to `[1,1,1024,512]`.

The operation order is adapted from MLX's MIT-licensed `rms_norm.metal` and
`rope.metal`; the complete notice is retained as `MLX_LICENSE.txt` beside the
source.

## Exactness

Real layer-2 KV norm weights, the released model's YaRN frequency array, offset
8192, and random BF16 Q/KV projection outputs were checked at four boundaries:

| Heads | Q normalized | Q rotated | KV normalized | KV rotated |
|---:|---:|---:|---:|---:|
| 24 | array-equal | array-equal | array-equal | array-equal |
| 32 | array-equal | array-equal | array-equal | array-equal |
| 40 | array-equal | array-equal | array-equal | array-equal |

Every maximum absolute difference was zero.

Storage-bit equality (rather than NaN-semantic array comparison) also passed
for every head count with both frequency families—local base-10000 and
compressed base-160000 YaRN—at offsets 0 and 8192. This covers the 224 explicit
infinity no-RoPE pairs without replacing them with a copy shortcut.
An H24 stress row containing positive/negative zero, positive/negative
infinity, and NaN also matched every BF16 storage bit at all four boundaries.

## M3 Ultra result

Balanced medians from the three-shape run were:

| Heads | Q current | Q fused | Q speedup | Combined current | Combined fused | Combined speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1.982 ms | 1.010 ms | 1.963x | 1.846 ms | 1.112 ms | **1.661x** |
| 32 | 1.807 ms | 1.012 ms | 1.785x | 1.956 ms | 1.005 ms | **1.946x** |
| 40 | 1.879 ms | 1.007 ms | 1.866x | 1.884 ms | 1.024 ms | **1.839x** |

KV alone is neutral because it is only one 1 MiB row bank; the win is Q's
large normalized intermediate and its eliminated second dispatch.

A separate 12-cycle H24 confirmation measured 1.683 ms current versus 1.020 ms
fused, or **1.650x**, saving 0.663 ms per layer. This clears the requested
1.10x finalizer gate.

## Byte and full-layer value

The current path writes then rereads the normalized Q and KV tensors. The
candidate removes:

| Heads | Removed norm write + RoPE read |
|---:|---:|
| 24 | 50 MiB/layer |
| 32 | 66 MiB/layer |
| 40 | 82 MiB/layer |

At the confirmed H24 delta, 43 layers save an isolated 28.52 ms per 1024-token
pass. If rank 0 remains on the critical path and none of that work overlaps,
the 628.76 tok/s reference projects to about 639.97 tok/s, a 1.018x full-pass
gain. This is a quantified bound, not a distributed throughput claim.

## Shipping state

The exact pair is available through a strict default-off production seam:
`OMLX_DSV4_ATTN_FINALIZER_PREFILL=1`. Before either native graph node is
created, one shared preflight requires BF16 B1/M1024/D512 Q and KV, H24/H32/H40,
the supplied contiguous FP32 `[256]` frequency buffer, BF16 KV norm weight,
non-negative scalar offset, non-training/non-verification mode, and both native
symbols. Any rejection builds the unchanged four-operation stock graph. There
is no exception-based retry after selection.

The cluster hostfile exports an explicit `0` by default, preventing rank-local
selection drift. Caches and collectives are unchanged. The seam is ready for a
full TP exactness/A-B, but remains off because only the local M3 rank was timed
here; H40 still needs physical M5 measurement.

Reproduce the local gate with:

```bash
python benchmarks/bench_ds4_attention_finalizer.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --layer 2 --heads 24 32 40 --offset 8192 --strict
```
