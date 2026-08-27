# DS4 M=1024 projection campaign, 3:5 TP

This is an isolated exactness and tile-selection campaign. It adds no model
dispatch and does not change serving, caches, collectives, or checkpoint
loading.

## Decision

Projection grouping and the classic M3 Steel tile sweep do not meet the 1.30x
bucket gate. The remaining plausible projection-wide lever is the portable M5
NAX tile sweep: stock MLX hardcodes BM64/BN64/BK64, WM2/WN2 for every M=1024
MXFP8 projection despite radically different Q-B, O-A, and O-B shapes.

The optional NAX source in this campaign compiles ten exact-arithmetic tile
variants, including stock as variant 0. A subsequent M5 confirmation promoted
one narrow slice: variant 0 for rank-1 O-A with prepared shape
`[1,8,1024,2560]`. It is wired behind default-off
`OMLX_DSV4_NAX_OA_PREFILL=1`; every other projection and shape remains stock.

## Exact shapes

Both ranks use BF16 activations and group-32 MXFP8 projection weights.

| Projection | K | Rank-0 N / geometry | Rank-1 N / geometry |
|---|---:|---:|---:|
| Q-A | 4096 | 1024 | 1024 |
| Q-B | 1024 | 12,288 / 24 heads | 20,480 / 40 heads |
| raw KV | 4096 | 512 | 512 |
| O-A | 1536 / 8 groups | 8 × 1024 | — |
| O-A | 2560 / 8 groups | — | 8 × 1024 |
| O-B | 8192 | 4096 | 4096 |

Ratio-4 additionally has two BF16 4096→1024 compressor projections, two
4096→256 index-compressor projections, row-local MXFP8 index Q-B, and a small
BF16 index-weight projection.

## Real layer-2 profile

Clean M3 medians for the ratio-4 representative were:

| Stage | Rank 0, 24 heads | Rank-1 shape, 40 heads |
|---|---:|---:|
| Q-A | 0.909 ms | 0.824 ms |
| Q-B | 1.932 ms | 3.043 ms |
| raw KV | 0.597 ms | 0.567 ms |
| main compressor pair | 1.730 ms | 1.639 ms |
| index-compressor pair | 0.790 ms | 0.771 ms |
| O-A | 1.996 ms | 3.103 ms |
| O-B | 4.872 ms | 4.744 ms |

O-B dominates rank 0. Q-B and O-A grow with local head width and together
dominate the rank-1 shape. Therefore an O-B-only win cannot close both ranks.

## Grouping controls

- The M=1024 Q-A/raw-KV plus safe compressor grouping is array-equal, but only
  1.060x on rank 0 and 1.019x on the rank-1 shape. Across the full projection
  bucket it produces 1.015x and 1.011x respectively.
- Main plus indexer Q-B concatenation is exact but flat: 1.003x M3 and 1.001x
  M5.
- One width-2560 gate/up MXFP4 gather is exact through LimitedSwiGLU, but only
  1.005x M3 and 1.034x M5. A production representation would make the combined
  bank canonical, release the source banks, and expose fallback slices as
  views, so steady-state duplicate memory is zero.
- Concatenating five BF16 compressor projections reaches 1.121x but is not
  array-equal. It is rejected.
- The existing B=1 native QKV/compressor bundle is also rejected: 0.959x M3 and
  0.953x M5.

## Classic Steel tile result

The isolated primitive calls MLX's own `fp_qmm_t_impl` with unchanged MXFP8
dequantization, K walk, FP32 accumulators, and BF16 store. All ten M3 variants
were array-equal to `mx.quantized_matmul`. BM64/BK32/BN32 was best for every
large bank:

| Rank shape | Q-B | O-A | O-B | Projected bucket |
|---|---:|---:|---:|---:|
| rank 0 / 24 heads | 1.054x | 1.090x | 1.060x | **1.054x** |
| rank 1 / 40 heads | 1.102x | 1.081x | 1.121x | **1.092x** |

This is useful tuning evidence but not a promotable 1.30x primitive.

## ds4-metal source audit

ds4-metal commit `78269ce` has no active large-M Q-A/WKV/compressor fusion.
Its Q-A/KV pair and six-projection bundle are decode-only; batch prefill still
launches those projections separately. The directly transferable ideas are:

- paired Q-latent/KV RMSNorm (`metal/norm.metal`, around lines 180-242);
- Q-head RMSNorm plus RoPE (`metal/dsv4_rope.metal`, around lines 352-429);
- concurrent Q/KV finalizer scheduling (`ds4_metal.m`, around 21446-21625).

An oMLX port must use MLX's BF16 rounding frontier, precise RMSNorm reduction,
explicit FP32 `_freqs` including infinity-valued no-RoPE prefixes, and MLX RoPE
trigonometry. Reconstructing DwarfStar's runtime YaRN formula is not exact.

That finalizer can remove about 50 MiB/layer on rank 0 and 82 MiB/layer on rank
1, plus two dispatches. It remains a worthwhile secondary campaign, but norms
and RoPE are outside the measured projection bucket and cannot by themselves
support a 1.30x projection claim.

## Portable M5 gate

Build the optional kernels, then run on the M5:

```bash
python benchmarks/bench_ds4_projection_campaign.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --rank 1 --layers 2 --tile-sweep --warmup 3 --cycles 6
```

Every Q-B/O-A/O-B candidate must be array-equal. Apply each best exact timing
to the same balanced bucket median and require at least 1.30x. If the NAX sweep
also fails, there is no evidence-backed projection primitive at this gate; the
next work should be the exact norm/RoPE finalizer as an independent full-layer
optimization, not further concatenation.

The production-facing dispatch remains deliberately narrower than the sweep:
BF16, non-training, non-verification, M=1024, 40-head O-A; MXFP8 weight
`[8,1024,640]`, scales `[8,1024,80]`; NAX artifact and device capability both
required. All checks happen before the native graph is created. The cluster
hostfile exports an explicit `0` by default so both ranks share the rollback.
