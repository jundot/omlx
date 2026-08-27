# DS4 expert-blocked M5 NAX routed-MoE probe

This is an isolated, default-unreachable prototype for the signed DS4 Flash
3:5 rank-1 prefill shape. It changes no model dispatch, server, registry,
collective, or checkpoint representation.

## Decision

The first BM32 NAX primitive passed every lossless boundary and the physical
M5 performance gate. A production-facing seam is now available behind
`OMLX_DSV4_NAX_MOE_BLOCKS=1`; the shipped and cluster-hostfile default is `0`.

| Component | Stock M5 NAX | Expert-blocked NAX | Speedup |
|---|---:|---:|---:|
| Gate/up pair | 9.230 ms | 5.915 ms | **1.560x** |
| Pair + LimitedSwiGLU | 9.286 ms | 5.991 ms | **1.550x** |
| Down, fixed input | 4.452 ms | 3.135 ms | **1.420x** |
| Composed routed projection | 13.490 ms | 8.916 ms | **1.513x** |

The composed result clears the 1.45x isolated gate. The runtime seam remains
opt-in until the full 3:5 TP cold-prefill A/B clears independently.

## Structural opportunity

MLX 0.32's stock M5 MXFP4 gather kernel is
`BM64/BN64/BK64, WM2/WN2`. It owns global route tiles. For every expert
segment inside one tile it clears the accumulator and repeats a complete
64-row TensorOps matmul, then stores only that expert's row slice.

The fixed M=1024/top-6 fixture has 6,144 routes and exactly 24 routes per each
of 256 experts:

| Work item | Stock global BM64 | Expert-local BM32 |
|---|---:|---:|
| Route tiles / expert blocks | 96 | 256 |
| Expert segments / weight segments | 320 | 256 |
| Row-MMA equivalents | 20,480 | 8,192 |
| Relative row work | 2.50x | 1.00x |

The candidate reuses the existing `[row_start, expert, rows]` BM32 block plan.
One threadgroup owns one expert block, eliminating the stock kernel's
multi-expert loop. It retains the stock per-SIMDgroup geometry: BM32/WM1 and
BM64/WM2 both give `SM=32`; BN64/WN2 gives `SN=32`; BK64 walks K in the same
32-wide TensorOps steps.

## Frozen first ABI

The symbol is `deepseek_mxfp4_gather_qmm_blocks_nax`:

- BF16 sorted input `[6144,1,K]`;
- 256-expert, group-32 MXFP4 U32 weights `[256,N,K/8]`;
- U8 E8M0 scales `[256,N,K/32]`;
- BM32 metadata `[448,3]` and scalar block count;
- gate/up `(K,N)=(4096,1280)` or down `(1280,4096)`; and
- BF16 output `[6144,1,N]`.

The Metal kernel uses MLX's `QuantizedBlockLoader<bfloat>` and NAX
`tile_matmad_nax` with FP32 accumulators. It does not fuse activation or alter
the route order. The initial implementation deliberately dispatches gate, up,
and down separately so stock M5 BF16 rounding boundaries remain observable.

## Exactness

The physical M5 comparison used real layer-20 5/8 checkpoint slices and stock
BF16 `mx.gather_qmm` as the oracle. Every boundary was array-equal with zero
mismatches and zero maximum absolute difference:

1. up projection;
2. gate projection;
3. post-LimitedSwiGLU activation;
4. down projection with a fixed stock activation;
5. composed sorted down rows;
6. inverse-sorted original route rows; and
7. BF16-scored, fixed-slot-reduced local output.

The full machine-readable ABBA result is
`ds4_nax_moe_blocks_m5_2026-08-22.json`.

## Physical rank-1 stage profile

The same normal-power M5 then ran the existing three-representative-layer
stage profiler at the exact 3:5 rank-1 shape. Normalized to the measured
628.76 tok/s distributed reference, its wall attribution is:

| Component | Physical M5 rank-1 wall |
|---|---:|
| Attention/compressor projections | 43.15% |
| Routed MoE pair | 27.48% |
| Routed MoE down | 11.00% |
| Routed MoE total | **38.48%** |
| JACCL collectives, modeled | 7.56% |
| Sparse indexer | 1.57% |
| Miscellaneous | 9.25% |

Applying the independently measured pair and down factors to that physical
mix gives:

```text
speedup = 1 / (1 - 0.27476 - 0.11002
               + 0.27476 / 1.55002
               + 0.11002 / 1.42026)
        = 1.1495x
```

That is a projected **+14.95%** full-pass gain, or about 722.8 tok/s from the
628.76 reference before stacking the separate O-A/finalizer candidates. The
physical profile is saved as
`ds4_tp_stage_profile_3x5_m5_physical_2026-08-22.json`.

## Remaining gates and pitfalls

The seam fails closed before building the candidate block plan unless all of
the following agree:

- explicit flag `OMLX_DSV4_NAX_MOE_BLOCKS=1`;
- the established exact DeepSeek-V4-Flash-0731 config fingerprint;
- TP size 2, rank 1, routed-MoE weights `(3,5)`;
- physical device name `Apple M5 Max` and NAX capability;
- BF16 M=1024/top-6 input, FP32 scores, and the exact 5/8 MXFP4 weight shapes;
- non-training and non-DSpark-verification state; and
- native Python symbol, optional NAX metallib, and NAX device-artifact guards.

After the complete preflight, the candidate alone constructs the existing
BM32 plan. It dispatches separate BF16 up and gate projections, preserves the
stock LimitedSwiGLU boundary, dispatches BF16 down, and rejoins the unchanged
inverse-sort, score, shared-expert, and rank-order all-sum path. There is no
catch-and-retry after candidate graph construction.

Focused coverage exercises a fixed 6,144-route distribution containing an
inactive expert plus experts with 1, 31, 32, and 33 rows. The actual production
block builder produces no block, one partial block, one full block, and two
blocks respectively, while all other experts carry realistic 24/25-row loads.
The full-model gate should still record naturally occurring per-layer counts.

Remaining work:

1. Run fixed-content and independent cold 14K TP prompts with O-A enabled;
   require identical output hashes and at least +10% over the post-O-A control.
2. Record naturally occurring router-count distributions during that gate.
3. Do not substitute the M3 FP16 tail kernel as the oracle. Stock M5 NAX uses
   BF16 projection and activation boundaries and is not array-equal to that
   Steel path.
4. Keep the M3 asymmetric tail candidate available once the faster M5 ceases
   to be the MoE straggler.

An expert-subset/route-ownership layout, like ds4-metal's Mac TP path, is a
plausible later occupancy lever. It is not a drop-in lossless replacement for
the current intermediate-row TP: strict parity requires separately
accumulating the current 3/8 and 5/8 down-K partitions, preserving both BF16
rounding points and rank-order addition, while also exchanging ownership-routed
rows. That larger algebra/collective change should not replace this already
qualified scheduling-only primitive.
