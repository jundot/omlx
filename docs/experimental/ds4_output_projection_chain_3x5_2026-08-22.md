# DS4 M=1024 exact output-projection chain, 3:5 TP

This is an isolated M3 Steel prototype. It is not selected by the model and it
does not alter serving, collectives, checkpoint loading, or the live cluster.

## Result

One native primitive now executes the DeepSeek-V4-Flash O-A and O-B
projections back to back while retaining the mandatory BF16 boundary. The O-A
epilogue writes that boundary directly in O-B's token-major layout. This
removes the separate transpose/materialization destination and keeps the
intermediate private to the command encoder.

On real layer-2 weights, the BM64/BK32/BN32 Steel variant was bitwise equal at
the O-A BF16 boundary and final O-B output for three independent inputs on
both 3:5 ranks. A 20-cycle A-B-B-A confirmation measured:

| Local shape | Stock chain | Native chain | Speedup |
|---|---:|---:|---:|
| H24 / O-A K=1536 | 6.675 ms | 6.140 ms | **1.087x** |
| H40 / O-A K=2560 | 7.763 ms | 7.201 ms | **1.078x** |

The O-A-plus-layout boundary alone improved from 2.045 to 1.886 ms on H24
(1.085x) and from 3.154 to 2.919 ms on H40 (1.080x). The shorter eight-cycle
run independently measured 1.062x and 1.079x full-chain speedups. Both runs
selected the same BM64/BK32/BN32 pair.

A separate cross-layer screen used real layers 0, 2, and 3 (the model's
ratio-0, ratio-4, and ratio-128 attention classes). All six rank/layer cases
were exact and selected the same variant; short-run gains ranged from 1.076x
to 1.095x.

This is a real exact gain on both ranks, but it is not yet routed into the
model. Full-model promotion must wait for a clean cluster A/B and the M5 NAX
extension described below.

## Recovered real layout

Both ranks receive BF16 attention output in eight O groups:

| Boundary | H24 / rank 0 | H40 / rank 1 |
|---|---:|---:|
| O-A input | `[1,8,1024,1536]` BF16 | `[1,8,1024,2560]` BF16 |
| O-A packed weight | `[8,1024,384]` U32 | `[8,1024,640]` U32 |
| O-A scales | `[8,1024,48]` U8 | `[8,1024,80]` U8 |
| O-A logical result | `[1,8,1024,1024]` BF16 | same |
| O-B input | `[1,1024,8192]` BF16 | same |
| O-B packed weight | `[4096,2048]` U32 | same |
| O-B scales | `[4096,256]` U8 | same |
| O-B result | `[1,1024,4096]` BF16 | same |

Each U32 packs four MXFP8 values. Each U8 scale covers a 32-value contraction
group. The logical O-A weights are therefore 8 independent
`1024 x {1536,2560}` matrices; O-B is one `4096 x 8192` matrix.

The 16 MiB O-A result is a mandatory numerical boundary. Stock O-A stores it
group-major as `[1,8,1024,1024]`. The model then transposes groups behind the
token axis and flattens to `[1,1024,8192]`. That view is not row-contiguous for
O-B, so the stock graph materializes a second 16 MiB destination. An isolated
copy control costs about 0.27 ms on this M3 Ultra.

## Exact arithmetic contract

The Metal reduction body is the MLX `fp_qmm_t_impl` algorithm with one change:
the result-store row stride and group offset are explicit. Everything before
the store remains unchanged:

- group-32 MXFP8 dequantization;
- BK32 K walk;
- FP32 `BlockMMA` accumulators and accumulation order;
- BF16 result conversion in the O-A epilogue;
- the same BK32 FP32/BF16 contract for O-B.

The boundary-probe symbol returns O-A's private token-major BF16 tensor. The
benchmark requires `mx.array_equal` at three points for every tile variant and
seed: O-A BF16, stock O-B fed by native O-A, and the native chain's final O-B.
All checks have zero maximum absolute error.

## Why the prototype has two dispatches

Removing the BF16 boundary itself would be lossy. Algebraically composing O-A
and O-B moves the BF16 rounding frontier and is disallowed.

A single-dispatch streaming kernel also has a hardware-sharing problem. Each
O-A latent tile is consumed by all 4,096 O-B output columns, but independent
O-B threadgroups cannot share threadgroup memory. A strictly exact kernel must
therefore choose one of two bad schedules:

1. recompute O-A once per O-B N tile, multiplying O-A weight traffic; or
2. keep one threadgroup per M tile and spill/reload the full FP32 O-B partial
   sheet between latent tiles, sharply reducing occupancy and adding hundreds
   of MiB of scratch traffic per layer.

The current design avoids both traps. It launches O-A once, stores exactly one
BF16 intermediate directly in its consuming layout, and launches O-B in the
same primitive. It removes only redundant storage traffic and dispatch/graph
surface; it does not claim impossible cross-threadgroup SRAM sharing.

## M5 NAX extension path

The primitive ABI and private-intermediate layout are backend-neutral. The
clean M5 follow-up is an optional NAX O-A epilogue that uses the same
token-major address mapping, followed by an independently gated O-B tile. The
known rank-1 O-A NAX shape is `[1,8,1024,2560]` with
BM64/BK64/BN64/WM2/WN2. Promotion still requires:

1. bitwise equality at the private O-A BF16 boundary;
2. bitwise equality after O-B;
3. M5 A-B-B-A timing for H40;
4. a fallback to this Steel primitive or the untouched stock graph before any
   native node is enqueued.

No M5 benchmark or remote artifact change was made in this campaign.

## Reproduction

After rebuilding the GLM kernel extension in this worktree:

```bash
python benchmarks/bench_ds4_output_projection_chain.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --layers 2 --ranks 0 1 --warmup 5 --cycles 20 \
  --parity-seeds 3 --strict
```

The strict gate fails unless every numerical boundary is array-equal and each
rank has at least one faster exact variant.
