# DS4 TP prefill MoE phase B: bounded deterministic down

This is an isolated follow-on to the phase-A shared-X campaign. It freezes a
lossless bounded-output schedule for DS4 Flash at `M=1024`, top-6, equal TP2
(TP4/4), local intermediate 1024, and hidden width 4096. Nothing here is wired
into CMake or production dispatch.

## Why a bounded tile

The expert-major down GEMM is efficient because all 24 routed rows for an
expert reuse its output-weight tile. Writing those results directly into the
token row with atomics loses BF16 slot order. Running one expert-major down
pass per slot preserves order but activates all 256 experts in every slot for
the fixed benchmark fixture, rereading down weights exactly 6.0x.

The bounded schedule preserves both properties:

1. Partition the 4096 down-output columns into disjoint supertiles.
2. For one supertile, run the existing BM32/BN32/BK32 expert-major down order
   into an FP16 `[6144, supertile]` scratch tensor.
3. In the next dispatch, each `(token, column)` thread reads original slots
   0 through 5 via `inverse_order`, applies the current casts and BF16 score,
   accumulates BF16 in slot order, and writes the local `[1024,4096]` slice.
4. Reuse the same scratch allocation for the next supertile.

Every global output column belongs to exactly one down dispatch. Total expert
weight and activation-tile workgroups are identical to the current down GEMM;
down-weight read amplification is 1.0x. Dispatch ordering, not an atomic or a
cross-threadgroup spin protocol, separates scratch production, consumption,
and reuse.

## Frozen ABI

The future isolated Python/native symbol is:

```text
deepseek_mxfp4_down_top6_tiled(
    activated_f16,       # [R, 1, I_local]
    down_weight_u32,     # [E, H, I_local/8]
    down_scales_u8,      # [E, H, I_local/32]
    block_meta_i32,      # [max_blocks, 3]
    block_count_i32,     # [1]
    inverse_order_u32,   # [R], route_id -> expert-sorted row
    scores_f32,          # [B, L, 6]
    supertile: int,
) -> output_bf16         # [B, L, H]
```

One primitive owns a single `[R,supertile]` FP16 temporary and encodes all
down/reduction dispatch pairs in one command buffer. The two unbuilt Metal
kernels are sketched in
`benchmarks/prototypes/ds4_tp_prefill_moe_phase_b.metal`.

## Exactness contract

For every route output and every hidden column:

1. down accumulates with the existing Steel MMA K order;
2. the down result narrows to FP16 in scratch;
3. the reduction casts that FP16 value to BF16;
4. its FP32 route score casts to BF16;
5. multiplication produces BF16; and
6. BF16 accumulation runs in original slot order 0, 1, 2, 3, 4, 5.

No atomic operation is permitted. Promotion requires `mx.array_equal` against
the current full down + unsort + cast + score + sum result at `[1,1024,4096]`
for every candidate on both M3 Ultra and M5 Max.

## Scratch and dispatch sweep

Each row is the complete per-layer primitive, including down and reduction:

| Supertile | Scratch | Tile pairs | Dispatches/layer | Dispatches/43 layers |
|---:|---:|---:|---:|---:|
| 128 | 1.5 MiB | 32 | 64 | 2,752 |
| 256 | 3 MiB | 16 | 32 | 1,376 |
| 512 | 6 MiB | 8 | 16 | 688 |
| 1,024 | 12 MiB | 4 | 8 | 344 |
| 2,048 | 24 MiB | 2 | 4 | 172 |
| 4,096 | 48 MiB | 1 | 2 | 86 |

The 4,096 point is the full-route scratch control. The useful tradeoff is
expected to lie between 512 and 2,048, but that is only a hypothesis. The
128/256 points quantify dispatch sensitivity; none may be promoted without
the two-device GPU gate.

## Analytical byte ceilings

At the fixed shape the full FP16 route tensor is 48 MiB and the local BF16
output is 8 MiB. Counting explicit current graph payload boundaries gives:

- current down-tail payload: 392 MiB per layer;
- bounded candidate payload: 104 MiB per layer;
- tail-only payload ceiling: 3.77x;
- down compute plus tail ceiling: about 1.13x, because 544 MiB of down weights
  and 1,536 MiB of logical activation-tile reads are unchanged.

With routed MoE modeled as half of prefill time, phase B alone has only about
a 1.02x shape-derived end-to-end ceiling. Combined with the phase-A shared-X
byte model, bounded A+B is about 1.16x end to end. These are optimistic byte
ceilings that ignore extra dispatch cost and cache behavior, not speed claims.
The ideal no-mid A+B ceiling from the parent campaign remains separate; this
phase deliberately retains the 12 MiB activated mid until a persistent expert
kernel proves faster.

Metadata upper bounds are also explicit: a naive per-output implementation
logically addresses 96 MiB of inverse ids and 96 MiB of scores per layer, even
though both should cache or broadcast heavily. Native profiling must confirm
that the reduction kernel does not turn those small logical tables into a new
bandwidth bottleneck.

## Later GPU gate

After a native agent wires only the isolated symbol, run the same sweep on
each machine:

```bash
python benchmarks/bench_ds4_tp_prefill_moe_phase_b.py \
  --model /path/to/DS4-Flash --device-label m3-ultra --rank 0 --strict

python benchmarks/bench_ds4_tp_prefill_moe_phase_b.py \
  --model /path/to/DS4-Flash --device-label m5-max --rank 1 --strict
```

The harness benchmarks only activated-mid through local routed output, uses
pairwise ABBA against the current path, reports every supertile separately,
and requires exact parity. No result is claimed by this document because the
prototype has not been compiled or run on either GPU.
