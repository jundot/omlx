# DS4 TP2 cold-prefill taper attribution

Date: 2026-08-23. Device: Apple M3 Ultra. Branch:
`feat/cluster-v2-dflash2`.

## Result

The ratio-4 sparse indexer explains essentially all measured context-dependent
prefill taper. The other large buckets (attention projections and routed MoE)
remain important for the absolute 850–1,000 tok/s target, but their work does
not grow with the retained prefix.

An isolated BF16 replay used the production MMA score kernel, deterministic
top-512, final temporal sort, 512 query rows per TP rank, and all 21 ratio-4
layers. Nine interleaved A/B repetitions produced:

| context | pooled rows | 21-layer indexer | observed TP2 | observed 1,024-token wall | indexer share of wall |
|---:|---:|---:|---:|---:|---:|
| 30K | 7,500 | 83.328 ms | 870 tok/s | 1,177.011 ms | 7.1% |
| 100K | 25,000 | 251.074 ms | 784 tok/s | 1,306.122 ms | 19.2% |
| 250K | 62,500 | 608.929 ms | 619 tok/s | 1,654.281 ms | 36.8% |

From 30K to 100K, measured indexer growth is 167.746 ms while live wall
growth is 129.111 ms (1.30x). From 100K to 250K, indexer growth is 357.855 ms
while live wall growth is 348.159 ms (1.03x). The residual after subtracting
the indexer stays approximately flat: 1,094 / 1,055 / 1,045 ms. Ratios above
one are not an Amdahl claim; they indicate normal isolated-vs-live variance and
make the narrower conclusion stronger: no second context-growing component is
needed to explain the observed slope.

## Audited alternatives

- The ThunderMLX top-k reuse pattern applies to decode, where nearby steps may
  intentionally reuse a selection under its model-specific contract. Cold
  prefill has a distinct query for every row; reusing those selections would
  change model output and is not a lossless option.
- ds4-metal's tiers are device/layer placement plus per-tier scratch ownership.
  They reduce allocation and movement overhead but still scan the full indexer
  key extent, so they do not remove this context slope.
- oMLX's certified hierarchical indexer remains correctly default-off. Its
  exact certificate is fail-closed, but the current screen still performs
  full-width upper-bound construction/selection and periodically rebuilds
  derived state. There is not yet physical evidence to enable it.
- Folding the final temporal sort into the deterministic native top-k writer
  was implemented and verified against adversarial cutoff ties in BF16/FP16.
  The 21-layer physical A/B was neutral at 100K/250K and regressive at 30K, so
  both source and metallib were reverted.

The next lossless taper campaign should therefore target a fused score/select
primitive that does not write and reread the complete `[rows, pooled]` score
sheet, or a certified screen whose total full-pool work is demonstrably below
the saved exact scan. Projection/MoE work remains the parallel campaign for
raising the context-independent floor.

## Score/select decomposition and bounded streaming rejection

A later production-runtime replay isolated the real `L=512, P=62,500` graph:
the qualified MMA score kernel took **28.005 ms**, deterministic top-512 took
**0.749 ms**, and temporal sorting took **0.252 ms** (**28.660 ms combined**).
Score generation is therefore 97.7% of this measured path. Eliminating only a
top-k dispatch or the final score-sheet reread cannot materially change the
250K taper.

An exact chunk-streamed implementation was also physically rejected. It
scores contiguous pooled-key chunks, selects each exact local top-512, merges
the candidates in global temporal order, and preserves the external causal
mask. At the same `L=512, P=62,500` shape, the complete full-sheet graph was
**28.51 ms**; 16K chunks took **30.05 ms** and the best 32K chunks took
**29.08 ms**, with identical final indices. The next hierarchy must reduce
score arithmetic through a certified native screen; graph chunking by itself
only adds launch/merge overhead.

The existing rank-48 certificate remains the useful mathematical foundation,
but its Python/MLX implementation is not a promotion candidate. A productive
native version must emit grouped upper bounds or compact candidates without a
full `[L,P]` approximate sheet, reuse the existing BF16 exact MMA rescore, and
fail closed to the full scan whenever every omitted key is not proven below
the exact cutoff.

## Exact M3 MMA partition follow-up

A bounded follow-up kept the production MMA kernel's 64x64 output tile,
128-thread group, 16KB resident K panel, MMA reduction order, and learned-head
reduction order, but repartitioned its four simdgroups from WM2xWN2 to WM4xWN1.
That halves device Q-fragment loads in exchange for rereading B fragments from
threadgroup memory. Complete BF16 score sheets and deterministic top-512
indices were bit-identical to both WM2xWN2 and Steel, including unaligned
boundary tiles.

On the M3 Ultra (`applegpu_g15d`), a 15-repetition interleaved physical gate
measured positive medians at every taper context:

| context | WM2 score | WM4 score | score gain | WM2 21-layer graph | WM4 21-layer graph | graph gain |
|---:|---:|---:|---:|---:|---:|---:|
| 30K | 4.874 ms | 4.699 ms | 1.0373x | 87.350 ms | 86.506 ms | 1.0098x |
| 100K | 12.949 ms | 12.863 ms | 1.0067x | 267.321 ms | 262.302 ms | 1.0191x |
| 250K | 31.368 ms | 30.967 ms | 1.0130x | 653.971 ms | 643.753 ms | 1.0159x |

The identical raw candidate was also gated on M5 Max (`applegpu_g17s`) and
rejected: its 21-layer ratios were 0.9930x / 0.9921x / 0.9924x. Dispatch is
therefore fail-closed on the exact architecture string: only
`applegpu_g15d` requests WM4xWN1; M5 and every unknown architecture retain the
qualified WM2xWN2 path in both Python and the native primitive.

## Reproduce

The profiler now chooses the same qualified MMA-vs-Steel route as serving,
checks MMA scores bit-for-bit against Steel before timing, emits JSON, and can
attribute measured live rates directly:

```bash
/Users/jonathanspangler/omlx-v2-build/bin/python \
  benchmarks/bench_ds4_indexer_threshold.py \
  --query-tokens 512 \
  --logical-chunk-tokens 1024 \
  --pooled-tokens 7500,25000,62500 \
  --observed-tps 870,784,619 \
  --score-kernel auto \
  --output /tmp/ds4-indexer-taper.json
```

Re-run the WM partition gate on an idle qualified M3 Ultra with:

```bash
/Users/jonathanspangler/omlx-v2-build/bin/python \
  benchmarks/bench_ds4_mma_indexer_partition.py \
  --query-tokens 512 \
  --logical-chunk-tokens 1024 \
  --pooled-tokens 7500,25000,62500 \
  --layers 21 --warmup 4 --repeats 15 \
  --output /tmp/ds4-wm4-gate.json
```

For a single-node 1,024-token chunk, use `--query-tokens 1024` with the same
`--logical-chunk-tokens 1024`. Run physical profiling only while the serving
GPU is idle.
