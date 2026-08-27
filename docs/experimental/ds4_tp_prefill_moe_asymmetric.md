# DS4 3:5 TP routed-MoE prefill campaign

This campaign measures the exact asymmetric TP slice selected for the M3 Ultra
rank in a 3:5 plan. It uses real layer-20 checkpoint weights and M=1024 without
stopping or replanning the live server and without touching the remote node.

## Exact slice

The full expert intermediate is 2048. Rank 0 owns rows `[0,768)` and rank 1
owns `[768,2048)`:

| Rank | Weight | Gate/up rows | Down packed columns | Down scale groups |
|---:|---:|---:|---:|---:|
| 0 (M3) | 3/8 | 768 | 96 | 24 |
| 1 | 5/8 | 1280 | 160 | 40 |

The harness is `benchmarks/bench_ds4_tp_prefill_moe_asymmetric.py`. Its rank-1
loader and native shape guards are wired, but only rank 0 was measured here.

## Existing-kernel sweep

All five block variants were array-equal to stock at pair, down, and composed
boundaries. BM32/BN32 variant 2 was the clear winner:

| Variant | BM/BN | Pair | Down | Composed |
|---:|---:|---:|---:|---:|
| 0 | 8/32 | 6.614 ms | 3.552 ms | 9.957 ms |
| 1 | 16/32 | 6.673 ms | 3.485 ms | 9.899 ms |
| 2 | 32/32 | **6.121 ms** | **3.214 ms** | **9.071 ms** |
| 3 | 16/64 | 6.682 ms | 3.559 ms | 10.055 ms |
| 4 | 32/64 | 6.251 ms | 3.270 ms | 9.376 ms |

Variant 2 was 1.123x faster than stock pair, 1.084x faster than stock down,
and 1.121x faster composed. Pair is the dominant primitive at 1.90x the down
time.

## Tail8 asymmetric result

The existing tail8 Metal kernels already take intermediate width as a runtime
constant; only their native ABI had been fixed to 1024. The isolated wrappers
now accept the exact DS4 TP widths `{768,1024,1280}` while the production gate
remains restricted to the already-qualified equal-TP2 width 1024.

Tail8 was compared directly with BM32/BN32 variant 2, not stock. Two runs were
bit-exact at pair, down, and composed boundaries and produced composed
speedups of 1.211x and 1.233x, clearing the requested 1.10x gate. The live
cluster introduced visible thermal/scheduling variance in absolute medians;
the balanced ABBA ratios were nevertheless consistently above the threshold.
Machine-readable results are in
`ds4_tp_prefill_moe_asymmetric_results_2026-08-22.json`.

## Shipping state

There is no asymmetric production dispatch. The symbols remain isolated and
the existing `OMLX_DSV4_MOE_TAIL8=0` gate only accepts B1/M1024 with local
intermediate 1024 on pre-NAX hardware. Rank-1 width 1280 is available for a
later isolated measurement, not automatically selected. Decode is unchanged.
