# DS4 TP 3:5 rank-local stage profile — 2026-08-22

This is a default-off, isolated diagnostic. It did not import into serving,
start a server, or contact the M5. The harness lazy-loaded three real DS4
checkpoint layers on the M3 Ultra, sliced them to the signed 3:5 TP geometry,
and measured a complete 1,024-token block forward after an 8,192-token cache
prefix. Ratio 0/4/128 representatives were weighted by the real 2/21/20 layer
schedule. JACCL time was modeled separately from the measured 6.2 GB/s and
30 us link, including 86 activation all-sums and 21 high-context indexer
all-gathers.

Rank 0 is the real M3 3/8 geometry. Rank 1 is the M5 5/8 **shape replayed on
the M3**, so its absolute milliseconds are not an M5 claim. The similar
normalized mixes make the priorities useful now; the same script must run on
the M5 to identify the exact distributed bottleneck rank.

## Real-layer medians

| 3:5 rank shape | local heads | local MoE width | ratio-0 layer | ratio-4 layer | ratio-128 layer |
|---|---:|---:|---:|---:|---:|
| rank 0 / 3 units | 24 | 768 | 25.26 ms | 34.39 ms | 30.90 ms |
| rank 1 / 5 units | 40 | 1,280 | 32.74 ms | 46.70 ms | 43.84 ms |

## Wall attribution normalized to 628.76 tok/s

| component | rank-0 mix | rank-1-shape mix | useful envelope |
|---|---:|---:|---:|
| attention + compressor projections | 45.3% | 44.1% | **44–46%** |
| routed MoE pair | 22.2% | 24.1% | 22–24% |
| routed MoE down | 9.8% | 11.1% | 10–11% |
| sparse indexer | 4.6% | 3.5% | 3–5% |
| JACCL collectives | 7.6% | 7.6% | ~8% |
| norms, HC, routing, attention kernel, shared expert, other | 10.5% | 9.6% | 10–11% |

The projection bucket includes Q-A/Q-B, KV, compressor projections, and the
O-A/O-B chain. It excludes SDPA itself, which remains in `misc`. Routed MoE is
31–35% of exposed wall, not enough to close the target by itself.

## Amdahl table: 628.76 -> 1,000 tok/s

The required wall ratio is 0.62876. “Infinite ceiling” means every nanosecond
in that component disappears while all other work stays unchanged.

| intervention | required speedup / projection | rank-0 result | rank-1-shape result |
|---|---:|---:|---:|
| attention projections alone | 5.53–6.33x | 1,000 tok/s | 1,000 tok/s |
| routed MoE alone | impossible | 924.4 ceiling | 970.4 ceiling |
| indexer alone | impossible | 659.4 ceiling | 651.8 ceiling |
| collectives alone | impossible | 680.2 ceiling | 680.2 ceiling |
| attention + MoE, same factor | **1.81–1.83x** | target | target |
| attention 2x, then MoE | **1.75–1.83x MoE** | target | target |
| MoE 2x, then attention | **1.79–1.88x attention** | target | target |
| attention 2x + MoE 2x | fixed scenario | 1,024.8 tok/s | 1,041.8 tok/s |
| fully hide collectives, then attention + MoE | **1.59–1.62x** | target | target |

## Decision

The fastest credible route to 1,000 is a **joint projection + routed-MoE
campaign**, not MoE alone and not transport alone:

1. Fuse/group the Q-A, KV, compressor, and output-LoRA projection chains. This
   is the largest exposed bucket at 44–46%.
2. Build the full routed-MoE prefill primitive (shared-X gate/up, LimitedSwiGLU,
   down scatter/ordered score reduction). It must approach roughly 1.8–2.0x
   together with the projection work.
3. Overlap the two per-layer activation reductions once the compute kernels are
   faster. Today perfect collective removal is only a move from 629 to ~680;
   after kernel speedups it lowers the required compute factor to ~1.6x.
4. Keep the exact NAX indexer work bounded. Its whole current bucket is only
   3–5%, so it cannot be the lead campaign.

Reproduce rank 0 locally:

```bash
/Users/jonathanspangler/omlx-v2-build/bin/python \
  benchmarks/bench_ds4_tp_stage_profile.py \
  --model /Users/jonathanspangler/.lmstudio/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --rank 0 --shard-weights 3,5 --tokens 1024 --prefix-tokens 8192
```

Run the same command with `--rank 1` on the M5 to turn the shape replay into a
physical-rank critical-path measurement. Reports are JSON plus a rendered
Amdahl table; `--output PATH` saves the JSON for comparison.
