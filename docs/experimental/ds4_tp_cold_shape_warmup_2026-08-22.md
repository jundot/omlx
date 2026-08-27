# DS4 TP cold-prefill shape warmup

## Problem

A freshly loaded 3:5 DS4 TP process served its first independent 14,106-token
prompt at 516.8 tok/s. Two later prompts in the same process, both with
`cached_tokens=0`, served at 676.4 and 656.4 tok/s. The missing time is about
5.8 seconds and must not be reported as steady-state model throughput.

## Rank-local isolation

The real-layer profiler was run in a fresh MLX process on the M3 rank with the
production 3:5 tensor shapes, 1,024 tokens, no prefix, no warmup cycles, and
`mx.clear_cache()` between measured cycles. The progressive test loader calls
`mx.eval(layer.parameters())` before any forward. The output is preserved at
`/private/tmp/ds4-first-shape-r0.json` on the test machine.

| Compression ratio | First 1K forward | Hot range | First/hot |
|---:|---:|---:|---:|
| 0 | 79.17 ms | 31.75-32.58 ms | 2.46x |
| 4 | 40.12 ms | 35.20-39.50 ms | 1.12x |
| 128 | 42.86 ms | 37.80-39.07 ms | 1.11x |

The ratio-0 first forward spent 42.78 ms in Q-A versus 1.81-2.13 ms after the
shape was hot. Later layers used different weight arrays but inherited the hot
Q-A shape. That is the signature of first-use Metal compute-pipeline creation,
not a lazy parameter left unevaluated.

Two fresh-cache 14 x 1K sweeps then separated first-use shape cost from growing
context cost:

| Layer | Chunk 1, first sweep | Chunk 1, second sweep | Chunks 2-14 result |
|---|---:|---:|---|
| ratio 4 | 62.41 ms | 25.81 ms | matched within ordinary run noise |
| ratio 128 | 70.03 ms | 25.02 ms | matched within ordinary run noise |

The ratio-4 first chunk recorded no major page faults. New cache objects and an
allocator purge were used for the second sweep, yet the first-use gain
persisted. Later first-sweep context sizes (2K through 14K) did not pay another
shape penalty. These controls reject paged-cache metadata allocation, reusable
allocator growth, and per-context attention-shape compilation as the dominant
cause. Weight residency may still contribute on a full rank, but the same
single forward also touches every resident shard and therefore covers it.

## Implementation

`install_runtime_optimizations` now publishes the exact 1K inner shape only
when DS4 adaptive TP prefill is active. After all ranks publish `rank_ready`,
the existing supervisor release marker provides the safe load barrier. Each
rank then executes one lockstep full-model forward with a temporary prompt
cache and `skip_lm_head=True` before the HTTP listener starts. It evaluates all
cache state, drops every Python reference, and calls `mx.clear_cache()`; Metal's
in-process pipeline-state cache remains hot.

The path is bounded to at most 4,096 tokens, does not run on single-node or
pipeline-parallel deployments, and is disabled with
`OMLX_CLUSTER_PREFILL_SHAPE_WARMUP=0`. Unknown models do not publish shape
metadata and are unchanged.

The full distributed acceptance gate remains an unload/reload A/B: the first
14,106-token `cached_tokens=0` request must move into the same band as the next
two independent prompts, with unchanged output hashes and no rank-memory or
JACCL regression.
