# DS4 MTP attention/indexer handoff — 2026-08-23

## Goal

Continue the lossless DS4 TP2 MTP depth-5 lane toward **>=130 tok/s** while
keeping the parallel targets visible: **>=1,000 tok/s cold prefill**,
**50–80 tok/s non-MTP decode baselines**, and the MTP decode target.  The
previous B=6 full-MoE candidate was rejected end-to-end (66.3 vs 66.7 tok/s;
completion hash changed), so this iteration is limited to the remaining
attention/indexer/hotset work.

## Current status

- Qualified fixed-depth-5 MTP baseline remains about **69.4 tok/s**
  (**70.13 peak**), with an 86.46 ms six-token cycle.
- Existing M5 attribution puts attention projections at **11.74 ms/cycle**,
  sparse indexer at **5.24 ms/cycle**, and the combined attention+indexer
  hotset at **19.63% of wall (16.98 ms/cycle)**.  Even removing that entire
  bucket would only reach about 86 tok/s; it cannot independently provide a
  15% E2E speedup.
- This iteration changed no production dispatch, defaults, runtime, deployment,
  server state, or kernel source.

## Iteration evidence

### Physical DSpark stage profile on M3

Command (benchmark-only, exact verification arm):

```bash
/Users/jonathanspangler/omlx-v2-build/bin/python \
  benchmarks/bench_ds4_tp_stage_profile.py \
  --model /Users/jonathanspangler/model-staging/DeepSeek-V4-Flash-0731-FP8 \
  --rank {0,1} --shard-weights 3,5 --tokens 6 --prefix-tokens 65 \
  --layers 0 2 3 --warmup 1 --cycles 2 --projection-detail --fine-detail \
  --dspark-verify --baseline-tps 69.4 --target-tps 130
```

Both rank geometries ran on `applegpu_g15d` (Apple M3 Ultra) with
`dspark_verify=true`, strict local array evaluation, and representative real
layers 0/2/3 (compression ratios 0/4/128).  The profiler reports modeled TP
collectives separately; it does not claim a distributed critical-path wall.

| M3 local shape | attention projections | indexer | routed pair | routed down | misc | modeled collectives |
|---|---:|---:|---:|---:|---:|---:|
| rank 0, 24 heads / I=768 | 18.49 ms isolated | 8.70 | 26.28 | 15.76 | 178.89 | 3.26 ms |
| rank 1, 40 heads / I=1280 | 18.45 ms isolated | 9.30 | 28.61 | 16.46 | 191.68 | 3.26 ms |

These short-row measurements are diagnostic only: the profiler's projection
brackets do not capture all quantized decode operations at six rows, so the
qualified M5 normalized roofline remains the serving attribution for Amdahl
decisions. Raw local outputs were written during the run to:

- `/tmp/ds4-mtp-depth5-attn-indexer-m3.json`
- `/tmp/ds4-mtp-depth5-attn-indexer-m3-r1.json`

### Exact projection/indexer grouping probe

The existing `bench_ds4_tp_projection_profile.py` grouping probe was run with
real layer 2, `tokens=6`, `prefix_tokens=65`, and 2 warmups / 4 interleaved
repetitions on both signed 3:5 rank shapes.

| candidate | rank 0 median | rank 1 median | parity | scope |
|---|---:|---:|---|---|
| `concat_indexer_dense_triplet` | **1.181x** | **1.152x** | all three output arrays equal on both ranks | indexer compressor WKV + gate + weights projection |
| `compile_concat_q_input_chain` | 1.104x | 1.077x | all q/raw/index-q arrays equal on both ranks | Q-A + raw KV + Q-B + indexer-Q graph |
| `compile_output_chain` | 0.981x | 0.985x | output array equal | O-A/O-B only; no gain |
| `concat_main_compressor_pair` | not eligible | not eligible | second output differs | rejected as lossy |
| `concat_all_dense_direct_x` | not eligible | not eligible | second output differs | rejected as lossy |

The indexer dense-triplet candidate is the best measured exact candidate, but
the M5 indexer bucket is only 6.06% of the qualified MTP wall.  At the slower
5-unit measured 1.152x local speedup, Amdahl projects only about **69.96 tok/s**
from a 69.4 tok/s baseline (about +0.56 tok/s).  It is therefore retained as
a benchmark prototype, not a promotion candidate.

## M5 status and applicability

The available M5 evidence for this grouping probe is the existing qualified
TP2 stage/roofline and transport artifacts; no new physical M5 grouping run was
completed in this iteration. The DS4 MXFP4 checkpoint is in fact staged on the
M5 at the same `.lmstudio` path as the Studio and was used by the separate
physical full-MoE gates. The exact grouping code is single-node/rank-local and
must still be re-gated on physical M5 before any promotion; the M3 result is
not an M5 performance claim.

## Decision / next iteration

No attention/indexer candidate in this cycle plausibly closes a **>=15% E2E**
gap while preserving strict array/token parity.  Keep the grouping probe
benchmark-only.  Next work should either:

1. repeat the depth-5 grouping gate on the already-staged M5 checkpoint with
   balanced A/B timing, or
2. prototype a purpose-built exact Metal projection bank that can cover the
   full attention+indexer 19.63% bucket; stock `mx.compile` and broad dense
   concatenation are already ruled out by the evidence above.

Promotion still requires real-layer parity, balanced physical M3/M5 timings,
the cold TP2 depth-5 gate, unchanged completion hash, and repeated steady
measurements.  Do not alter serving defaults until all gates pass.
