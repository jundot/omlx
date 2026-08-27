# DS4 TP2 MTP verification roofline

Date: 2026-08-23. Hardware: Apple M3 Ultra rank 0 plus Apple M5 Max
rank 1, signed 3:5 tensor split. Model:
`DeepSeek-V4-Flash-0731-MXFP4-MLX`. All experiments were lossless.

## Result

The remaining MTP cycle is target-verification bound. The qualified fixed
depth-5 rate was about **69.4 tok/s steady** (**70.13 tok/s peak**). A depth-5
cycle can emit at most six tokens, so 130 tok/s requires a cycle no longer
than 46.15 ms even at perfect acceptance. The qualified rate corresponds to
86.46 ms per six-token cycle: at least 40.31 ms must be removed.

Existing rank telemetry agrees on the owner of that budget. Across 21 cycles
it recorded 1,972.09 ms in the target-backbone bucket, 137.25 ms in DSpark
head dispatch, 36.29 ms in cache operations, and 1.58 ms in sampling. That is
93.91 / 6.54 / 1.73 / 0.08 ms per cycle. The head is asynchronously queued,
so some unfinished head work may resolve under the following backbone sync;
the safe conclusion is that the combined target/head GPU cycle dominates,
not that every millisecond is a pure target-layer measurement.

ThunderMLX's proven speculative architecture does not expose a missing loop
optimization here. oMLX already follows the same important structure:
constant K+1 verification geometry, coordinator-owned token decisions, one
tiny fixed decision packet, identical target forwards on every rank, and
rank-local rollback from the synchronized accepted prefix. Rewriting the
engine loop would not address the measured weight-streaming cost.

## Exact M=6 stage attribution

`bench_ds4_tp_stage_profile.py --dspark-verify` now arms the same exact
short-row projection/attention path and rollback stash as serving. Real
checkpoint layers 0/2/3 represent compression ratios 0/4/128; their measured
mix is normalized to the 69.4 tok/s wall. JACCL traffic uses the measured
6.2 GB/s and 30 us constants.

| bucket | M3 wall | M5 wall | M5 ms/cycle |
|---|---:|---:|---:|
| routed gate/up | 38.70% | **40.46%** | 34.98 |
| routed down | 9.96% | **12.49%** | 10.80 |
| attention projections | 14.74% | 13.57% | 11.74 |
| sparse indexer | 5.82% | 6.06% | 5.24 |
| collectives | 3.77% | 3.77% | 3.26 |
| misc | 27.02% | 23.64% | 20.44 |

The M5 straggler therefore spends **52.96% / 45.79 ms** of the normalized
cycle in routed MoE. Doubling routed MoE alone projects only 94.39 tok/s;
removing it entirely projects 147.53 tok/s and still requires an implausible
8.35x speedup to reach 130 with everything else fixed. The full measured
kernel hotset (routed MoE + attention + indexer) needs about **2.79x**. Even
including perfect collective removal, the required group speedup is 2.57x.

An exclusive fine-detail run splits the M5 misc bucket approximately as
follows. These rows are a prioritization aid, not additive physical wall
measurements: extra barriers deliberately serialize each nested boundary,
then the result is scaled back into the original 20.44 ms misc wall.

| misc sub-bucket | normalized M5 ms/cycle |
|---|---:|
| HyperConnection | 5.06 |
| attention output O-A/O-B chain | 3.74 |
| norms | 2.18 |
| shared expert | 1.71 |
| attention Q-B | 1.57 |
| attention Q/KV bank | 1.36 |
| router | 1.30 |
| remainder | 3.52 |

No non-MoE sub-bucket can close the target independently. Eliminating the
largest one, HyperConnection, would move 69.4 to only about 73.7 tok/s.

## Rejected bounded candidates

The real-weight vocabulary-head sweep changed only output ownership; every
candidate retained the production K-to-lane map and reduction order. All 36
shape/schedule boundaries were array-equal on both Macs. M3 saved about
0.13 ms per cycle. On the critical M5 rank, the best absolute M=6 target and
M=5 draft schedules saved only 0.047 and 0.017 ms; the production Markov
schedule remained fastest. The total **0.064 ms/cycle (<0.1%)** bound rejected
the change without a live reload.

A one-dispatch M=6 HyperConnection expansion was storage-bit exact and 1.16x
faster on M3, but no tested reduction order was storage-bit exact on M5. It
was rejected and no runtime/kernel source change remains.

The previously available verify attention-finalizer gate was also left off:
its prior fixed-depth TP2 canary measured 68.5-69.7 tok/s against about 69.5
control with an identical completion hash.

## Required next design: B=6 routed-MoE plus verify hotset

The next campaign must target the actual fixed-depth-5 geometry, not promote
a B<=4 result by extrapolation:

1. Qualify `deepseek_mxfp4_full_decode` at exactly B=6 on both 3:5 slices.
   Keep gate/up, limited SwiGLU, down, BF16 route-output cast, score cast, and
   six-slot ordered accumulation array-equal to the composed reference.
2. Tune the pair and down row tiles independently for M3 I=768 and M5 I=1280.
   The current primitive uses one M3 row only for I=768 and two elsewhere;
   B=6 needs its own occupancy gate.
3. Cover both disjoint/high-entropy 36-route blocks and repeated/skewed expert
   blocks. Expert grouping may improve cache locality, but per-route outputs
   must be restored before the existing slot-ordered reduction so arithmetic
   does not change.
4. Promote only after real-layer array equality, balanced physical A/B on both
   chips, then a cold TP2 fixed-depth-5 HTTP gate with identical completion
   hash, acceptance/depth counters, and at least five steady repetitions.
5. After routed MoE, attack attention projection/indexer together. The M5
   roofline says the complete kernel hotset needs roughly 2.79x; a routed-only
   win cannot by itself guarantee 130 tok/s.

## Reproduce

```bash
python benchmarks/bench_ds4_tp_stage_profile.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --rank 1 --tokens 6 --prefix-tokens 65 \
  --warmup 2 --cycles 5 --projection-detail --fine-detail \
  --dspark-verify --baseline-tps 69.4 --target-tps 130 \
  --output /tmp/ds4-mtp-verify-stage.json

python benchmarks/bench_ds4_mtp_vocab_head_tiles.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --rank 1 --warmup 8 --cycles 40 \
  --output /tmp/ds4-mtp-vocab-head.json
```

Raw confirmation artifact SHA-256 values:

- M3 stage: `8b8a1ee491a201bb9e8a9a543c4ac4bf743b97286a1493d24bb7ed6095935c6c`
- M5 stage: `9a65aa0e32e84ebd9f7b620be8f0c65e43d205f016d96358a2780280ca3ed13e`
- M3 head: `a48c7329537c9bcf5c84780e1a7e95ae9662a37135e25053c8bb4f93be49c19b`
- M5 head: `6689b4d5739a1fe7739ea748e099d8656f732dacd2f72539ea411771bfd9434b`
