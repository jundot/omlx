# DeepSeek V4 Flash prefill optimization on M3 Ultra

Validated: 2026-08-07

Baseline: `49ec271676ba9c14bbebb75da1912e3fcb5fb0f4`

Machine: Mac Studio, M3 Ultra, 512 GB unified memory
Model: DeepSeek V4 Flash 0731, official native mixed FP4/FP8 weights

## Result

This branch is the approved M3 Ultra candidate. It improves the controlled
17,219-token cold-prefill workload by 14.18% with no measured objective
accuracy loss.

| Metric | Untouched baseline | Candidate | Change |
|---|---:|---:|---:|
| Median cold prefill, 5 runs | 35.97 s | 30.87 s | -14.18% |
| Median prompt throughput | 478.67 tok/s | 557.81 tok/s | +16.53% |
| Fixed 5,476 + 512 token total | 25.27 s | 22.73 s | -10.05% |
| Fixed decode throughput | 41.12 tok/s | 43.53 tok/s | +5.86% |
| Paired objective accuracy | 717/864 (82.99%) | 718/864 (83.10%) | +0.12 points |

The paired 95% bootstrap interval for the accuracy delta is -0.58 to +0.81
percentage points, which passes a -1 point non-inferiority margin. Prediction
agreement was 92.94%, and exact raw-response agreement was 81.25%. The
optimized numerical path therefore changes some generated wording, but did not
reduce measured aggregate correctness.

## Controls

- identical prompt and model artifact on both sides;
- exactly 17,219 API prompt tokens for the primary benchmark;
- five cold repetitions per route with `cached_tokens=0`;
- temperature 0 and exact `READY` output assertion;
- native kernels enabled;
- balanced memory guard;
- upstream 2,048-token prefill step;
- embedded DSpark enabled with three stages and draft width three;
- MTP enabled with `mtp_num_draft_tokens=3`;
- no model download and no quantization change.

## What changed

The candidate combines six separately screened changes:

1. Keep the faster MXFP4 BM16 route through the 12,288-route production shape.
2. Route ratio-128 compressed attention through the existing native
   local-plus-pooled sparse kernel with sequential pooled indices.
3. Use that native route for ratio-4 first-chunk all-pooled attention.
4. Skip indexer scoring while all pooled rows fit under top-k, while preserving
   the indexer compressor/cache update.
5. Fuse hyperconnection prefill expansion in a Metal kernel.
6. Fuse 256-expert route counting, ordering, inverse mapping, and MXFP4 block
   plan construction.

The changes are gated to the proven DeepSeek V4/MXFP4 shapes and retain the
existing fallbacks elsewhere.

## Bottleneck evidence

Synchronized component profiling on untouched upstream attributed 33.91% of
prefill time to routed `SwitchGLU`, 25.66% to sparse attention, and 24.33% to
compressed attention. The indexer was not the primary bottleneck. Metal System
Trace also showed a GPU-compute-bound request rather than a CPU or I/O-bound
path.

## Correctness and regression evidence

- Complete test suite: 7,926 passed, 25 skipped, 74 deselected, 0 failed.
- Focused DeepSeek tests: 239 passed; new dispatch tests: 4 passed.
- Seven boundary prompts passed with exact output and zero cached tokens.
- Live cache-on testing: both routes returned exact `READY` on one cold and two
  repeated requests; repeated requests hit 16,896 cached tokens.
- Checked Pi smoke pack: both routes passed the applicable hard gates with
  score 100 and clean lifecycle teardown.
- Complete six-task creative suite: upstream passed 5/6 objective hard gates;
  the candidate passed 4/6 in the full-suite sample. The extra candidate miss
  passed an immediate same-route diagnostic rerun, so it was not repeatable.
  Human visual review remains outside this objective approval.

## Scope

This is an M3 Ultra result, not evidence for an unconditional cross-chip
default. Before enabling the thresholds universally, collect per-chip results
or add runtime autotuning.

Machine-readable evidence is under
`benchmarks/evidence/deepseek_v4_flash_prefill_m3_ultra/`.
