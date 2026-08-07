# DeepSeek V4 Flash prefill optimization on M3 Ultra

Validated: 2026-08-07

Baseline: `49ec271676ba9c14bbebb75da1912e3fcb5fb0f4`

Machine: Mac Studio, M3 Ultra, 512 GB unified memory
Model: DeepSeek V4 Flash 0731, official native mixed FP4/FP8 weights

## Result

This branch is an M3 Ultra draft candidate. It improves the controlled
17,219-token cold-prefill workload by 14.18% with no measured aggregate
accuracy loss in the recorded evaluation.

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

The changes stay inside the DeepSeek V4 patch and retain the existing
shape/dtype fallbacks. MXFP4 tile selection and route counting are restricted
to the native MXFP4 path. The 2/3-bit affine path retains its prior 8,192-route
large-block threshold. DeepSeek V4 checkpoints whose top-level quantization is
below 4 bits keep the upstream dense pooled-attention and compiled
hyperconnection expansion paths; the other native MoE improvements remain
enabled.

## Bottleneck evidence

Synchronized component profiling on untouched upstream attributed 33.91% of
prefill time to routed `SwitchGLU`, 25.66% to sparse attention, and 24.33% to
compressed attention. The indexer was not the primary bottleneck. Metal System
Trace also showed a GPU-compute-bound request rather than a CPU or I/O-bound
path.

## Correctness and regression evidence

- Complete test suite: 7,900 passed, 70 skipped, 74 deselected, 0 failed.
- Focused DeepSeek tests: 210 passed, including eleven low-bit guard,
  propagation, and reference-path checks.
- Exact oQ2.5e checkpoint gate: upstream and candidate returned the same
  50-token response ending in `45`, with 17 prompt tokens and zero cached
  tokens. Candidate generation took 2.62 s versus upstream's 2.80 s; this
  single short request is correctness evidence, not a performance benchmark.
- Seven boundary prompts passed with exact output and zero cached tokens.
- Live cache-on testing: both routes returned exact `READY` on one cold and two
  repeated requests; repeated requests hit 16,896 cached tokens.
- Checked Pi smoke pack: both routes passed the applicable hard gates with
  score 100 and clean lifecycle teardown.
- Complete six-task creative suite: upstream passed 5/6 objective hard gates;
  the candidate passed 4/6 in the full-suite sample. The extra candidate miss
  passed an immediate same-route diagnostic rerun, so it was not repeatable.
  Human visual review remains outside this objective approval.

Additional PR coverage exercises the production 12,288-route affine path at
2-bit and 3-bit in FP16/BF16, compares the native compressed-attention path to
its causal MLX reference, compares the all-pooled attention/indexer shortcuts
to their reference paths, checks hyperconnection expansion against its
equation, and compares route-counting output to the prior `argsort` route.

## Maintainer low-bit finding and draft gate

The maintainer reported a deterministic first-token regression with
`Jundot/DeepSeek-V4-Flash-0731-oQ2.5e` at revision `5c8b681` while official FP8
and mixed MXFP4/MXFP8 oQ4e preserved their checked outputs. Against upstream
`49ec2716`, the exact request returned `45`; the original candidate `4d3d8e2`
returned `135`. First-token isolation showed that the candidate attention path
and fused hyperconnection path each independently flipped the greedy token.
See the exact report in
[Discussion #2499](https://github.com/jundot/omlx/discussions/2499#discussioncomment-17936475).

The oQ2.5e config declares top-level 2-bit affine quantization. This revision
now derives a numerical-safety gate from that checkpoint metadata: DeepSeek V4
configs below 4 bits use the upstream dense pooled-attention calculation and
compiled hyperconnection expansion. Unit tests cover the automatic 2/3-bit
dispatch as well as both reference paths.

The exact 103.08 GiB checkpoint was downloaded at revision
`5c8b6811550e8ef0c02c006a3fdfc4e724ffec1c` and exercised sequentially against
isolated upstream and candidate servers with prefix caching disabled. Both
returned the byte-identical response reported above, resolving the reproduced
regression locally. The PR remains a draft for cross-chip review.

## Reproduction

Build each revision with the native kernels and verify that the extension is
active:

```bash
git checkout 49ec271676ba9c14bbebb75da1912e3fcb5fb0f4  # baseline
OMLX_WITH_CUSTOM_KERNEL=1 python -m pip install -e .
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"

git checkout agent/deepseek-v4-prefill-m3-ultra             # candidate
OMLX_WITH_CUSTOM_KERNEL=1 python -m pip install -e .
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

For each revision, start a cache-disabled server with a fresh state directory.
`MODEL_PARENT` must contain the `DeepSeek-V4-Flash-0731` model directory.

```bash
omlx serve \
  --model-dir "$MODEL_PARENT" \
  --base-path "$RUN_STATE" \
  --no-cache \
  --memory-guard balanced \
  --port 8000
```

The headline comparison used five cold requests per revision. The request body
was 71,092 characters, tokenized to exactly 17,219 API prompt tokens, had
SHA-256 `a1465f4b5ee68dbd173c138dd65718bd06957439b66e99069e91f50364bf81f1`,
used temperature 0 and `max_tokens=8`, and asserted the exact response
`READY`. Send the same saved prompt with:

```bash
jq -n --rawfile prompt "$PROMPT_FILE" '{
  model: "DeepSeek-V4-Flash-0731",
  messages: [{role: "user", content: $prompt}],
  temperature: 0,
  max_tokens: 8,
  stream: false
}' > /tmp/deepseek-v4-prefill-request.json

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/deepseek-v4-prefill-request.json
```

The bundled admin benchmark provides a self-contained nearby reproduction with
the repository's `code_python` corpus (16,384 prompt tokens rather than the
17,219-token headline input):

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/admin/api/bench/start \
  -H 'Content-Type: application/json' \
  --data '{"model_id":"DeepSeek-V4-Flash-0731","prompt_lengths":[16384],"generation_length":8,"context_profile":"code_python"}'

curl --no-buffer \
  http://127.0.0.1:8000/admin/api/bench/"$BENCH_ID"/stream
```

The maintainer's low-bit regression uses the non-MTP oQ2.5e checkpoint at
revision `5c8b6811550e8ef0c02c006a3fdfc4e724ffec1c`, server prefix caching
disabled, and this exact request:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model":"DeepSeek-V4-Flash-0731-oQ2.5e",
    "messages":[{"role":"user","content":"Return only the integer: Convert binary 101101 to decimal."}],
    "temperature":0,
    "top_p":1,
    "max_tokens":64,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Run the focused regression and numerical checks with:

```bash
python -m pytest \
  tests/test_deepseek_v4_patch.py \
  tests/test_glm_moe_dsa_patch.py \
  tests/test_deepseek_v4_dspark.py
```

## Scope

This is an M3 Ultra result, not evidence for an unconditional cross-chip
default. Before enabling the thresholds universally, collect per-chip results
or add runtime autotuning.

Machine-readable evidence is under
`benchmarks/evidence/deepseek_v4_flash_prefill_m3_ultra/`.
