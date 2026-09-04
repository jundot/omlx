# DeepSeek V4 Vision stability validation

Status: experimental. This note records an incident and defines evidence gates;
it does not establish a root cause. Production rollout remains contingent on a
physical two-Mac validation.

## Refactor scope

Based on PR #3354 head `cc113f9d` (fetched 2026-09-04). Current upstream main
was fetched for comparison; this work does not merge unrelated main changes.

- Queue the sequential Vision backend in the proxy. Waiting clients cannot
  start a backend inactivity timer or cancel the request ahead of them.
- Use backend SSE progress for both public streaming and non-streaming calls;
  preserve one final public response for non-streaming callers.
- Keep activity accounting correct through repeated cancellation. Map launch
  and recovery failures to HTTP 503; preserve pending unload and memory
  accounting if deferred cleanup fails.
- Agree cache prefix lengths and successful SSD restores across ranks before
  reuse. Convert local memory-probe failures into an agreed rejection.
- Signal local and remote ranks before waiting for exit, retain worker identity
  for verification, and honor quarantine even if writing its marker fails.
- Bound fallback indexer score tiles and remove per-token GPU sentinel checks
  using shared image spans. Release image preparation state as it is consumed.

These address concrete failure paths, not a proven root cause for this incident.
Actual throughput and post-crash driver recovery still require hardware evidence.

Image admission now rejects more than 16 images per request, payloads over
50 MiB (base64 size is checked before decoding), source images over 64 Mi-pixels,
and resized inputs above 8192 ViT patches per image. These limits bound image
preprocessing and encoding before large allocations.

## Incident evidence

The 2026-09-04 run used oMLX 0.6.4, MLX 0.32.2, mlx-lm 0.31.3, two ranks,
JACCL, and `DeepSeek-V4-Flash-Vision-Exp`. The deployment became ready at
12:46:48. Before failure, requests completed through prompts of 67,165 tokens;
the logged 64,722-token request took 49.96 s and the 67,165-token request took
44.81 s. These records do not include per-request wall-clock timestamps or
vision stage markers.

The primary recorded symptom is `rank-zero inference stream timed out: no
rank-zero data for 300s while the cluster was ready`. The supervisor then
stopped the deployment and reaped remote rank 1 as killed. No rank-zero exit
status, signal, Metal/IOGPU diagnostic, or JACCL transport error is present.

The follow-on errors are recovery containment: reload was quarantined because
both rank processes had exited while live memory ceilings remained low. The
reported reload requirements were 86.5 GiB (rank 0) and 88.6 GiB (rank 1),
while immediately available capacity was about 42 GiB on each. Later requests
returned 500, and unload returned 503. At 13:24:52 the service reported four
consecutive failed health checks and restarted the service.

Possible explanations include a rank-zero long-prefill/decode stall, a
rank/transport failure, or retained driver memory after termination. The log
cannot distinguish these. Python cannot force reclaim of memory retained by a
Metal driver; the recovery quarantine must therefore remain authoritative until
a capacity probe demonstrates recovery. Do not clear it by deleting markers or
by changing system settings.

## Cheapest local gates

These commands use repository code and synthetic fixtures; they do not require
the full checkpoint or a cluster. Run from the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_cluster_launch.py -k 'quarantine or recovery'
.venv/bin/python -m pytest -q tests/test_cluster_runtime.py -k 'marker or recovery'
.venv/bin/python -m pytest -q tests/test_deepseek_v4_vision.py tests/test_deepseek_v4_patch.py
.venv/bin/python -m pytest -q tests/test_distributed_engine.py tests/test_cluster_engine_pool.py
.venv/bin/python -m pytest -q tests/test_cluster_prefill_guard.py tests/test_cluster_telemetry.py
```

The first two gates cover persisted quarantine and proof-of-recovery behavior.
The vision tests cover tensor/embedding contracts with test fixtures. Route and
seam tests cover failure responses without starting ranks. Record commit,
Python, package versions, exit status, and the complete test output.

The broader local regression command used for this refactor is:

```bash
.venv/bin/python -m pytest -q tests/test_cluster_*.py tests/test_distributed_engine.py \
  tests/test_deepseek_v4_vision.py tests/test_deepseek_v4_patch.py \
  tests/test_deepseek_v4_indexer_dispatch.py tests/test_deepseek_v4_wsdpa.py \
  tests/test_server.py tests/test_engine_pool.py --disable-warnings
```

The repository's default pytest selection excludes slow/model-loading and live
integration tests. MLX import still requires local Metal access; the test suite
is not evidence of physical JACCL or full-model performance validation.

Final local run (2026-09-04, Python 3.12.13): **1400 passed, 62 skipped** in
80.74 seconds. `python -m compileall -q omlx tests` and `git diff --check` passed.
Ruff reported no new diagnostics relative to the PR head; existing diagnostics
in the large model patch and server files remain. The shared transport, queue
cancellation, deferred teardown and HTTP error changes also received an
independent Sol review with no actionable findings.

If a local Apple-silicon hardware check is available, treat this as a separate
gate and capture its JSON output:

```bash
.venv/bin/python -m omlx.cli cluster hardware-canary --json
```

It is intentionally small, but it still exercises Metal. It does not reproduce
a full-checkpoint leak or validate physical JACCL.

## Staged physical validation: 2x M5 Max, 128 GiB, JACCL

Use identical commit, absolute checkpoint path, Python, MLX, mlx-lm, and oMLX
configuration. Mac A is rank 0. Preserve the signed planner JSON and dashboard
status for every stage. Do not proceed when either planned resident total is
above capacity minus reserve.

1. Record baseline memory, capacity, route/interface, JACCL version, and process
   identities on both Macs. Run worker, collective, and pipeline smokes.
2. Run the planner with two 128 GiB nodes and a conservative 16 GiB reserve.
   Verify pipeline support, tensor parallel size 1, rank-0 coordinator bytes,
   rank-1 zero coordinator bytes, contiguous ranges, and positive headroom.
3. Load once and prove both ranks reach ready. Capture planned, measured, peak,
   reserve, and headroom bytes, plus proof that both rank processes exit on
   teardown.
4. Send text-only, then one short image, then streamed one-image requests.
   Require the expected vision encode, multimodal embedding, distributed
   prefill, and first-token markers; rank 0 alone reports image ownership.
5. Exercise context bands near 64k, 67k, and near 128k only after short tests
   pass. Record TTFT, prefill/decode rates, expanded sequence length, peak
   memory, and time since the previous stage. A request must fail before launch
   when planner admission is insufficient; no rank may enter pressure teardown.
6. Alternate two same-dimension but different-content images. Require distinct
   content-keyed cache identities and one vision encode per image; no stale KV
   reuse is acceptable.
7. Cancel an in-flight streamed request at short, 64k, and 67k contexts. Require
   bounded cancellation, rank-zero and rank-one proof of exit, and a recovered
   memory baseline before retrying. A cancellation timeout is a failure even if
   the HTTP client disconnects.
8. Unload, prove complete rank/process-group exit, wait for capacity recovery,
   reload, and repeat text plus one-image requests. Answers, stage counts, and
   memory baselines must match the first pass within an agreed tolerance.

Also queue two Vision requests behind a long request and cancel one waiter.
Only the running request should reach the backend; the remaining waiter should
complete afterward without worker reload. Test a public non-streaming response
lasting longer than 300 seconds while rank-zero keeps reporting progress. In
small-model fault tests, remove one rank's cached prefix or invalidate its SSD
snapshot: both ranks must recompute together, then successfully serve a new
request. A local admission probe error must reject on all ranks without a hang.

For each request retain request size/type, context band, stream/cancel outcome,
stage timestamps, rank exit evidence, per-rank memory samples, and the exact
error. Do not retain prompt contents, images, or tensor dumps.

Use identical prompts and cache state at `cc113f9d` and the refactor, recording
five warm runs per context band, with native indexer availability recorded.
Target median decode throughput within 5% and short-request TTFT within 10% of
baseline; report long-prefill changes separately because bounding the fallback
indexer may trade throughput for lower peak memory. Flag any missing
rank-zero event for 300 s, any unexplained multi-minute gap, monotonic memory
growth after completed requests, or a post-teardown baseline that does not
recover. Treat the observed 300-second timeout as a hard failure, not a normal
long-context result.

## Optional fault testing

Only after small synthetic fault tests pass, an operator may run hazardous
full-size crash/kill tests on an isolated pair. These can retain driver memory,
kill ranks, and require a reboot; they must not be used to justify production
readiness. Capture supervisor escalation, complete process-group exit,
quarantine persistence, capacity recovery, and refusal to reload while memory
is retained. Never force this path with destructive shell commands or a sysctl
change.

Until the physical sequence above passes, classify the implementation as
experimental and keep reload quarantine enabled.
