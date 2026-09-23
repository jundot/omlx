# macOS readability theme tests

Run `xcodebuild -project apps/omlx-mac/oMLX.xcodeproj -scheme oMLX -destination 'platform=macOS' -only-testing:oMLXTests/ThemeTests test` to check theme colors. The readability case renders a probe through `.omlxThemed()` with isolated saved preferences, checking disabled and enabled colors in light and dark appearances. It covers the shared theme path used by popovers, not app relaunch or full-screen layout.

# Text-only VLM loading tests

Run `python -m pytest -q tests/test_vlm_vision_fallback.py` to check strict loading and logits with a small quantized DiffusionGemma checkpoint, unchanged loaders for unreadable shards or retained vision, and patch restoration after loading errors. No model download is required.

# Test timing

CI runs all default tests on Python 3.11, 3.12, and 3.13, reports the 50 slowest phases, and uploads `test-results.xml` as `test-results-py<version>`. Use `python -m pytest --durations=50 --junitxml=test-results.xml` to collect the same timing data locally. Compare runner queue time separately from test execution.

# First-token burst release

Run `python -m pytest -q tests/test_engine_core.py tests/test_output_collector.py`
to check first-chunk delivery across the executor boundary, late admission,
multi-token chunks, output ordering, later burst limits and request cancellation.
Burst decode releases each request's first generated chunk before continuing
with the configured burst policy. This adds one executor hand-off per request;
it does not shorten prefill or bypass parser, stop-string or stream-interval
buffering. Subsequent chunks still follow the selected Burst Decode setting.

For a real-server comparison, use the same model, prompt, output length and
cache state on main and the branch. Measure client-observed first content and
complete-response time separately from producer-side token timestamps, with
balanced (0.1 s) and aggressive (0.2 s) burst settings. Include short replies,
long replies, a second request admitted during decode, and disconnect/recovery.
Report any custom budgets separately from the stock modes.

Cluster process-group tests use the `mock_cluster_ssh` fixture; remote teardown and serve-marker tests retain their own transport assertions. Mock-model engine tests skip explicit GC, while `test_engine_teardown.py` and `test_per_engine_threads.py` retain teardown and reclamation coverage. GLM5 execution tests reuse the eight-layer KDA/DSA fixture with dense and MoE layers; checkpoint-key tests retain the 45-layer configuration. The SDPA memory test retains the 8K/32K length ratio, head dimension 256, and 6:1 GQA ratio with fewer heads. DeepSeek V4.1 direct and converted engine checks run sequentially in one isolated subprocess with separate checkpoint directories.

# Cache cleanup logging tests

Run `python -m pytest -q tests/test_vision_feature_cache.py tests/test_paged_ssd_cache.py -k "cleanup_unlink_failure or corrupt_block_cleanup_logging"` to check failed-delete warnings and cache cleanup state. These cases use the existing cache fixtures to close writer threads.

# Cache-preserving engine teardown tests

Run `python -m pytest -q tests/test_engine_teardown.py tests/test_engine_core.py tests/test_scheduler.py tests/test_paged_ssd_cache.py tests/test_hot_cache.py tests/test_engine_pool.py tests/test_batched_engine.py tests/test_vlm_engine.py` to check the 60-second teardown budget and one progress-gated extension to 120 seconds. Clock tests cover the deadlines; short-budget subprocesses exercise fatal exits. Real writer-thread cases verify primary/draft flushes and in-flight prefix stores survive a saturated queue and can be reused after reload. Async cases cover event-loop responsiveness, cancellation, and leases during another model's unload.

# Claude launcher tests

Run `python -m pytest -q tests/test_cli.py tests/test_integrations.py` to check model selection, saved tier precedence, and the final Claude Code environment. Mixed-context cases verify that the selected initial model is passed through `ANTHROPIC_MODEL` and that the process-wide context and auto-compact limits use the smallest known context window across the initial model and configured tiers. Larger models consequently compact at that shared limit; models without reported limits cannot contribute a limit.

# ModernBERT embedding tests

Run `python -m pytest -q tests/test_modernbert_attention.py tests/test_embedding.py tests/test_mlx_embeddings_compat.py` to check finite padded attention, single-input equivalence, local-window masking, and embedding integration. The attention regression covers fp16, bf16, and fp32 at lengths around the affected SDPA tile boundaries.

# QSA reservation tests

Run `python -m pytest -q tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

# Prefill memory accounting tests

Run `python -m pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `python -m pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Cluster join recovery tests

Run `python -m pytest -q tests/test_cluster_pairing_session.py tests/test_cluster_pairing.py tests/test_cluster_ui_integration.py tests/ui/test_cluster_v2_wizard.py` to check joining, cancellation, and approval. Session tests recreate a manager with the same base path to verify that the original code and cancellation proof survive a restart. They also cover offline cancellation, switching peers while cleanup is pending, rejected requests versus lost responses, and storage failures. Wizard tests check that delayed responses cannot restore a cancelled join and that pending cleanup leaves new pairing controls available.

For a two-Mac smoke test, start isolated servers with separate base paths. Request a join, restart only the joining server, then cancel and retry; the coordinator must remove the original pending request. Repeat with the coordinator offline: cancellation must return `state: idle` with `cleanup_pending: true`, and joining a different reachable Mac must work. Restore the coordinator and poll the join endpoint to verify cleanup. Keep existing approval/cancel race and token-ownership tests in the run; never relax the coordinator's token check to make a stale request disappear.

# DeepSeek V4.1 offline tests

CED scheduler prefill, short suffix positions, full-logit scoring, and DSpark
continuity: `python -m pytest tests/test_deepseek_v41_ced.py tests/test_deepseek_v41_ssd.py -q`.
Replay is approximate; cache restoration is compared with the same chunk boundaries.

Run `python -m pytest -q tests/test_deepseek_v41.py` for the text/vision port. Synthetic weights and small recorded official outputs cover prefill/decode, DSpark, Engram and vision numerics without a checkpoint download or vendored reference implementation. See `tests/fixtures/deepseek_v41_expected.md` for provenance and reference corrections. PyTorch is optional for the independent FP8/FP4 arithmetic checks. Other cases cover BatchGenerator admission, cache restoration, DSML and VLM engine execution.

FP8 activation boundary and layout checks: `python -m pytest tests/test_deepseek_v41_activation.py -q`. These compare fused FP8 and weighted SwiGLU kernels against reference arithmetic across rounding ties, scale transitions, clipping, intermediate casts and floating-point dtypes.

MoE activation reuse and per-expert reference checks: `python -m pytest tests/test_deepseek_v41_moe.py -q`. These cover sorted prefill, decode, independent routed/shared activation policies, and repeated outputs with small quantized weights.

DeepSeek V4.1 Metal arithmetic and sparse-addressing checks: `python -m pytest tests/test_deepseek_v41_kernels.py -q` (no checkpoint required).

Packed attention rounding: `python -m pytest tests/test_deepseek_v41_attention_rounding.py -q`. An independent MLX oracle checks 64-key online maxima, FP32 denominators, BF16 PV probabilities, masking, sink placement, and growing or strided KV across the threadgroup capacity boundary. This does not execute the official CUDA kernels.

Engram storage and prefetch lifecycle: `python -m pytest tests/test_deepseek_v41_offload.py -q`. Modal state, forced-toggle behavior, and save payload: `node --test tests/deepseek_v41_offload_ui.test.cjs`. Both use synthetic fixtures and require no checkpoint download.


### MoE expert residency

`tests/test_deepseek_v41_moe_offload.py` exercises original and converted
V4.1 expert reads, MXFP4/MXFP8 and mixed-bit affine arithmetic, repeated
evictions, sorted routes, load/inference thread separation, Engram coexistence,
draft-weight exclusion, and memory estimates. The load probe rejects whole
expert reads from shared shards and any expert slab read through the Engram
mapping. Further cases check that consumed read buffers are released within the
in-flight byte window and pin the serial LRU order under concurrent reads,
expert-boundary chunking of sorted routes, and the fit-to-budget residency
helper against the admission arithmetic. `tests/test_deepseek_v41_affine_source.py`
covers community `mlx_lm` affine source checkpoints: packed and declared-dense
projections, exact force-dense dequantization, the affine Engram table spec, a
convert round-trip against a direct load, the declared-format resolver, and
offload eligibility. Run alongside `test_deepseek_v41_offload.py`,
`test_moe_expert_offload.py`, and the engine-pool/model-settings suites.
`node tests/moe_expert_offload_ui.test.cjs` checks the actual dashboard
save/reopen payload and speculative-decoding toggle exclusion.

`tests/test_moe_expert_offload.py` also exercises Qwen4-Exp MoE routing with
512 experts, top-k 10, 64 resident slots, shared experts, and repeated
evictions. `tests/test_moe_offload_compat.py` covers the model-type allowlist,
checkpoint completeness, dense-model exclusion, API/runtime rejection, and
PLE/Engram metadata after expert savings.

# macOS port persistence tests

Run the `AppConfigTests`, `ServerProcessIntegrationTests`, `ServerScreenVMStorageDiffTests`, `AppServicesPathTests`, and `MenubarControllerPortTests` targets with `xcodebuild test`. For process tests, set `TEST_RUNNER_OMLX_INTEGRATION=1`, `TEST_RUNNER_OMLX_PYTHON_OVERRIDE` to a Python interpreter with oMLX dependencies, `TEST_RUNNER_PYTHONPATH` to the repository root, and `TEST_RUNNER_OMLX_BASE_PATH` to a fresh temporary directory so the test host does not start the user's configured server. Leave `TEST_RUNNER_OMLX_DEV_SERVER_SCRIPT` unset to exercise the real Python API and CLI with empty model directories. Setting it to `apps/omlx-mac/Scripts/dev_server.py` instead runs the lightweight process fixture.

The port cases cover offline Apply, recovery from an occupied port, web API saves followed by manual and automatic restarts, native Apply while running, client endpoint synchronization, and persistence across subsequent starts. Each child uses a temporary settings directory and an OS-selected port, and teardown stops the children. Endpoint-only persistence also checks that unrelated settings and corrupt files are preserved.

# Accuracy benchmark worker tests

Run `python -m pytest -q tests/test_eval_worker_pool.py tests/test_eval.py tests/test_accuracy_benchmark.py tests/test_admin_external_accuracy_diagnostics.py tests/test_accuracy_upload.py`. Worker tests cover slot refilling, thinking-mode retries, sequential code scoring without blocking generation, and repeated cancellation during scoring. Real HumanEval, MBPP, and LiveCodeBench subprocess cases verify normal completion, cancellation draining, and temporary-file cleanup with a controlled engine; no model checkpoint is required.

# Profile API exposure tests

Run `python -m pytest -q tests/test_admin_new_profile_expose_as_model.py tests/test_admin_profiles_api.py tests/test_model_settings_profiles.py`. The new-profile tests check toggle bindings, the OFF reset, and request serialization. Existing API tests cover the edit form, persistence, exposed model IDs, and name collisions.

### Lightning MTP with XTC sampling

Run `python -m pytest tests/test_mtp_xtc_sampling.py -q` for request sampler changes, late-joining mixed batches, row removal, and greedy sampling. These tests use a small MLX model and observe the MTP eligibility boundary; they do not execute a trained MTP head.

# VLM cache boundary tests

Run `python -m pytest -q tests/test_vlm_cache_boundaries.py tests/test_vlm_engine.py tests/test_prefix_cache.py tests/test_paged_cache.py` to check image-aware prefix keys. Boundary cases cover reasoning-dependent template prefixes, final grid token positions, adjacent images, multiple images per turn, block edges, invalid metadata, and isolation when earlier or later images change. These tests use synthetic processor inputs and KV arrays; checkpoint preprocessing and inference comparisons require local models.


### Profile consistency

Run `python -m pytest tests/test_model_settings_profiles.py tests/test_admin_profiles_api.py tests/test_admin_model_settings_template.py -q`. The editor behavior test executes the dashboard JavaScript with Node.js and is skipped if Node.js is unavailable. The cases cover latest-template application, same-name model preservation, renamed and deleted template references, persistence rollback, create-and-apply behavior, and preserving edits during focus refresh. The macOS `ProfileScopeTests` cover source references, stale copies, orphan visibility, and display names. For a manual cross-client check, edit a global profile in the web UI, apply it from the app, and verify the model's effective settings; repeat with an independent same-name model profile and with an unsaved editor open.

Global and model profiles may share display names while retaining separate IDs. Applying a global template reads its latest settings; deleting it preserves model copies as independent profiles. Both editors save and apply new profiles, and focus refresh preserves unsaved edits.

Startup reference repair is covered in `tests/test_model_settings_profiles.py`: missing references are cleared without replacing saved values, originals are backed up under `<base_path>/profile-reference-backup-*`, and subsequent loads do not write again. Retries with unchanged originals reuse the same content-hash backup directory and fill missing files. Mismatched backups prevent repair. Invalid storage, unsupported versions, backup failures, and failed writes must not persist inferred repairs. A rollback failure aborts startup and logs the backup path.

## Qwen tool-call recovery

Run `python -m pytest tests/test_tool_calling.py tests/integration/test_e2e_streaming.py -q`. Final Qwen parsing preserves unknown tool names for client feedback, recovers complete functions missing only the outer envelope close at normal EOF, and reports unrecoverable siblings without a successful stop. Cases cover all three streaming APIs, chunk boundaries, repeated calls, literal tags in arguments, schema validation and length stops. Other parser families and reasoning-channel promotion keep their existing rules.

For a real-server check, request a small `write(content: string)` call with thinking disabled and greedy sampling. Compare the normal result with a request using `stop: ["</tool_call>"]`: the complete function should still arrive once with identical arguments. Then supply an assistant call to an unknown tool followed by a matching tool-error message naming `write`; verify the next model turn uses `write`. Use an isolated port and base path, and do not execute model-supplied file operations during the check.

# Streamed oQ calibration tests

Run `python -m pytest tests/test_oq.py -k TestStreamedCalibration` for streamed calibration. The small BF16 Qwen4 fixture exercises GDN, sparse attention, mmap PLE and the MTP head. It compares imatrix statistics and fused sensitivity with resident collection, verifies cache reuse with and without MTP, and converts and reloads the artifact with its shared PLE scale intact. A small MiniMax decoder fixture also compares dense and MoE collection. These cases replace the separate streaming test modules and need no external checkpoint.

## Prism runtime optimizations

Run `python -m pytest tests/test_prism_runtime.py tests/test_prism_decode.py
tests/test_active_models_visibility.py tests/test_model_loading.py
tests/test_vlm_engine.py tests/test_vlm_cache_boundaries.py`.

The pinned mlx-vlm loader owns schema-2 Prism model loading and processors.
oMLX applies its runtime changes after native VLM loading; there is no duplicate
checkpoint loader or copied Hadamard implementation. Tiny native checkpoints
verify unchanged stored weights, default logits, FP16 activation/KV boundaries,
FP32 recurrent state, tied embeddings and isolation from other model types.
Decoder tests compare logits and used cache contents with upstream across
single-row decode, padding, multiple rows, prefill and capture options.

For a real-checkpoint check, serve `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`
through the VLM engine with a separate base path and port. Check text, vision,
reordered image reuse, concurrent requests and at least 192 generated tokens.
Repeat a prompt longer than the scheduler's 4096-token Prism block floor and
verify the answer and nonzero prefix reuse. Repeat with
`OMLX_PRISM_FP16_ACTIVATIONS=1`. The opt-in activation mode must use
`:prism_fp16_v2` in the scheduler cache namespace while retaining the public
model ID in dashboard progress. It must not reuse FP32 states or the older
custom-loader adapter's `:prism_fp16_v1` states. Assess FP16 task outputs
separately; it changes rounding and is not a bit-exact precision mode.

For performance measurements, exercise the actual serving transition. oMLX
already keeps regular `KVCache` for standalone requests; a direct
`BatchGenerator` with a synthetic one-row cache does not represent that path.

1. Start isolated oMLX servers from main and the candidate, with identical
   native builds, dependencies, checkpoint, settings and FP32 activations.
   Leave `OMLX_PRISM_FP16_ACTIVATIONS` unset. Use concurrency two and stock
   balanced decode bursting, with thinking/speculation and KV quantization off.
2. Warm a real prompt of roughly 8K tokens through HTTP. Measure a standalone
   256-token reply as a control. Then start another 256-token reply and submit
   a 32-token reply after receiving its first content chunk. Both requests use
   the same messages and deterministic sampling settings.
3. Observe two-row decoding followed by one unpadded `BatchKVCache`. Reject
   trials that never batch or whose measured tail still includes multiple rows.
   Inspect the backing allocation as well as the used prefix: main tightens it
   every token, whereas the candidate preserves spare capacity. The standalone
   control must retain regular `KVCache` in both implementations.
4. Run fresh-process A/B/B/A phases, discarding a warmup for each case and
   retaining repeated samples. Compare output token IDs, prefix reuse, HTTP
   first-content/complete latency and a fixed tail window using the engine's
   producer timestamps. Report memory separately and avoid extrapolating the
   affected tail to all requests or to cold prefill.

[Recorded HTTP benchmark and exact reproduction harness](https://gist.github.com/samfenwick/476245f50cb037b9eb798acaa146d554)
compare main `8288884d` and runtime head `3be7a24e` on M1 Max / 64 GB, with four
retained samples per arm/scenario. The 8,386-token prompt reuses 8,379 tokens.
After the batch shrinks, the 128-token tail improves from 12.86 to 19.18
tokens/sec; the complete overlapping reply takes 22.37 versus 16.81 seconds.
Standalone tail throughput is 18.55 versus 18.51 tokens/sec. All retained output
token IDs and text match. Peak MLX allocation is 21.06 GiB in both arms, including
load and warmup; this is not a measured peak-memory reduction or a broad quality
evaluation. The optional FP16 activation path is outside this comparison.

The cache ownership behavior originates in mlx-vlm's Qwen3.5 singleton path.
An upstream dependency fix can replace the scoped runtime patch; these results
quantify its present serving impact rather than deciding that maintenance
boundary. The FP32 attention fallback already on main remains covered by
`tests/test_sdpa256_attention.py`.
