# Test layout

Server tests live in `tests/` and must not depend on `apps/`: after the full run, CI deletes `apps/`, collects the server tests again, and starts `omlx serve` to check the API-only fallback. Web UI tests live in `apps/omlx-web/tests/` and the macOS app's Python tests in `apps/omlx-mac/Tests/python/`. A plain `pytest` runs all three through `testpaths`. The standalone dashboard script tests are the `apps/omlx-web/tests/*.test.cjs` files except `network_auth_ui.test.cjs`, which `test_admin_network_auth_ui.py` runs with its input.

# Qwen3-ASR audio quantization tests

Run `python -m pytest -q tests/test_audio_stt.py -k Qwen3ASRAudioQuantization` to check Qwen3-ASR audio quantization through the STT loader. Small local checkpoints cover mixed 4-bit text and 8-bit audio weights and floating-point audio layers. No model download is required.

# Cluster test filesystem isolation

The autouse `cluster_home` fixture gives each test a temporary directory for cluster interpreter shims and SSH files. It preserves explicit shim `home` arguments and leaves `HOME` unchanged so model discovery paths still work. The shim unit tests import the original function directly and provide their own temporary paths or patch `HOME` to verify the real default-path behavior.

The audio model-list smoke test checks that app startup publishes its shim inside the isolated directory. The cluster GET-route smoke test checks that `/ssh-key` creates its key pair there.

# macOS readability theme tests

Run `xcodebuild -project apps/omlx-mac/oMLX.xcodeproj -scheme oMLX -destination 'platform=macOS' -only-testing:oMLXTests/ThemeTests test` to check theme colors. The readability case renders a probe through `.omlxThemed()` with isolated saved preferences, checking disabled and enabled colors in light and dark appearances. It covers the shared theme path used by popovers, not app relaunch or full-screen layout.

# Text-only VLM loading tests

Run `python -m pytest -q tests/test_vlm_vision_fallback.py` to check strict loading and logits with a small quantized DiffusionGemma checkpoint, unchanged loaders for unreadable shards or retained vision, and patch restoration after loading errors. No model download is required.

# Test timing

CI runs all default tests on Python 3.11 for PRs and pushes to `main`. A daily scheduled run, which can also be started manually, covers Python 3.11, 3.12, and 3.13. Each run reports the 50 slowest phases and uploads `test-results.xml` as `test-results-py<version>`. Use `python -m pytest --durations=50 --junitxml=test-results.xml` to collect the same timing data locally. Compare runner queue time separately from test execution.

The automatic Qwen FP16/BF16 decode route has numerical, cache-state and
fallback tests in `tests/test_qwen35_fp16_decode.py`. Run it with
`tests/test_qwen35_gdn_prework.py` to check that the existing BF16 Qwen4 and
speculative routes remain intact.

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

# Decision model tests

Run `python -m pytest -q tests/test_systemone.py` to check the `/v1/systemone` decision models. Clef cases cover the prompt layout and question/option spans, state truncation, answer formatting, the joint head parameter names against the released checkpoint layout, and one logit per option for each question. OpenJev cases cover the text and screenshot prompt layouts, image sources, the two-stage readout for more than 52 options, and the calibration and confidence formulas. A tiny random Qwen3.5 backbone checks that chunked prefill matches a single pass and that a copied prefix cache gives the same readout as a full prefill. Detection, pool dispatch, the endpoint and the oQ head passthrough are tested in `test_model_discovery.py`, `test_engine_pool.py`, `integration/test_server_endpoints.py` and `test_oq.py`. No model download is required.

# QSA reservation tests

Run `python -m pytest -q tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

# Qwen4 verify attention row tests

Run `python -m pytest -q tests/test_qwen4_verify_attention_rows.py` to check that row-exact Lightning MTP verify windows through Qwen4 attention give every row the bits of the serial one-row decode step and leave the same KV and QSA indexer state. The tests build one attention layer at the real Flash-Next shapes with synthetic 6-bit weights. Masked-arm windows (past the 2,048-token QSA budget, rank-three positions) cover 2 to 8 rows at 2,060, 16,382 and 24,000 cached tokens and compare each row's FP32 block scores and token mask; a rollback case accepts one draft and decodes on. Dense windows below the budget include rows on both sides of MLX's one-pass/two-pass vector SDPA switch at 1,024 keys.

`test_mlx_vlm_qwen4_exp_compat.py::test_qwen4_mtp_one_row_step_is_the_serial_decode_step` checks that a one-row Lightning MTP window (the activation step and depth-0 cycles) runs the serial decode step: equal logits and cache state, no speculative transaction, and a following verify window that rolls back as usual.

# Prefill memory accounting tests

Run `python -m pytest -q tests/test_scheduler_prefill_eviction_progress.py` to check usage after prefill pauses. Cold and warm requests retain token progress across repeated pauses, report only restored prefix tokens as cached, and include locally computed tokens in prompt throughput.

`python -m pytest -q tests/test_engine_preflight.py` checks route admission after eviction or reclaim. Both batched wrappers must refresh their cached MLX sample on the owning executor, including when the pool reports that no action was necessary. Controlled memory readings cover newly available headroom, insufficient headroom, and requests that already fit without executor work.

Run `python -m pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `python -m pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, exact SpecPrefill static-prefix sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Batch KV capacity tests

Run `python -m pytest -q tests/test_vlm_batch_kv_capacity.py` after changing `omlx/patches/vlm_batch_kv_capacity.py` or bumping the mlx-lm or mlx-vlm pin. The tests load the installed, unpatched cache module next to the patched one and compare cache state and attention output bit for bit through random appends, trims, rollbacks, merges, joins, filters and restores. They cover the mlx-vlm class, the mlx-lm class that prefix restore builds, and a batch that mixes both.

# Cluster join recovery tests

Run `python -m pytest -q tests/test_cluster_pairing_session.py tests/test_cluster_pairing.py tests/test_cluster_ui_integration.py apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` to check joining, cancellation, and approval. Session tests recreate a manager with the same base path to verify that the original code and cancellation proof survive a restart. They also cover offline cancellation, switching peers while cleanup is pending, rejected requests versus lost responses, and storage failures. Wizard tests check that delayed responses cannot restore a cancelled join and that pending cleanup leaves new pairing controls available.

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

Engram storage and prefetch lifecycle: `python -m pytest tests/test_deepseek_v41_offload.py -q`. Modal state, forced-toggle behavior, and save payload: `node --test apps/omlx-web/tests/deepseek_v41_offload_ui.test.cjs`. Both use synthetic fixtures and require no checkpoint download.


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
`node apps/omlx-web/tests/moe_expert_offload_ui.test.cjs` checks the actual dashboard
save/reopen payload and speculative-decoding toggle exclusion.

`tests/test_moe_expert_offload.py` also exercises Qwen4-Exp MoE routing with
512 experts, top-k 10, 64 resident slots, shared experts, and repeated
evictions, plus the resident Lightning MTP head (`mtp.*`) and its admission
pricing. `tests/test_moe_offload_compat.py` covers the model-type allowlist,
checkpoint completeness, dense-model exclusion, API/runtime rejection, and
PLE/Engram metadata after expert savings.

# macOS port persistence tests

Run the `AppConfigTests`, `ServerProcessIntegrationTests`, `ServerScreenVMStorageDiffTests`, `AppServicesPathTests`, and `MenubarControllerPortTests` targets with `xcodebuild test`. For process tests, set `TEST_RUNNER_OMLX_INTEGRATION=1`, `TEST_RUNNER_OMLX_PYTHON_OVERRIDE` to a Python interpreter with oMLX dependencies, `TEST_RUNNER_PYTHONPATH` to the repository root and `apps/omlx-web`, and `TEST_RUNNER_OMLX_BASE_PATH` to a fresh temporary directory so the test host does not start the user's configured server. Leave `TEST_RUNNER_OMLX_DEV_SERVER_SCRIPT` unset to exercise the real Python API and CLI with empty model directories. Setting it to `apps/omlx-mac/Scripts/dev_server.py` instead runs the lightweight process fixture.

The port cases cover offline Apply, recovery from an occupied port, web API saves followed by manual and automatic restarts, native Apply while running, client endpoint synchronization, and persistence across subsequent starts. Each child uses a temporary settings directory and an OS-selected port, and teardown stops the children. Endpoint-only persistence also checks that unrelated settings and corrupt files are preserved.

# Accuracy benchmark worker tests

Run `python -m pytest -q tests/test_eval_worker_pool.py tests/test_eval.py tests/test_accuracy_benchmark.py apps/omlx-web/tests/test_admin_external_accuracy_diagnostics.py tests/test_accuracy_upload.py`. Worker tests cover slot refilling, thinking-mode retries, sequential code scoring without blocking generation, and repeated cancellation during scoring. Real HumanEval, MBPP, and LiveCodeBench subprocess cases verify normal completion, cancellation draining, and temporary-file cleanup with a controlled engine; no model checkpoint is required.

# Profile API exposure tests

Run `python -m pytest -q apps/omlx-web/tests/test_admin_new_profile_expose_as_model.py tests/test_admin_profiles_api.py tests/test_model_settings_profiles.py`. The new-profile tests check toggle bindings, the OFF reset, and request serialization. Existing API tests cover the edit form, persistence, exposed model IDs, and name collisions.

### Lightning MTP with XTC sampling

Run `python -m pytest tests/test_mtp_xtc_sampling.py -q` for request sampler changes, late-joining mixed batches, row removal, and greedy sampling. These tests use a small MLX model and observe the MTP eligibility boundary; they do not execute a trained MTP head.

### Batched DFlash drafter

Run `python -m pytest tests/test_dflash_batched.py tests/test_mlx_lm_mtp_patch.py -q -k "dflash_batched or block_drafter"`. `test_dflash_batched.py` builds a tiny DFlash2 drafter with random weights and checks that rows drafted together match the same rows drafted alone across ring wrap-around, ragged context segments and cohort changes, plus the prefill seed window slicing and block-size clamping. The `block_drafter` cases in `test_mlx_lm_mtp_patch.py` drive the Lightning MTP verify path with a table drafter on the CountingModel harness and require token parity with standard decoding, one context entry per committed position (including late joins) and release of finished rows. Real drafter acceptance and throughput need a Qwen3.5-family VLM checkpoint with its `z-lab` DFlash draft and are measured against the standard batched engine.

# VLM cache boundary tests

Run `python -m pytest -q tests/test_vlm_cache_boundaries.py tests/test_vlm_engine.py tests/test_prefix_cache.py tests/test_paged_cache.py` to check image-aware prefix keys. Boundary cases cover reasoning-dependent template prefixes, final grid token positions, adjacent images, multiple images per turn, block edges, invalid metadata, and isolation when earlier or later images change. These tests use synthetic processor inputs and KV arrays; checkpoint preprocessing and inference comparisons require local models.


### Profile consistency

Run `python -m pytest tests/test_model_settings_profiles.py tests/test_admin_profiles_api.py apps/omlx-web/tests/test_admin_model_settings_template.py -q`. The editor behavior test executes the dashboard JavaScript with Node.js and is skipped if Node.js is unavailable. The cases cover latest-template application, same-name model preservation, renamed and deleted template references, persistence rollback, create-and-apply behavior, and preserving edits during focus refresh. The macOS `ProfileScopeTests` cover source references, stale copies, orphan visibility, and display names. For a manual cross-client check, edit a global profile in the web UI, apply it from the app, and verify the model's effective settings; repeat with an independent same-name model profile and with an unsaved editor open.

Global and model profiles may share display names while retaining separate IDs. Applying a global template reads its latest settings; deleting it preserves model copies as independent profiles. Both editors save and apply new profiles, and focus refresh preserves unsaved edits.

Startup reference repair is covered in `tests/test_model_settings_profiles.py`: missing references are cleared without replacing saved values, originals are backed up under `<base_path>/profile-reference-backup-*`, and subsequent loads do not write again. Retries with unchanged originals reuse the same content-hash backup directory and fill missing files. Mismatched backups prevent repair. Invalid storage, unsupported versions, backup failures, and failed writes must not persist inferred repairs. A rollback failure aborts startup and logs the backup path.

## Qwen tool-call recovery

Run `python -m pytest tests/test_tool_calling.py tests/integration/test_e2e_streaming.py -q`. Final Qwen parsing preserves unknown tool names for client feedback, recovers complete functions missing only the outer envelope close at normal EOF, and reports unrecoverable siblings without a successful stop. Cases cover all three streaming APIs, chunk boundaries, repeated calls, literal tags in arguments, schema validation and length stops. Other parser families and reasoning-channel promotion keep their existing rules.

For a real-server check, request a small `write(content: string)` call with thinking disabled and greedy sampling. Compare the normal result with a request using `stop: ["</tool_call>"]`: the complete function should still arrive once with identical arguments. Then supply an assistant call to an unknown tool followed by a matching tool-error message naming `write`; verify the next model turn uses `write`. Use an isolated port and base path, and do not execute model-supplied file operations during the check.

# Streamed oQ calibration tests

Run `python -m pytest tests/test_oq.py -k TestStreamedCalibration` for streamed calibration. The small BF16 Qwen4 fixture exercises GDN, sparse attention, mmap PLE and the MTP head. It compares imatrix statistics and fused sensitivity with resident collection, verifies cache reuse with and without MTP, and converts and reloads the artifact with its shared PLE scale intact. A small MiniMax decoder fixture also compares dense and MoE collection. These cases replace the separate streaming test modules and need no external checkpoint.

# Fused routed-expert decode tests

Run `python -m pytest -q tests/test_qwen35_moe_routed_decode.py tests/test_qwen35_moe_router.py tests/test_qwen35_moe_gate_up.py` to check the one-token routed-expert kernels. Real `Qwen3_5MoeSparseMoeBlock` instances laid out like Qwen3.8-Flash-Next oQ (quantized routed experts, 8-bit shared expert and shared-expert gate, bf16 router) must match the served body bit for bit, with the shared expert and its gate folded into the two launches: 5-bit (oQ5e) and 4-bit experts at the Flash-Next shape (hidden 2560, intermediate 640, top-k 10), and 5-bit gs32, 6-bit gs128 and 8-bit experts at smaller shapes. A bf16 shared expert stays composed and must match too. Both launches are also run in FP32 against MLX's FP32 mat-vecs (routed and shared gate+up after SwiGLU, the gate row, every routed and shared down row), because BF16 outputs hide one-ulp FP32 differences (a fast-math `exp` in the SwiGLU sigmoid passes most BF16 cases but fails these). The kernels bind a one-expert view of the stacked weights; routing to experts 500+ of 512 checks that the view still reads the stacked buffer, and replacing the expert or shared-expert arrays must rebuild the cached plan. The other cases check that shapes where MLX would pick a different mat-vec partition, 3-bit experts, top-k 8, prefill and verify rows, float16, blocks without the gate+up fusion and a kernel failure all keep the served body. The one-launch router softmax + top-k must return the indices and scores of the softmax and top-k launches for random logits and engineered near-ties (every logit repeated eight times, logits on adjacent bf16 values, two-valued rows), a block whose router rows repeat eight times must route like the served block, and the softmax runs in FP32 against MLX's FP32 softmax (a fast reciprocal or a precise `exp` still routes identically but fails there). The router gemv must return MLX's `x @ W.T` logits bit for bit at 512x2560, 256x2048 and 128x1024, and its FP32 row sums must equal MLX's FP32 gemv on the same values (a `simd_sum` in place of MLX's shuffle-down tree changes only a few BF16 logits but every FP32 sum); shapes where MLX reduces K differently (K >= 16 N, a guarded K tail) keep `nn.Linear`. Where the router has 128 to 4,096 experts, its softmax + top-k run inside the gate+up launch for one to three rows: 512-expert blocks (with a folded or a bf16 shared expert) must still match the served body, and in FP32, with router logits tied in pairs, the launch must return the routing launch's indices and scores and the gate+up rows of the launch fed with them. `tests/test_qwen35_moe_verify_window.py` runs every window size with the folded and with the separate routing launch.

# Qwen3.5 fused verifier norm

Run `python -m pytest -q tests/test_qwen35_gdn_norm_gate.py` on a Metal-capable
Mac to compare the fused gated RMS norm with the served SiLU graph. Tests cover
FP16/BF16, varied RMS weights, three epsilon values, every gate encoding, and
fallback when the installed MLX arithmetic is unsupported. The check supports
both released and nightly MLX builds; it does not assume the exponential from
the version number.

# Native Qwen video input

Run `python -m pytest -q tests/test_video.py tests/test_image_utils.py tests/test_vlm_engine.py -k "video or media"` to check video input for Qwen3.5-family checkpoints without weights. The cases cover data-URI extraction and the unchanged rejection for other models, the video processor attached after load (official pixel budget, checkpoint config, frame cap), the prompt-token count against mlx-vlm's real processor both under and over the pixel budget, rejection of undecodable or degenerate clips, the order of video and text in the formatted prompt, per-clip prefix-cache ranges, video feature caching, temporary-file cleanup when preprocessing fails, and the preflight budget. Changes to the video path also need a real-model check with Lightning MTP and the batched DFlash drafter, including a follow-up turn on the same clip that reuses the cached prefix.

# Engine idle timing tests

Run `python -m pytest -q tests/test_engine_pool.py tests/test_active_models_visibility.py` to check that lease completion refreshes the LRU/TTL timestamp and that busy models report zero idle time. The cases cover a request longer than its TTL, release with a pending unload, cancelled release, redundant releases, and models with a held lease, an active request, or a waiting request.

# Structured output Unicode tests

Run `python -m pytest -q tests/integration/test_server_endpoints.py -k structured_output` to check that JSON cleanup keeps Unicode keys and values readable. The cases cover `json_object` and `json_schema` on both OpenAI endpoints, including buffered chat streaming and Responses completion events. Responses streaming must finish with the same text as the emitted deltas. They also check that an unpaired surrogate escape in the model JSON is returned unchanged instead of failing response serialization.

# Batched Qwen4 QSA decode and verify tests

Run `python -m pytest -q tests/test_qwen4_qsa_decode_gather.py -k "batch_bank or batch_gather"` to check the batched QSA paths. The row-bank cases drive one attention layer through right-padded chunked prompt prefill, decode, verify windows, ragged vector rollback, reorder and drop filters, extend, trim and state restore, and require layer outputs and cache state to be bit-identical with fresh per-row indexer caches. The gathered cases compare batched decode steps and verify windows with the dense masked path: outputs allclose, cache state identical, every query bounded to the QSA budget plus its tail, and each fail-closed case kept on the dense path. Both groups include the served ratio 4 / top-k 512 geometry.

For a real-server comparison, use the same model, prompts and cache state on main and the branch, with sessions of different lengths well past the QSA crossover (e.g. a 100K shared prompt plus 2K unshared tokens per session). Report aggregate decode while every stream is decoding, per batch size, with Lightning MTP on and off, and check a temperature-0 cold/warm identity and a multi-session codeword fan-out with prompts of different lengths.
