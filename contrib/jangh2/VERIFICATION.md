# Upstream packaging verification

The following is the historical Linux packaging record. The subsequent Apple
Metal acceptance run is recorded below; neither run establishes current-main
oMLX server integration.

- The vendored bridge archive: 264 tests pass.
- Six added transport regressions fail against the uncorrected diagnostic, then
  pass with the already supplied negative-stride/UINT64 fix. Restrictions are
  enforced by a CPU test double; no GPU execution is claimed.
- Eight default-installer tests fail before implementation and pass afterwards.
  These check path quoting, shell syntax, forwarding, safe symlink replacement,
  independent service identity and installer help without MLX.
- Complete contributed suite: 289 tests pass; no skips, failures or errors.
- 99 Python files compile and parse using Python 3.11 grammar.
- Production bridge sources are byte-identical to the vendored bridge archive
  except the files listed in `SOURCE-MANIFEST.json`
  (`runtime_changes_from_archive`): `device_selftest.py` (both previously
  confirmed fixes), `scheduler_compat.py` (non-JANGH2 VLM-native batch
  conversion plus `_wrap_caches` mixed-list passthrough) and
  `tests/test_scheduler_cache_compat.py` (regression tests for that change).
- The scheduler fallback is verified by CPU test doubles that mirror
  mlx-lm 0.31.2 `to_batch_cache()` and the mlx_vlm 0.7.8 batch cache classes;
  live mixed-model serving still needs Mac hardware acceptance.
- The optional installer does not upgrade packages, rewrite active bridge code,
  move model files, read API keys or automatically start/stop a process.

The ordinary Git diff is add-only and is round-trip checked in a fresh local Git
repository, then the contributed tests are rerun there. A full network clone of
upstream could not be obtained in this container; no claim is made that its full
existing test suite ran. Current-main interfaces were inspected via upstream
source, and they differ from the legacy private hooks. This is an RFC, not a
verified native current-main integration.

No macOS launchd process was started here. The installer emits a plist whose
schema and payload were round-tripped with plistlib; actual login service and
mixed-model inference need verification on the working Mac.

The source manifest records the two input artifacts and contributed source
hashes. Earlier user-supplied real-Mac results are useful reported evidence, but
not a substitute for these explicit test boundaries.

## Mac regression verification

Executed on Apple M5 Max with Python 3.11.17, MLX 0.32.4, MLX-LM 0.31.2,
MLX-VLM 0.7.8 and the existing PR #364 oMLX 0.3.0rc1 runtime.

- Complete contributed CPU suite: 295 tests pass.
- The original Allosaurus failure was reproduced at relative L2
  0.056308435 and maximum absolute logit error 1.817502141. Disabling device
  caching reproduced the same error; repeated reference execution was exact.
- Matching FP32 expert scale placement and summation order, and using MLX's
  precise exponential, made all three fixed-prefix/cached-continuation cases
  exact: relative L2, maximum error and KL divergence are zero, with 100%
  top-token agreement over 36 sampled positions. Acceptance limits are unchanged.
- The complete `prepare.py --long-profile` workflow passed qualification,
  tuning, whole-model comparison, and fresh-cache 1024/4096-token profiling.
  Qualification includes 24 exact FP32 reference-alignment cases across
  decode, sorted prefill, grouped tiles, row tuning, mixed bits and rotations.
- Standalone long profiling measured about 249/212 prefill tokens per second
  and 32/30 decode steps per second for 1024/4096-token prompts respectively.
  These are standalone timings, not concurrent HTTP-server throughput.
- Real Qwen3.6-35B-A3B-4bit and Qwen3.8-27B-MLX-4bit checkpoints passed through
  the PR VLM adapter and boundary-snapshot batch generator with bridge hooks
  active: chunked singleton prefill/decode, unequal-length batching/filtering,
  and cached continuation. Generated token counts matched and log probabilities
  were finite. The native hybrid-cache fix preserves offset arrays and cache
  list identity.
- A real BatchGenerator smoke check reproduced the cleanup stream ownership
  failure before the lifecycle fix. Closing on its MLX worker during scheduler
  shutdown cleared the wired-memory marker and allowed later main-thread
  destruction without the stream error.

This acceptance run did not test launchd, HTTP tool calling, image inputs,
mixed-model concurrent serving or every discovered checkpoint.
