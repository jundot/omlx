# JANGH2 diagnostics and legacy serving adapter

Current oMLX loads JANGH2 Qwen4-Exp checkpoints through its ordinary VLM backend.
Use `omlx serve --model-dir /path/to/models`; see [native support and optional
thinking](../../docs/JANGH2.md). The production numerical implementation lives in
`omlx/models/jangh2`. The bridge modules here import that same implementation so
kernel qualification and native serving cannot silently diverge.

This directory retains qualification, tuning, workload and full-model comparison
tools, plus the legacy PR #364 serving adapter and reversible command installer.
The adapter targets oMLX `0.3.0rc1`, MLX-LM `0.31.2` and MLX-VLM `0.7.8`.
Its private scheduler hooks are unnecessary for current native serving. Do not
install those legacy hooks into a current server.

## Compatibility boundary

The historical PR reference is `a77d410f3f5948854bb42e5363239883282ac1d4`.
Native integration is based on oMLX `0.7.3.dev1` with the dependencies pinned by
the root project. Both paths have separate hardware acceptance records in
[VERIFICATION.md](VERIFICATION.md). The legacy installer retains its selected
checkout, environment, API key and settings and does not install packages.

## Contents

- `jangh2_bridge/metal_ops.py`: packed expert Metal decode kernels.
- `grouped_ops.py`: opt-in grouped prompt-processing kernels.
- `reference_ops.py` and `experts.py`: reference execution and module integration.
- `checkpoint.py`, `metadata.py`, `loader.py`: strict format checks, reversible
  namespace/layout conversion and model-scoped loading; no checkpoint rewriting.
- `ple.py`, `row_cache.py`, `device_rows.py`: bounded SSD mappings, exact CPU
  packed-row cache and optional exact device-row cache.
- `scheduler_compat.py`: legacy batching and worker-stream compatibility, plus
  batch conversion of `mlx_vlm`-native caches for non-JANGH2 VLM checkpoints
  (Qwen3.5 family) that stock mlx-lm cannot batch.
- `chat_policy.py`, `chat_compat.py`: target-model template/reasoning controls.
- Qualification, tuning, workload, profiling, full-model comparison and snapshot
  CLIs, plus CPU regression tests using real NumPy where possible.
- `scripts/install_default.py`: optional user-local command/LaunchAgent installer.
- `compat/`: historical PR reproduction notes and its dependency/dispatch diff.

No speculative decoding/MTP is enabled. Device-row caching defaults off because
reported concurrent benchmarks did not demonstrate a consistent speed benefit.
No model weights, generated machine profiles, credentials, captured prompts,
cache data, virtual environments, or private absolute paths are included.

## Test without Apple MLX

From this directory, with Python 3.11+ and NumPy installed:

```sh
python -m unittest discover -s tests -v
```

These tests include CPU numerical checks, API doubles, temporary safetensors,
resource-limit checks and local synthetic HTTP servers. They do not establish
Apple Metal compilation, full-model equivalence, or support for new upstream
APIs. See `VERIFICATION.md` for the recorded test boundaries.

## Use with an existing known-good runtime

Use the interpreter that already runs the working server. Do not upgrade the
runtime as part of packaging. Supply the paths explicitly:

```sh
/path/to/working-omlx/.venv/bin/python serve.py \
  --checkout /path/to/working-omlx \
  --model /path/to/models/Org/Allosaurus-v0.1-125B-A6B-JANGH2 \
  --backend metal \
  serve --base-path /path/to/existing-server-state \
  --model-dir /path/to/models --host 127.0.0.1 --port 8500
```

`--model` selects the one checkpoint intercepted by the JANGH2 loader. It is not a
model whitelist. `--model-dir` is passed to oMLX discovery, which still loads
other supported models through its normal engines. Existing bridge hooks alter
some process-level scheduling/stream objects, so mixed-model serving requires
its own hardware acceptance tests; do not infer universal compatibility. When
stock mlx-lm refuses to batch a model's `mlx_vlm`-native caches, the scheduler
adapter converts them with MLX-VLM's own batch classes (Qwen3.5 family).
Those hybrid cache lists retain native offset arrays and list identity, which
Qwen3.6/Qwen3.8 need when extracting and merging cached rows. Scheduler shutdown
closes the batch generator on the MLX worker that owns its generation stream.

On the reference PR, discovery scans direct model directories and one level of
organisation directories. Pointing it at an LM Studio `models` root, rather than
one publisher's subdirectory, exposes all compatible models at those levels.
This does not add arbitrary GGUF support or architectures missing from the
installed loaders. Models are loaded on demand; do not pin the entire collection.

## Make the working environment the default command

Stop the foreground server, disable the old Homebrew service (if used), and quit
any app-managed server on the same port. In a shell where this directory exists:

```sh
/path/to/working-omlx/.venv/bin/python scripts/install_default.py \
  --checkout /path/to/working-omlx \
  --bridge /path/to/already-qualified-bridge \
  --model-root /path/to/models \
  --target-model /path/to/models/Org/Allosaurus-v0.1-125B-A6B-JANGH2 \
  --base-path /path/to/existing-server-state --port 8500 \
  --profile /path/to/existing-server-state/validation/measured-profile.json \
  --qualification /path/to/existing-server-state/validation/qualification.json
```

The default command is written atomically to `~/.local/bin/omlx`. Any existing
file or symlink there is backed up first; its symlink target is never overwritten.
Prepend `$HOME/.local/bin` to PATH in the relevant shell configuration. Existing
shell aliases/functions take precedence and should be removed if they shadow it.

The installer uses the existing bridge path rather than the newly packaged copy.
That preserves the runtime source fingerprint and tuned profile. A new bridge
copy or package reinstall may require fresh qualification; never edit a report
to bypass the fingerprint check. Profiles are data, not sourced shell code.

Run `omlx serve` in the foreground first. Non-serving commands use the selected
Python. This changes the user's Terminal command, **not** the Homebrew formula
or bundled macOS application. `~/.omlx-pr364` can remain the settings directory;
renaming it is not necessary to make the command the default.

### Optional login service

Re-run the installer with `--with-service` to create
`~/Library/LaunchAgents/local.omlx.jangh2.plist`. It is not started immediately,
but will start at login. The wrapper implements:

```sh
omlx start
omlx status
omlx logs
omlx restart
omlx stop
```

These lifecycle commands control **only** this LaunchAgent, never Homebrew or
the app. `omlx stop` does not stop a manually launched foreground process.
Background defaults are embedded in the wrapper: launchd does not source
`.zshrc`. Editing shell exports will not change an already-running service.
The service's combined output is in the existing base path's
`logs/jangh2-service.log`. Bootstrap success is not a health check: inspect the
logs and `/v1/models` using the existing API key.

Do not run this LaunchAgent and Homebrew/app-managed oMLX on the same port.
To remove login startup:

```sh
launchctl bootout "gui/$(id -u)/local.omlx.jangh2"
rm "$HOME/Library/LaunchAgents/local.omlx.jangh2.plist"
```

Restore the backed-up `~/.local/bin/omlx` file or remove the wrapper to return to
another PATH installation. No runtime, checkpoint, settings or environment
installation is removed by that operation.

## Hardware acceptance

Packed expert arithmetic follows the FP32 reference's weight scaling and
summation order. Small differences that pass isolated expert tolerances can
change later top-k routing and fail whole-model validation. Qualification also
checks exact FP32 reference agreement at representative GEMV dimensions for
decode, sorted prefill, grouped kernels and row tuning. Whole-model acceptance
limits remain unchanged.

For changed bridge code, stop the server and run `check_device_cache.py`, then
`prepare.py --model /checkpoint --long-profile`. Configure `--checkout` on
individual diagnostics where supported; legacy helper defaults reference the
original `~/src/omlx-pr364` layout. Do not activate candidate grouped/device paths
until the complete workflow passes. A copied qualification is not a substitute.

Re-test ordinary chat, streaming, thinking on/off, tools, target-model unload /
reload, a non-JANGH2 model, mixed-model concurrency, cancellation recovery,
prefix-cache restoration and stable memory usage. Benchmark without cancellation
for throughput comparisons; run cancellation separately for stability. Report
actual server settings, not just a free-form benchmark label.

## Licence and provenance

The shared numerical implementation is covered by its [MIT licence](../../omlx/models/jangh2/LICENSE). `NOTICE.md` describes the
original implementation and upstream interfaces consulted. Referenced oMLX,
MLX-LM, MLX-VLM and optional JANG components retain their original licences.
No upstream JANG kernel implementation or model-weight licence is redistributed
as part of this patch.
