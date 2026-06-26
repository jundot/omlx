# oMLX unload memory retention investigation

Date: 2026-06-26

## Problem

The observed oMLX backend state was inconsistent with the UI and admin cache
surfaces:

- The UI showed no active work and the hot cache was empty.
- Only one model was loaded: `Qwen3.6-35B-A3B-nvfp4`.
- `/api/status` reported no active or waiting requests.
- The process still held a very large resident footprint: `64.8G`.
- `vmmap -summary` attributed most of it to Metal/IOAccelerator memory:
  `IOAccelerator (graphics): 61.9G`.

Clearing the hot cache reclaimed nothing:

- `/admin/api/hot-cache/clear`: `total_cleared=0`, `bytes_reclaimed=0`.

Unloading the only loaded model released model-weight memory, but not all Metal
memory:

- `POST /v1/models/Qwen3.6-35B-A3B-nvfp4/unload`
- Log result: `freed=19.01GB`, `active_memory: 42.89GB (settled)`.
- After unload, `/api/status` showed `models_loaded=0` and
  `model_memory_used=0B`.
- `vmmap` still showed footprint around `45.6G`, with `IOAccelerator 42.9G`.

Restarting the server released the memory:

- New process footprint: about `109.5M`.
- `IOAccelerator`: about `48K`.

This proves the retained memory was not hot cache, not model weights, and not
visible as a loaded model. It was process-lifetime Metal/MLX memory retained
after engine unload.

Related upstream issue comment:

- https://github.com/jundot/omlx/issues/1691#issuecomment-4809293896

## Evidence from logs

The incident followed a long-context generation with repeated chunked prefill
memory pressure:

- Many log lines showed `Chunked prefill above max_bytes... 68.xGB > 67.9GB`.
- At `2026-06-26 17:10:29`, oMLX stored a boundary cache snapshot:
  `storing 28672/40512 tokens`.
- The same completion reported:
  `Chat completion: 174 tokens in 183.43s, prompt: 40338`.

Later, after model unload:

- Scheduler shutdown/deep reset completed.
- oMLX logged `freed=19.01GB`.
- Active MLX memory remained `42.89GB`.

This points to large temporary/prefix-cache/prefill Metal allocations surviving
the engine lifecycle even after request and model state were no longer visible
through the admin API.

## Analysis

The likely root cause is a teardown-order and executor/stream ownership bug in
`EngineCore.close()`.

Relevant behavior before the patch:

- Each engine owns a per-engine MLX executor and a thread-local MLX stream.
- Scheduler shutdown and deep reset run through that per-engine executor.
- After that, `EngineCore.close()` cleared the MLX compile cache and shut down
  the executor.
- Only after executor handling did it clear output collectors and set
  `self.model`, `self.tokenizer`, and `self.scheduler` to `None`.
- There was no final `mx.synchronize(stream)` plus `mx.clear_cache()` pass on
  the engine worker thread after dropping those references.
- Other admin/global clear paths use the global MLX executor, not necessarily
  the per-engine executor/stream that owns the retained buffers.

One subtle reference-retention issue also existed:

- The `for fn in (self.scheduler.shutdown, self.scheduler.deep_reset)` loop left
  the local variable `fn` bound to the last scheduler method after the loop.
- That bound method can keep the scheduler object alive until `close()` returns.
- A final GC/reclaim pass before dropping this reference would not see the true
  reclaimable object graph.

Why this matches the runtime evidence:

- Model unload released about `19GB`, consistent with model weights being
  dropped.
- The remaining `42.9GB` was Metal/IOAccelerator memory, consistent with large
  prefill/cache/temporary MLX buffers.
- Restart released everything, proving it was process-local retained MLX/Metal
  state rather than external files or hot cache entries.

Secondary observation:

- `VLMBatchedEngine.has_active_requests()` only checks output collectors.
- It may report no active requests even while scheduler-side async cleanup or
  deferred removals exist.
- This can make the UI look idle earlier than the scheduler/MLX memory lifecycle
  is fully settled. This is probably not the primary cause of the unload-after
  residual memory, but it can hide cleanup activity from users.

## Current patch

Files changed:

- `omlx/engine_core.py`
- `tests/test_per_engine_threads.py`
- `tests/test_engine_core.py`

Patch intent:

1. Add `_final_engine_thread_reclaim(stream)`.
2. In that helper:
   - run `gc.collect()`;
   - call scheduler's `_sync_and_clear_cache(stream)`;
   - run `gc.collect()` again.
3. In `EngineCore.close()`:
   - run scheduler shutdown and deep reset on the engine executor;
   - clear the leftover bound-method local `fn`;
   - close and detach the SSD cache manager;
   - clear output collectors and request-side stream state;
   - set `self.model`, `self.tokenizer`, and `self.scheduler` to `None`;
   - submit `_final_engine_thread_reclaim(self._mlx_stream)` to the same
     per-engine executor;
   - only then clear the MLX compile cache and shut down or immortalize the
     executor according to the existing compile-cache policy.

The important design choice is that final reclaim happens after engine-owned
references are dropped, and on the same worker thread/stream that owned the MLX
allocations.

Test changes:

- `test_close_clears_compile_cache_then_shuts_down` now asserts:
  - final reclaim is called before compile-cache clear;
  - by the time final reclaim runs, `engine.model`, `engine.tokenizer`, and
    `engine.scheduler` are already `None`.
- `test_close_fatal_exits_when_compile_cache_clear_times_out` now accounts for
  the extra executor submission before compile-cache clear.

## Verification update

Initial direct test command:

```bash
uv run pytest tests/test_per_engine_threads.py tests/test_engine_core.py
```

failed while preparing dependencies:

```text
Failed to download mlx-metal==0.31.2
Failed to extract archive: mlx_metal-0.31.2-py3-none-macosx_26_0_arm64.whl
I/O operation failed during extraction
Failed to download distribution due to network timeout.
Try increasing UV_HTTP_TIMEOUT (current value: 30s).
```

Then the test was moved into tmux:

```bash
tmux new -A -d -s omlx
tmux send-keys -t omlx 'cd /Users/zhouwei/code/zw/omlx && UV_HTTP_TIMEOUT=300 uv run pytest tests/test_per_engine_threads.py tests/test_engine_core.py 2>&1 | tee /tmp/omlx-pytest.log' C-m
```

That command ran for several minutes with no stdout and
`/tmp/omlx-pytest.log` remained empty, so it was interrupted and split into
environment setup plus test execution.

Current tmux command:

```bash
cd /Users/zhouwei/code/zw/omlx && UV_HTTP_TIMEOUT=300 uv sync 2>&1 | tee /tmp/omlx-uv-sync.log
```

As of the latest check:

- `uv sync` was still running.
- `/tmp/omlx-uv-sync.log` was still `0` bytes.
- `lsof -p <uv-pid>` showed that `uv` was active, not deadlocked:
  - it had HTTPS connections open;
  - it was writing temporary files under `~/.cache/uv`;
  - observed files included `mlx/lib/mlx.metallib` and `cmake`.

So the original blocker was dependency download/extraction speed or network
behavior, not a known code/test failure.

Because full `uv sync` was too slow on the GitHub-backed dependencies, the
verification path was narrowed to the target tests:

1. Stop the slow `uv sync` / `git fetch` process.
2. Use the existing `.venv`.
3. Install only the minimum packages needed for these tests from the Tsinghua
   PyPI mirror:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  pytest pytest-asyncio mlx==0.31.2 fastapi uvicorn numpy pyyaml \
  requests psutil setproctitle transformers tokenizers huggingface-hub \
  jinja2 itsdangerous pillow rich sentencepiece tiktoken protobuf tqdm \
  jsonschema python-multipart tabulate socksio

uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  mlx-lm mlx-vlm
```

Notes:

- The domestic PyPI mirror fixed the wheel/package download bottleneck.
- It does not replace `git+https://github.com/...` dependencies from
  `pyproject.toml`; those still need GitHub or an explicit Git mirror.
- Running MLX imports inside the sandbox failed with:
  `RuntimeError: [metal::load_device] No Metal device available`.
- Running the same test command outside the sandbox allowed MLX to access Metal.

Targeted verification passed:

```bash
.venv/bin/python -m pytest tests/test_per_engine_threads.py tests/test_engine_core.py
```

Result:

```text
78 passed in 4.31s
```

## Recommended next steps

1. If full dependency sync is still needed later, prefer the domestic PyPI
   mirror for wheel packages, but handle GitHub dependencies separately:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
```

2. For this patch, the targeted tests have already passed:

```bash
.venv/bin/python -m pytest tests/test_per_engine_threads.py tests/test_engine_core.py
```

3. Run basic diff hygiene:

```bash
git diff --check
```

This passed after the patch.

4. Perform a live memory validation:

- Start patched oMLX.
- Load `Qwen3.6-35B-A3B-nvfp4`.
- Reproduce a long-context request similar to the observed `~40k` prompt case.
- Confirm idle state through `/api/status`.
- Record `vmmap -summary <pid>` before unload.
- Unload the model.
- Record:
  - `/api/status`;
  - admin active model/cache stats;
  - `vmmap -summary <pid>`.
- Expected result after the patch:
  - model-weight memory drops as before;
  - retained `IOAccelerator`/`mx.get_active_memory()` should be much lower than
    the previous `42.89GB` residual;
  - if it is not lower, the next suspect is scheduler-side request/cache
    reference retention after deep reset rather than missing final worker-thread
    reclaim.

5. Consider a follow-up UI/admin fix:

- Make active-request reporting include scheduler cleanup states such as pending
  async removals, in-flight store futures, or deferred clear state.
- This would not replace the memory fix, but would make "idle" status more
  honest during cleanup.

## Historical blocked commands

These commands were part of the blocked full-sync path and are kept here for
auditability.

Check whether the tmux `uv sync` command completed:

```bash
tmux capture-pane -pt omlx -S -220
ls -l /tmp/omlx-uv-sync.log
```

If `uv sync` is still running with empty output, inspect the active process:

```bash
ps aux | rg '[u]v sync|[u]v run pytest'
lsof -p <uv-pid>
```

Once `uv sync` completes, run the targeted tests separately:

```bash
UV_HTTP_TIMEOUT=300 uv run pytest tests/test_per_engine_threads.py tests/test_engine_core.py
```

or in the existing tmux session:

```bash
tmux send-keys -t omlx 'cd /Users/zhouwei/code/zw/omlx && UV_HTTP_TIMEOUT=300 uv run pytest tests/test_per_engine_threads.py tests/test_engine_core.py 2>&1 | tee /tmp/omlx-pytest.log' C-m
```
