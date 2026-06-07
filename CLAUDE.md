# omlx (project notes for Claude Code)

LLM inference server optimized for Apple Silicon. This checkout is a **fork**, not a clean upstream clone. Read this before bumping versions or resolving merges.

## Repo layout & install
* Editable install: `~/omlx/.venv/bin/omlx` (shell alias `omlx`). Code is live; no rebuild needed after edits.
* Python 3.14 venv at `.venv`. Run tests/CLI via `.venv/bin/python` and `.venv/bin/omlx`.
* Version is dynamic (`pyproject.toml` `dynamic = ["version"]`), derived from git tags. Check with `.venv/bin/omlx --version`.

## Fork topology (important)
* `origin` = upstream `jundot/omlx` (the moving project).
* `fork` = `khsd6327/omlx` (our fork, where local `main` is pushed).
* Local `main` tracks `fork/main` and carries our own commits on top of upstream.
* Workflow is: periodically `git merge origin/main` into local main, keeping a small set of fork customizations layered on top. We track origin/main (which moves past rc tags), not the tags themselves, so "latest" usually means `origin/main`, not the newest `vX` tag.

## Bumping to latest (the procedure that worked for 0.3.12 -> 0.4.0rc1)
1. `git fetch origin --tags && git fetch fork`. Compare with `git rev-list --left-right --count main...origin/main`.
2. `git merge origin/main --no-edit`. Expect conflicts in files our fork commits touched.
3. Resolve conflicts (see conflict map below). After resolving, confirm none remain: `git diff --name-only --diff-filter=U` should be empty, and grep for real markers with `git grep -nE '^(<<<<<<< |>>>>>>> |=======$)'` (note: decorative `====` underline lines in docstrings are false positives).
4. Reinstall to sync dependency changes: `.venv/bin/pip install -e . --quiet`.
5. **Force-reinstall git-pinned deps.** pip SKIPS a `git+...@<commit>` dependency if the package version string is unchanged (e.g. `mlx-vlm` stayed `0.5.0` across a commit-pin bump), so the new commit silently does not install. Run:
   `.venv/bin/pip install --force-reinstall --no-deps "mlx-vlm @ git+...@<commit>" "dflash-mlx @ git+...@<commit>" "paroquant==<ver>"`
   using the exact pins from the merged `pyproject.toml`. Then verify with `pip check` and `pip show`.
6. Run the full suite: `.venv/bin/python -m pytest -q -p no:cacheprovider` (about 4 minutes; ~4900 tests). `pytest.ini` is the active config (it ignores the `[tool.pytest]` block in pyproject, which prints a harmless warning).

## Conflict map / decisions
Our fork's heavy customizations live in `omlx/scheduler.py`. Key fork commits:
* `3653101` grouped sampling and MLX stream integration (`_grouped_sample`, `_sampler_key`, `_SAMPLER_BATCH_*`)
* `3d80802` streaming and cache-store hot-path opt (`_pending_cache_stores`, deferred cache store)
* `37f5c6f` scheduler fairness + incremental prefill budget (4-tuple external-prefill return)
* Plus: KMMLU eval one-indexed answer fix, MTP-patch guard against mocked mlx-lm imports, Playwright gitignore.

Resolution conventions:
* Additive-method conflicts (both sides add new methods at the same spot): keep both groups.
* `modify/delete` conflicts where upstream deleted a file our fork patched (e.g. upstream removed `omlx/patches/gated_delta_advance.py` in `3d2fef5` when bumping the mlx-vlm pin, since the fix moved into mlx-vlm itself): follow the deletion with `git rm`, after grepping that nothing still imports it. Our local edits to such files are moot once the file is gone upstream.

**Decision on 2026-06-01 (0.4.0rc1 bump):** the external-prefill conflict was a deep architectural divergence (our 4-tuple incremental prefill vs upstream's 2-tuple + TurboQuant + adaptive guard + memory monitoring). Per Ted's call we took **upstream `scheduler.py` wholesale** (`git checkout --theirs omlx/scheduler.py`), dropping the fork's scheduler fairness / incremental-prefill-budget / streaming hot-path / grouped-sampling features in exchange for upstream TurboQuant et al. The original fork work survives in commits `37f5c6f`, `3d80802`, `3653101` if it needs re-porting onto the new upstream structure.

## After taking a side wholesale, prune orphaned tests
Dropping a feature leaves fork tests that assert now-absent internals (they fail with `AttributeError` on `_SAMPLER_BATCH_*`, `_pending_cache_stores`, etc., or on changed behavior). Confirm a test is fork-only before deleting: `git show origin/main:tests/<file> | grep "def <test>"` (absent = fork-only). The 0.4.0rc1 bump removed 6 fork tests plus the whole fork-only file `tests/test_sampler_batching_patch.py`.

## Bump history & what to expect now
* `0.3.12 -> 0.4.0rc1`: hard merge, scheduler taken wholesale, 6+1 orphan tests pruned.
* `0.4.0rc1 -> 0.4.0rc2 -> 0.4.2.dev2 -> 0.4.2.dev3`: **conflict-free** apart from one `modify/delete` (upstream removed `gated_delta_advance.py`). Since we stopped carrying scheduler customizations, the only recurring conflict class is `modify/delete` on patches upstream folds into its mlx-* pins. Expect clean `git merge origin/main` going forward; still run the full suite.
* Recurring dep churn: `mlx-vlm` git pin bumps almost every release (same `0.x.y` version string, new commit -> always force-reinstall). New runtime deps arrive as plain pip pins (e.g. `markitdown[pdf,docx,pptx]==0.1.6` in dev3) and install via `pip install -e .`.

## Deployment / runtime (this is the live inference backend on the MacBook)
The repo is also the **running production server** on the M5 Max laptop, not just a checkout. Edits are live (editable install) but the running process only picks them up on restart.

* **Service:** launchd agent `com.omlx.serve` (`~/Library/LaunchAgents/com.omlx.serve.plist`), `KeepAlive=true`, `RunAtLoad=true`, serving `omlx serve --host 0.0.0.0 --port 8000`. Logs: `~/.omlx/logs/omlx-serve.{log,err}`.
* **Restart after a bump:** `launchctl kickstart -k gui/$(id -u)/com.omlx.serve`, then poll `curl -s http://127.0.0.1:8000/health`. Health reports `default_model` + loaded count (`loaded_count:0` right after restart is normal — models lazy-load on first request).
* **Kokoro TTS is a SEPARATE server, not part of omlx:** `com.kokoro.tts.serve` on `:8001`, its own Python 3.12 + uvicorn `server:app`, independent of this repo (omlx is `:8000`, Python 3.14). Bumping omlx / mlx-vlm / mlx-audio does NOT touch it. omlx's own audio changes are audio-INPUT (Gemma4 understanding) + its built-in TTS model support (chatterbox etc.), distinct from the standalone Kokoro service.

## Runtime config lives in `~/.omlx/` (NOT the repo)
* `~/.omlx/settings.json` (server/auth/memory/cache/claude_code) and `~/.omlx/model_settings.json` (per-model settings, managed via the admin page). Models live in `~/.omlx/models/`.
* **Default model = the `model_settings.json` entry with `is_default:true`** (`get_default_model_id()`); server.py falls back to `available_models[0]` with a `Default model '<x>' not found, using first model` warning if that id isn't discovered. **Gotcha:** when a model is re-downloaded under a new name (oMLX quant-naming changes, e.g. `Qwen3.6-35B-A3B-oQ6` -> `Qwen3.6-35B-A3B`, `...-oQ4` -> `...-6bit`), the stale `is_default` pointer AND `settings.json -> claude_code.{opus,sonnet,haiku}_model` silently break. Fix by repointing to a currently-served id (cross-check against `GET /v1/models`), then restart. Back up the json before editing; the admin write-back can otherwise clobber a live edit.
* **Admin login uses the API key AS the password** (`auth.api_key` in settings.json; `POST /admin/api/login` compares against it). Shortcut: `GET /admin/auto-login?key=<API_KEY>`. The admin API needs a session cookie, so it can't be driven by curl with just the key — edit the json + restart instead.
