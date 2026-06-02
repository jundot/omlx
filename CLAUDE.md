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
* `gated_delta_advance.py`: upstream's `__call__` added `target_verify`; our fork added `**_` for forward-compat kwargs. Keep BOTH: `target_verify: bool = False, **_: Any`. The function body uses `target_verify`, so dropping it breaks.
* `_sync_and_clear_cache(...)`: our fork passes `self._stream`; keep that AND any upstream additions (e.g. requeue logic) at the same site.
* Additive-method conflicts (both sides add new methods at the same spot): keep both groups.

**Decision on 2026-06-01 (0.4.0rc1 bump):** the external-prefill conflict was a deep architectural divergence (our 4-tuple incremental prefill vs upstream's 2-tuple + TurboQuant + adaptive guard + memory monitoring). Per Ted's call we took **upstream `scheduler.py` wholesale** (`git checkout --theirs omlx/scheduler.py`), dropping the fork's scheduler fairness / incremental-prefill-budget / streaming hot-path / grouped-sampling features in exchange for upstream TurboQuant et al. The original fork work survives in commits `37f5c6f`, `3d80802`, `3653101` if it needs re-porting onto the new upstream structure.

## After taking a side wholesale, prune orphaned tests
Dropping a feature leaves fork tests that assert now-absent internals (they fail with `AttributeError` on `_SAMPLER_BATCH_*`, `_pending_cache_stores`, etc., or on changed behavior). Confirm a test is fork-only before deleting: `git show origin/main:tests/<file> | grep "def <test>"` (absent = fork-only). The 0.4.0rc1 bump removed 6 fork tests plus the whole fork-only file `tests/test_sampler_batching_patch.py`.
