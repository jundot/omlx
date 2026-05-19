# Discussion: DFlashEngine double-engine refactor (Path A) — solving 2x weight memory + concurrency degradation

Labels: `discussion`

---

Hi @jundot — first, thanks for oMLX. We've been running it on M2 Max / M5 Max (128 GB) for tool-calling and reasoning workloads and it's been the cleanest local serving story we found.

This is a **discussion**, not a PR. We hit two problems with the current `DFlashEngine` long-context fallback and prototyped a refactor (`Path A`) in our fork. Before doing any cleanup or pushing a PR, we'd like to gauge whether the direction is something you'd consider upstreaming, or whether you already have a different fix in mind.

We're happy to keep this entirely in our fork if the design doesn't fit your roadmap — no expectation that you accept it.

---

## 1. Problem statement

Current `DFlashEngine` keeps an `_in_fallback_mode` flag and a sibling fallback engine for the long-context path. From `docs/experimental/dflash_mlx_integration.md`:

> `DFlashEngine.start()` ... `start fallback engine` → `VLMBatchedEngine (if model detected as VLM)` / `BatchedEngine (otherwise)`

In practice this means two consequences we'd like to discuss:

**1.1 — Two copies of the target model weights in Metal memory.**
`dflash_mlx.runtime.load_target_bundle()` loads the target inside the DFlash bundle, **and** the fallback `VLMBatchedEngine` loads the same target a second time (separate `mlx_vlm` model object, separate weights). For `gemma-4-26b-a4b-it` quantized to int4 (~15 GB resident) this is ~30 GB on a single host. On a 128 GB Mac that's manageable; on 64 GB it's the difference between "DFlash works" and "DFlash OOMs".

**1.2 — `dflash_max_ctx` collapses a 2D decision into 1D.**
The actual decision space we care about is `(context length, concurrency, KV pressure)`. Today only `ctx_len ≥ DFLASH_MAX_CTX` triggers fallback, and the trigger is **one-way**: once flipped, the engine stays in fallback mode for its lifetime. So a single 5K-prompt request permanently disables DFlash for the engine, even after that request finishes. The single-request constraint (limitation §1 in the integration doc) is real, but the all-or-nothing engine-lifetime gate is what we wanted to relax.

---

## 2. What we tried (Path A)

Repo: `panwudi/omlx`, branch `feat/gemma4-dflash`, main commit `ef00c8b` (already rebased onto your `9749c40`).

**Core idea.** Make `DFlashEngine` a thin **container** around a long-lived `VLMBatchedEngine`, and have the DFlash drafter attach to the **same target weights** the batched engine already loaded.

```python
class DFlashEngine:
    def start(self):
        # Eagerly construct the embedded batched engine (was lazy before)
        self._embedded_vlm = VLMBatchedEngine(self._model_name, ...)
        self._embedded_vlm.start()

        # Wrap mlx_vlm target so dflash_mlx's Gemma4TargetOps accepts it
        wrapped = DFlashVLMTargetWrapper(self._embedded_vlm._vlm_model)

        # New factory: run dflash_mlx's draft load + hook install,
        # but reuse the already-loaded target (no second weight copy)
        self._bundle = attach_dflash_to_loaded_target(
            target_model=wrapped,
            draft_path=self._dflash_draft_path,
            ...
        )

    def generate(self, request):
        if self._route(request) == "dflash":
            return self._dflash_generate(request)
        return self._embedded_vlm.generate(request)
```

ID-level weight sharing verified: `engine._target_model._vlm is engine._embedded_vlm._vlm_model == True`.

Per-request routing replaces the lifetime gate:

```python
def _route(self, request) -> Literal["dflash", "bg"]:
    if active_requests >= self._max_dflash_concurrent:
        return "bg"          # reason="concurrency"
    if kv_pressure > self._kv_pressure_threshold:
        return "bg"          # reason="kv_pressure"
    if ctx_len > self._dflash_max_ctx:
        return "bg"          # reason="max_ctx" (kept for compat)
    return "dflash"
```

The `_in_fallback_mode` one-way gate is removed; both paths coexist for the engine's lifetime, and every request gets a fresh decision.

We also added `dflash_lazy_drafter: bool` after observing that even when the drafter doesn't participate in decode it still occupies Metal memory and degrades concurrent throughput (see §3).

---

## 3. Bench data — `gemma4-moe-26b-a4b` on M5 Max 128 GB

Same prompt (structured JSON output, ~400-token completion), `temperature=0`, 5-run average.

| Workload | DFlash OFF | DFlash ON (eager, mc=1) | Delta |
|---|---|---|---|
| **c=1 sequential** | 84.8 tok/s | **94.0** tok/s | **+10.9%** |
| **c=4 concurrent** (eager drafter, mc=1) | 180 tok/s | 115 tok/s | **−36%** |
| **c=4 concurrent** (eager drafter, mc=0 — drafter loaded but never used) | 180 tok/s | 129 tok/s | −28% |
| **c=4 concurrent** (`lazy_drafter=True`, mc=0) | 180 tok/s | **165 tok/s** | −8% |
| **c=12 concurrent** (lazy_drafter, mc=0) | 269 tok/s | 202 tok/s* | −25% |

\* c=12 still shows wrapper-level overhead under heavy concurrency; remaining gap is `DFlashEngine`/`VLMBatchedEngine` indirection, not the drafter.

Two findings we'd like to highlight:

1. **`install_verify_linears` is not the bottleneck.** mc=0 (drafter loaded but never participates) vs hypothesis test with verify-linears installed but no drafter: same throughput (113 vs 115). The drafter **occupying Metal memory** is the culprit.
2. **`lazy_drafter=True` recovers ~70% of the loss.** Combined with `dflash_max_concurrent=0` it gives a "DFlash available for single-user workflows, batched throughput preserved for concurrent" mode that we couldn't get from the current architecture without unloading the whole engine.

Resulting recommended config matrix:

| Workload | Config |
|---|---|
| Single user / dev | `enabled=True, mc=1, lazy=False` (+11% latency) |
| Multi-user server | `enabled=True, mc=0, lazy=True` (−8% vs OFF, no DFlash path) |
| DFlash off | `enabled=False` (baseline) |

---

## 4. Files we changed (scope)

All changes confined to:

- `omlx/engine/dflash.py` — refactor (+362 −177), removed `_in_fallback_mode` gate, added `_route()`, `_load_drafter_bundle()`, `_ensure_drafter_loaded()`
- `omlx/speculative/dflash_factory.py` — new (~200 LOC), `attach_dflash_to_loaded_target()`
- `omlx/speculative/dflash_vlm_target_wrap.py` — new (~140 LOC), wraps `mlx_vlm.Gemma4` to satisfy `Gemma4TargetOps.supports_model()`
- `omlx/speculative/__init__.py` — adds `detect_fallback_engine_type()` + the monkey-patch noted in §5.2
- `omlx/metrics/dflash_routing.py` — new (~90 LOC), jsonl with `routed_to` / `reason` / `kv_usage_ratio` for tuning
- `omlx/engine_pool.py` — 4 new ctor kwargs (defaulted, backward compatible)
- `omlx/model_settings.py` — adds `dflash_lazy_drafter`, `dflash_kv_pressure_threshold`
- `tests/test_dflash_engine.py` — rewrote 2 routing tests, +8 cases. **34/34 pass.**

Scheduler, batched engine, VLM engine, admin UI: untouched.

---

## 5. Questions we'd like your take on

These are the points where we genuinely don't know the right answer, and where your call would change what a PR looks like — or whether there should be one.

### 5.1 — `DFlashVLMTargetWrapper` is currently Gemma 4 only

The wrapper bridges `mlx_vlm` → `mlx_lm`-shaped surface that `Gemma4TargetOps.supports_model` expects (`args.layer_types`, `_get_per_layer_inputs`, etc.). It's non-invasive (proxy + `__getattr__` passthrough, doesn't mutate the underlying model), but the property names it maps are Gemma-4-specific.

Qwen targets are already `mlx_lm`-shaped, so they don't need the wrapper — they'd flow straight through. But a generalized plugin (`TargetWrapperRegistry` keyed by model family) would be the clean version.

**Would you prefer a plugin/registry shape before considering merge, or is Gemma-4-only acceptable for an initial PR with a follow-up generalization?**

### 5.2 — Monkey-patching `dflash_mlx.runtime.get_stop_token_ids`

We hit a bug where `dflash_mlx.runtime.get_stop_token_ids` does `list(tokenizer.eos_token_ids)`, but `GemmaTokenizer.eos_token_ids` returns an `int`, not a list, so `list(int)` raises `TypeError`. Our `omlx/speculative/__init__.py` does an idempotent import-time monkey-patch coercing `int → [int]`.

This is obviously not the right long-term home. **Should we push the fix to `bstnxbt/dflash-mlx` instead and have oMLX bump the pin?** We're happy to open that PR if you'd consider bumping `dflash-mlx` after it lands.

### 5.3 — `PagedCacheManager.usage` semantics

While wiring KV-pressure routing we found `PagedCacheManager.usage` returns values that don't match our observation. The formula is roughly:

```python
1 - free_block_queue.num_free_blocks / (max_blocks - 1)
```

`free_block_queue` is a **bounded** queue (we observed ~256 entry cap), so `num_free_blocks` is not the count of unallocated blocks — it's the queue's current size, which saturates well below `max_blocks`. With an essentially empty cache we were getting `usage = 0.9974`.

We worked around it with `allocated_blocks_len / max_blocks` (0.0026 in the same scenario, matches reality). **Is this a known issue, an intentional semantic we're misreading, or worth a separate fix on your side?** If it's the first two we'd like to align on naming before relying on it.

### 5.4 — Path A vs scheduler-level routing

Your `vlm_mtp` integration takes the scheduler-level routing approach. Path A is deliberately the **engine-level** alternative — no scheduler touch, all logic in `DFlashEngine`. They're not necessarily exclusive (Path A could be one branch a scheduler-level router selects), but they are different design centers.

**Do you see Path A and scheduler-level routing as eventually merging, or as two parallel options for different decode techniques?** If the latter, we'd structure a PR very differently than if the former.

---

## 6. Open questions to maintainers

To save round-trips:

1. **Are you already working on a fix for the 2x-weight-memory issue?** If yes we'll happily defer; no point in parallel solutions.
2. **Between "double engine + weight sharing" and "scheduler-level per-request routing", do you have a preference for where DFlash routing should live long-term?**
3. **If Path A — after generalizing the wrapper (5.1), upstreaming the dflash-mlx fix (5.2), and aligning on `PagedCacheManager.usage` (5.3) — were submitted as a PR, would you consider it?** And if so, are there constraints you'd want stated up front (test coverage, benchmark requirements, design-doc-first, etc.)?

---

## 7. What we are explicitly **not** claiming

- The wrapper is Gemma-4-only today. Qwen support is not tested and would need its own validation path (even if it doesn't need wrapping).
- The `get_stop_token_ids` monkey-patch is a temporary measure; the real fix belongs in `dflash-mlx`.
- The c=4/c=12 concurrent results still have an 8–25% gap vs OFF baseline — engine-wrapper overhead, not solved by Path A.
- We haven't run the full oMLX test suite cross-platform; only the dflash + engine_pool tests on M5 Max.
- If this direction doesn't fit oMLX's roadmap, we're entirely fine maintaining it as a fork patch. Nothing here depends on upstream acceptance.

Happy to share the full bench scripts, routing jsonl samples, or a draft PR diff if any of the above is worth digging into. Thanks for reading.

---

*Refs:* fork branch [`panwudi/omlx@feat/gemma4-dflash`](https://github.com/panwudi/omlx/tree/feat/gemma4-dflash), main commit `ef00c8b`, rebased onto upstream `9749c40` at merge commit `47f4f26`. Internal design doc: `docs/dflash-pathA-spec.md`.
