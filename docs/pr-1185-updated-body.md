<!--
PR #1185 updated body draft.

INSTRUCTIONS:
1. Go to https://github.com/jundot/omlx/pull/1185
2. Click "Edit" (pencil icon, top right of PR description, NOT a comment)
3. Suggested new TITLE:
       feat(dflash): Gemma 4 backend + Path A double-engine + concurrency-aware routing
4. Replace the existing body with the markdown content below
   (everything between the two horizontal rules `---` below).
5. Click "Update" / "Save".
6. Tell me when done; I can verify the PR description was updated.
-->

---

## Summary

This PR brings Gemma 4 (26B-A4B MoE / dense 31B) up on `DFlashEngine` via `bstnxbt` HEAD, and **lands the internal weight-sharing follow-up that was originally deferred** — see "Path A" below. Five threads of work:

1. **Gemma 4 DFlash backend** — bstnxbt HEAD upgrade, event-API adaptation, `SummaryEvent` summary-access fix. Allows `gemma4` model_type through the DFlash loader. Admin benchmark batch path now runs against `DFlashEngine` instead of silently skipping.

2. **`admin/benchmark` metric symmetry** — Continuous Batching speedup ratios were comparing `batch.tg_tps` (wall-aggregate) against `baseline.gen_tps` (gen-only), producing fake sub-1x ratios when prefill was non-negligible. Both paths now share a `wall_tg_tps` metric. The omlx.ai community-board submission keeps `gen_tps` (peak decode rate) — that's the standard community metric and is annotated in code.

3. **`dflash_max_concurrent` admission cap** — a soft cap on simultaneously in-flight DFlash requests (default 4, `Optional[int]`), enforced via `asyncio.Semaphore`. Excess requests block at the gate until a slot opens.

4. **Path A — DFlashEngine double-engine refactor (NEW)** — what the original PR description said was "deferred to a follow-up PR" is now actually in this PR. DFlashEngine becomes a "smart container" that eagerly embeds a long-lived `VLMBatchedEngine`; the DFlash drafter attaches to the **same** target weights via a non-destructive wrapper (`DFlashVLMTargetWrapper`). Per-request routing on concurrency / KV pressure / max_ctx decides which path serves the request. The `_in_fallback_mode` one-way gate and `_evict_dflash_and_start_fallback` 71-line eviction helper are gone; both paths coexist for the engine's lifetime. Adds `dflash_lazy_drafter` opt-in for high-concurrency deployments that benefit from "framework present but drafter not loaded" (zero Metal contention).

5. **Upstream merge (56 commits) + post-merge fixes** — the branch was rebased onto `jundot/main` (paroquant integration, audio routes, etc.). One latent bug surfaced post-merge: `_load_drafter_bundle` was reading `self._draft_quant_bits`, an attribute the merge had replaced with upstream's 4-field quant config. Caught + fixed in `d80127b` — `_load_drafter_bundle` now uses `_build_quant_spec(weight, activation, group_size)` from the 4 dataclass fields; pytest 42/42 pass.

### Path A — what it solves and how

**Problem.** Pre-Path-A, `dflash_max_ctx` triggers eager eviction of the dflash bundle, then lazy-loads `VLMBatchedEngine`/`BatchedEngine` from scratch. The target weights are **re-loaded** in memory — Gemma 4 26B-A4B Q4 is ~15 GB, so context fallback doubles the resident footprint to ~30 GB. The fallback flag (`_in_fallback_mode`) is also one-way: once flipped, subsequent short-context requests stay on the fallback engine forever.

**Approach.** Refactor `DFlashEngine` so it owns both decode paths simultaneously:

```python
async def start(self):
    self._embedded_vlm = VLMBatchedEngine(...)   # canonical target owner
    await self._embedded_vlm.start()
    wrapped = DFlashVLMTargetWrapper(self._embedded_vlm._vlm_model)  # non-destructive proxy
    self._dflash_bundle = await attach_dflash_to_loaded_target(
        target_model=wrapped,                   # SAME _vlm_model object
        draft_path=...,
        ...,
    )
```

**Weight sharing verified at Python id level** in spike: `engine._target_model._vlm is engine._embedded_vlm._vlm_model == True`. Metal active memory peaks at ~15.6 GB (vs ~30 GB if we had two copies).

**Routing.** Per-request decision based on:

```python
def _route(self, prompt_tokens):
    if self._active_count >= self._max_dflash_concurrent: return "bg"
    if self._kv_pressure() > self._kv_pressure_threshold:  return "bg"
    if self._max_dflash_ctx and len(prompt_tokens) >= self._max_dflash_ctx: return "bg"
    return "dflash"
```

No hysteresis — when concurrency drops, the next request goes back to dflash immediately. Verified by burst-then-single test (`c=16` cold burst then single request at `t+0s`: 153 tok/s, in the dflash band; the prior same-prompt test showed t+30s drop to 119 tok/s, but with unique-prompt prompts the curve flattens, confirming the apparent "decay" was prefix cache aging, not routing state).

**Lazy drafter** (`dflash_lazy_drafter: bool = False`). Defers the drafter+wrapper+factory call until the first dflash-routed request. For workloads that almost never trigger dflash (high concurrency + bg-only), this avoids the Metal contention from the drafter being co-resident even when idle. Trade-off: first dflash request pays ~3s cold-start. Concrete impact on `c=4` with `mc=0`: 129 tok/s (eager, drafter loaded but unused) vs 165 tok/s (lazy, drafter not loaded) — ~28% throughput recovery.

### Empirical motivation

**VLM MTP on Gemma 4 26B-A4B Q4** (initial Gemma 4 bringup measurements; `/v1/chat/completions`, m5max):

| prompt | bare tg | VLM MTP tg | speedup |
| --- | --- | --- | --- |
| PROSE short story | 121 tok/s | 125 tok/s | 1.04x |
| CODE Fibonacci | 123 tok/s | 159 tok/s | **1.29x** |
| MATH arithmetic step-by-step | 121 tok/s | 174 tok/s | **1.41x** |
| JSON 3 capitals | 123 tok/s | 158 tok/s | **1.29x** |
| CCM 6K-token quote-parse (real production prompt) | 107 tok/s | 144 tok/s | **1.34x** (mean of 2 runs) |

**DFlash on Path A** (gemma4-moe-26b-a4b Q4, m5max, structured JSON output prompt, `max_tokens=300`, `temp=0.0`, 4 threads × 3 rounds for concurrent rows):

| concurrency | DFlash OFF | DFlash mc=4 (Path A) | speedup |
| --- | --- | --- | --- |
| c=1 (single) | 98 tok/s | **168 tok/s** | **+72%** |
| c=2 | 130 | 190 | +46% |
| c=4 | 195 | 195 | parity |
| c=8 | 236 | 192 | -19% |
| c=12 | 269 | 202 | -25% |
| c=16 | 259 | 204 | -21% |

`c=12` is roughly where BatchedEngine saturates on this hardware (Apple M5 Max, 100 GB Metal budget). Below saturation, dflash wins; above, BatchedEngine's continuous batching wins. The routing math (concurrency + KV pressure) keeps each request on the right path.

### Caveats — when DFlash is NOT a win

DFlash (and VLM MTP) is prompt-distribution-sensitive. The drafter's accept rate collapses outside its training distribution; once `accept_rate × tokens-per-round ≤ drafter forward cost`, you go negative. On Gemma 4 26B-A4B Q4 with `bstnxbt`'s drafter checkpoint:

| prompt type | DFlash speedup |
| --- | --- |
| Structured outputs (math, code, JSON, tool-calling) | **1.7 – 2.4x** |
| Real production prompts (CCM quote-parse, ~6 K tokens) | **2.2x** |
| General prose / explanation, EN or ZH | **0.58 – 0.78x — slower than bare BatchedEngine** |

Operators routing mixed workloads (e.g. an agent that alternates between tool-calling steps and natural-language summarization) should benchmark on their own prompt distribution. `dflash_max_concurrent` addresses memory + tail-latency under bursts; it does NOT paper over distribution sensitivity. A future PR could add an accept-rate-monitored auto-disable (drop to bare BatchedEngine if the moving-window accept rate falls below threshold), but that is out of scope.

### Files & scope

| file | role |
| --- | --- |
| `omlx/engine/dflash.py` | refactored: embedded VLM in `start()`, `_load_drafter_bundle` extracted, `_ensure_drafter_loaded` lazy path, new `_route` + `_kv_pressure` (corrected formula — `mgr.usage` upstream computes against the bounded free-block queue, which gives misleading 0.997 on empty caches; we use `allocated_blocks/max_blocks` instead), `_record_route` jsonl metric |
| `omlx/speculative/dflash_vlm_target_wrap.py` | NEW. mlx_vlm → mlx_lm Gemma 4 surface adapter (non-destructive proxy). Currently Gemma 4 only |
| `omlx/speculative/dflash_factory.py` | NEW. `attach_dflash_to_loaded_target` — partial bundle factory; loads only the drafter and binds, does NOT call `load_runtime_bundle` (which would re-load target) |
| `omlx/speculative/__init__.py` | NEW. Idempotent monkey-patch for `dflash_mlx.runtime.get_stop_token_ids` (HF GemmaTokenizer's `eos_token_ids` returns `int`, upstream `list(int)` throws TypeError); `detect_fallback_engine_type` helper |
| `omlx/metrics/dflash_routing.py` | NEW. jsonl routing decision writer with size guard + `DFLASH_METRIC_DISABLE` env |
| `omlx/engine_pool.py` | passes new ctor kwargs from `model_settings`; duck-type sniff (`hasattr(engine, "_dflash_bundle")`) replaces `type().__name__` string check |
| `omlx/model_settings.py` | `dflash_lazy_drafter` + `dflash_kv_pressure_threshold` fields (in addition to upstream's quant fields) |
| `tests/test_dflash_engine.py` | route tests rewritten to call `_route()` returning `"dflash"`/`"bg"`; 8 additional Gemma 4 compatibility tests from upstream merge auto-merged |

**Scheduler intentionally untouched.** `omlx/scheduler.py` has zero diff in this PR. Per-request routing inside `DFlashEngine` was preferred over `_route_to_dflash` in the scheduler because `dflash_mlx.SpeculativeSession.open()` does not currently accept a pre-prefilled cache — implementing the latter cleanly requires upstream `dflash_mlx` cooperation. Path A reaches "high concurrency degrades to BatchedEngine throughput" within the engine layer without touching scheduler dispatch.

### Open questions / discussion welcome

- `DFlashVLMTargetWrapper` is hardcoded against mlx_vlm Gemma 4 layout. Should this be generalized (e.g. a plugin registry keyed on family) so Qwen / other VLM families can layer on?
- The monkey-patch on `dflash_mlx.runtime.get_stop_token_ids` is a temporary fix. Happy to send a separate 1-line PR to `bstnxbt/dflash-mlx` upstream so omlx can drop the patch.
- `PagedCacheManager.usage` semantic: we found that the formula `1.0 - free_block_queue.num_free_blocks / (max_blocks - 1)` returns near-1.0 even on near-empty caches because `num_free_blocks` is the **bounded free queue size** (~256 cap), not the true unallocated block count. Our Path A `_kv_pressure()` computes `allocated_blocks / max_blocks` instead. Should `usage` itself be fixed upstream?
- Path A vs scheduler-level routing (à la `_route_to_vlm_mtp`): we kept these as separate strategies — would the project rather see them unified?

### Caveats on this PR's scope

- `DFlashVLMTargetWrapper` covers Gemma 4 only. Adding more families is mechanical (the 6 attribute-rename drift points are documented in the wrapper's module docstring), but not done here.
- Tests grew the unit-test surface (route, build_quant_spec, gemma4 compatibility variants) but **do not** cover the new factory body — its full execution requires a real drafter checkpoint. Validated end-to-end on m5max via `spike6` against `gemma4-moe-26b-a4b` + `z-lab/gemma-4-26B-A4B-it-DFlash`.
- Burst-load soak (100 concurrent against `mc=4`) deferred.
- Fork can stay as-is if the architectural direction here doesn't match the project's preferred shape — happy to maintain in `panwudi/omlx` long-term.

## Test plan

- [x] Server starts cleanly on `gemma4-moe-26b-a4b` with `dflash_enabled=True, dflash_max_concurrent=4`
- [x] `ModelSettings.dflash_max_concurrent` defaults to 4; round-trips through `from_dict` / `to_dict`
- [x] Admin GET `/api/models` exposes `dflash_max_concurrent`; PUT `/api/models/{id}/settings` accepts and persists
- [x] DFlash engine load log prints `max_concurrent=4`
- [x] Continuous Batching speedup ratio post-fix on VLM MTP path: bs 1/2/4/8 wall_tg = 105/107/164/221 tok/s, speedup 1.00/1.02/1.56/2.10x
- [x] Path A: weight sharing verified at Python `id` level (`_target_model._vlm is _embedded_vlm._vlm_model`)
- [x] Path A: end-to-end `gemma4-moe-26b-a4b` + real DFlash drafter on m5max — factory loaded, `Gemma4TargetOps` resolved, real chat completion 200 OK
- [x] Path A: routing recovery from c=16 burst → single returns to dflash band at t+0 (no hysteresis)
- [x] pytest `tests/test_dflash_engine.py` 42/42 pass post-merge + post-quant-fix
- [x] i18n key parity (en / zh / zh-TW) for `dflash_max_concurrent`, `dflash_max_concurrent_placeholder`, `dflash_max_concurrent_help`
- [ ] Burst-load soak (100 concurrent vs `mc=4`) — out of scope, would benefit from a dedicated stress-test harness
- [ ] Family generalization beyond Gemma 4 — deferred to follow-up

---
