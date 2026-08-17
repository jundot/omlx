# Technical Report — Single-State ArraysCache Prefix Cache Fix (LFM2.x compatibility)

**Branch**: `fix/arrays-cache-single-state-prefix-cache`
**Scope**: oMLX prefix cache, GDN sidecar boundary snapshots
**Status**: All 13 new regression tests pass; 756 tests pass in the broader cache
suite; 0 new regressions (the 8 failures observed are pre-existing
`mlx_vlm.speculative` import errors on `main`, reproduced via `git stash`).

---

## 1. Root cause

The investigation matches the brief: `BoundarySnapshotSSDStore.save()` already
serialises single-state ArraysCache layers (LFM2.x-style recurrent caches with
`state_count == 1`) — `state_count = "1"` is written to the V3 metadata and the
single element lands under `layer_{i}_state_0`. Likewise, `load()` reconstructs
a 1-element state tuple. The GDN sidecar path writes the recurrent member
correctly (the `_should_quantize_gdn_state` gate is currently `state_index == 1`,
which keeps the index-0 single state at fp32 — a safe and acceptable fallback
since the task states sidecars are already written correctly).

The blocker was **entirely on the prefix-cache read path**, in three guards
that all assumed `len(state) >= 2` for Arrays-family layers:

| # | Location | Guard | Impact on state_count=1 |
|---|----------|-------|-------------------------|
| 1 | `BlockAwarePrefixCache._validated_gdn_snapshot_layers` (`omlx/cache/prefix_cache.py` L349-356) | `if not isinstance(state, (list, tuple)) or len(state) < 2: return None` | GDN sidecar overlay rejected as invalid; chosen payload is `None`; the cache hit falls back to legacy fp32 or is rejected entirely |
| 2 | `BlockAwarePrefixCache._extract_block_tensor_slice` non-sliceable branch (L2328+) | `len(state) > 2` / `len(state) >= 2` else placeholder | The non-last block's recurrent state is replaced with `mx.zeros((1,))`, breaking walk-back truncation and forcing a full re-prefill |
| 3 | `BlockAwarePrefixCache._validate_block_cache_data` N-tuple branch (L4471) | `len(elements) < 2` | Block is rejected as having "invalid N-tuple state", blocking reconstruction |

The encoder side (`boundary_snapshot_store.py` `_serialize`, `_should_quantize_gdn_state`,
`_encode_gdn_state`) was correct: a single-state cache produces a 1-element
tensors_raw dict whose single `layer_{i}_state_0` key carries the recurrent
tensor, no decode failures occur, and the file is written successfully.

`__nstate__` (the existing `(marker, class_name, [elements])` triple) already
round-trips arbitrary `len(state) >= 2` N-tuple caches. The fix simply routes
single-state caches through the same marker — no new code path, no new
serialization, no new reconstruction logic.

---

## 2. Patch rationale

### 2.1 Surgical, additive only

Each guard is widened from `len(state) < 2` (or `len(elements) < 2`) to
`len(...) < 1` so that an empty tuple is still rejected as malformed, and a
single-element tuple is accepted. The 2-tuple legacy Mamba/SSM path is
preserved unchanged — `state_count == 2` still emits a plain `(conv, ssm)`
pair because reconstruction for that exact shape does not consult the
`__nstate__` marker (it predates the marker format).

### 2.2 Reuse of an existing mechanism

The `__nstate__` marker path is the canonical way to round-trip non-2-tuple
ArraysCache states in oMLX — it already supports state_count ≥ 3 (PoolingCache,
DeepSeek V4 composite, MambaCache with extra meta slots). Routing
`state_count == 1` through the same marker:

- avoids inventing a new wire format;
- keeps `ArraysCacheHandler.reconstruct_cache` unchanged (it already builds a
  `SizedArraysCache` from `state["states"]` of any length ≥ 0);
- keeps the `_read_state_tuple_raw` / `_read_state_tuple_arrays` deserialisers
  unchanged (they iterate `range(count)` and pack into a tuple);
- means a future model with `state_count == 0` would still need a separate
  guard — the `len(state) < 1` lower bound rejects it, exactly as before.

### 2.3 Why the encoder was not modified

The task statement is explicit:

> BoundarySnapshotSSDStore correctly serializes ArraysCache with state_count=1.
> GDN sidecars are correctly written.

The encoder's only state_count-sensitive gate is `_should_quantize_gdn_state`,
which currently fires at `state_index == 1`. For a 1-element tuple, the
recurrent member is at `state_index == 0`, so no GDN quantization happens and
the sidecar is written as raw fp32. This is acceptable: storage cost matches
the prior 2-state path's fp32 storage of the recurrent member, and the
quantization knob (`gdn_sidecar_state_dtype`) remains opt-in per the existing
config surface. Touching the gate would require knowing `state_count` at
quantize time, which would change the encoder contract for every other
caller — disproportionate risk for the POC objective.

### 2.4 Diff size

```
omlx/cache/prefix_cache.py | 67 +++++++++++++++++++++++++++++++++-------------
1 file changed, 49 insertions(+), 18 deletions(-)
```

Three small edits in one file, one new test file (`tests/test_arrays_cache_single_state.py`,
547 lines, 13 tests). No other source files modified. No model patches added
(LFM2.x is already supported by `mlx_lm.models.cache.ArraysCache` itself —
the bug was purely in oMLX's prefix-cache handling).

---

## 3. Files changed

### 3.1 `omlx/cache/prefix_cache.py`

Three guards, same fix pattern:

**Edit 1 — `_validated_gdn_snapshot_layers` (L349-365)**

Lower bound `len(state) < 2` → `len(state) < 1`; add explicit
`len(state) == 1` branch that emits the `__nstate__` marker.

```python
if not isinstance(state, (list, tuple)) or len(state) < 1:
    return None
if len(state) == 1:
    cache_data = ("__nstate__", type_name, list(state))
elif len(state) == 2:
    cache_data = (state[0], state[1])
else:
    cache_data = ("__nstate__", type_name, list(state))
```

**Edit 2 — `_extract_block_tensor_slice` non-sliceable branch (L2338-2385)**

`len(state) > 2` → `len(state) >= 1`; inline the `len(state) == 2` legacy
Mamba/SSM unpacking under a `len(state) == 2` sub-branch and route the
remaining cases (`== 1` and `> 2`) through the `__nstate__` marker. The dead
`elif len(state) >= 2` arm is removed.

**Edit 3 — `_validate_block_cache_data` N-tuple branch (L4468-4483)**

Lower bound `len(elements) < 2` → `len(elements) < 1`; safe access via
`elements[0] if len(elements) >= 1 else None` and `elements[1] if
len(elements) >= 2 else None`. The subsequent `if cache_type in
non_sliceable_types: continue` already skips the seq-len check for
ArraysCache, so a 1-element marker never triggers a spurious shape check.

### 3.2 `tests/test_arrays_cache_single_state.py` (new)

547 lines, 13 tests organised in three classes:

| Class | Purpose | Tests |
|-------|---------|-------|
| `TestValidatedGdnSnapshotLayers` | Pins `_validated_gdn_snapshot_layers` for state_count ∈ {0, 1, 2, 3} and a non-Arrays-family layer | 5 |
| `TestExtractBlockTensorSliceSingleState` | Pins `_extract_block_tensor_slice` non-sliceable branch for last/non-last block + state_count ∈ {1, 2, 3} | 4 |
| `TestSingleStateEndToEnd` | Full round-trip: `BoundarySnapshotSSDStore.save()` → V3 metadata → `load()` → `store_cache()` → `fetch_cache()` → `reconstruct_cache()` → `SizedArraysCache` equality check | 4 |

A minimal duck-typed `_Provider` class is used in place of importing
`omlx.scheduler._BoundarySnapshotProvider` so the e2e tests do not depend
on the unrelated `mlx_vlm.speculative` import. The duck-typed class mirrors
the interface used by `BlockAwarePrefixCache.store_cache()` and
`commit_gdn_checkpoint`.

---

## 4. Test results

### 4.1 New tests

```
tests/test_arrays_cache_single_state.py ..................... 13 PASSED in 3.34s
```

### 4.2 Regression scope

```
756 passed, 8 failed (cache + prefix_cache + boundary_snapshot + ntuple_state + ...)
```

The 8 failures are reproducible on `main` without my changes (verified via
`git stash`):

```
FAILED tests/test_cache_ntuple_state.py::TestPrefixCacheNTupleSubState::test_extract_cache_states_preserves_pooling_cache_state
FAILED tests/test_boundary_snapshot_store.py::TestBoundarySnapshotProvider::test_provider_loads_from_store
FAILED tests/test_boundary_snapshot_store.py::TestBoundarySnapshotProvider::test_provider_uses_pre_extracted_in_memory_snapshot
FAILED tests/test_boundary_snapshot_store.py::TestBoundarySnapshotProvider::test_provider_empty
FAILED tests/test_prefix_cache_cachelist_mixed.py::test_prefill_snapshot_decoupled_from_live_cache
FAILED tests/test_prefix_cache_cachelist_per_member.py::test_decode_snapshot_fallback_filters_kv
FAILED tests/test_paged_ssd_cache.py::TestSchedulerPlumbsBlockSizeToSSDCache::test_block_size_plumbed_to_ssd_manager
FAILED tests/test_paged_ssd_cache.py::TestSchedulerPlumbsBlockSizeToSSDCache::test_larger_block_size_shrinks_ssd_manager
```

All 8 share the same root cause:
`ModuleNotFoundError: No module named 'mlx_vlm.speculative'` from
`omlx/scheduler.py:68 → omlx/speculative/vlm_mtp.py:45`. They are independent
of this fix.

---

## 5. Risks and known limitations

### 5.1 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Empty-tuple state (length 0) might now pass through `len < 1` guard as `len < 1` | None | Guard is still `len(state) < 1`, not `<= 1`; empty tuple is still rejected with the same `return None` |
| `state_count == 1` GDN sidecar stored as fp32 (not quantized) | Low | Matches pre-existing behaviour for any non-`state_index == 1` member; storage cost is unchanged for the 2-state path's recurrent member; opt-in quantization knob remains untouched |
| A model that previously surfaced a placeholder for state_count=1 will now surface real state, slightly growing paged_ssd_cache files | Low | The walk-back truncation point will be found one block further back, increasing the cache hit rate; storage growth is bounded by `recurrent_tensor_size_per_layer × num_layers × num_blocks` |
| The `__nstate__` marker is now used more frequently; reconstruction must keep accepting it | Low | Reconstruction (`ArraysCacheHandler.deserialize_state` → `reconstruct_cache`) is already marker-agnostic — it iterates `state["states"]` of any length |
| `_validate_block_cache_data` accepts `len(elements) == 1` for non-ArraysCache layers too (e.g. a malformed KVCache with a 1-element tuple would pass) | Low | The non-sliceable `continue` skips the seq-len check anyway; the malformed tuple would have already failed to extract in the upstream code paths |

### 5.2 Known limitations (out of scope for POC)

- `_should_quantize_gdn_state` does not quantize the index-0 recurrent member
  of single-state caches. The GDN sidecar still works (fp32 storage) but
  consumes the same bytes as a 2-state path's recurrent member. Quantizing
  this member requires passing `state_count` to `_should_quantize_gdn_state`,
  which is a separate encoder refactor.
- CacheList layers that mix a single-state ArraysCache sub-cache with a
  KVCache sub-cache (e.g. an LFM2.x-style hybrid model) would need a
  sub-cache-level test; the current edit handles the top-level case but the
  nested case is exercised by `test_reconstruct_preserves_variable_length_arrays_state`
  only for `len(state) >= 3`. A future commit can add a `len == 1`
  CacheList sub-cache test if such a model is targeted.
- The diff deliberately does not touch the BoundarySnapshotSSDStore encoder
  or the model patches directory — the bug was entirely in the prefix-cache
  read path. Any encoder-side improvement is left for a follow-up.

---

## 6. Benchmark instructions

The fix is correctness, not performance. To verify the impact at the
benchmark level (prefix-cache hit rate on LFM2.x), use the existing
`benchmarks/` infrastructure:

```bash
# 1. Baseline: load an LFM2.x model, run a multi-turn session, measure
#    first-token latency with and without prefix-cache reuse.
python benchmarks/bench_prefix_cache.py \
    --model-path /path/to/lfm2-1.2b \
    --num-prompts 20 \
    --shared-prefix-tokens 1024 \
    --vary-tail-tokens 128 \
    --output /tmp/lfm2_prefix_cache_bench.json

# 2. Compare against main:
cd /Users/bot/02_dev/omlx
git checkout main
python benchmarks/bench_prefix_cache.py ... --output /tmp/lfm2_main.json
git checkout fix/arrays-cache-single-state-prefix-cache
python benchmarks/bench_prefix_cache.py ... --output /tmp/lfm2_fix.json

# 3. Expected: identical first-token latency when no prefix cache is used;
#    substantially lower first-token latency on the patched branch when
#    the multi-turn session shares the prefix (because the recurrent
#    state is now restored instead of being re-prefilled every turn).
```

Side-channel signal: `BlockAwarePrefixCache.get_stats_dict()["prefix_cache"]`
shows the cache hit/miss counters; `prefix_cache_hits` should rise from 0
to >= 1 on the second turn of an LFM2.x multi-turn session after the fix.

Manual smoke check (no benchmark required):

```bash
python -c "
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
cache = ArraysCache(1)
cache[0] = mx.ones((1, 8, 16))
mx.eval(cache[0])
# ... load the cache into a single-state-extracted boundary snapshot,
# round-trip through BoundarySnapshotSSDStore and BlockAwarePrefixCache
# as in tests/test_arrays_cache_single_state.py::TestSingleStateEndToEnd.
"
```

---

## 7. What was NOT changed (per task constraints)

- No model patches added (LFM2.x is already supported upstream).
- No unrelated code modified.
- No merge performed — branch is local-only, awaiting human review.
- BoundarySnapshotSSDStore encoder untouched (per task: sidecars correctly written).
- `_should_quantize_gdn_state` untouched (encoder contract preserved).
- Reconstruction paths untouched (already marker-agnostic via `ArraysCacheHandler.deserialize_state`).