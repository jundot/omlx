# DeepSeek V4 monkey patch (mlx-lm PR 1192)

This directory ports https://github.com/ml-explore/mlx-lm/pull/1192 onto the
pinned mlx-lm v0.31.3 (`ed1fca4`) without modifying the upstream package.

## Pinned PR HEAD

- PR: https://github.com/ml-explore/mlx-lm/pull/1192
- HEAD SHA: `5c10538136b9038b9626c134612b08afc18d697a` (2026-05-01)
- Source repo: `Blaizzy/mlx-lm` branch `pc/add-deepseekv4flash-model`

## Files

| File | Source | Notes |
|---|---|---|
| `deepseek_v4_model.py` | PR 1192 `mlx_lm/models/deepseek_v4.py` | 1:1 copy — do not edit |
| `hyper_connection.py` | PR 1192 `mlx_lm/models/hyper_connection.py` | 1:1 copy — do not edit |
| `cache_extras.py` | PR 1192 `mlx_lm/models/cache.py` lines 903-1447 | PoolingCache + BatchPoolingCache, 1:1 |
| `utils_patch.py` | adapted from PR 1192 `mlx_lm/utils.py` | replacement `load_model` + `_load_safetensors` |
| `generate_patch.py` | adapted from PR 1192 `mlx_lm/generate.py` | replacement `_make_cache` |
| `cache_handlers.py` | omlx-side, new | PoolingCache / BatchPoolingCache handlers for omlx CacheTypeRegistry |
| `__init__.py` | omlx-side, new | `apply_deepseek_v4_patch()` orchestration |

## Activation

The patch is gated on `model_type == "deepseek_v4"` in the model's
`config.json`, dispatched from:

- `omlx/utils/model_loading.py::load_text_model`
- `omlx/engine/batched.py::BatchedEngine.start` (before `mlx_lm.load`)

Other models pay zero cost — the patch never runs.

## Why monkey-patched, not vendored

omlx pins `mlx-lm` to a git commit; we never modify that package. Adding
DeepSeek V4 by editing `mlx_lm/models/` would diverge from upstream and
break re-pinning. The patch instead injects the new modules into
`sys.modules` and replaces a small number of `mlx_lm.utils` /
`mlx_lm.generate` entries.

## Removing this patch

When mlx-lm merges PR 1192 upstream, simply:

1. `rm -rf omlx/omlx/patches/deepseek_v4/`
2. Remove the `_maybe_apply_deepseek_v4_patch` calls from
   `omlx/utils/model_loading.py` and `omlx/engine/batched.py`.
3. Remove the `PoolingCache` / `BatchPoolingCache` lines from
   `omlx/cache/type_handlers.py::CacheType` and
   `omlx/cache/type_registry.py::_class_name_map`.
4. Repin `mlx-lm` in `pyproject.toml` to the commit that includes PR 1192.

## Synchronizing with PR 1192 updates

If PR 1192 adds new commits before merge, refresh the 1:1 sources:

```bash
SHA=<new_head_sha>
curl -sSL "https://raw.githubusercontent.com/Blaizzy/mlx-lm/$SHA/mlx_lm/models/deepseek_v4.py" > deepseek_v4_model.py
curl -sSL "https://raw.githubusercontent.com/Blaizzy/mlx-lm/$SHA/mlx_lm/models/hyper_connection.py" > hyper_connection.py
# cache_extras.py: extract lines 903-1447 of the new cache.py
# Update PR_HEAD_SHA in __init__.py.
```

Re-run the unit tests in `tests/test_deepseek_v4_patch.py` and the slow
e2e test against `mlx-community/DeepSeek-V4-Flash-mxfp8`.

## Known PR 1192 caveats (none blocking)

- `_load_safetensors` rewrites the safetensors header in-place to advertise
  `F8_E8M0` bytes as `U8`. mlx-community's published V4 weights are already
  converted to standard dtypes, so this fallback is mostly dead code in
  practice — but it's there for completeness.
- prefix-cache dedup is disabled for `PoolingCache` (the pool is compressed
  in fixed-ratio windows, partial slicing is meaningless). SSD eviction
  still works via the registered handlers.
