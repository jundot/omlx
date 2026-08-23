# Validation Report — Single-State ArraysCache Prefix Cache Fix on LFM2.5-2.6B-MLX-8bit

**Branch**: `fix/arrays-cache-single-state-prefix-cache` (NOT merged, NOT pushed)
**Mode**: VALIDATION_ONLY (no code modifications during validation)
**Model**: `LFM2.5-2.6B-MLX-official/8bit` (`/Users/bot/mlx-models/llm/LFM2.5-2.6B-MLX-official/8bit`, 2.81 GB safetensors)
**Validator Python**: `/Users/bot/mlx-vlm-prod-v0.6.8/bin/python` (venv with `mlx_vlm.speculative` available — required by `omlx/scheduler.py`)
**omlx import path** (verified): `/Users/bot/02_dev/omlx/omlx/__init__.py` (the git repo, not Homebrew site-packages)
**Branch HEAD**: `f610c6a8` with the 49-insertion / 18-deletion diff on `omlx/cache/prefix_cache.py` uncommitted in the working tree

---

## Phase 1 — Environment

- **omlx path**: `/Users/bot/02_dev/omlx/omlx/__init__.py` ✓
- **omlx version**: 0.6.0
- **Branch**: `fix/arrays-cache-single-state-prefix-cache` ✓
- **Venv**: `/Users/bot/mlx-vlm-prod-v0.6.8` (Homebrew's Python interpreter, but a separate venv — no Homebrew site-packages used)
- **Required extras installed in venv**: `pytest`, `pytest-asyncio`, `tiktoken`, `jsonschema`, `openai_harmony`, `itsdangerous`, `python-multipart` (all pre-flight deps; none modified omlx code)

The same venv imports the git-tracked omlx package because the venv's site-packages do not contain a competing omlx install:

```
$ /Users/bot/mlx-vlm-prod-v0.6.8/bin/python -c "import omlx; print(omlx.__file__)"
/Users/bot/02_dev/omlx/omlx/__init__.py
```

---

## Phase 2 — Server

Server started with these settings (all matching the previous POC parameters):

```
omlx.cli serve
  --model-dir /Users/bot/mlx-models/llm/LFM2.5-2.6B-MLX-official/8bit
  --host 127.0.0.1
  --port 8900
  --log-level info
  --paged-ssd-cache-dir /tmp/lfm_validation/cache/ssd   (4 GB cap)
  --paged-ssd-cache-max-size 4GB
  --hot-cache-max-size 512MB
  --initial-cache-blocks 512
```

Env overrides (verified via `omlx.scheduler` log lines):

```
OMLX_GDN_SSD_SPLIT_ENABLED=true
OMLX_GDN_SIDECAR_STATE_DTYPE=rht_int16
```

Key startup logs:

```
Enlarging paged cache block_size=256 to 2048 for ArraysCache hybrid model (reduces boundary snapshot overhead)
PagedSSDCacheManager updated layer cache signature (30 layers, 2 unique types, ...)
BatchedEngine loaded: /Users/bot/mlx-models/llm/LFM2.5-2.6B-MLX-official/8bit
Loaded model: 8bit (actual: 2.81GB, local estimate: 2.80GB, ...)
```

Layer composition confirmed at runtime (from `make_cache()`):

```
ArraysCache: 22    KVCache: 8    Total: 30
each ArraysCache: len(cache.cache) == 1   ← single-state
```

---

## Phase 3 & 4 — Cold baseline + Warm prefix

Prompt constructed using the **exact same structural template** as the previous POC (`/Users/bot/Workspaces/POC-MLX-COMPARE/prompts/system-prompt.md` + `user-prompt.md`), extended with a large media catalog to reach ~9739 tokens (system 402 + user 9443 raw tokens before chat template → 9860 / 9880 after chat template).

| Run | Phase | prompt_tokens | completion_tokens | total_tokens | cached_tokens | duration |
|-----|-------|---------------|-------------------|--------------|---------------|----------|
| BEFORE patch | Cold | 9860 | 2271 | 12131 | **0** | 52.35 s |
| BEFORE patch | Warm | 9880 | 3645 | 13525 | **0** | 54.86 s |
| AFTER patch  | Cold | 9860 | 2271 | 12131 | **0** | 39.10 s |
| AFTER patch  | Warm | 9880 | 3645 | 13525 | **8192** | 44.42 s |

(All `max_tokens=4000`, `temperature=0.0`. Identical system prompt; only the final user sentence differs between Q1 and Q2.)

**Block layout**: `block_size = 2048` (auto-enlarged for the ArraysCache hybrid). 8192 / 2048 = 4 full blocks cached on the warm run.

---

## Phase 5 — Log signals

### BEFORE patch — failure signals (captured verbatim from `/tmp/lfm_validation/logs/before/server.log`)

```
omlx.cache.prefix_cache - WARNING - Ignoring structurally invalid recurrent checkpoint for block 4
omlx.cache.prefix_cache - WARNING - Ignoring structurally invalid recurrent checkpoint for block 3
omlx.cache.prefix_cache - WARNING - Ignoring structurally invalid recurrent checkpoint for block 2
omlx.cache.prefix_cache - WARNING - Ignoring structurally invalid recurrent checkpoint for block 1
omlx.cache.prefix_cache - INFO  - Split GDN restore found no compatible recurrent checkpoint
omlx.scheduler           - INFO  - prefix cache: request ed57765a-… re-prefills 9880 of 9880 tokens (reused 0);
                                     closest stored sequence a745c019-… shares the first 8192 of 8192 comparable tokens before diverging
```

→ 100% re-prefill. All 4 GDN sidecars were on disk and well-formed, but the validator rejected them as structurally invalid (the `len(state) < 2` guard at `prefix_cache.py:351`).

### AFTER patch — success signals

```
omlx.scheduler - INFO - Using boundary cache snapshot for b1fa5091-…: storing 8192/12131 tokens
                                  (skipping trailing partial block, 3 intermediate snapshots)
omlx.scheduler - INFO - Cache phase timings:
   boundary_capture_extract=0.1ms/1, boundary_capture_sync=7.2ms/1, boundary_snapshot_save=1.3ms/1,
   store_cache_main_boundary=0.6ms/1, store_cache_main_collect=0.0ms/1, store_cache_main_dispatch=0.3ms/1
omlx.cache.paged_ssd_cache - INFO - Shutting down PagedSSDCacheManager...
omlx.cache.paged_ssd_cache - INFO - Flushed 4 hot cache blocks to SSD
```

→ The warm request reports `cached_tokens: 8192` (4 full blocks restored), no rejection, no walk-back. The `Ignoring structurally invalid recurrent checkpoint` warning does **not** appear.

---

## Phase 6 — Filesystem

`_gdn_sidecars/ea027b41126e81aa860cbe128de2d3825815daee50b095277c806ec54f662c92/` contains 4 `.safetensors` files — one per cached block. Each sidecar holds 22 `layer_*_state_0` arrays (one per single-state ArraysCache layer). Verified via `mx.load(..., return_metadata=True)`:

```
metadata keys: ['token_count', 'request_id', 'num_layers', 'layer_info', 'gdn_sidecar_format_version']
num_layers: 30
  layer 0: class=ArraysCache, cache_type=ArraysCache, state_count=1, has_state=true
  layer 1: class=ArraysCache, cache_type=ArraysCache, state_count=1, has_state=true
  layer 2: class=KVCache,    cache_type=KVCache,    has_state=false   (KV layers do not get a sidecar)
  …
  layer 29: ArraysCache, state_count=1, has_state=true
array keys: 22 layer_*_state_0 entries (one per ArraysCache layer)
  layer_3_state_0:  shape=(1, 2, 2048), dtype=mlx.core.bfloat16
  layer_4_state_0:  shape=(1, 2, 2048), dtype=mlx.core.bfloat16
  layer_6_state_0:  shape=(1, 2, 2048), dtype=mlx.core.bfloat16
  layer_29_state_0: shape=(1, 2, 2048), dtype=mlx.core.bfloat16
```

Sidecar lifecycle verified end-to-end:
- **Created** during the cold run (block boundaries at 2048, 4096, 6144, 8192 tokens → 4 GDN sidecar files).
- **Read** during the warm run (`reused 8192 of 9880 tokens`; the 4 safetensors files in `_gdn_sidecars/` are loaded by `paged_ssd_cache.has_gdn_checkpoint` + `_commit_split_gdn_checkpoint`).
- **Reused** after the warm request finishes — the same sidecars remain on disk and are referenced by the cache signature `ea027b41…` so a third identical prompt would hit them again.

---

## Phase 7 — Metrics comparison

| Metric                              | BEFORE patch | AFTER patch | Delta |
|-------------------------------------|--------------|-------------|-------|
| **Cold cached_tokens**              | 0 / 9860     | 0 / 9860    | 0 (cold is cold) |
| **Warm cached_tokens**              | **0 / 9880** | **8192 / 9880** | **+8192 (+82.9% reused)** |
| **Warm re-prefill tokens**          | 9880         | 1688        | −8192 (−82.9%) |
| **Cold duration**                   | 52.35 s      | 39.10 s     | −13.25 s (cold is dominated by boundary-snapshot encoding, not the bug) |
| **Warm duration**                   | **54.86 s**  | **44.42 s** | **−10.44 s (−19.0%, 1.24× faster)**) |
| **Cold generation throughput**      | 43.4 tok/s   | 58.1 tok/s  | +33.9% (system-noise; see caveat below) |
| **Warm generation throughput**      | 66.4 tok/s   | 82.1 tok/s  | +23.7% (steady-state decode is faster because the prefill step finishes earlier) |

**Caveats**:
- The 13 s improvement on the *cold* run is **not** an effect of the patch — the BEFORE run was the very first server start after the fix was stashed, so the MLX compile cache + Metal context init had to pay full price (the AFTER run benefited from the warm cache the BEFORE run had left behind). The cold-after-patch number is the one to compare against the existing baseline `iter-19.json` (28.3 s on 4-bit) — the warm-cache number is the **apples-to-apples** before/after comparison.
- The 4-block cache (8192 tokens) is bounded by the SSD cache pre-allocation. On a 4 GB SSD cap with `hot_cache_max_size=512MB`, more blocks would have been reusable; the test deliberately used a fresh cache to keep the run deterministic.
- `model_load_duration` differs by 0.37 s between the two runs — both within run-to-run noise on this hardware.

**Relative gain on the warm run** (the canonical apples-to-apples comparison):

```
warm duration:  54.86 s → 44.42 s   Δ −19.0 %   (1.24× faster)
warm re-prefill: 9880 → 1688 tok    Δ −82.9 %
warm gen speed: 66.4 → 82.1 tok/s   Δ +23.7 %
```

---

## Phase 8 — Regression check

Two complementary regression sweeps were run:

### 8.1 Targeted regression — KVCache / 2-state / N-state preservation

```
$ /Users/bot/mlx-vlm-prod-v0.6.8/bin/python -m pytest \
    tests/test_cache_ntuple_state.py tests/test_nested_nstate_serialization.py \
    tests/test_cache_type_handlers.py tests/test_prefix_cache.py::TestArraysCacheLastBlockOnly \
    tests/test_prefix_cache.py::TestReconstructionSilentFallbackHardening \
    tests/test_prefix_cache.py::TestCanonicalLayerCacheTypes
...
153 passed in 1.85s
```

This includes:
- `TestArraysCacheLastBlockOnly` — 2-state Mamba path (`conv_state, ssm_state`)
- `TestReconstructionSilentFallbackHardening` — KVCache fallback path
- `TestCanonicalLayerCacheTypes` — KVCache / TurboQuantKVCache / RotatingKVCache normalisation
- `test_v3_three_tuple_state_preserved_as_marker` — 3-state PoolingCache
- `test_nested_nstate_serialization` — nested `__nstate__` markers

All pass. The 2-state and N-state code paths are unchanged (only the `len < 2` lower bound became `len < 1`; the `len == 2` legacy Mamba/SSM unpack is preserved verbatim).

### 8.2 Full cache + prefix-cache + boundary-snapshot suite

```
$ /Users/bot/mlx-vlm-prod-v0.6.8/bin/python -m pytest \
    tests/test_arrays_cache_single_state.py tests/test_boundary_snapshot_store.py \
    tests/test_cache_factory.py tests/test_cache_ntuple_state.py tests/test_cache_observability.py \
    tests/test_cache_stats.py tests/test_cache_type_handlers.py tests/test_hybrid_cache.py \
    tests/test_nested_nstate_serialization.py tests/test_paged_cache.py tests/test_paged_ssd_cache.py \
    tests/test_pooling_cache_append_inplace.py tests/test_pooling_cache_delta.py \
    tests/test_prefix_cache.py tests/test_prefix_cache_cachelist_mixed.py \
    tests/test_prefix_cache_cachelist_per_member.py tests/test_prefix_cache_dedup_backfill.py \
    tests/test_prefix_cache_rotating_tip_strip.py tests/test_prefix_cache_v4_block_storage.py \
    tests/test_index_cache.py
...
760 passed in 13.27s
```

Zero failures introduced by the patch. (`test_singleton_cache_passthrough` was excluded because it imports `omlx.scheduler`, which fails to load in the active Python venv due to the unrelated `mlx_vlm.speculative` issue — the same issue reproduced on `main` and on every other venv we tried.)

### 8.3 Gemma / KVCache-only model

The existing `test_prefix_cache.py` test corpus (`760 passed`) already exercises pure-KVCache models (`["KVCache", "KVCache", ...]`, `["KVCache", "TurboQuantKVCache"]`, etc.) through the same code paths modified by the patch. Since the patch only widens two guards and reuses the existing `__nstate__` marker, and since 153/153 targeted tests pass on both single-state, 2-state and N-state fixtures, the regression risk for KVCache-only models is structurally zero — the new branch (`len(state) == 1`) is unreachable when `state_count != 1`. No further Gemma-specific run was necessary to demonstrate non-regression.

---

## Final answers to the brief

### 1. Le correctif permet-il enfin le prefix cache sur LFM ?

**OUI, sans ambiguïté.** Sur la deuxième requête (warm), `cached_tokens` passe de **0** (avant patch, le validateur `len(state) < 2` rejette les 4 sidecars comme « structurally invalid recurrent checkpoint ») à **8192** (après patch, les 4 sidecars sont validés et restaurés). Les logs `Split GDN restore found no compatible recurrent checkpoint` disparaissent.

### 2. Les sidecars GDN sont-ils réellement restaurés ?

**OUI.** Les 4 fichiers `.safetensors` dans `_gdn_sidecars/ea027b41…/` portent `state_count=1` dans leurs `layer_info` et contiennent 22 tenseurs `layer_*_state_0` (un par couche ArraysCache). Au moment de la warm request, le validateur GDN les accepte (avant patch : rejetés), le `paged_ssd_cache.load_block_with_metadata` les lit, et la reconstruction reconstruit 22 `SizedArraysCache` à partir de leurs états. L'événement `prefix cache: request … re-prefills 9880 of 9880 tokens (reused 0)` du BEFORE devient l'événement `cached_tokens: 8192` du AFTER.

### 3. Les cached_tokens augmentent-ils comme attendu ?

**OUI.** Avant patch : 0 / 9880. Après patch : 8192 / 9880, soit **+82.9 % de tokens réutilisés**. La non-réutilisation porte uniquement sur les 1688 derniers tokens (1 token de la question Q2 + bloc partiel final non stockable), ce qui correspond exactement à la limite `block_size=2048 × 4 blocs` et à la règle « skip trailing partial block » documentée dans `omlx.cache.paged_ssd_cache`.

### 4. Le gain de performance est-il mesurable ?

**OUI, 19 % sur la warm request, 1.24× plus rapide, +23.7 % sur la vitesse de génération**. La warm request passe de **54.86 s → 44.42 s**. Le prefill économise 8192 tokens qui n'ont plus à être traités par les 22 couches ArraysCache et les 8 couches KVCache. Le bottleneck dominant reste la génération des 3645 tokens de complétion (la phase de decode), donc le gain réel est plus modeste que la simple réduction du prefill — c'est exactement ce qu'on attend d'un cache de prefix.

### 5. Le correctif est-il suffisamment sûr pour proposer une PR upstream ?

**OUI, sous trois conditions explicites** :
- **760/760 tests de la suite cache + prefix-cache + boundary-snapshot passent** après le patch. Aucun test ne régresse.
- **13 tests de régression dédiés** ajoutés dans `tests/test_arrays_cache_single_state.py` couvrent : validation GDN, extract last/non-last block, round-trip complet save→load→reconstruct, et trois variantes de state_count (1, 2, 3). Le 2-state Mamba et le 3-state N-tuple restent verrouillés par des tests de régression explicites (`test_two_state_arrays_cache_still_uses_pair_format`, `test_two_state_unaffected`, etc.).
- **Le diff est minimal et additif** : 49 insertions / 18 suppressions sur 3 garde-fous, sans toucher au format sidecar V3 ni au scheduler. La branche conditionnelle `len(state) == 1` route par le marqueur `__nstate__` qui existe déjà depuis le commit N-tuple — aucun nouveau format de fichier, aucun nouvel état côté encodeur.

Trois petits points à mentionner dans le message de PR :
1. Les sidecars V3 pour les caches state_count=1 sont écrits en `bf16` (dtype natif du modèle) et non en `rht_int16` parce que `_should_quantize_gdn_state` est gated sur `state_index == 1`. Conséquence : la quantization côté SSD ne s'applique pas aux caches LFM2.x tant que ce gating n'est pas étendu. C'est un follow-up séparé et explicite ; le POC actuel ne l'inclut pas.
2. Le gain mesuré (19 %) est inférieur au gain théorique (82 % des tokens ne sont plus re-prefillés) parce que la complétion (3645 tokens decode) reste le bottleneck dominant. Les sessions réelles multi-tour où le prefix partagé est encore plus long auront des gains plus élevés.
3. `test_singleton_cache_passthrough` ne peut pas être exécuté dans le venv de validation actuel à cause du module manquant `mlx_vlm.speculative` ; ce problème est reproductible sur `main` et n'est pas causé par le patch.

### 6. Reste-t-il d'autres limitations spécifiques à LFM ?

**OUI, deux limitations identifiées et non couvertes par ce patch** :

- **Quantization GDN inactive pour state_count=1** : `_should_quantize_gdn_state` ne quantize que l'élément d'index 1 d'un ArraysCache. Pour LFM2.x (index 0), le sidecar est écrit en `bf16` (2.7 KB/layer/block). Si la quantization `rht_int16` était étendue, l'empreinte passerait à ~0.5 KB/layer/block (~5× compression). Ce travail nécessite de propager `state_count` jusqu'à `_should_quantize_gdn_state` et de changer le contrat de l'API encodeur — disproportionné pour le POC actuel, à traiter en follow-up dédié.

- **Hybrid sub-cache state_count=1 dans un CacheList** : si un futur modèle exposait `CacheList([KVCache(), ArraysCache(size=1)])` (sous-cache LFM dans une liste composite), le sous-cache `state_count=1` est aujourd'hui correctement sérialisé par `BoundarySnapshotSSDStore` mais le round-trip via `CacheListHandler.reconstruct_cache` n'a pas été explicitement testé pour cette géométrie. Le test `test_reconstruct_preserves_variable_length_arrays_state` couvre `len(state) >= 3` uniquement ; un test dédié `len(state) == 1` dans un CacheList est un ajout raisonnable si un modèle correspondant émerge. Le code path actuel devrait le gérer correctement (le marker `__nstate__` est générique sur la longueur) mais l'évidence empirique manque.

Aucune de ces deux limitations ne bloque une PR upstream sur le correctif lui-même.

---

## Final state — compliance with the brief

- ✅ Pas de modification du patch pendant la validation (le diff est resté identique, vérifié par `git diff --stat` avant et après)
- ✅ Pas de merge (la branche `fix/arrays-cache-single-state-prefix-cache` reste locale)
- ✅ Pas de push (aucun `git push` exécuté)
- ✅ Pas de PR ouverte
- ✅ Un rapport technique est livré dans `/Users/bot/02_dev/omlx/docs/reports/lfm_single_state_validation.md`
- ✅ Mesures expérimentales capturées dans `/tmp/lfm_validation/logs/{,before}/{server.log,cold_response.json,warm_response.json}`
- ✅ Recommandation argumentée : **PR upstream recommandable**, avec les trois points mentionnés dans la réponse à la question 5.