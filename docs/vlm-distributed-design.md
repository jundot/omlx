# Design: VLM models in distributed cluster inference

Status: investigation / design proposal (no code changes yet)
Scope: why VLM checkpoints are rejected by cluster activation, what actually
breaks under the distributed path, and a tiered plan to lift the restriction.
All claims below are grounded in code with `file:line` anchors, and in a byte
level inspection of the two motivating checkpoints:

- `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` (`model_type=qwen3_5_moe`, non-empty
  `vision_config`, `text_config.model_type=qwen3_5_moe_text`, 40 LM layers,
  `mtp_num_hidden_layers=1`)
- `scottlowry/Qwen3.8-27B-oQ4e-mtp` (`model_type=qwen3_5`, non-empty
  `vision_config`, 64 LM layers, `mtp_num_hidden_layers=1`)

**TL;DR.** The pinned mlx-lm (0.31.3, `pyproject.toml:51`) already does the
hard part: `mlx_lm.models.qwen3_5_moe.Model.sanitize` natively drops the
vision tower and loads these VLM-shaped checkpoints as text models, and
`qwen3_5.Model.shard()` is a single top-level layer loop that passes oMLX's
own AST safety proof (verified by running
`tensor_strategies.native_shard_is_layer_local` against the real source —
returns `True`). Tier (a) — "run the language model distributed, ignore the
vision tower" — therefore reduces to removing oMLX's own gates, fixing a
latent worker validation bug that affects the whole Qwen3.5/3.6 family (text
checkpoints included), and correcting planner byte accounting. Recommended
first increment: tier (a), tensor-parallel only, behind an explicit
"deploy text-only" flag. Effort: M. No upstream changes required.

---

## 1. The exact gate

The rejection is raised by `EnginePool.resolve_cluster_model_id`:

- **`omlx/engine_pool.py:668-674`** — after resolving the deployment's model
  path to a discovered entry:

  ```python
  if entry.engine_type != "batched":
      raise ValueError(
          f"Model '{model_id}' is a {entry.model_type} model. "
          "Distributed cluster inference currently supports text LLM "
          "models only."
      )
  ```

  `entry.engine_type` is `"vlm"` for any checkpoint whose config carries a
  non-empty vision sub-config (`omlx/model_discovery.py:1473`, classification
  via `_has_vision_subconfig`, `omlx/model_discovery.py:549-572` — checks
  `vision_config` / `vit_config` / `mm_vision_tower`, non-empty on purpose so
  stripped text-only quants with a `vision_config: {}` stub are not
  misclassified, issue #2385).

- Reached from the activation path at **`omlx/cluster/routes.py:3091`**
  (`activate_cluster_deployment`, `routes.py:2976`) and the deactivation path
  at `routes.py:3225`. The `ValueError` is mapped to **HTTP 400** at
  `routes.py:3187-3188` (`except (OSError, PlanningError, ValueError)` →
  `HTTPException(status_code=400, ...)`). Note: the failure observed as
  "403/400" is in fact a 400; there is no 403 in this path.

- Asserted by `tests/test_cluster_engine_pool.py:174`.

Two sibling gates enforce the same policy and must change together:

1. **`omlx/engine_pool.py:706-711`** — `register_cluster_model` (the path for
   a rank-0-partial model dir that local discovery didn't index) rejects
   entries already registered with a non-batched engine type: *"stop or
   remove that local model before activating it as a text cluster model."*
2. **`omlx/engine_pool.py:186`** — `_distributed_deployment_for_entry`
   returns `None` for `entry.engine_type != "batched"`, so even a registered
   deployment never dispatches a VLM entry to `DistributedBatchedEngine`
   (dispatch site: `engine_pool.py:2137-2141`).

### What the gate is guarding against

The distributed rank runtime is **the pinned mlx-lm server, not oMLX's engine
stack**: `omlx/cluster/inference_worker.py:836-837` imports
`mlx_lm.server.{ModelProvider, ResponseGenerator, run}`, loads the model via
`ModelProvider.load_default()` wrapped by oMLX's progressive loader
(`inference_worker.py:985-1019`), and serves rank 0 over mlx-lm's private
HTTP API (`inference_worker.py:1130`). The coordinator-side
`DistributedBatchedEngine` (`omlx/engine/distributed.py:31`) keeps only a
tokenizer (`distributed.py:124-154`, `mlx_lm.utils.load_tokenizer`) and
proxies `/v1/chat/completions` / `/v1/completions` as JSON text payloads
(`distributed.py:360-426`, `:568`, `:872`). Nothing in this chain can load an
mlx-vlm model, run a vision tower, or carry image inputs. Before the gate
existed, the failure surfaced late and expensively — the planner comment at
`omlx/cluster/planner.py:697-707` records that trusting the text module's
capabilities "offered pipeline for Qwen3.5/3.6-family VLMs and then failed at
load" after staging tens of GiB. The gate converts that into an upfront
refusal. It is honesty, not a fundamental incapability — and the codebase's
recent history treats *silently dropping vision* as a bug (issues #1261,
#1426 in `omlx/utils/model_loading.py:759-767`, `:909-926`), which is why the
gate refuses rather than degrades.

---

## 2. Root cause: what actually breaks for a VLM under the distributed path

### 2.1 Anatomy of the target checkpoints (measured)

`Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` — 2052 tensors, 20.13 GiB total:

| component | prefix | tensors | bytes |
|---|---|---|---|
| LM backbone | `language_model.model.layers.{0..39}.*`, `language_model.lm_head`, embeddings | 1677 | 18.83 GiB |
| vision tower | `vision_tower.blocks.{0..26}.*`, `vision_tower.{merger,patch_embed,pos_embed}` | 333 | 0.83 GiB |
| MTP drafter | `language_model.mtp.*` (incl. `language_model.mtp.layers.0.*`) | 42 | 0.47 GiB |

Config: top-level `model_type=qwen3_5_moe` **with** `vision_config`;
`num_hidden_layers` (=40) and `mtp_num_hidden_layers` (=1) live only under
`text_config`. The scottlowry Qwen3.8 checkpoint is the dense sibling
(`model_type=qwen3_5`, 64 layers), same layout conventions.

### 2.2 The two loaders and their shapes

- **mlx-vlm** (what `VLMBatchedEngine` uses; pinned commit 78b96eb,
  `pyproject.toml:122`): the wrapper is
  `Model(vision_tower=VisionModel, language_model=LanguageModel)` — see the
  bundled copy at
  `packaging/_build/framework-mlx-base/lib/python3.11/site-packages/mlx_vlm/models/qwen3_5_moe/qwen3_5_moe.py:10-17`.
  It exposes **no** `model.model`, no `pipeline()`, no `shard()`. Every
  distributed sharding hook misses it by construction.

- **mlx-lm 0.31.3** (the rank runtime; inspected from the identical release
  in the MTPLX runtime venv —
  `.../site-packages/mlx_lm/models/qwen3_5.py` — oMLX pins the same release
  as a git commit, `pyproject.toml:51`; re-verify at the exact pin during
  implementation):
  - `qwen3_5_moe.ModelArgs.from_dict` wraps a *flat* config into
    `text_config` (`qwen3_5_moe.py:10-19`), so the `Model` class is **always
    the wrapper** `Model(language_model=TextModel(...))`
    (`qwen3_5.py:367-372`) — for text-only conversions too.
  - `qwen3_5_moe.Model.sanitize` (`qwen3_5_moe.py:23-52`) **drops
    `vision_tower.*` / `model.visual.*` keys and remaps
    `model.language_model.*` → `language_model.model.*`** — i.e. the pinned
    mlx-lm natively loads a VLM-shaped Qwen3.5/3.6 checkpoint as a text
    model. This is exactly what oMLX's single-node `force_lm` /
    `model_type_override` path already relies on
    (`omlx/engine_pool.py:1296-1307`, #2385: "loads only its language
    weights").
  - `Model.shard(group)` exists (`qwen3_5.py:400`) and is one top-level
    `for layer in self.layers:` loop. **Verified**: running
    `native_shard_is_layer_local` (`omlx/cluster/tensor_strategies.py:165-229`)
    over its real source returns
    `(True, "native shard is confined to one layer loop")`.
  - There is **no `pipeline()` anywhere** in `qwen3_5.py` / `qwen3_5_moe.py`
    / `qwen3_next.py` (grep verified). Pipeline parallelism is off the table
    for this family at the current pin, VLM-shaped or not.

### 2.3 Where each distributed component stands today

Walking the activation-to-serving chain for a Qwen3.6 VLM checkpoint, with
the gate hypothetically removed:

1. **Planner capability flags** — already mostly correct:
   - `_supports_pipeline` (`omlx/cluster/planner.py:669-707`) returns `False`
     for any config with a vision sub-config (the guard added by commit
     `eb05f6…` after the staged-then-failed incident). Correct.
   - `_supports_tensor_parallel` (`planner.py:583-608`) imports
     `mlx_lm.models.qwen3_5_moe`, finds the inherited `Model.shard`, runs the
     AST proof → **`True`**. Correct — and this is why the dashboard, since
     commit `784860f9` ("make a 2-node cluster of a single-node-fitting VLM
     plan tensor"), already snaps such models to N-way tensor and
     `_create_cluster_plan` names the N-way-tensor remedy in its 400
     (`omlx/cluster/routes.py:753-763`). **Planning succeeds today;
     activation is the wall.**
   - `_tensor_parallel_divisors` (`planner.py:611-632`) reads
     `num_attention_heads`, `num_key_value_heads`,
     `linear_num_{key,value}_heads` through `_config_int`, whose candidate
     list includes nested `text_config` (`planner.py:566-570`). Correct for
     these checkpoints.

2. **Planner byte accounting — wrong for VLM-shaped checkpoints.**
   `inspect_safetensors_layout` (`planner.py:910-1013`) classifies every
   tensor by `_tensor_layer_index` (`planner.py:508-512`) using
   `_LAYER_PATTERNS = (r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)…",)`
   (`planner.py:29`). Consequences, measured on the Jundot checkpoint:
   - `vision_tower.blocks.{0..26}.*` (324 tensors, 0.83 GiB) **matches the
     regex** and is merged into decoder-layer indices 0–26.
   - `language_model.mtp.layers.0.*` (38 tensors, most of the 0.47 GiB MTP
     block) is merged into decoder layer 0. (The `num_hidden_layers` trim at
     `planner.py:973-983` only removes *extra-index* MTP layers, not
     `mtp.layers.0`-style keys.)
   - Net effect: layer 0 is accounted as **960 MiB vs ~456 MiB** for a real
     mid-stack layer, layers 0–26 are inflated, and total planned weight
     bytes include ~1.3 GiB that the mlx-lm text load will never
     materialize. For TP this only over-reserves (safe direction), but it
     distorts headroom, catalogue `weight_bytes`
     (`omlx/cluster/catalogue.py:352`), and the per-layer proportional KV
     reservation; for any future pipeline split it would misplace stage
     boundaries.

3. **Progressive loading — works for TP, by luck of the pinned sanitize.**
   `progressive_sharded_load` (`omlx/cluster/progressive_loading.py:72-249`)
   calls `mlx_lm.utils.load_model(strict=False, lazy=True)`
   (`progressive_loading.py:122-127`), which builds the text wrapper and
   sanitize-drops vision. Then:
   - `has_pipeline = hasattr(model, "model") and hasattr(model.model, "pipeline")`
     (`progressive_loading.py:128`) — `False` for the wrapper (it has
     `language_model`, never `model`). Consistent with the planner's answer.
   - `has_native_tensor = callable(getattr(model, "shard", None))`
     (`progressive_loading.py:129`) — `True`.
   - The tensor path (`progressive_loading.py:196-237`) materializes fixed
     weights, then `apply_tensor_strategy`
     (`omlx/cluster/tensor_strategies.py:533-554`) takes the native branch:
     `_common_layer_owner` (`tensor_strategies.py:96-119`) **already
     traverses a `language_model` attribute** to find the mutable layer list
     (`Qwen3_5TextModel.layers`), and `_native_layerwise_shard`
     (`tensor_strategies.py:122-163`) exposes one layer at a time to
     `Model.shard`. No change needed here.

4. **Worker stage validation — broken, and not only for VLMs.**
   `_validate_loaded_stage` (`omlx/cluster/inference_worker.py:540-575`)
   starts with `pipeline_model = getattr(model, "model", None)` and raises
   `"loaded model does not expose an MLX pipeline model"` when absent. Since
   the qwen3_5-family wrapper never has `.model`, **pure-TP deployment of
   any Qwen3.5/3.6 checkpoint — text conversions included — dies here
   today.** (TP was proven on `qwen3_next` / `nemotron_h`, whose `Model`
   classes do have `self.model` — `mlx_lm/models/qwen3_next.py:429`.) This
   is a pre-existing bug that tier (a) fixes for free, not a VLM-specific
   cost. `_loaded_stage` (`inference_worker.py:578-598`) has the same
   assumption (fail-soft: reports `None`s in the marker).
   `_validate_measured_weight_bytes` (`inference_worker.py:629-659`) is
   layout-agnostic (`tree_flatten`) and — once the planner stops counting
   vision/MTP bytes — becomes a *tighter*, correct check that the rank holds
   only its text shard.

5. **MTP (`-mtp` checkpoints) — safe, but inactive on ranks.** The worker
   applies pre-load patches with no model settings
   (`inference_worker.py:953`, `maybe_apply_pre_load_patches` at
   `omlx/utils/model_loading.py:352-405`): the mlx-lm MTP patch is installed
   for **sanitize correctness** (stock `TextModel.sanitize` at
   `qwen3_5.py:307-331` applies a +1 norm shift whenever it sees `mtp.*`
   keys — on an already-converted MLX checkpoint that double-shift corrupts
   output; the patch gates the shift correctly,
   `model_loading.py:595-609`), while `mtp_enabled=False` means no MTP head
   is attached and `num_draft_tokens=0` is hard-coded in the rank server
   contract (`inference_worker.py:305`). So `-mtp` weights load correctly
   and are dropped; **MTP speculative decode is inactive under distribution
   for every model family today** — a known limitation to document, not a
   VLM blocker.

6. **Tokenizer / processor.** Both the coordinator
   (`omlx/engine/distributed.py:131-154`) and each rank
   (`progressive_loading.py:117-121`) use `mlx_lm.utils.load_tokenizer`,
   which works on these checkpoints (standard `tokenizer.json`). The HF
   `AutoProcessor` / `preprocessor_config.json` (image preprocessing) is
   never loaded anywhere in the distributed path — fine for text-only, a
   real work item for full VLM.

7. **Image inputs.** The oMLX API keeps image parts only for VLM engines
   (`omlx/api/utils.py:174-176`, `omlx/api/responses_utils.py:198-224`);
   `DistributedBatchedEngine` subclasses `BatchedEngine`, so a distributed
   deployment is treated as text and images are flattened/dropped rather
   than rejected. Tier (a) must make this an explicit, documented behavior
   (reject with a clear 400, or document the drop).

8. **Worker environments.** A rank env installed "for text" may lack
   mlx-vlm; `omlx/cluster/autoconfigure.py:740-804` already derives, from
   the AST of `maybe_apply_pre_load_patches`, exactly which imports a given
   model's patches need per `for_vlm` polarity, so tier (a) (text path,
   `for_vlm=False`) needs nothing new here, and tier (b) has the guard
   machinery ready.

---

## 3. Support tiers

### Tier (a) — text-only distributed serving of a VLM checkpoint

Run only the language model across ranks; vision weights stay on disk. For
the motivating Qwen3.5/3.6/3.8 checkpoints — which many users run purely as
text LLMs — this is the highest-value, lowest-effort win, and it is the same
semantics the single-node `force_lm` / `model_type_override` path already
provides (`engine_pool.py:2106-2108`, `:583-597`).

**Feasibility: high.** The rank loader (pinned mlx-lm) natively loads these
checkpoints text-only and shards them (§2.2). Planning already offers N-way
tensor (§2.3.1). **TP-only** at the current pin (no pipeline for the family);
world size must divide the planner's divisors (measured from the checkpoints'
`text_config`: Jundot 35B — heads 16, kv 2, linear k/v 16/32; scottlowry 27B
— heads 24, kv 4, linear k/v 16/48). A 2-Mac TP plan divides everything on
both; note Jundot's `num_key_value_heads=2` makes the planner's divisor gate
refuse tp=4 even though mlx-lm's `shard()` would handle it by repeating KV
heads (`repeat_kv_layer_inplace`, `qwen3_5.py:409-423`) — a planner/loader
capability mismatch to be aware of, not a tier-(a) blocker.

**Required changes (all oMLX-only):**

1. **Policy + gate** — `omlx/engine_pool.py:668-674`: accept
   `engine_type == "vlm"` entries **when the deployment request carries an
   explicit text-only opt-in** (e.g. `ClusterDeploymentRequest.text_only`
   surfaced as a "Deploy text-only (vision disabled)" checkbox the dashboard
   pre-checks for VLMs). Rationale for explicit-over-automatic: the repo's
   own history treats silent vision-drop as a bug (#1261/#1426); the flag
   keeps the honest refusal for un-flagged requests while making the happy
   path one click. Same relaxation in `register_cluster_model`
   (`engine_pool.py:706-711`) and `_distributed_deployment_for_entry`
   (`engine_pool.py:186`) — the latter can key off the deployment record
   (persist `text_only` on `ClusterDeployment`,
   `omlx/cluster/deployment.py`) rather than the entry type.
2. **Capability check, not family allow-list** — gate the opt-in on "the
   pinned mlx-lm can build and shard this architecture": reuse
   `_supports_tensor_parallel` / `_supports_pipeline`
   (`planner.py:583-608`, `:669-707`) — i.e. a VLM entry is deployable
   text-only iff its layout reports at least one strategy. A Gemma-family
   VLM with no mlx-lm text module correctly stays refused.
3. **Worker validation** — `_validate_loaded_stage` and `_loaded_stage`
   (`inference_worker.py:540-598`): resolve the pipeline model via a
   fallback chain mirroring `_common_layer_owner`
   (`tensor_strategies.py:96-119`) — `model.model`, then
   `model.language_model.model`, then the owner of the mutable `layers`
   list — instead of hard-coding `model.model`. This also unblocks pure-TP
   for *text* Qwen3.5/3.6 conversions (pre-existing bug, §2.3.4).
4. **Planner text-only layout** — in `inspect_safetensors_layout`
   (`planner.py:910-1013`), when the model config has a vision sub-config
   (reuse `_has_vision_subconfig`, `model_discovery.py:549`), exclude
   vision-tower tensors (`vision_tower.`, `model.visual.`,
   `language_model.model.visual.` — reuse/centralize the prefix constants at
   `omlx/utils/model_loading.py:17-21` rather than new literals) from both
   fixed and per-layer accounting, and exclude `*.mtp.*` tensors from layer
   sums (they are dropped or never attached at load). This makes
   `planned_weight_bytes`, catalogue `weight_bytes`, KV proportionality and
   `_validate_measured_weight_bytes` all describe what the rank actually
   holds. Note `complete_model_layout` runs **on the peer** for remote
   holders (`planner.py:1048-1080` snippet), so this change has a
   mixed-version hazard (see Risks).
5. **Coordinator sizing** — cluster registration/admission should use the
   existing `text_only_size` (`model_discovery.py:376`,
   `engine_pool.py:78`) instead of full-checkpoint `estimated_size` for
   text-only VLM deployments (activation's fallback estimate at
   `routes.py:3096-3102` then also follows the corrected plan bytes).
6. **Image-request behavior** — for a text-only distributed deployment,
   reject image content parts with a clear 400 naming the text-only
   deployment (seams: `omlx/api/utils.py:174` extraction /
   `DistributedBatchedEngine._chat_payload`, `distributed.py:360`).
7. **(Optional) staging skip** — skip safetensors files that contain only
   excluded tensors (`omlx/cluster/staging.py` manifest); ~0.8 GiB/rank on
   the 35B checkpoint. Cheap but not required.

**Effort: M** (roughly: gate+policy S, worker validation S, layout filtering
M with tests, plumbing/dashboard S). **Risk: low-medium** — the load path is
already exercised by single-node force_lm; the new surface is validation and
accounting. **Upstream: none.**

### Tier (b) — full VLM distributed (vision tower on rank 0, LM sharded)

What it takes, per component:

- **Rank runtime**: `mlx_lm.server` cannot ingest images. Either (i) extend
  the private rank-0 server with an image-bearing request contract and run
  mlx-vlm's processor + vision tower on rank 0, injecting
  `input_embeddings` into the sharded text model (the seam exists:
  `TextModel.__call__(…, input_embeddings=…)`, `qwen3_5.py:287-293`), or
  (ii) replace the rank runtime for VLM deployments with an oMLX-owned
  worker speaking the same protocol. Both are large.
- **TP complication — deepstack**: Qwen3.6's `vision_config` declares
  `deepstack_visual_indexes` — visual features are injected at *multiple* LM
  layers. Under TP every rank runs every layer, so per-request visual
  embeddings must be broadcast to all ranks, not just rank 0. That means a
  new collective on the request path (`mx.distributed` broadcast of image
  embeddings before prefill), which mlx-lm's server loop has no hook for.
- **Loading**: `progressive_sharded_load` would need an mlx-vlm branch:
  load the wrapper via mlx-vlm (with oMLX's existing sanitize/MTP/nested
  visual patches, `model_loading.py:539-806`), shard
  `model.language_model` (the native `shard()` lives on the *mlx-lm* class;
  mlx-vlm's `LanguageModel` has none — either port it or upstream one), keep
  `vision_tower` only on rank 0 and blank it elsewhere.
- **Planner**: per-rank asymmetric fixed bytes (vision tower resident on
  rank 0 only) — `ModelLayout` (`planner.py:92-157`) has a single
  `fixed_weight_bytes`; needs a rank-0 surcharge concept, plus vision
  activation memory in the prefill guard (`omlx/cluster/prefill_guard.py`).
- **Coordinator**: image parts must survive
  `DistributedBatchedEngine._chat_payload` (`distributed.py:360-426`) into
  the rank protocol; prompt-cache keying must include image hashes
  (`prompt_snapshot_cache.py`).
- **Worker envs**: mlx-vlm becomes a hard rank dependency for these
  deployments — `autoconfigure.py`'s `for_vlm` guard machinery
  (`autoconfigure.py:740-804`, `:1106-1126`) already models this.

**Effort: L–XL. Risk: high** (new request-path collective, per-architecture
sharding of mlx-vlm classes). **Upstream:** ideally an mlx-vlm (or mlx-lm)
`shard()`/pipeline story for VLM language models; feasible oMLX-only but as
vendored patches. Not the first increment.

### Tier (c) — middle grounds

- **(c1) Reject-with-remedy, S:** keep the gate but make its message
  actionable once tier (a) exists ("re-deploy with text-only to run this
  model across Macs"). Included in tier (a)'s policy work.
- **(c2) Hybrid routing, S–M (after (a)):** for a text-only distributed
  deployment of a VLM whose full weights also fit some single node
  (`catalogue.py` already computes `standalone_node_id`,
  `catalogue.py:433-475`), route image-bearing requests to a local/standalone
  `VLMBatchedEngine` instance and text requests to the cluster. Operationally
  attractive (vision keeps working at single-node speed), but two resident
  copies of the LM weights; opt-in at most.
- **(c3) Vision-on-coordinator with embedding forwarding** is *not* a cheap
  middle ground: it needs the same rank-protocol extension and deepstack
  broadcast as tier (b). Listed to pre-empt the suggestion.

---

## 4. Recommended first increment (tier (a)) — implementation checklist

1. Re-verify §2.2 at the exact pin `ab1806e` (`pyproject.toml:51`):
   `qwen3_5_moe.Model.sanitize` vision-drop + prefix remap, `Model.shard`
   AST proof, absence of `pipeline()`. (Findings above were verified against
   the identical 0.31.3 release from a sibling install.)
2. Add `text_only: bool` to `ClusterDeploymentRequest` +
   `ClusterDeployment` (persisted, part of the plan/deployment record);
   dashboard surfaces it for VLM entries (pre-checked, labeled "vision
   disabled").
3. Relax the three gates (`engine_pool.py:668-674`, `:706-711`, `:186`)
   for `text_only` deployments whose layout reports
   `supports_tensor_parallel or supports_pipeline`; keep the current error
   (now with the text-only remedy sentence) otherwise.
4. Fix `_validate_loaded_stage` / `_loaded_stage`
   (`inference_worker.py:540-598`) to locate the pipeline model via the
   `_common_layer_owner`-style fallback chain. Unit tests: wrapper-shaped
   model (attrs `language_model.model.layers`) passes complete-TP-stage
   validation; classic shape unchanged; mismatch still fails closed.
5. Text-only layout accounting in `inspect_safetensors_layout`
   (`planner.py:910-1013`): exclude vision prefixes and `*.mtp.*` from
   layer/fixed sums when `_has_vision_subconfig(config)`; centralize prefix
   constants with `model_loading.py:17-21`. Unit test against a synthetic
   index mirroring the Jundot layout (`vision_tower.blocks.N` collision,
   `language_model.mtp.layers.0` collision); assert layer 0 ≈ layer 30.
6. Use `text_only_size` for cluster admission/registration of text-only VLM
   deployments (`engine_pool.py:1296-1307` already does this for the
   non-distributed case; extend to the deployment estimate at
   `routes.py:3096-3102`).
7. Reject image content parts on text-only distributed deployments with a
   clear 400 (`distributed.py:360` / API extraction seam).
8. Tests, end to end:
   - `resolve_cluster_model_id` accepts a VLM entry iff `text_only` and
     shardable; message otherwise names the remedy (update
     `tests/test_cluster_engine_pool.py:174`).
   - 2-rank TP activation smoke of the Jundot checkpoint (pattern:
     `omlx/cluster/pipeline_smoke_worker.py`): rank load passes
     `_validate_loaded_stage` **and** `_validate_measured_weight_bytes`
     (which becomes tighter once vision/MTP bytes leave the approved plan —
     measured expectation: ~18.8 GiB text weights, not 20.1), canary
     generates, output sane vs single-node force_lm output.
   - Image request against the deployment → 400 naming text-only.
9. Docs: note that `-mtp` heads are inactive under distribution (all
   families, `inference_worker.py:305` + `model_settings=None` at `:953`)
   and that vision is disabled by construction for text-only deployments.

### Risks / mitigations

- **Mixed-version clusters**: layout filtering changes
  `complete_model_layout`, which peers execute remotely
  (`planner.py:1048-1080`). A new coordinator + old peer would compute
  different layouts for the same checkpoint. Mitigation: layouts already
  travel as JSON with a `source` field; add the accounting variant to the
  plan hash inputs, or gate activation on both sides reporting the same
  `total_weight_bytes`.
- **Family generality**: tier (a) is proven for qwen3_5/qwen3_5_moe because
  mlx-lm ships a text-capable, shardable class. The capability probe (item
  3) keeps other VLM families fail-closed instead of failing at load.
- **Sanitize drift on pin bumps**: the whole tier rests on mlx-lm's
  vision-dropping sanitize; add a `KEEP:`-style comment (as
  `omlx/patches/mlx_lm_sharded_load.py:16-18` does) tying the gate
  relaxation to re-verification on pin bumps.

---

## 5. Open questions / spikes

1. **Spike (small): 2-rank TP numerics for qwen3_5_moe.** The native
   `shard()` passes the AST proof, but no evidence in-repo that this family
   has run TP end-to-end (validation would have crashed first, §2.3.4).
   Run the strategy benchmark / smoke worker on two Macs before committing
   to the dashboard default.
2. **MTP under distribution**: is attaching the MTP head on ranks
   (`model_settings` plumbed into the worker, `set_mtp_active(True)`,
   mlx-lm server draft loop) worth pursuing separately? Orthogonal to VLM
   but shares the `-mtp` checkpoints' value proposition.
3. **Prompt-cache interaction**: rank-local mlx-lm prompt cache keys on
   token IDs only; confirm no image-token placeholders can reach the rank
   in text-only mode once item 7 (reject images) lands.
4. **KV/state reservation for linear-attention layers (pre-existing, not
   VLM-specific).** ~3 of every 4 qwen3_5-family layers are linear-attention
   (gated delta), whose per-request state is an `ArraysCache` of
   conv/SSM state (`TextModel.make_cache`, `qwen3_5.py:305`), not a
   token-proportional K/V tensor — while the planner's
   `_kv_bytes_per_token_per_layer` (`planner.py:761-779`) applies the
   uniform `num_kv_heads * head_dim * 2 * dtype` formula to every layer.
   Whether that over/under-reserves, and whether the SSM state divides by
   the TP degree the way KV heads do, is unverified — it affects text
   Qwen3.5/3.6 TP deployments identically; verify during the 2-rank smoke.
5. **Should `model_type_override`→`llm` entries deploy without the flag?**
   A user who already overrode the model to LLM has opted in; their entry's
   `engine_type` is `"batched"` (`engine_pool.py:583-597`) and passes the
   gate today — decide whether that back door stays, and make the dashboard
   present both paths consistently.
6. **Tier (b) protocol design** (only if/when pursued): where does the
   image→embedding computation live (rank 0 vs coordinator), and how do
   deepstack embeddings reach all TP ranks — new pre-prefill collective vs
   embedding-in-request broadcast. Needs a dedicated design doc.

## Appendix: verification notes

- Gate + message: `omlx/engine_pool.py:668-674`; HTTP mapping
  `omlx/cluster/routes.py:3187-3188` (400).
- AST proof run: `native_shard_is_layer_local(qwen3_5.Model.shard source)` →
  `(True, "native shard is confined to one layer loop")` (executed against
  mlx-lm 0.31.3 site-packages copy, 2026-08-19).
- Checkpoint measurements from safetensors headers of
  `~/.omlx/models/Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` (figures in §2.1).
