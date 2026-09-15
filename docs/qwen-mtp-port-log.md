# qwen-mtp-challenge → oMLX port: correctness ledger

Per-submission record for the `perf/mlx-fast-27b` port of the
[Layr-Labs/qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)
Qwen 3.8 27B native-MTP Swift optimizations (`Sources/MLXFastModel/`,
`Vendor/mlx-swift-lm/`, `Vendor/mlx-swift/`) into oMLX's Python MLX Lightning
MTP path (`omlx/patches/mlx_lm_mtp/`).

Each commit on this branch that ports a challenge submission is labeled with
that submission's UUID (matching the organizer's own `Validate submission
<uuid>` commit labels) and MUST update this doc with the submission's
token/bit-exactness status. The port bar is **token-exact parity of the MTP
decode against the serial trajectory** (the challenge's token-fidelity gate:
"every emitted token must equal the token serial decode would have produced")
plus, where measurable, a win before anything ships default-on. Anything that
is NOT token-exact, or that carries a token-correctness risk, is recorded in
the Concern register with the challenge commit that introduced it.

## Pinned artifacts (verified, see `tests/test_qwen38_mtp_assets.py`)

| Artifact | Repo / revision | Records | Digest |
|---|---|---|---|
| backbone | `EigenLabs/Qwen3.8-27B-4bit` @ `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` | 10 | ✅ all matched (fixtures/reference_qwen3_8_27b_4bit.sha256) |
| head | `EigenLabs/Qwen3.8-27B-MTP-bf16` @ `26a328e070875b0314d652a039b6b59902690f03` | 4 | ✅ all matched (fixtures/qwen3_8_27b_mtp_head.sha256) |
| upstream | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | — | present in HF cache (reference lineage) |

The merged model (backbone + head tensors under the `mtp.` prefix, the Swift
loader's merge contract) loads through omlx's Lightning MTP machinery
(`apply_mlx_lm_mtp_patch` + `set_mtp_active`/`set_mtp_depth`) with the head
attached; the token-exactness gate is
`tests/test_qwen38_mtp_token_exact.py` (skipped when the merged checkpoint is
absent).

## Submission registry (Qwen 3.8 MTP era, chronological)

| # | Date | Submission | Challenge commit | Port | Token-exact | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-08-14 | `95611e60-4f77-473d-9a06-5727fa97d81f` | `b72fa41` — draftPolicy constant 1 → 0 (pure serial control) | — | ✅ (serial == serial) | no port: depth-0 = serial control, omlx MTP-off path |
| 2 | 2026-08-15 | `0f835e36-0744-427e-9d0b-d854bac51e1a` | `97921a3` — nConfirmed:1 K=1 eager checkpoint + `restoreAfterSingleDraftReject` (skip repair forward on reject) | — | ✅ | already covered: omlx chain verify runs `n_confirmed=1` + `mtp_partial_rollback` restores the eager checkpoint (`qwen35_model.mtp_partial_rollback`) |
| 3 | 2026-08-15 | `088f763b-163b-4a5f-a942-6b778ca1fd3e` | `c9e32f7` — packed Q/K/V concat on N: one affine-4 GEMM for Q+gate/K/V | `PORT-3` | ✅ token-exact (96-tok check; rows independent so concat-on-N is bit-exact) | packed, default ON (`OMLX_QWEN_PACKED_QKV=0` ablation); −10% decode (93 vs 103 ms/tok, 128-tok serial) |
| 4 | 2026-08-15 | `75c8d33b-450a-4122-a916-b637745f2cf1` | `ec0ba7d` — seed-tail vocab projection (last row only) + fused K=1 verify with lazy post-primary boundary | — | ✅ | no port: both halves already present. Fused verify = `GatedDeltaNet` n_confirmed body (one unsplit chunk + pre-forward stash, `mtp_partial_rollback` replay) — the K=1 design generalized to all widths. Seed-tail projection = mlx-lm batch prefill never evaluates the full `[S, V]` lm_head slab (dead lazy graph; only cache state is eval'd) |
| 5 | 2026-08-15 | `e5051ba4-dde7-4829-90b4-7ff321fc0e25` | `ab62cea` — two-stage exact top-2 reduction kernel (replaces per-row argPartition+gather) | — | ✅ | no port: top-2 ledger is trusted-parent machinery absent in omlx; omlx uses in-graph argmax with identical ordering (value-desc, id-asc) |
| 6 | 2026-08-15 | `b71bb354-4aa4-4ea5-b902-b67936205182` | `5c2441b` — top-2 reducer's first id IS the row argmax (drop separate argMax launch) | — | ✅ | already covered: omlx computes verify targets in-graph (`mx.argmax` per row in one sync), no separate host argMax pass |
| 7 | 2026-08-15 | `84583d41-0417-437f-8482-5663cbddf51a` | `b6c7251` — `qmv_fast_crossrow_affine4_g64` multi-row (M=2..5) QMV kernel | — | ✅ | no port: C++ Metal kernel in vendored mlx-swift; Python mlx ships prebuilt kernels. M=2..5 verify widths verified empirically: depth-4 (M=5) token-exact vs serial (96-tok check) — no multi-row decode arithmetic drift in the Python path |
| 8 | 2026-08-15 | `7b33621c-1d1a-4540-834d-d89c7f402b0f` | `61936f2` — compact proposal vocabulary: draft lm_head narrowed to 98,304 prefix + 26 control rows (~98k of 248k) with device-side id remap | `PORT-8` | ✅ token-exact (96-tok check) | compact draft lm_head, default ON (`OMLX_QWEN_COMPACT_DRAFT=0` ablation); −2.2% MTP decode (86.2 vs 88.2 ms/tok); acceptance preserved (50/44 cycles vs 49/45 full-vocab) |
| 9 | 2026-08-15 | `aec00005-e85f-4a92-be34-822123d60b57` | `62174db` — cost-model depth schedule: per-position acceptance EMAs + greedy marginal rule Πpᵢ > h·(1+S)/(1+kh); seed-prefill shape warm; phase trace | — | ✅ | covered: omlx `_DepthController` implements the same per-position-EMA + marginal-cost depth selection (score = expected tokens / t_est(d), measured machine economics, warmup sweep, depth-0 escape) — a strict superset of the one-EMA Swift schedule. Trace/warm are harness-level, not model ports |
| 10 | 2026-08-15 | `06e8c9d4-add5-4052-8ccb-85decf2cb1f6` | `09eda55` — prefix-replay tape for K>=2: exact pre-verify recurrent inputs kept compact; partial accept replays only the committed prefix; K=1 keeps the eager checkpoint | — | ✅ | covered: `mtp_partial_rollback` stashes exact pre-forward inputs per GDN layer (`_mtp_draft_stash`) and replays `1 + accepted` rows through `_process_chunk` on rejection — the same tape design, applied at every width |
| 11 | 2026-08-15 | `daf42d3e-df44-4117-a965-b34882dd999b` | `d819641` | — | ✅ | identical diff to 10 (re-submission; only index hashes differ) — covered as 10 |
| 12 | 2026-08-15 | `5b656013-a2aa-44bb-bfa1-13471d966063` | `08897af` — width-wall chunking: verify widths 6..9 split into [S-4, 4] sub-chunks to avoid wide-shape kernel drift | — | ✅ | no port: the width wall is a Swift/Metal kernel-selection property; Python mlx has no wide-shape drift — verified empirically: depth-8 verify (M=9, the widest legal round) is token-exact vs serial (96-tok check). The chunk-boundary refactor has no Python equivalent to port |
| 13 | 2026-08-15 | `6cb6c963-a793-4f34-987e-94c05182eeea` | `8f41fa6` — compiled GDN fusions: fp32 g+beta producer + post-norm SiLU product as compiled passes | `PORT-13` | ✅ token-exact (96-tok depth-2, fusions ON) | partial port, opt-in OFF: `OMLX_QWEN_COMPILED_FUSIONS=1`. `compute_g` already compiled in mlx-lm (shapeless); two-output g+beta fusion NOT ported (laguna C1 ULP risk; beta feeds the verify recurrence). Single-output post-norm SiLU compile token-exact but measured 196.9 vs ~87 ms/tok (2.3x slower, M4 Max) — stays OFF |
| 14 | 2026-08-15 | `91743270-d5e2-4a8e-b61d-451bde3121e7` | `deb63ad` — declares a remote MTP head (amal-david q2-q4-rerank) in `mtp-head.manifest.json` | — | ✅ | no port: head-declaration only — the proposal head is swappable artifact, not model code; omlx keeps the organizer-pinned bf16 head. Draft-side only: the target decides every emitted token |
| 15 | 2026-08-16 | `c08eb406-7383-4681-b12f-62e2fc35bf29` | `b6ce964` — 512-prefix attention warm + optimistic-decaying policy priors | — | ✅ | no port: warmup-shape + policy-prior tuning. omlx has no timed decode window (JIT cost is not inside a scored region) and `_DepthController` learns priors from measurement, not hand-set constants |
| 16 | 2026-08-16 | `be6b63f5-ab13-4fb1-bdee-cd49ffb5756a` | `1033e1a` — `draftTokenID` single-dispatch proposal (compact projection + argmax + id map in one dispatch) | — | ✅ | covered by port-8: `_chain_next_drafts` draws compact drafts with argmax + `map_draft_token_ids` in one host graph; the JIT-warm hazard this submission fixes is a timed-window artifact omlx does not have |
| 17 | 2026-08-16 | `a3104b04-715a-4a7e-a58a-8445b68b54a8` | `cf35029` — KV-only MTP-head history flush (leading rows append K/V only; final row runs full) + packed K/V concat | `PORT-17` | ✅ token-exact (96-tok depth-2) | ported, opt-in OFF: `OMLX_QWEN_KV_ONLY_HISTORY=1`. Token-exact but measured neutral on M4 Max (87.30 vs 87.43 ms/tok) — opt-in per the port bar. Padded-K/V pack folded into the packed-QKV path (port-3) |
| 18 | 2026-08-16 | `9ef7b7f1-7373-43d0-9a60-3a4e885f1198` | `4eb5448` — fused full-attention Q/K RMSNorm+partial-RoPE Metal kernel | — | ✅ | no port: C++ Metal kernel in vendored mlx-swift (MLXFast.metalKernel); Python mlx ships prebuilt kernels. The equivalent q/rms-norm/rope math runs in mlx's graph with identical primitives |
| 19 | 2026-08-16 | `22ce3162-41e8-4d8b-8199-bbbc555e00a6` | `e3b4531` — packed GDN prework mixer Metal kernel (conv+SiLU+split+norm+g in one launch; beta stays graph) | — | ✅ | no port: C++ Metal kernel. Python GDN prework runs the same primitives as graph ops; the one-input sigmoid ULP hazard (0xC0DB) is Metal-kernel-specific |
| 20 | 2026-08-16 | `aa7c3e0c-20d1-4b27-a80c-e622e7880999` | `df404e0` — SIMD-shuffle draft top-1 reduction kernel | — | ✅ | no port: C++ Metal kernel; the draft proposal argmax is a graph `mx.argmax` in omlx with the same total order (value-desc, id-asc) |
| 21 | 2026-08-16 | `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` | `7351e62` — margin-confidence gate in the cost model + fused gate-up SiLU Metal pass | — | ✅ | policy half covered by `_DepthController` (acceptance EMAs already condition on observed margins); SiLU fusion is a C++ Metal kernel — no port |
| 22 | 2026-08-16 | `f03469a9-d889-4bca-8061-a4ad3178c7d2` | `c7468c5` — `segmentedStreakGate = 2` (draft-policy constant) | — | ✅ | covered: streak/segmented gating is a Swift-session policy constant; omlx `_DepthController` owns depth choice with its own gates (hysteresis, exit streaks) |
| 23 | 2026-08-16 | `776d7168-e968-40c4-808e-f643ec1953a7` | `4aacc53` — `asyncEval` the replayed recurrent states after a wide-prefix replay | — | ✅ | covered: omlx dispatches the draft chain and rollback with `mx.async_eval` (single-sync cycle design); replay roots are consumed by the next cycle's graph build |
| 24 | 2026-08-17 | `5c523482-452c-4662-a303-a3b359c81030` | `cdb06b7` — packed GDN prework mixer with in-kernel beta (0xC0DB sigmoid ULP map) | — | ✅ | no port: C++ Metal kernel; the sigmoid ULP hazard is Metal-kernel-internal and does not exist in mlx's graph sigmoid |
| 25 | 2026-08-17 | `03dedda8-fc70-4e3e-881f-5384a17af405` | `32b94cb` — publish the verify forward's post-norm block so accepted head-history rows do not each repeat a row-local RMSNorm | — | ✅ | covered: `_chain_next_drafts` applies the trunk norm ONCE to the whole hidden block before the head fold (`_HEAD_HIDDEN_POST_NORM` + `_trunk_norm_module`), the published-block benefit is inherent — no per-row repeated normalization exists in the Python path |
| 26 | 2026-08-17 | `a451f3b3-20c6-4dcd-9a9f-e7f21d937a58` | `d077e68` — cost model refined to per-draft-row marginal costs hᵢ | — | ✅ | covered: `_DepthController` scores each depth with measured per-row marginal cost (warmup sweep + measured slope between depths) — the hᵢ generalization, machine-measured |
| 27 | 2026-08-17 | `febb7e27-935c-4a1a-bee4-d7387950ea2d` | `cbdc3a8` — proposal-head precision islands (selected exact BF16 rows in the declared head) | — | ✅ | no port: head-artifact + proposal-side model feature tied to a declared head; omlx runs the pinned bf16 head. Draft-side only, target decides emitted tokens |
| 28 | 2026-08-17 | `316bd671-e06a-4aaf-a276-8d59f3fedc5d` | `111b757` — warm the `callWithHiddenAndNormed` shapes + top-2 kernels at every row count | — | ✅ | no port: warmup-only (timed-window JIT avoidance); omlx has no scored timed window |
| 29 | 2026-08-17 | `39fdbf62-60e4-4ab7-bf09-0d1b5a0b618a` | `ed4dfd6` — warm additions + head precision islands re-declared | — | ✅ | no port: warmup-only + head declaration (as 27/28) |
| 30 | 2026-08-18 | `b1e2591b-13f2-4b17-baf1-2956ca9242df` | `036fd9c` — draft rerank: coarse affine-2 compact readout picks 32 rows, affine-4 compact readout reranks them; head declaration | — | ✅ | no port: rerank Metal kernel + head artifact (draft-side). omlx keeps the pinned head; the proposal readout stays graph argmax |
| 31 | 2026-08-18 | `824dc272-b560-4dc6-bf6c-42f58944f4cb` | `8dabcfb` — affine2/g64 single-row fast QMV for the coarse compact readout | — | ✅ | no port: C++ Metal kernel in vendored mlx-swift (2-bit qmv); Python mlx has no 2-bit quantized_matmul and the coarse readout is head-artifact-side |
| 32 | 2026-08-18 | `72ce82dc-f751-485d-a7b3-94ab6471cf87` | `dccba74` — affine2/g64 32-values-per-lane QMV variant | — | ✅ | no port: C++ Metal kernel (as 31) |
| 33 | 2026-08-18 | `4f76de6e-d9cf-4a52-8aa1-57dc4c0e2a16` | `c0e34af` — Metal residency-set resize to hold the post-warm footprint | — | ✅ | no port: Metal runtime memory policy (wired-limit / residency set) — Python mlx owns its own memory management; omlx cache-limit guard (set_cache_limit) already covers working-set sizing |
| 34 | 2026-08-18 | `b0994092-554a-452c-8d4c-78fecda724b4` | `1cb1f43` — verify-concat JIT warm (host primary + device draft ids concat kernels) | — | ✅ | no port: warmup-only (timed-window JIT avoidance); omlx has no scored timed window |
| 35 | 2026-08-19 | `59b321ee-eb5c-40ec-bb49-5218e4b8cd31` | `9e1ff9e` — later-window SDPA compile (throwaway FA K/V to kL>=1024, qL=1/5/4 shapes) | — | ✅ | no port: warmup-only (timed-window JIT avoidance) |

## Submission registry — Accept-labeled merged submissions (chronological)

Main also merges submissions under an "Accept submission <uuid>" label (the
organizer's acceptance path; same editable-surface imports). Rows 36–54 close
those out with the same port bar.

| # | Date | Submission | Challenge commit | Port | Token-exact | Status |
|---|---|---|---|---|---|---|
| 36 | 2026-08-15 | `3e8b6f1a-0d16-490d-b233-14adac4527ca` | `fe88292` — persistent MTP-head KV cache + seed-hidden history fold | — | ✅ | covered: omlx `prompt_priming` folds the prompt into the persistent head cache during prefill (the same committed-history design) and `state.mtp_cache` persists across rounds |
| 37 | 2026-08-15 | `45c257f1-f249-4fc0-945d-ee330bf9865c` | `1167008` — fused GDN in-proj (qkv+z+b+a) + fused SwiGLU gate/up matmuls; GDN mid-kernel; fused QK prep; head surface | `PORT-37` | ✅ token-exact (96-tok depth-2) | in-proj + gate/up fused, default ON (`OMLX_QWEN_FUSED_PROJ=0` ablation); 48/48 GDN + 64/64 MLP engage; −1% decode (89.5 vs 90.4 ms/tok). GDN mid-kernel + fused QK prep are Metal kernels (N); scale-const memo covered by constant folding |
| 38 | 2026-08-15 | `55fa8d31-7a40-4390-ba4e-4e906ead1e3d` | `3e157ad` — per-boundary eager checkpoints at every width + warmup | — | ✅ | covered: this is the transient pre-fused state; the final design (and omlx) run the fused unsplit verify with lazy replay (`mtp_partial_rollback`) — see rows 4/10 |
| 39 | 2026-08-16 | `12b1c699-febb-4ed6-9b24-c19018e5f006` | `033f622` — policy constants (headStepCostRatio 0.18, width-wall cap 5, streak gate 3) + fused gate/up & fused-QK dispatch thresholds | — | ✅ | covered: policy constants are `_DepthController` territory (machine-measured); no width wall in Python (row 12); dispatch thresholds are Metal-kernel selection |
| 40 | 2026-08-17 | `d3caa5fe-9aa7-4203-bb96-249aaafb4801` | `6209702` — declares the dwsdubey 4-bit re-quantized MTP head | — | ✅ | no port: head-declaration only (proposal-side; the target decides every emitted token) |
| 41 | 2026-08-17 | `ba493f74-c0fe-440a-a956-f77d26232e54` | `156b5b7` — cost-model schedule (as row 9) + quantized kernel tweak | — | ✅ | covered: cost-model depth selection is `_DepthController`; kernel tweak is C++ Metal |
| 42 | 2026-08-17 | `14b53255-e585-44bd-84d9-37b7b29c0be9` | `79683c6` — affine-4 qdot nibble-extraction kernel change + head manifest | — | ✅ | no port: C++ Metal kernel; head declaration |
| 43 | 2026-08-17 | `1d7876fd-d0e7-4e8a-a6dd-432a321084e9` | `be3361b` — warm `callWithHiddenAndNormed` shapes | — | ✅ | no port: warmup-only |
| 44 | 2026-08-17 | `1235f4ba-0a48-4f9a-a0fa-8c9ed6880fd7` | `0824e0e` — quantized kernel 2-line | — | ✅ | no port: C++ Metal kernel |
| 45 | 2026-08-17 | `f9ea43fd-e2b1-453a-b638-b58d2946115c` | `1abe636` — warmup + top-2 kernel compile + quantized tweak | — | ✅ | no port: warmup-only + C++ Metal kernel |
| 46 | 2026-08-17 | `bd007bc7-e8ab-4919-baf4-d5e90068dd83` | `d1530a4` — quantized kernel 4-line | — | ✅ | no port: C++ Metal kernel |
| 47 | 2026-08-17 | `caec88d4-c566-4d5c-ab80-5ce6e9c9e86d` | `0d800b2` — quantized kernel 6-line | — | ✅ | no port: C++ Metal kernel |
| 48 | 2026-08-18 | `578535f7-95e6-4f95-a34c-281b9dbbbffc` | `12d3756` — crossrow QMV lane partitioning (4+4 vs 3+3+2) | — | ✅ | no port: C++ Metal kernel |
| 49 | 2026-08-18 | `0792c757-e04b-4449-b9a5-e4ab9b64a396` | `868cde8` — memoized q/k norm scale constants + fused residual+rmsnorm kernel + quantized | — | ✅ | scale-const memo covered (Python constant folding); fused residual+rmsnorm is a Metal kernel (two-output compile is C1-risky in Python — see C1) |
| 50 | 2026-08-18 | `12864bc1-9c9e-4e3b-8964-e8b9e4da8d31` | `369cc05` — proposal-side top-32 shortlist kernel (replaces argPartition sort) + invScale hoist | — | ✅ | no port: rerank shortlist is head-artifact + Metal kernel; invScale hoist covered by constant folding |
| 51 | 2026-08-18 | `3a995c2b-3c42-48e8-b982-f36a8abda0e7` | `86fb1f0` — wired-limit residency-set resize + scale-const memo + fused residual+rmsnorm + quantized | — | ✅ | memory policy is Metal runtime (as row 33); rest as row 49 |
| 52 | 2026-08-18 | `942e5ab2-1c46-4c50-b7c3-eaf948878ed0` | `474c750` — affine2/g64 32-values-per-lane QMV kernel + memory policy | — | ✅ | no port: C++ Metal kernel + Metal memory policy (as rows 31/32/33) |
| 53 | 2026-08-18 | `11863aa9-0dc0-4703-b7a4-eacd473810cb` | `5068eb8` — empty commit (no editable-path diff) | — | ✅ | no-op: administrative accept, nothing to port |
| 54 | 2026-08-19 | `0cd0a6b4-b539-4705-a1c7-cb271c1f9d3b` | `0c90733` — wired-limit overwrite (memory policy) + session cleanup | — | ✅ | no port: Metal runtime memory policy (as row 33); session cleanup mirrors covered machinery |

## Concern register

### C1 — Two-output compiled g+beta fusion not ported (ULP-divergence risk)

- **Submission / challenge commit:** `6cb6c963-...` / `8f41fa6` (`qwen35CompiledGatedDeltaGBeta`).
- **Optimization:** compile the fp32 `g` and `beta` producers of the GDN recurrence into one two-output compiled pass.
- **Token-exactness issue:** the laguna port's C1 finding (MLX 0.32) shows two-output compiled functions that consume a shared intermediate can diverge from eager at ULP (5.96e-8–1.19e-7). `g`/`beta` feed the verify recurrence where the challenge requires absolute fidelity, so the fusion is not ported; mlx-lm's `compute_g` is already compiled (single output, shapeless) and `beta` stays a graph `sigmoid`. The single-output post-norm SiLU compile IS ported (opt-in, token-exact, measured 2.3× slower on M4 Max — see row 13).
- **MLX-version dependence:** property of MLX 0.32.0 compiled kernels; re-verify after any MLX bump.

## How to update

Every ported submission updates its registry row with the port commit and a
one-line evidence summary. Anything that fails the token-exactness bar is
documented in the Concern register before the corresponding code lands.
