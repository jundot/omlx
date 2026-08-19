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
| 7 | 2026-08-15 | `84583d41-0417-437f-8482-5663cbddf51a` | `b6c7251` — `qmv_fast_crossrow_affine4_g64` multi-row (M=2..5) QMV kernel | ⏳ | ⏳ | N: C++ Metal kernel in vendored mlx-swift; Python mlx ships prebuilt kernels. Verify M=2..5 decode dispatch equivalence empirically |
| 8 | 2026-08-15 | `7b33621c-1d1a-4540-834d-d89c7f402b0f` | `61936f2` — compact proposal vocabulary: draft lm_head narrowed to 98,304 prefix + 26 control rows (~98k of 248k) with device-side id remap | `PORT-8` | ✅ token-exact (96-tok check) | compact draft lm_head, default ON (`OMLX_QWEN_COMPACT_DRAFT=0` ablation); −2.2% MTP decode (86.2 vs 88.2 ms/tok); acceptance preserved (50/44 cycles vs 49/45 full-vocab) |
| 9 | 2026-08-15 | `aec00005-e85f-4a92-be34-822123d60b57` | `62174db` — cost-model depth schedule: per-position acceptance EMAs + greedy marginal rule Πpᵢ > h·(1+S)/(1+kh); seed-prefill shape warm; phase trace | — | ✅ | covered: omlx `_DepthController` implements the same per-position-EMA + marginal-cost depth selection (score = expected tokens / t_est(d), measured machine economics, warmup sweep, depth-0 escape) — a strict superset of the one-EMA Swift schedule. Trace/warm are harness-level, not model ports |
| 10 | 2026-08-15 | `06e8c9d4-add5-4052-8ccb-85decf2cb1f6` | `09eda55` — prefix-replay tape for K>=2: exact pre-verify recurrent inputs kept compact; partial accept replays only the committed prefix; K=1 keeps the eager checkpoint | — | ✅ | covered: `mtp_partial_rollback` stashes exact pre-forward inputs per GDN layer (`_mtp_draft_stash`) and replays `1 + accepted` rows through `_process_chunk` on rejection — the same tape design, applied at every width |
| 11 | 2026-08-15 | `daf42d3e-df44-4117-a965-b34882dd999b` | `d819641` | — | ✅ | identical diff to 10 (re-submission; only index hashes differ) — covered as 10 |
| 12 | 2026-08-15 | `5b656013-a2aa-44bb-bfa1-13471d966063` | `08897af` | ⏳ | ⏳ | pending |
| 13 | 2026-08-15 | `6cb6c963-a793-4f34-987e-94c05182eeea` | `8f41fa6` | ⏳ | ⏳ | pending |
| 14 | 2026-08-15 | `91743270-d5e2-4a8e-b61d-451bde3121e7` | `deb63ad` — declares a remote MTP head (amal-david q2-q4-rerank) in `mtp-head.manifest.json` | — | ✅ | no port: head-declaration only — the proposal head is swappable artifact, not model code; omlx keeps the organizer-pinned bf16 head. Draft-side only: the target decides every emitted token |
| 15 | 2026-08-16 | `c08eb406-7383-4681-b12f-62e2fc35bf29` | `b6ce964` — 512-prefix attention warm + optimistic-decaying policy priors | — | ✅ | no port: warmup-shape + policy-prior tuning. omlx has no timed decode window (JIT cost is not inside a scored region) and `_DepthController` learns priors from measurement, not hand-set constants |
| 16 | 2026-08-16 | `be6b63f5-ab13-4fb1-bdee-cd49ffb5756a` | `1033e1a` — `draftTokenID` single-dispatch proposal (compact projection + argmax + id map in one dispatch) | — | ✅ | covered by port-8: `_chain_next_drafts` draws compact drafts with argmax + `map_draft_token_ids` in one host graph; the JIT-warm hazard this submission fixes is a timed-window artifact omlx does not have |
| 17 | 2026-08-16 | `a3104b04-715a-4a7e-a58a-8445b68b54a8` | `cf35029` | ⏳ | ⏳ | pending (adds Qwen35MTP.swift model) |
| 18 | 2026-08-16 | `9ef7b7f1-7373-43d0-9a60-3a4e885f1198` | `4eb5448` — fused full-attention Q/K RMSNorm+partial-RoPE Metal kernel | — | ✅ | no port: C++ Metal kernel in vendored mlx-swift (MLXFast.metalKernel); Python mlx ships prebuilt kernels. The equivalent q/rms-norm/rope math runs in mlx's graph with identical primitives |
| 19 | 2026-08-16 | `22ce3162-41e8-4d8b-8199-bbbc555e00a6` | `e3b4531` — packed GDN prework mixer Metal kernel (conv+SiLU+split+norm+g in one launch; beta stays graph) | — | ✅ | no port: C++ Metal kernel. Python GDN prework runs the same primitives as graph ops; the one-input sigmoid ULP hazard (0xC0DB) is Metal-kernel-specific |
| 20 | 2026-08-16 | `aa7c3e0c-20d1-4b27-a80c-e622e7880999` | `df404e0` — SIMD-shuffle draft top-1 reduction kernel | — | ✅ | no port: C++ Metal kernel; the draft proposal argmax is a graph `mx.argmax` in omlx with the same total order (value-desc, id-asc) |
| 21 | 2026-08-16 | `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd` | `7351e62` — margin-confidence gate in the cost model + fused gate-up SiLU Metal pass | — | ✅ | policy half covered by `_DepthController` (acceptance EMAs already condition on observed margins); SiLU fusion is a C++ Metal kernel — no port |
| 22 | 2026-08-16 | `f03469a9-d889-4bca-8061-a4ad3178c7d2` | `c7468c5` — `segmentedStreakGate = 2` (draft-policy constant) | — | ✅ | covered: streak/segmented gating is a Swift-session policy constant; omlx `_DepthController` owns depth choice with its own gates (hysteresis, exit streaks) |
| 23 | 2026-08-16 | `776d7168-e968-40c4-808e-f643ec1953a7` | `4aacc53` — `asyncEval` the replayed recurrent states after a wide-prefix replay | — | ✅ | covered: omlx dispatches the draft chain and rollback with `mx.async_eval` (single-sync cycle design); replay roots are consumed by the next cycle's graph build |
| 24 | 2026-08-17 | `5c523482-452c-4662-a303-a3b359c81030` | `cdb06b7` — packed GDN prework mixer with in-kernel beta (0xC0DB sigmoid ULP map) | — | ✅ | no port: C++ Metal kernel; the sigmoid ULP hazard is Metal-kernel-internal and does not exist in mlx's graph sigmoid |
| 25 | 2026-08-17 | `03dedda8-fc70-4e3e-881f-5384a17af405` | `32b94cb` | ⏳ | ⏳ | pending |
| 26 | 2026-08-17 | `a451f3b3-20c6-4dcd-9a9f-e7f21d937a58` | `d077e68` — cost model refined to per-draft-row marginal costs hᵢ | — | ✅ | covered: `_DepthController` scores each depth with measured per-row marginal cost (warmup sweep + measured slope between depths) — the hᵢ generalization, machine-measured |
| 27 | 2026-08-17 | `febb7e27-935c-4a1a-bee4-d7387950ea2d` | `cbdc3a8` — proposal-head precision islands (selected exact BF16 rows in the declared head) | — | ✅ | no port: head-artifact + proposal-side model feature tied to a declared head; omlx runs the pinned bf16 head. Draft-side only, target decides emitted tokens |
| 28 | 2026-08-17 | `316bd671-e06a-4aaf-a276-8d59f3fedc5d` | `111b757` — warm the `callWithHiddenAndNormed` shapes + top-2 kernels at every row count | — | ✅ | no port: warmup-only (timed-window JIT avoidance); omlx has no scored timed window |
| 29 | 2026-08-17 | `39fdbf62-60e4-4ab7-bf09-0d1b5a0b618a` | `ed4dfd6` — warm additions + head precision islands re-declared | — | ✅ | no port: warmup-only + head declaration (as 27/28) |
| 30 | 2026-08-18 | `b1e2591b-13f2-4b17-baf1-2956ca9242df` | `036fd9c` — draft rerank: coarse affine-2 compact readout picks 32 rows, affine-4 compact readout reranks them; head declaration | — | ✅ | no port: rerank Metal kernel + head artifact (draft-side). omlx keeps the pinned head; the proposal readout stays graph argmax |
| 31 | 2026-08-18 | `824dc272-b560-4dc6-bf6c-42f58944f4cb` | `8dabcfb` — affine2/g64 single-row fast QMV for the coarse compact readout | — | ✅ | no port: C++ Metal kernel in vendored mlx-swift (2-bit qmv); Python mlx has no 2-bit quantized_matmul and the coarse readout is head-artifact-side |
| 32 | 2026-08-18 | `72ce82dc-f751-485d-a7b3-94ab6471cf87` | `dccba74` — quantized kernels | ⏳ | ⏳ | pending |
| 33 | 2026-08-18 | `4f76de6e-d9cf-4a52-8aa1-57dc4c0e2a16` | `c0e34af` — session + RuntimeStartupMemoryPolicy | ⏳ | ⏳ | pending |
| 34 | 2026-08-18 | `b0994092-554a-452c-8d4c-78fecda724b4` | `1cb1f43` — session +19 | ⏳ | ⏳ | pending |
| 35 | 2026-08-19 | `59b321ee-eb5c-40ec-bb49-5218e4b8cd31` | `9e1ff9e` — session +70 | ⏳ | ⏳ | pending |

## Concern register

(to be filled as ports land)

## How to update

Every ported submission updates its registry row with the port commit and a
one-line evidence summary. Anything that fails the token-exactness bar is
documented in the Concern register before the corresponding code lands.
