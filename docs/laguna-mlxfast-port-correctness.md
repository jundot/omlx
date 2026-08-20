# mlxfast-challenge → oMLX port: correctness ledger

Per-submission record for the `perf/mlx-fast-laguna` port of the
[Layr-Labs/mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge)
Laguna XS 2.1 DFlash Swift optimizations (`Sources/MLXFastModel/`) into oMLX's
Python MLX Laguna path (`omlx/patches/laguna/laguna_model.py`).

Each commit on this branch that ports a challenge submission is labeled with
that submission's UUID (`Validate submission <uuid>`, matching the organizer's
own commit labels) and MUST update this doc with the submission's
token/bit-exactness status. The port bar is **bit-exact parity against the
stock vendored model** plus a measured win before anything ships default-on;
anything that is NOT bit-exact, or that carries a token-correctness risk, is
recorded in the Concern register with the challenge commit that introduced it.

Challenge baseline read at `layr-labs/main` head `d9459e4`. Only the
**Laguna XS 2.1** submissions are in scope (the challenge's earlier
DeepSeek V4 Flash / Gemma 4 31B eras were replaced by the organizer's Laguna
migration `4799830` and are not part of the current model surface).

## Submission registry (Laguna era, chronological)

| # | Date | Submission | Challenge commit | Port | Bit/token-exact | Measured | Status |
|---|---|---|---|---|---|---|---|
| 93 | 2026-07-23 | `8b4de42b-d6bd-4da8-814d-b0b3ae6cf2f2` | `c9e1043` — compiled softplus gate, compiled SiLU product | `2a6fbe92` | ✅ bit-exact (single-output compile, identical expression tree) | +3.96% decode aggregate | compiled, default ON |
| 94 | 2026-07-23 | `613aaf69-9016-4d57-b799-bdd22d51c5c9` | `62c6697` — fused routed + shared gate/up NVFP4 banks; fused QKV | `35592b93` | ✅ bit-exact (per-row independence of gather-QMM/qmm) | routed neutral; shared −2.3% | opt-in OFF; fused QKV NOT ported (Swift ablation: no decode gain) |
| 95 | 2026-07-23 | `8adb56be-8f8f-4611-8914-8daf052b5f21` | `f8848e0` — compiled top-k normalize; compiled two-output router tail | `f48c5323` | ✅ top-k normalize bit-exact / ⛔ router tail NOT bit-exact if compiled (C1) | n/a | normalize ON; router tail kept eager (C1) |
| 96 | 2026-07-24 | `9a37e4dc-b518-446c-a3f0-e4e90a581674` | `b424bc8` — compiled weighted expert combine | `6181a829` | ✅ bit-exact (same reduction order) | +3.96% decode aggregate | compiled, default ON |
| 97 | 2026-07-24 | `eb76e2b8-de50-44d5-9137-953c6e40d28e` | `4d9eecb` — folded-normalized expert combine (deferred top-k) | `90c997ed` | ✅ bit-exact (pinned: router-normalize + combine ≡ folded) | n/a (covered by 95+96) | reproduced equivalently, no re-ported code |
| 98 | 2026-07-24 | `dc738a8d-a8b9-4187-abc3-68f61099fb67` | `7e61f8d` — residual-variant expert combines | `4b27cc88` | ✅ bit-exact (IEEE add commutative) | +3.96% decode aggregate | compiled, default ON |
| 99 | 2026-08-02 | `a02330a7-430d-45b1-82f3-9314e115555e` | `018eb60` — compiled fusions re-applied to vendored `Laguna.swift` + `CausalMaskCache` in KVCache | (see 93–98) | ✅ compiled fusions covered by 93–98 / ⛔ CausalMaskCache NOT ported (C3) | n/a | compiled fusions covered; mask cache documented (C3) |
| 100 | 2026-08-02 | `e23551d8-87aa-4544-962a-32da86f094e2` | `e8ede96` — group-32 affine INT8 re-quantization of attention projections | — | ⛔ NOT bit-exact (LOSSY requant, C4) | removes ~1.25 GB/step weight traffic | NOT ported (C4) |

## Concern register (token/bit-exactness issues, with challenge commits)

### C1 — Compiled two-output router tail is not portably bit-exact in Python MLX

- **Submission / challenge commit:** `8adb56be-8f8f-4611-8914-8daf052b5f21` / `f8848e0` (`lagunaCompiledRouterTail`).
- **Optimization:** compile `[sigmoid(logits), -(sigmoid(logits)+bias)]` into one kernel (four elementwise launches per router call, 39 per token).
- **Token-exactness issue:** a compiled function returning TWO outputs that consume the same `sigmoid(a)` intermediate is bit-exact on the GitHub `macos-14-arm64` runner but diverges from eager at ULP on an M3 Ultra (5.96e-8–2.38e-7, deterministic, isolated to the multi-consumer shape). Single-output compiled fusions reusing the same sigmoid are bit-exact, so the trigger is the two-output shape on affected GPUs. It feeds `argpartition` expert selection — a ULP flip at a near-tie boundary changes WHICH experts are gathered (a different forward, not a small perturbation), i.e. the challenge's own documented correctness cliff ("Rank is the wrong metric").
- **Mitigation:** kept eager in the port for portable token exactness; bounded numerical parity is pinned by `test_two_output_compiled_tail_is_numerically_close` on both bit-exact and ULP-divergent GPUs.
- **Hardware and MLX-version dependence:** property of MLX 0.32.0 compiled kernels varies by Apple GPU; re-verify across supported GPU generations after any MLX bump.

### C2 — `logits_last_only` head slicing is ULP-divergent (frame divergence)

- **Challenge commit:** `4799830` (`lagunaLastTokenHidden`) — the Laguna migration, not a submission; recorded for completeness.
- **Optimization:** slice post-norm hidden to the last position before `lm_head` so prefill never computes the `[L-1, vocab]` slab.
- **Token-exactness note (not a bug):** a `[1,1,H]` head matmul is ULP-divergent from the `[B,L,H]` full matmul (measured ~1.8e-7) — the same matmul-width **frame divergence** the challenge contract documents. The DFlash reference layer tolerates it; the real-checkpoint greedy trajectory is token-identical with and without the slice. oMLX's DFlash target path already implements it (`logits_last_only`), pinned by `test_target_ops_logits_last_only_slices_before_lm_head`.

### C3 — Causal-mask memo is NOT portable to mlx-lm's rotating cache

- **Submission / challenge commit:** `a02330a7-430d-45b1-82f3-9314e115555e` / `018eb60` (`CausalMaskCache` in `MLXLMCommon/KVCache.swift`).
- **Optimization:** memoize the sliding-window causal mask keyed on `(n, offset, windowSize)` — the Swift asserts a saturated rotating ring rebuilds a byte-identical mask every decode step, so the memo skips the per-step rebuild (two host→device index uploads + GreaterEqual/Add/Less/And; 5× per DFlash drafter round).
- **Token-exactness issue (why NOT ported):** mlx-lm's `RotatingKVCache.make_mask` is NOT constant across saturated decode steps — the rolled window mask advances with the ring's wrap state (verified on MLX 0.32: two consecutive saturated-ring decode steps produce different masks). A memo keyed on `(n, idx, window)` would return a STALE mask → wrong attention → token corruption. Additionally, oMLX's stock Laguna config sizes the ring to the window (`make_cache` → `RotatingKVCache(max_size=sliding_window)`), so the decode mask is `None` and there is nothing to memoize at all.
- **Status:** documented, not ported. The compiled-fusions half of the same submission is already covered by rows 93–98.

### C4 — Group-32 affine INT8 attention re-quantization is LOSSY

- **Submission / challenge commit:** `e23551d8-87aa-4544-962a-32da86f094e2` / `e8ede96` (`lagunaAttentionINT8Enabled`, default ON with `LAGUNA_ATTENTION_INT8=0` escape).
- **Optimization:** re-represent the BF16 attention projections (`q/k/v/o/g_proj`, ~2.9 GB of the ~4.3 GB per-decode-step weight traffic) as group-32 affine INT8 at init — 9 bits/weight vs BF16's 16, removing ~1.25 GB/step.
- **Token-exactness issue:** this is an explicitly **lossy** re-quantization (the submission itself documents it; it lives inside the track envelope's permitted "representation set" but is not bit-exact vs the reference). It is not reproduced in oMLX: the port bar requires bit-exact parity or a documented exception. If oMLX ever adopts it, it must be a default-OFF toggle with lossiness documented and a token-diff gate against the BF16 reference.

## Laguna commits beyond the 8 submissions (all examined)

Every other commit that touched the Laguna model surface (migrations, vendored
model, harness, kernels, docs) was examined for safely-portable logic:

| Commit | What it changed | Disposition |
|---|---|---|
| `4799830` (07-21) | Laguna migration (base model) | 📋 pieces documented: `lagunaLastTokenHidden` → C2; constructor warmup → N3-style note; NVFP4-only-expert + YaRN mscale already verified in oMLX |
| `3d9ec53`/`25f8a50`/`67513ac`/`3b21af3` (07-22) | editablePaths fixes, Gemma naming cleanup | 📋 structural, no optimization |
| `6d679f4` (07-22) | NVFP4 v2 quantization layout | 📋 oMLX `sanitize()` already handles NVFP4 loading |
| `b00280b` (07-24) | vendored Laguna.swift header (reference vs scored) | 📋 docs, no logic |
| `ca6149b` (07-26) | low-memory startup profile | 📋 N1: not ported (oMLX `set_cache_limit(total)` is a load-bearing #300 panic guard) |
| `7632313d`/`2ac117b`/`a1914e5`/`78f6c12` (07-29/30) | DFlash vendor + Criterion-E harness | 📋 N4: benchmark-integrity / harness invariants, not model optimizations |
| `55aec0f` (08-01) | editable-surface: expose sort/reduce kernels + dispatch wrappers | 📋 challenge-structure (moves stock MLX kernels into editable scope), no model logic; the header change only clarifies which path is scored |
| `8c6e218` (08-01) | docs audit | 📋 docs |
| pre-Laguna shared-file commits (`f5ed2be`, `165d3d1`, `ef0a57b`, `fe9d166`, `75ca2a0`, `2ebae10`, `f310c09`, `2a5428a`, `5fc87e3`, `c118b23`) | harness hardening, streaming weight load (`DenseTensorStore`), Gemma-era kernel vendoring | 📋 techniques documented; streaming load is not a decode optimization and mlx-lm loads the cache |

## How to update

Every ported submission updates its registry row with the port commit and a
one-line evidence summary. Anything that fails either half of the bar
(bit-exactness or a measured win) is documented in the Concern register before
the corresponding code lands.
