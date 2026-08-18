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

## Concern register (token/bit-exactness issues, with challenge commits)

### C1 — Compiled two-output router tail is not bit-exact in Python MLX

- **Submission / challenge commit:** `8adb56be-8f8f-4611-8914-8daf052b5f21` / `f8848e0` (`lagunaCompiledRouterTail`).
- **Optimization:** compile `[sigmoid(logits), -(sigmoid(logits)+bias)]` into one kernel (four elementwise launches per router call, 39 per token).
- **Token-exactness issue:** a compiled function returning TWO outputs that consume the same `sigmoid(a)` intermediate diverges from eager at ULP (5.96e-8–1.19e-7, deterministic, isolated to the multi-consumer shape). Single-output compiled fusions reusing the same sigmoid are bit-exact, so the trigger is the two-output shape. It feeds `argpartition` expert selection — a ULP flip at a near-tie boundary changes WHICH experts are gathered (a different forward, not a small perturbation), i.e. the challenge's own documented correctness cliff ("Rank is the wrong metric").
- **Mitigation:** kept eager in the port; pinned by `test_two_output_compiled_tail_diverges_documented` (fails if MLX ever fixes it).
- **MLX-version dependence:** property of MLX 0.32.0 compiled kernels; re-verify after any MLX bump.

### C2 — `logits_last_only` head slicing is ULP-divergent (frame divergence)

- **Challenge commit:** `4799830` (`lagunaLastTokenHidden`) — the Laguna migration, not a submission; recorded for completeness.
- **Optimization:** slice post-norm hidden to the last position before `lm_head` so prefill never computes the `[L-1, vocab]` slab.
- **Token-exactness note (not a bug):** a `[1,1,H]` head matmul is ULP-divergent from the `[B,L,H]` full matmul (measured ~1.8e-7) — the same matmul-width **frame divergence** the challenge contract documents. The DFlash reference layer tolerates it; the real-checkpoint greedy trajectory is token-identical with and without the slice. oMLX's DFlash target path already implements it (`logits_last_only`), pinned by `test_target_ops_logits_last_only_slices_before_lm_head`.

## How to update

Every ported submission updates its registry row with the port commit and a
one-line evidence summary. Anything that fails either half of the bar
(bit-exactness or a measured win) is documented in the Concern register before
the corresponding code lands.
