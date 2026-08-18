# mlxfast-challenge → oMLX port: correctness ledger

Per-submission record for the `perf/mlx-fast-laguna` port of the
[Layr-Labs/mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge)
Laguna XS 2.1 DFlash Swift optimizations (`Sources/MLXFastModel/`) into oMLX's
Python MLX Laguna path (`omlx/patches/laguna/laguna_model.py`).

Each commit on this branch is labeled with the challenge submission it ports
(`Validate submission <uuid>`, matching the organizer's own commit labels) and
MUST update this doc with that submission's token/bit-exactness status. The
port bar is **bit-exact parity against the stock vendored model** plus a
measured win before anything ships default-on; anything that is NOT bit-exact,
or that carries a token-correctness risk, is recorded here with the challenge
commit that introduced it.

Challenge baseline read at `layr-labs/main` head `d9459e4`; the optimization
features entered the frozen baseline through the organizer's "Validate
submission" commits, mapped below.

## Submission registry (chronological)

| # | Submission (challenge commit) | Port commit | Bit/token-exact | Measured | Status |
|---|---|---|---|---|---|
| 93 | `8b4de42b-d6bd-4da8-814d-b0b3ae6cf2f2` (`c9e1043`: compiled softplus gate, compiled SiLU product) | `Validate submission 8b4de42b…` | ✅ bit-exact (identical expression trees; compile fuses elementwise only) | +3.96% decode aggregate (with 95–98) | compiled, default ON |
| 94 | `613aaf69-9016-4d57-b799-bdd22d51c5c9` (`62c6697`: fused routed + shared gate/up banks; fused QKV) | `Validate submission 613aaf69…` | ✅ bit-exact (per-row independence of gather-QMM/qmm) | routed: neutral; shared: −2.3% | opt-in OFF; fused QKV NOT ported (Swift ablation: no decode gain) |
| 95 | `8adb56be-8f8f-4611-8914-8daf052b5f21` (`f8848e0`: compiled top-k normalize; compiled two-output router tail) | `Validate submission 8adb56be…` | ✅ bit-exact (top-k normalize is single-output) / ⛔ NOT bit-exact if compiled (router tail, C1) | n/a | top-k normalize compiled, ON; router tail kept eager (C1) |
| 96 | `9a37e4dc-b518-446c-a3f0-e4e90a581674` (`b424bc8`: compiled weighted expert combine) | `Validate submission 9a37e4dc…` | ✅ bit-exact (single-output; same reduction order) | +3.96% decode aggregate (with 93–95, 97–98) | compiled, default ON |

## Concern register (token/bit-exactness issues, with challenge commits)

### C1 — Compiled two-output router tail is not bit-exact in Python MLX

- **Submission / challenge commit:** `8adb56be-8f8f-4611-8914-8daf052b5f21` / `f8848e0` (`lagunaCompiledRouterTail`).
- **Optimization:** compile `[sigmoid(logits), -(sigmoid(logits)+bias)]` into one kernel (four elementwise launches per router call, 39 per token).
- **Token-exactness issue:** a compiled function returning TWO outputs that consume the same `sigmoid(a)` intermediate diverges from eager at ULP (5.96e-8–1.19e-7, deterministic, isolated to the multi-consumer shape). Single-output compiled fusions reusing the same sigmoid are bit-exact, so the trigger is the two-output shape. It feeds `argpartition` expert selection — a ULP flip at a near-tie boundary changes WHICH experts are gathered (a different forward, not a small perturbation), i.e. the challenge's own documented correctness cliff ("Rank is the wrong metric").
- **Mitigation:** kept eager in the port; pinned by `test_two_output_compiled_tail_diverges_documented` (fails if MLX ever fixes it).
- **MLX-version dependence:** property of MLX 0.32.0 compiled kernels; re-verify after any MLX bump.

## How to update

Every ported submission appends a row to the registry with its port commit and
a one-line evidence summary. Anything that fails either half of the bar
(bit-exactness or a measured win) is documented in the Concern register before
the corresponding code lands.
