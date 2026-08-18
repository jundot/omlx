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

## Concern register (token/bit-exactness issues, with challenge commits)

_Currently empty — the first ported submissions are bit-exact. Any future
submission whose port is not bit-exact gets a row here with its challenge
commit._

## How to update

Every ported submission appends a row to the registry with its port commit and
a one-line evidence summary. Anything that fails either half of the bar
(bit-exactness or a measured win) is documented in the Concern register before
the corresponding code lands.
