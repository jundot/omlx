# DFlash in oMLX

## Overview

DFlash is block diffusion speculative decoding (arXiv:2602.06036). A small draft model reads hidden states captured from a few target layers and proposes a whole block of tokens in one forward pass; the target verifies the block in a single forward and keeps the longest accepted prefix.

oMLX runs DFlash inside its standard engines. `BatchedEngine` (mlx-lm text models) and `VLMBatchedEngine` (mlx-vlm models) attach a DFlash block drafter to the Lightning MTP decode loop, so DFlash requests use the scheduler, continuous batching, and the paged hot/SSD prefix cache like any other request. The external dflash-mlx engine (`DFlashEngine`) that earlier versions used is gone.

## Supported targets

| Family | Engine | Speculation |
|---|---|---|
| Qwen3.5 / Qwen3.6 / Qwen3.8 (`qwen3_5`, `qwen3_5_moe`) | VLM or text | concurrent rows |
| Qwen3, Qwen3-MoE (`qwen3`, `qwen3_moe`) | text | single row |
| Laguna (`laguna`) | text | single row |
| MiMo V2 (`mimo_v2`, `mimo_v2_flash`) | text | single row |
| GLM-5.3 (`glm5_next`) | VLM | concurrent rows, block of at most 8 |
| Gemma 4 (`gemma4`, `gemma4_unified`) | VLM | single row |
| Muse Glimmer (`muse_glimmer`) | VLM | single row |

"Single row" means a request speculates while it decodes alone; a batch of two or more decodes normally, since rows that accept different counts would need a ragged rollback of sliding-window caches. MiMo V2 checkpoints with the omnimodal vision sidecar load on the VLM engine and run without DFlash; other model types also load without DFlash and log a warning. The gate lives in `omlx/utils/model_loading.py` (`dflash_batched_supported`, `dflash_compatibility`).

## Settings

| Setting | Meaning |
|---|---|
| `dflash_enabled`, `dflash_draft_model` | Turn DFlash on with a draft checkpoint (local path or discovered model id). Without a draft model, a draft bundled in the target's `dflash/` directory (MiMo V2) is used. |
| `dflash_draft_quant_enabled`, `dflash_draft_quant_weight_bits`, `dflash_draft_quant_group_size` | Affine-quantize a bf16 draft at load. |
| `dflash_block_size` | Block size (anchor plus proposals); clamped to the trained block and to the Lightning MTP verify limit (9). |
| `dflash_verify_mode` | `adaptive` (default) lets the Lightning MTP depth controllers choose how many drafts of each block to verify from measured cost and acceptance; `dflash` and `off` verify the whole block. A stored `ddtree` loads as `adaptive`. |
| `dflash_draft_window_size` | Context tokens the drafter attends to (default: the draft's `sliding_window`, else 2048). |

DFlash is mutually exclusive with Lightning MTP, VLM MTP and MoE expert offload. Changing any DFlash setting on a loaded model reloads it.

Removed with the external engine: the dflash-mlx L1/L2 prefix caches (`dflash_in_memory_cache*`, `dflash_ssd_cache*`), `dflash_max_ctx`, `dflash_draft_sink_size`, activation-bit quantization and the `ddtree` verifier. Older settings files with these keys still load; the keys are ignored.

## How a request runs

1. **Prefill.** The scheduler runs the prompt through the target in chunks and asks for the drafter's target layers (`capture_layer_ids`). The captures seed the request's drafter context. After a prefix-cache hit, the drafter restores the projected context of a recently finished request with the same prefix.
2. **Draft.** The drafter projects new captures (`fc` + norm), writes them into a per-row context ring, and drafts a block from `[anchor, mask, ..., mask]`. Rows of a batch draft in one forward with per-row RoPE offsets and masks.
3. **Verify.** Lightning MTP verifies the drafts with one target forward, which also returns the captures for the verified positions. Greedy rows accept by prefix match; sampled rows use the draft distribution for Leviathan acceptance.
4. **Rollback.** Rejected positions are trimmed from the target caches (KV trim with an undo log for rotating and pooling caches, GDN state replay for Qwen3.5).
5. **Ordinary decode.** When a step decodes normally (for example a batch of single-row targets), the same forward feeds the committed token's captures to the drafter, so speculation can resume without a gap.

## Code map

| Path | Role |
|---|---|
| `omlx/speculative/dflash/__init__.py` | Load, bind and attach the drafter; block size and verify mode. |
| `omlx/speculative/dflash/drafter.py` | `DFlashDrafter`: per-row context rings, cohorts, predraft, prefix-hit restore. |
| `omlx/speculative/dflash/variants.py` | Per-checkpoint pieces of the draft forward (Laguna fused QKV, gate and per-capture norms; Muse encoder; MiMo mask embedding; partial RoPE). |
| `omlx/speculative/dflash/selection.py`, `kernels.py` | Proposal sampling, the DFlash2 candidate selector and its Metal kernels. |
| `omlx/patches/mlx_lm_mtp/` | Lightning MTP verify, acceptance, rollback and depth controllers; `drafter_target.py` adds captures and rollback to plain-attention targets (Qwen3, Laguna, MiMo V2, Muse Glimmer). |
| `omlx/patches/mlx_vlm_mtp/*_vlm_runtime.py` | Captures for Qwen3.5, GLM-5.3 and Gemma 4 VLM targets. |
| `omlx/scheduler.py` | Prefill capture and seeding (`_dflash_prefill_capture`, `_dflash_seed_prefill`). |
| `omlx/engine_pool.py` | Gate, drafter load after engine start, memory estimate. |

Draft checkpoints load through mlx-vlm's drafter classes (DFlash v1, DFlash2, Laguna, Muse Glimmer assistant). oMLX rebuilds partial RoPE from the checkpoint config because mlx-vlm rotates the whole head, and applies MiMo's attention value scale and trained mask embedding, which the v1 class does not read.

## Tests

See the "Batched DFlash drafter" section of `docs/TESTING.md`.
