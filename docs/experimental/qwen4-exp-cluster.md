# Qwen4-Exp cluster support (experimental)

oMLX can serve Qwen4-Exp as a pipeline-parallel cluster when the checkpoint is
loaded with the Qwen4 rank adapter. The adapter is text-only: the vision tower
is not loaded on cluster ranks, and the embedded Lightning MTP head is supported
on rank 0 with a verify-only role on the other ranks.

## What is supported

- A text-only `mlx_lm.models.qwen4_exp` adapter for cluster ranks.
- Stage slicing over the Qwen4-Exp decoder layers.
- Hyper-connection (HC) residual handoff between ranks, with the pending write
  flushed before the stage send.
- The final HC mixer and LM head on rank 0 (the tail rank).
- PLE n-gram tables accounted as SSD-backed lookup tables and kept on the head
  rank.
- Lightning MTP on rank 0; other ranks replay the verify cycles.
- Pipeline capability reporting through the normal cluster planner.

Vision and image inputs remain single-Mac features in this version. The
adapter requires the Qwen4-Exp checkpoint layout with the embedded `mtp.*`
head and the PLE table in the layer named by `ple_layer_ids`.

## Requirements

- The cluster must be provisioned with the Qwen4-capable oMLX pin.
- The Qwen4 checkpoint must be present at the same path on every rank.
- The PLE table may be mmap'd from SSD; the planner charges it as
  `ssd_weight_bytes` and the head rank needs room for its row cache.
- MTP requires a plan with room for the draft head. Set MTP explicitly; do not
  leave one node on a different MTP state.

## Prefill working set of the table stage

Loading a stage is only half of admitting it. Measured live on a 128 GiB +
32 GiB split (2026-11-12), the stage holding the PLE n-gram layer grows well
above its loaded footprint during one prefill chunk:

| Chunk | Growth above load | Note |
|---|---|---|
| 1024 | +7.25 GiB | measured peak 16 GiB |
| 2048 | +13.25 GiB | measured peak 22 GiB |
| 4096 | ≤ 16.25 GiB | completed a 26k-token prefill at the 25 GiB Metal ceiling |

The chunk sets the size; the context saturates above ~13k tokens. The growth
is the table stage's per-chunk working set — not attention (the stage has one
full-attention layer and grows 1.8× as much as an 11-attention-layer tail) and
not the table being paged in (one lookup over the prompt reads ~0.8 GB of file
data, an order of magnitude below the spike).

The plan reserves this as `ModelLayout.table_prefill_transient_bytes(chunk)`:
piecewise-linear through the three measured points, scaling down below 1024
and keeping the largest measured envelope above 4096 (larger chunks are
unmeasured; the rank guard stays the request-time authority). Only the stage
whose layers include a `ple_layer_ids` layer pays it, and only when the plan
offloads the table to SSD; the reservation is `max(generic attention
transient, envelope)` there — the measured growth already contains the
attention share, so the two are never summed. Every other stage keeps the
generic price. A checkpoint without a lookup table plans byte-for-byte as
before. With this reservation the split that completed live (layers 0–4 on
the 32 GiB Mac) plans, and the split that ran out of memory (layers 0–8) is
refused at plan time.

## Validation

The 2026-11-01 two-node validation (M5 Max 128 GiB + M1 Max 32 GiB, cut 6)
covered short prompts, a 4,239-token sequence, a 10,553-token needle and a
63,053-token needle with MTP on and the PLE table on SSD. At 63k, 4096-token
prefill chunks reached about 864 tok/s prefill and 43.8 tok/s decode. See
`TEST-QWEN4.md` in the localinferenz test tree for the full matrix.

## Not covered

- Vision and image preprocessing on cluster ranks.
- Four-node and alternate-cut validation.
- Exact long-context token parity with single-Mac bf16 execution.
- Quantization or checkpoint conversion.

## Implementation map

- `omlx/patches/qwen4_exp_mlx_lm/` — rank adapter and registration.
- `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`
  — pipeline forward and HC send/receive.
- `omlx/cluster/planner.py` — cold planner registration, SSD table accounting
  and PLE-aware stage starts.
- `omlx/cluster/pipeline_compat.py` — pipeline capability reporting.
- `omlx/utils/model_loading.py` — pre-load adapter registration.
- `tests/test_qwen4_cluster_adapter.py`, `tests/test_qwen4_pipeline.py` —
  regression tests.
- `tests/test_cluster_table_prefill.py` — the measured table-stage working
  set and the two live split outcomes.