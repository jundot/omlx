# DS4 non-MTP B1 full routed-MoE decode probe

Date: 2026-08-23. Hardware: local Apple M3 Ultra. Model:
`DeepSeek-V4-Flash-0731-MXFP4-MLX`, real layer 20 weights, signed 3:5
intermediate slices. This is an isolated kernel/benchmark gate; no serving
dispatch, runtime, or deployment default was changed.

## Exact baseline

After rebuilding the native extension against the current source, the
existing `deepseek_mxfp4_full_decode` primitive accepts both real TP2 slices.
The composed `mx.gather_qmm` reference and native result were
`mx.array_equal` for BF16 B1/top-6, with zero maximum absolute difference.
The stale extension previously rejected these same asymmetric shapes before
queuing a kernel.

Using 12 warm repetitions and 12 measured ABBA repetitions on layer 20:

| slice | stock median | native median | exact | speedup |
|---|---:|---:|---|---:|
| M3 3/8, I=768 | 0.3426 ms | 0.2803 ms | yes | 1.222x |
| M5-shaped 5/8, I=1280 (replayed on M3) | 0.3979 ms | 0.3428 ms | yes | 1.161x |

These are isolated routed-MoE timings, not end-to-end decode throughput.

## Row-tile probe

The kernel now has an opt-in `OMLX_DSV4_FULL_DECODE_ROWS=4` schedule. It adds
no model-serving dispatch path, remains benchmark-only/default-off, and
preserves the existing shape/device choice when the variable is absent. The
added B1 parity test covers row tiles 1, 2, and 4.
Twenty measured repetitions on the same real layer gave:

| slice | row setting | stock median | candidate median | exact | speedup |
|---|---:|---:|---:|---|---:|
| M3 3/8, I=768 | 4 | 0.3064 ms | 0.2552 ms | yes | 1.201x |
| M5-shaped 5/8, I=1280 (replayed on M3) | 4 | 0.5301 ms | 0.4359 ms | yes | 1.216x |

The M5-shaped row-4 result is a useful candidate relative to its row-2 probe,
but the M3 row-1 default remains competitive. These are not physical M5
measurements; the M5 rank must be rerun on its actual NAX device before any
promotion. Results are shape-sensitive and have not been promoted to the
model seam.

The physical M5 follow-up rejected row 4 after a clean metallib rebuild:
candidate median **0.478646 ms** versus stock **0.435417 ms** (**0.910x**),
with bit-exact output. The M3 shape replay did not predict the M5
scheduler/occupancy result. Row 4 therefore remains a benchmark-only rejected
probe and must not enter model-serving dispatch.

## Amdahl and next gate

At the current non-MTP TP2 rate (about 31.2 marker / 31.95 API tok/s), this
lane can only contribute its measured per-layer saving to the routed-MoE
share. The isolated 16--22% kernel reduction is not evidence of a 16--22%
end-to-end gain; a full layer attribution and cold two-rank B1 decode gate
are required before claiming a 10% E2E improvement.

Next gate: retain the current qualified row schedules and run five or more cold
TP2 single-token repetitions with identical prompt/cache state and completion
hash while recording per-rank routed-MoE wall. A different physical-M5 kernel
is required for the next production candidate; row 4 is closed.

Reproduce the isolated probe:

```bash
OMLX_DSV4_FULL_DECODE_ROWS=4 \
  .venv/bin/python benchmarks/bench_ds4_full_moe_decode.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --layer 20 --rank 0 --tokens 1 --warmup 6 --iterations 20
```
