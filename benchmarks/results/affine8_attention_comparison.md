# Affine8 attention comparison

Apple M5 Pro, MLX 0.32.2. Cache construction excluded; timings cover attention dispatch, evaluation, and synchronization. Fifty measured iterations after twenty warmups. Inputs use FP16, four KV heads, and the same random tensors for all formats in each geometry.

## 250K context

| Head dim | Query rows | Format | Cache MiB | Median ms | Relative L2 vs native FP16 |
|---:|---:|---|---:|---:|---:|
| 128 | 1 | Native FP16 | 488.5 | 2.005 | 0 |
| 128 | 1 | TurboQuant4 | 125.9 | 2.210 | 0.1446 |
| 128 | 1 | TurboQuant8 | 248.0 | 2.602 | 0.0194 |
| 128 | 1 | Affine4 | 129.7 | 1.597 | 0.1623 |
| 128 | 1 | Affine8 | 251.8 | 1.986 | 0.0093 |
| 128 | 4 | Native FP16 | 488.5 | 4.852 | 0 |
| 128 | 4 | TurboQuant4 | 125.9 | 5.026 | 0.1381 |
| 128 | 4 | TurboQuant8 | 248.0 | 5.081 | 0.0185 |
| 128 | 4 | Affine4 | 129.7 | 1.851 | 0.1622 |
| 128 | 4 | Affine8 | 251.8 | 2.746 | 0.0095 |
| 256 | 1 | Native FP16 | 977.0 | 4.258 | 0 |
| 256 | 1 | TurboQuant4 | 248.0 | 4.710 | 0.1436 |
| 256 | 1 | TurboQuant8 | 492.1 | 7.125 | 0.0192 |
| 256 | 1 | Affine4 | 251.8 | 2.074 | 0.1807 |
| 256 | 1 | Affine8 | 495.9 | 3.839 | 0.0102 |
| 256 | 4 | Native FP16 | 977.0 | 12.167 | 0 |
| 256 | 4 | TurboQuant4 | 248.0 | 29.120 | 0.1373 |
| 256 | 4 | TurboQuant8 | 492.1 | 29.492 | 0.0183 |
| 256 | 4 | Affine4 | 251.8 | 6.480 | 0.1708 |
| 256 | 4 | Affine8 | 495.9 | 8.375 | 0.0098 |

Affine8 uses 50.8–51.5% of native FP16 cache bytes in these geometries. Its relative L2 error is 1.87–2.07× lower than TurboQuant8's and 16.5–17.8× lower than Affine4's in these samples. Native M5 Affine8 attention was selected for every tested 8K, 32K, 128K, and 250K case at head dimensions 128 and 256 with one or four query rows.

At 250K, TurboQuant8/Affine8 median-latency ratios are 1.31×, 1.85×, 1.86×, and 3.52× for head-dim 128/one row, 128/four rows, 256/one row, and 256/four rows, respectively. Against native FP16, Affine8 has 1.0%, 43.4%, 9.8%, and 31.2% lower median latency in those same four geometries. This attention-only result does not predict complete model throughput; server trials measure prefill, full-layer dispatch, memory guard behavior, and sustained generation separately.
