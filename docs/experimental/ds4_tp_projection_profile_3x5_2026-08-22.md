# DS4 TP 3:5 projection profile — rank 0, M1024

This experiment is benchmark-only. It did not start a server, contact the M5,
or install production dispatch. Three real checkpoint blocks (ratio 0/4/128)
were sliced to the signed rank-0 3/8 geometry and replayed on the M3 Ultra at
M=1024 after an 8,192-token prefix. Their medians were weighted by DS4's real
2/21/20 layer schedule and normalized to the measured 628.76 tok/s wall.

## Projection attribution

| projection | attributed wall | attributed time / 1K chunk | infinite single-component ceiling |
|---|---:|---:|---:|
| O-A grouped projection | **18.87%** | 307.27 ms | 775.0 tok/s |
| O-B replicated projection | **13.61%** | 221.73 ms | 727.9 tok/s |
| Q-B local-head projection | 5.55% | 90.37 ms | 665.7 tok/s |
| Q-A projection | 4.02% | 65.41 ms | 655.1 tok/s |
| compressor WKV | 1.85% | 30.20 ms | 640.6 tok/s |
| compressor gate | 1.83% | 29.82 ms | 640.5 tok/s |
| raw WKV | 1.66% | 27.04 ms | 639.4 tok/s |
| indexer Q | 1.13% | 18.45 ms | 636.0 tok/s |
| indexer compressor WKV | 0.65% | 10.58 ms | 632.9 tok/s |
| indexer compressor gate | 0.51% | 8.24 ms | 632.0 tok/s |
| indexer weights | 0.32% | 5.27 ms | 630.8 tok/s |

The O-A/O-B chain is 32.48% of total observed wall and **64.96% of all timed
projection work**. All Q-path projections together are 12.36%; all direct
compressor/indexer dense projections together are 5.16%.

## Exact stock opportunities

Every timing below is a balanced multi-order median over 48 samples per path.
An exact candidate needs at least 1.02x locally before it is treated as evidence.

| candidate | array-equal | local speedup | projected end-to-end | outcome |
|---|---:|---:|---:|---|
| `mx.compile` O-A -> reshape -> O-B | yes | 1.0123x | 631.26 tok/s | below gate |
| concat Q-A/WKV and Q-B/indexer-Q chains | yes | 1.0112x | 629.62 tok/s | below gate |
| compile the concatenated Q chain | yes | 1.0049x | 629.14 tok/s | below gate |
| concat Q-B + indexer Q | yes | 1.0078x | 629.09 tok/s | below gate |
| concat Q-A + raw WKV | yes | 1.0057x | 628.96 tok/s | below gate |
| concat main compressor WKV + gate | yes | 1.0037x | 628.84 tok/s | below gate |
| concat indexer dense triplet | **no** | 1.3334x | forbidden | reduction geometry drift |
| concat all dense direct-X banks | **no** | 1.0976x | forbidden | reduction geometry drift |

There is no stock grouping/concat/compile win worth promoting. The two visually
fast dense concats change BF16 results, and every array-equal stock path is
inside noise or below the 2% evidence floor.

## Amdahl-ranked implementation candidates

| candidate family | wall exposed | TPS if that family reaches 2x | infinite ceiling | target alone? |
|---|---:|---:|---:|---:|
| all projection work | 50.00% | 838.37 | 1,257.63 | 3.88x required |
| **native or distributed O-A/O-B chain** | **32.48%** | **750.68** | 931.24 | no |
| native Q input chain | 12.36% | 670.17 | 717.42 | no |
| native compressor projection bank | 5.16% | 645.43 | 663.00 | no |

The lead projection candidate is therefore the **O-A/O-B output chain**, not
another Q/KV concat. A useful implementation must change the actual kernel or
distributed ownership:

1. A tiled native O-A -> BF16 boundary -> O-B path that avoids the 8,192-wide
   intermediate round trip while preserving the existing BF16 store and exact
   reduction order; or
2. an exact output-ownership/reduce-scatter design that removes replicated O-B
   work without changing rank-order summation.

Algebraically pre-composing the two weight matrices is not lossless because it
removes the current BF16 intermediate rounding boundary. `mx.compile` already
proves that graph wrapping alone is not the answer.

Even perfect removal of the entire output chain tops out near 931 tok/s, so
this remains a joint campaign with routed MoE. The earlier stage profile found
attention/projection plus MoE need roughly 1.8x together; this drill-down says
where the projection half must start.

Reproduce:

```bash
/Users/jonathanspangler/omlx-v2-build/bin/python \
  benchmarks/bench_ds4_tp_projection_profile.py \
  --model /Users/jonathanspangler/.lmstudio/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --rank 0 --shard-weights 3,5 --tokens 1024 --prefix-tokens 8192 \
  --warmup 2 --cycles 5 --grouping-cycles 12 --min-speedup 1.02
```
