# Qwen ANE prefill with DFlash: benchmark report

> Temporary PR validation report (2026-09-19). Remove after review if these
> findings are retained in the PR discussion or permanent documentation.

This report follows the work built on [PR #3058](https://github.com/jundot/omlx/pull/3058),
in the context of [issue #2986](https://github.com/jundot/omlx/issues/2986): an
initial benchmark series exposed two lifecycle problems, which were reproduced
and fixed before repeating the comparison.

With the tested settings, the repeated runs show an 8K prefill benefit but a
4K regression. Enabling ANE costs about 3.83 GB of additional peak memory.
The results below compare ANE off and on in the same fixed build.

## Test setup

- Hardware: Apple M5 Pro, 64 GiB unified memory, macOS 26.7.
- Target: `mlx-community/Qwen3.8-27B-4bit`.
- Drafter: `incoai/Qwen3.8-27B-DFlash2`.
- Runtime: source-built oMLX `0.7.0.dev4`, including fix `96eb965e`.
- Benchmark: Auto engine, Code (Python), single requests, 128 generated tokens.
- ANE: 2048-token tile; MLP fraction 0.35 / 64 layers; GDN fraction 0.375 /
  48 layers; dual ANE enabled; fused down projection and CPU sharing disabled.
- DFlash L1 cache enabled, L2 disabled, every measured request logged zero cached tokens.

## Initial series: promising numbers, then two failures

The first ANE-on run reached 525.9 / 529.3 prefill tok/s at 4K / 8K, compared
with 512.9 / 501.9 for the preceding GPU baseline. A later accidental **Tune
again** run stopped after 8 of 10 tests, reporting that ANE had compiled but
never executed. Logs showed zero MLP ANE operations while GDN still ran.
Delivered chunks were 2048 tokens, ruling out the suggested chunk-size mismatch
for this run. Later DFlash benchmarks inherited the missing MLP dispatch,
restarting the service restored it.

Independently, benchmark cleanup timed out with about 18.01 GB of MLX active
memory still referenced. This already affected the first ANE-on run, before
the tuner failure, and restarting did not resolve it.

The exploratory series is excluded from the final comparison: it mixed valid
and broken dispatch states and included an accidentally unaligned run.

## Benchmark comparison (after the fixes)

Three alternating pairs were run: A1/B1, A2/B2, A3/B3. A keeps DFlash enabled
with ANE off; B enables ANE without changing the tuning values. 

Arithmetic means of three runs per mode:

| Prompt | ANE | Prefill (tok/s) | Generation (tok/s) | Total time (s) | Peak memory (GB) |
|---|---|---:|---:|---:|---:|
| 4097 | Off | 513.9 | 35.9 | 11.568 | 18.35 |
| 4097 | On | 454.1 | 36.3 | 12.669 | 22.18 |
| 8193 | Off | 493.7 | 39.6 | 19.845 | 19.24 |
| 8193 | On | 536.5 | 35.1 | 18.934 | 23.07 |

<details>
<summary>Individual measured runs</summary>

| Run | Prompt | Prefill (tok/s) | Generation (tok/s) | Total time (s) | Peak memory (GB) |
|---|---:|---:|---:|---:|---:|
| A1 | 4097 | 511.9 | 32.0 | 12.001 | 18.35 |
| A1 | 8193 | 488.0 | 40.0 | 19.998 | 19.24 |
| B1 | 4097 | 481.2 | 35.1 | 12.156 | 22.18 |
| B1 | 8193 | 533.4 | 34.0 | 19.121 | 23.08 |
| A2 | 4097 | 514.9 | 38.8 | 11.259 | 18.35 |
| A2 | 8193 | 497.0 | 42.1 | 19.522 | 19.24 |
| B2 | 4097 | 384.6 | 38.0 | 14.024 | 22.17 |
| B2 | 8193 | 537.2 | 33.2 | 19.101 | 23.07 |
| A3 | 4097 | 514.8 | 36.8 | 11.443 | 18.35 |
| A3 | 8193 | 496.1 | 36.7 | 20.014 | 19.24 |
| B3 | 4097 | 496.5 | 35.8 | 11.828 | 22.18 |
| B3 | 8193 | 538.8 | 38.1 | 18.580 | 23.05 |

</details>

## Conclusion

- **8K:** mean prefill throughput increased **8.7%**. Generation throughput
  decreased **11.4%**, leaving a **4.6% reduction in total request time**
  (19.845 to 18.934 seconds).
- **4K:** mean prefill throughput decreased **11.6%**, and total request time
  increased **9.5%**. B2 was particularly slow – its cause remains unproven.
  The other two ANE-on runs also had lower prefill throughput than their
  paired GPU baselines.
- **Memory:** ANE added about **3.83 GB** of peak memory at both prompt lengths.

I do not know why performance differed or whether the benchmark setup
affected the results. These results need independent verification.
