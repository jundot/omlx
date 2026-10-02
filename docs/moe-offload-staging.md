# Optional expert offload staging

Build the optional transport extension, then enable it before starting oMLX:

```sh
OMLX_WITH_MOE_STAGING=1 python -m pip install -e .
OMLX_MOE_OFFLOAD_STAGING=1 omlx serve --model-dir /path/to/models
```

Staging is disabled by default. Unset `OMLX_MOE_OFFLOAD_STAGING`, or set it to
`0`, and reload the model to use the original bytes transport. Only the exact
value `1` enables it. The initial scope is offloaded `qwen4_exp` (Flash Next)
with MTP off; other families and resident MTP retain their current transport.
The C++ extension uses the project's pinned MLX/nanobind ABI and requires
CMake and Command Line Tools, without additional Metal shader compilation.
Missing extensions, failed pool initialization and unsupported layouts use
the bytes path. Read errors still propagate through the existing cleanup.

One model shares a fixed pool of 40 expert equivalents, initialized on the
first inference owner. MLX allocations rounded to 16 KiB pages must fit within
128 MiB. The measured Flash Next layout uses 360 buffers / 108.75 MiB. An
occupied pool falls back immediately to bytes, without waiting or worker-side
MLX allocation. Calls on another owner use bytes. A lease remains retained by
MLX's shared array Data until lazy graphs and GPU completion release it; its
C++ deleter never takes the Python GIL.

Each missing expert still issues nine independent parallel tensor reads in
the original IO executor. Prefetch window, serial fallback, resident-gather
barrier, install order, LRU, free slots, maps, counters, quantized computation,
and output stream are unchanged. Staging only replaces bytes-to-MLX copying
with a view of the filled shared buffer, including BF16 reinterpretation.

## Local benchmark evidence

The preceding benchmark transport experiment was rebased onto main
`5dcfe2430b73a86e871de13019c93e047f0aba9a`, including #4052. Five fresh-process,
uninstrumented pairs per length reversed stock/native order on alternating
pairs. These are one local series, separate from earlier measurements.

| Input tokens | Stock mean TPOT (ms) | Staged mean TPOT (ms) | Decode gain | Positive pairs |
| ---: | ---: | ---: | ---: | ---: |
| 1025 | 141.020 | 132.910 | +6.10% | 5/5 |
| 4097 | 125.282 | 120.152 | +4.27% | 5/5 |
| 8193 | 136.926 | 130.654 | +4.80% | 5/5 |

Decode gain is `mean(stock TPOT) / mean(staged TPOT) - 1`; throughput is
`1000 / mean(TPOT)`. Individual 8k gains ranged from +0.54% to +6.87%; all
measurements were retained. This is no population or significance claim.
Full-request process CPU increased by 0.7–2.3%; total CPU savings are not
established. The additional memory makes this an explicit opt-in.

M1 Max / 64 GiB, Flash Next oQ4e, 128/512 experts resident (25% resident,
75% streamed), R=1, mmap PLE, MTP/lookahead off, overlap=1, IO workers=12,
batch=48; identical app Python 3.11.10, MLX 0.32.2, model and native assets
in both arms. Max 128 tokens/EOS, prompt seed 1, eight-token warmup, fresh
KV and APFS checkpoint clones. 8k used standard 4096+4096 prefill.

#4052's new Flash Next fused attention and resident routed-expert/top-k-fold
paths were inactive in this workload: `applegpu_g13s` fails the attention
architecture gate, and offload fails the resident expert layout guard. The
existing fused router, R1 GDN and native long-prefill QSA remained active.
MTP-only unrolled/deferred verification was inactive. Separate exact GPU
selfchecks and forced mismatch tests verified the rolled fallback. The
observed gain therefore comes from the expert transport experiment.

Strict qualification compared emitted and successor probability bytes,
tokens, routes, read plans, nine-read task counts, install slots/order and
cache/LRU/maps, including stock-v-stock at 8k. No tolerances were weakened.
Production integration validation is recorded separately below.

## Production implementation: separate repeat

The production switch was tested directly in a clean-main checkout, with
the same pinned runtime and workload above. Only
`OMLX_MOE_OFFLOAD_STAGING=0/1` differed between the arms. This new series is
separate from the preceding benchmark transport experiment; results are not
pooled. Five fresh-process reversed pairs were run per length after strict
16-token pilots and full 128/EOS qualification.

| Input tokens | Stock mean TPOT (ms) | Staged mean TPOT (ms) | Decode gain | Positive pairs |
| ---: | ---: | ---: | ---: | ---: |
| 1025 | 142.708 | 132.688 | +7.55% | 5/5 |
| 4097 | 126.976 | 120.650 | +5.24% | 5/5 |
| 8193 | 138.894 | 132.224 | +5.04% | 5/5 |

| Pair | Arm order | 1025 gain | 4097 gain | 8193 gain |
| ---: | --- | ---: | ---: | ---: |
| 1 | stock/staged | +9.36% | +5.51% | +1.49% |
| 2 | staged/stock | +7.49% | +4.53% | +5.28% |
| 3 | stock/staged | +6.82% | +4.51% | +9.40% |
| 4 | staged/stock | +5.73% | +7.10% | +4.99% |
| 5 | stock/staged | +8.39% | +4.56% | +4.17% |

The 1k/4k prompt order also reversed on even pairs; 8k used its own series.
All 15 pairs were retained and positive. The mean benefit persisted, with
4k/8k near 5% and 1k higher. The spread, especially 8k's +1.49% to +9.40%,
does not establish a tight 4–6% band for individual pairs.

Production probability-byte, token, route, read-plan, nine-read task,
install-order and cache/LRU/map gates passed in both 1k/4k prompt orders and
against two stock processes at 8k. The 128/EOS runs retained 43/95/50 scored
decode inputs and covered all 48 layers. Every rate request matched its
same-order fulltrace token/cache/counter control. The native pool drained to
zero busy buffers after each request; staged plus fallback reads equaled
nine times the misses. Observer/timer hooks were absent from rate runs.

The production full CI selection gave 16,143 passed, 11 failed, 1,136
skipped and 84 deselected: exactly the clean-main failure set below, plus
30 passing new tests. All 30 staging tests also passed with app Python
3.11.10; the existing development environment uses Python 3.12.14. The
native extension and wheel/sdist were built and checked. Tests cover the
kill switch, missing helper, bounded fallback, all transport dtypes,
serial/parallel IO, exact output/cache/install state, read/partial-install
errors, lazy lease retention and asynchronous completion without a GIL
deadlock. No embedding/GDN source, comparison tolerance or test was changed.

## Existing main failures

On clean detached main `5dcfe243`, the full CI selection produced 16,113
passes, 11 failures, 1,136 skips and 84 deselections in the existing local
Python 3.12 environment. Ten failures are BF16 GDN verify/rollback comparisons
in `tests/test_qwen35_gdn_prework.py`. The remaining failure is
`tests/test_embedding.py::TestNativeEmbeddingLoading::test_embed_batches_long_inputs_in_input_order`.

The embedding target passes alone, but fails in the full embedding module
(105 passed / 1 failed) and its own class (10 passed / 1 failed), on clean
main without staging. This bounds the order-dependent reproducer to that
class; the underlying numerical cause is unproven. Staging does not modify
embedding code or loosen its comparison. The suite is not claimed green.

```sh
python -m pytest tests/test_embedding.py::TestNativeEmbeddingLoading -q
python -m pytest tests/ -m "not slow and not integration" --durations=50
```
