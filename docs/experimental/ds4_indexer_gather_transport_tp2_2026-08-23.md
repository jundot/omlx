# DS4 indexer TP2 transport gate — 2026-08-23

## Incident attribution

The failed long-context lifetime at 17:43:28 was reported by the patched JACCL
progress guard as `all_gather made no progress for 30001ms`; oMLX then marked
both endpoints unserviceable. The rank-local JSONL trace was not enabled for
that later request, so the native operation classification is the strongest
available evidence, not proof of the missing completion's lower-level cause.
The earlier captured seq=5808 failure was a separate one-element `all_sum`
after vocab transport and has already been replaced by point-to-point token
delivery.

`_gather_indexer_rows()` is the only DS4 serving path that issues 21 compact
`all_gather` operations per prompt chunk. For the production 1,024-token
chunk, each equal TP rank contributes `(512, 1, 512)` uint32 values (1 MiB).

## Physical gate

Hardware was the signed M3 Ultra rank 0 + M5 Max rank 1 JACCL/RDMA pair. The
probe ran the real 1 MiB/rank geometry for 2,058 iterations per round, equal to
21 sparse layers times `ceil(100000 / 1024)` prompt chunks. Three independent
launcher sessions each ran three measured rounds, alternating transport order.
Both ranks verified the complete reconstructed row tensor exactly.

| equal 512/512 rows | all-gather | strict ordered P2P | one-eval graph P2P |
|---|---:|---:|---:|
| session 1 median | 0.3430 s | 1.1542 s (3.365x) | 0.7938 s (2.314x) |
| session 2 median | 0.3442 s | 1.1508 s (3.343x) | 0.7912 s (2.299x) |
| session 3 median | 0.3424 s | 1.1452 s (3.344x) | 0.7952 s (2.322x) |

Across 18,522 measured exchanges per transport and rank, all variants were
exact and completed without a JACCL progress timeout. P2P therefore fails the
promotion limit of no more than 2% regression for equal production rows.

The previous 3:5 row experiment paid for all-gather padding to 640 rows on both
ranks. A secondary 384/640 probe (256 iterations, three rounds) found:

| weighted 384/640 rows | rank 0 | rank 1 |
|---|---:|---:|
| padded all-gather | 0.123786 s | 0.123784 s |
| strict ordered P2P | 0.147047 s (1.188x) | 0.147058 s (1.188x) |
| one-eval graph P2P | **0.110526 s (0.893x)** | **0.110429 s (0.892x)** |

The graph P2P route is about 10.8% faster than padded all-gather for the uneven
transport itself. This is not a whole-model promotion: weighted row compute
and transport must still clear a cold-prefill gate together.

## Shipping decision

- `OMLX_DSV4_INDEXER_GATHER_P2P=0` remains the rank default.
- Equal rows always use `all_gather`, even if the P2P experiment is enabled.
- P2P requires pure TP2, non-empty unequal rows, and both
  `OMLX_DSV4_WEIGHTED_INDEXER_ROWS=1` and
  `OMLX_DSV4_INDEXER_GATHER_P2P=1`.
- Non-TP2 worlds and every rollback case retain the original padded
  `all_gather` implementation.

Reproduce with:

```bash
mlx.launch --hostfile HOSTS.json --backend jaccl -- \
  python benchmarks/bench_ds4_indexer_gather_transport.py \
  --iterations 2058 --rounds 3
```

Use `--rank-rows 384,640` for the explicit weighted-row transport probe.
