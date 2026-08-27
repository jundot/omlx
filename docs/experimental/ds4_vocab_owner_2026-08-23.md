# DS4 TP2 vocabulary-owner gate

Date: 2026-08-23. Model: `DeepSeek-V4-Flash-0731-MXFP4-MLX`, BF16
vocabulary head `[129280,4096]`. This was an isolated real-weight comparison;
no serving or deployment route was retained.

## Result

Moving the complete vocabulary head to rank zero/M3 is rejected. Reconstructing
the two current row shards was bitwise equal to the full projection on both
Macs, with logits SHA-256
`00a006dcb3363b6a35aaec962c7f1d1d552b6021fdcf3a3dbbc622c06366b635`
and argmax token `128816`. The ownership transformation is therefore exact,
but its physical critical stage is slower.

| Device | Full head | First half | Second half |
|---|---:|---:|---:|
| M3 Ultra | 1.654 ms | 0.973 ms | 0.977 ms |
| M5 Max | 2.142 ms | 1.318 ms | 1.313 ms |

The current TP2 critical half is the M5 second half at 1.313 ms. A rank-zero
owner would replace it with the M3 full head at 1.654 ms, a 26.0% regression
before accounting for the current shard transfer (only about 0.07 ms at the
measured fabric rate). It cannot provide a 10% end-to-end decode gain or close
the 31.2 ms/token backbone budget. Equal vocabulary sharding remains the
qualified layout.

The temporary owner-only loader/runtime seam and its tests were removed after
this gate. Reproduce independently on either Mac with:

```bash
python benchmarks/bench_ds4_vocab_owner.py \
  --model /path/to/DeepSeek-V4-Flash-0731-MXFP4-MLX \
  --warmup 8 --cycles 60
```
