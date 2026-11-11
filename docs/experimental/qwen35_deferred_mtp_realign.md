# Qwen3.5/3.6/3.8 Deferred MTP Rollback Realignment (Experimental)

When two or more requests decode together with native MTP, their rows usually
accept different numbers of draft tokens. mlx-vlm's speculative rollback then
trims the batch KV cache to the longest row and rolls every lagging row over
the whole cache width, so all rows end at the same column again. At long
context that roll is paid on every full-attention layer of every cycle
(about 3.6 ms per layer at 24.7K + 100.7K tokens on an M5 Max), which a lone
request never pays.

With this option on, a ragged rollback records a per-row *tail* instead: each
row keeps its keys contiguous and ends `tail` columns before the shared end,
its next block is written right after its own last key, and the padded-row
attention reads each row only up to its own end. The cache is realigned (the
same roll as before, all tails at once) only when a tail would exceed 8
columns, or right before anything else uses the cache (row joins and
finishes, cache extraction for the prefix and SSD caches, any read of
`keys` / `values` outside the attention), so everything outside the steady
MTP loop sees the usual layout. Multi-token forwards keep tails only inside mlx-vlm's
speculative verifier; any other multi-token forward realigns first.

It is disabled by default.

## Requirements and limits

- Qwen3.5/3.6/3.8 models with native MTP, served with concurrency 2 or more.
  Single requests and MTP-off batches are unaffected.
- Builds on the padded-row attention patch
  (`omlx/patches/qwen35_padded_row_attention.py`); only the full-attention
  caches that patch serves defer realignment.
- Attention over a row with a tail is bit-identical to the same row after
  realignment (`tests/test_qwen35_deferred_realign.py`).

## Enable

Set the environment variable before starting the server:

```bash
OMLX_DEFER_REALIGN=1 omlx serve
```

The server log shows
`qwen3_5 deferred ragged-rollback realignment patch applied` at model load.
