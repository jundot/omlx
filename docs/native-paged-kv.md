# Native paged KV with tiered prefix caching

This opt-in integration uses the existing oMLX prefix index and RAM/SSD cache.
The active model owns a GPU page pool. Stored snapshots use the existing KVCache
schema; prefix hits are imported into the active pool before native generation.
Hybrid recurrent states keep the existing checkpoint boundary rules.

## Runtime requirements

The default dependency pins and ordinary serving path are unchanged. Enabling
native paging currently requires the ECO MLX-LM runtime with PagedKVCacheManager,
cache_factory and admission_controller hooks. Its upstream integration is pending.
A compatible paged-attention MLX primitive or ECO extension is required for the
native kernel path; gather mode uses existing SDPA. This is a review candidate,
not a standalone native-paging installation from the stock dependency lock.

With that runtime installed:

```sh
omlx serve --model-dir /path/to/models --native-paged-kv-pages 640
```

Keep the normal tiered cache enabled. `--no-cache` remains an explicit user choice.
`--native-paged-prefix-cache-pages` configures the experimental process-local
radix cache only when tiered caching is disabled.

## Validation and limits

Tests cover SSD round trips, hot caching, GDN sidecars, warm generation, cancellation,
pool teardown and reload, partial-allocation rollback, and snapshot stability after
page reuse. A Qwen3.6-35B-A3B NVFP4 run reused 2048 of 2224 input tokens and produced
identical output after a pool reload. This run used the ECO source MLX runtime.

Persistence gathers pages into portable tensors; restoration copies them into the
current pool. Decode continues to use page tables. This does not claim zero-copy
SSD restoration or shared physical pages across separate SSD-restored requests.

Before upstream acceptance, agree on a distributable runtime dependency and verify
its supported MLX versions. Existing packaged previews need rebuilding to include
this bridge.
