# MLX runtime tuning for MTP decode

The oMLX server applies a throughput-oriented Metal policy before importing
`mlx.core`:

- `MLX_METAL_FAST_SYNCH=1` reduces the CPU/GPU synchronization latency paid by
  speculative verify cycles.
- On hosts with at least 64 GiB of physical memory,
  `MLX_MAX_MB_PER_BUFFER=512` and `MLX_MAX_OPS_PER_BUFFER=100` allow a longer MTP
  chain to remain in each command buffer.

On Qwen3.8-27B oQ4e MTP on M3 Ultra, the combined policy increased the official
code benchmark median from 75.6 to 85.6 generated tokens/s. The larger command
buffers trade some prefill and high-concurrency scheduling latency for singleton
decode throughput, so they are memory-gated.

## Overrides

Operator-provided MLX variables always win because oMLX uses `setdefault`.
For example:

```bash
MLX_MAX_MB_PER_BUFFER=320 MLX_MAX_OPS_PER_BUFFER=50 omlx serve
```

Disable all oMLX runtime defaults with:

```bash
OMLX_MLX_RUNTIME_TUNING=0 omlx serve
```

These variables must be set before MLX initializes. Changing them after the
server imports an engine module is not supported.
