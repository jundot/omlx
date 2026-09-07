# Native paged KV with tiered prefix caching

This opt-in integration uses the existing oMLX prefix index and RAM/SSD cache.
The active model owns a GPU page pool with request page tables, reference counts,
copy-on-write and admission control. Snapshots use the existing KVCache schema;
prefix hits are imported into the active pool before generation. Hybrid recurrent
states keep the existing checkpoint boundary rules.

## Install the optional runtime

The normal installation retains the upstream MLX-LM pin. Native paging requires
additional BatchGenerator and model hooks supplied by the fixed MLX-LM revision
in `docs/native-runtime.txt`. This is an explicit dependency override for the
experimental native configuration, not a requirement for ordinary serving.

From this checkout, with Python 3.12, Xcode Metal tools and a C++ compiler:

```sh
uv venv --python 3.12 .venv
uv --no-config pip install --python .venv/bin/python --overrides docs/native-runtime.txt \
  . ./extensions/eco-paged-attention
.venv/bin/omlx serve --model-dir /path/to/models --native-paged-kv-pages 640
```

`--no-config` prevents the ordinary `[tool.uv]` MLX-LM override from competing
with the explicit native runtime revision.

The bundled extension builds against official MLX 0.32.2. Its C++ bridge depends
on the MLX ABI, so upgrading MLX requires rebuilding and requalification. See
[the extension contract](../extensions/eco-paged-attention/README.md).

Keep the normal tiered cache enabled. `--no-cache` remains an explicit user choice.
`--native-paged-prefix-cache-pages` configures the process-local radix cache only
when tiered caching is disabled. GPU page size and persistent block size remain
independent; hybrid SSD checkpoints retain their existing alignment.

## Optional NVFP4 prefill computation

Set `OMLX_NVFP4_PREFILL=1` before loading a text engine to enable transient BF16
MLP projections for the qualified dense Qwen 27B layout: 64 layers, hidden size
5120, intermediate size 17408, and NVFP4 gate/up/down projections. Each loaded
model is modified independently. Global MLX methods and stored weights retain
their original definitions and data.

For BF16 inputs with at least 1024 matrix rows, each eligible projection expands
its packed weight, performs a dense matrix multiplication, and evaluates its
output before the next projection. The final layer MLP is left unchanged so
cache-only prefill can omit its work. Short inputs, decode and other layouts use
the original quantized computation. This option works independently of paging.

A controlled M4 Pro 48 GB experiment with Qwen3.8-27B NVFP4, batch 1 and 2049
input tokens measured 6.85% (English) and 7.33% (Chinese) lower prefill-plus-last-
token latency. Four symmetric blocks per input gave 95% intervals of
4.55–9.09% and 6.86–7.80%; first-token logits matched exactly. These are warm
model/allocator direct-model measurements, not HTTP throughput results. Larger
batch trials encountered swapping and power-setting changes and are exploratory.

## Validation and limits

Tests cover SSD round trips, hot caching, GDN sidecars, cancellation, pool teardown
and reload, partial-allocation rollback, and snapshot stability after page reuse.
The prefill tests cover opt-in behavior, per-model isolation, unchanged decode,
parameter identity, idempotence and unsupported-model fallback.

Persistence gathers pages into portable tensors; restoration copies them into
the active pool. Decode uses page tables. Separate SSD-restored requests receive
independent physical pages. Full application restart, sustained HTTP load and
hardware beyond the measured M4 Pro require further qualification. The optional
MLX-LM dependency remains a maintained fork while upstream interface work is paused.

For a short direct-model reproduction (two warmups followed by one ABBA block):

```sh
.venv/bin/python benchmarks/nvfp4_prefill.py --model /path/to/Qwen3.8-27B-nvfp4
```

Use a fixed power mode, stable AC supply and idle GPU; record system swapping.
One block is exploratory and does not reproduce the confidence intervals above.
