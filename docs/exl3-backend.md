# Native Qwen4-Exp packed EXL3 experts

Opt in with `OMLX_EXL3_ENABLED=1`. Supported checkpoints declare
`expert_quant` with format `exl3`, codebook `mcg`, output scales `svh`,
window 8–16 and an even halfword tile rate from 32 to 64. Initial validation
covers the published Qwen3.8-Flash-Next-Sushi-2.6bpw pack (k=2.625/window=15).
Other architectures/codebooks and packed MTP are explicitly unsupported.

The custom loader validates packed tensor headers before allocating weights,
uses the vendored native Qwen4-Exp model, and replaces only its routed
SwitchGLU projections. Packed U16 trellises remain compressed in memory.
Native attention, vision, tool parsing, scheduler, KV cache, prefix cache and
model unload continue through oMLX. Checkpoint Python is never imported.
Embedded MTP is disabled for this loader; this contribution makes no claim
about speculative decoding speed or parity.

## Kernels

A scaled normalized Hadamard kernel brackets each packed projection. Decode
uses a cooperative four-simdgroup tile kernel. Prefill uses sorted row windows
and simdgroup matrix multiplication, sharing unpacked tiles across matching
expert rows. Unsorted rows are sorted and inverse-permuted on the GPU.
Invalid expert ids are clamped before accessing scale/weight banks and produce
NaN outputs. Floating-point summation differs from a serial reference, so
numerical tests use explicit FP16-stage tolerances rather than bitwise claims.

## Disk-backed n-gram table

Flat `ngram_table.bin` files stamped `mlx-serve-ngram` expose virtual shards to
the existing DiskBackedShardedEmbedding reader. No 32 GB table copy or full GPU
upload is needed. The adapter validates metadata, quantization agreement,
geometry, dtypes, tensor bounds/overlap and model-directory containment.
Existing mmap, prefetch, row dequantization and close behavior are retained.

## Validation

Local hardware: M1 Max, 64 GiB. MLX 0.32.2. Load and API checks ran under a
separate process watchdog: >=49 GiB available before load, >=4.5 GiB headroom,
and no significant new swap. Other inference processes were unloaded first.

- Full native load: 10.86 seconds, 43.09 GiB active MLX memory.
- Short text decode before matrix-prefill addition: 18.7–20.9 tokens/sec.
- Cooperative real-expert projection: 0.493 ms; CPU oracle cosine approximately
  1.0, maximum absolute difference 3.05e-5 on the bounded fixture.
- Actual oMLX API: 736-token request 3.89 seconds cold / 0.60 seconds warm,
  729 cached tokens; both answered correctly.
- Two reference images: identified red then blue correctly.
- Tool use: returned structured get_weather(city=London), not source text.
- Admin dashboard rendered; native unload request accepted. Tiny native reader
  tests separately confirm exact rows across shard boundaries and idempotent
  resource cleanup. Completion of the queued API unload is a separate lifecycle
  gate, not implied by request acceptance.

Do not extrapolate small-prompt throughput or peak memory to a 128K prompt.
Context admission and memory guards must stay enabled; long-context and
multi-request behavior need additional measurement on each target machine.

Run the numerical/storage tests with `python -m pytest tests/test_exl3_reference.py
 tests/test_flat_ngram_views.py`. `benchmarks/exl3_native_smoke.py MODEL_PATH`
is an explicit opt-in full-model smoke test; ensure adequate headroom first.

## Provenance and licences

Format decoding, cooperative projection and matrix-prefill arithmetic adapted
from Sushi commit 27ca1c8684c01d1970bf576d8cef80ebea617928,
`src/expert_exl3.zig` / `src/expert_exl3_kernels.zig`.
Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu.
The [full MIT notice](licenses/sushi-MIT.txt) is retained. Sushi identifies these
as its own implementation of the ExLlamaV3 format. Other components from its
NOTICE are not incorporated by this change.

Downloaded model weights retain their own publisher licences and are not
redistributed here. The inspected pack is under Qwen Community License 1.0;
its additional commercial MaaS/work-assistant conditions are distinct from
Sushi's MIT source licence.
