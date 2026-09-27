# Native Qwen4-Exp packed EXL3 experts

Opt in with `OMLX_EXL3_ENABLED=1`. Supported checkpoints declare
`expert_quant` with format `exl3`, codebook `mcg`, output scales `svh`,
window 8–16 and an even halfword tile rate from 32 to 64. Initial validation
covers the published Qwen3.8-Flash-Next-Sushi-2.6bpw pack (k=2.625/window=15).
Other architectures/codebooks are explicitly unsupported. The published one-layer
packed MTP head is supported through the existing Lightning draft/verify path.

The custom loader validates packed tensor headers before allocating weights,
uses the vendored native Qwen4-Exp model, and replaces only its routed
SwitchGLU projections. Packed U16 trellises remain compressed in memory.
Native attention, vision, tool parsing, scheduler, KV cache, prefix cache and
model unload continue through oMLX. Checkpoint Python is never imported.
MTP remains opt-in through the per-model Lightning MTP setting. Plain loads
drop the head, while MTP loads validate and install its packed expert banks
and retain the existing attention, cache transaction and rollback implementation.

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
- Actual oMLX API: 736-token request 3.40 seconds cold / 0.60 seconds warm,
  729 cached tokens; both answered correctly.
- Two reference images: identified red then blue correctly.
- Tool use: returned structured get_weather(city=London), not source text.
- Admin dashboard rendered; queued native unload completed after activity drained. Tiny native reader
  tests separately confirm exact rows across shard boundaries and idempotent
  resource cleanup.

Do not extrapolate small-prompt throughput or peak memory to a 128K prompt.
Context admission and memory guards must stay enabled; long-context and
multi-request behavior need additional measurement on each target machine.

Run the numerical/storage tests with `python -m pytest tests/test_exl3_reference.py
 tests/test_flat_ngram_views.py`. `benchmarks/exl3_native_smoke.py MODEL_PATH`
is an explicit opt-in full-model smoke test; ensure adequate headroom first.

## Packed MTP

Enable **Lightning MTP** in the model settings (`mtp_enabled: true`), with
`OMLX_EXL3_ENABLED=1` still set. This uses the checkpoint's ordinary embedded
head; no altered model weights or correction adapters are required. No external
assistant model is loaded. `vlm_mtp_enabled` is the separate assistant drafter
and is not needed here.

For the M1 Max test, fixed depth one (`mtp_fixed_depth: 1`) was more consistent
than the existing adaptive depth up to three. This is a hardware/workload
recommendation, not a changed global default. Discovery reserves the head;
runtime admission/unload accounting excludes it when MTP is off. Reserve
approximately another 0.87 GiB of weights plus verification workspace when on.

Isolated oMLX API comparison, ordinary published 2.6bpw checkpoint, MLX 0.32.2,
M1 Max 64 GiB, temperature zero, 512-token configured context and 128-token
output cap. These are single-run wall times including prefill, after one load
warmup; they are not decode-only throughput or a long-context benchmark.

| Request | MTP off | Fixed depth one | Accepted drafts |
| --- | ---: | ---: | ---: |
| Interval-merge coding | 7.45 s | 5.50 s | 61/65 |
| Short conversational reaction | 3.07 s | 2.61 s | 13/23 |
| Bat-and-ball reasoning | 6.95 s | 5.18 s | 54/62 |
| LRU cache coding | 7.65 s | 5.71 s | 55/71 |
| Structured weather tool call | 3.77 s | 3.33 s | 12/12 |

Loaded MLX allocation was 43.44 GiB off / 44.31 GiB on. Rejected drafts were
observed and generation continued across subsequent requests. A blue reference
image was identified correctly. A 203-token prefix request took 2.03 s cold /
0.54 s warm, with 196 cached tokens. Unit regressions exercise QSA and PLE
partial rollback against sequential replay, cache restoration, head dimensions
and strict header checks.

**Parity limitation:** the sampled reasoning/chat responses and structured tool
arguments matched at depth one, but the two coding responses differed in typing
or wording. The semantic spot checks passed; byte-identical greedy output is
not established. Do not treat high acceptance as a parity proof or these small
requests as a guaranteed speedup. Adaptive depth up to three was mixed (coding
faster, short chat/tool requests slower). MTP remains an experimental opt-in.

`python benchmarks/exl3_mtp_api.py MODEL --label off --output off.json`,
then reload with MTP enabled and run with `--label depth1 --output on.json
--compare off.json`. Authentication uses `OMLX_API_KEY`; the harness never
changes services or model settings. It reports wall time and message equality;
read the server's `MTP[...]` logs to establish acceptance and actual drafting.
402 scoped loader, projection, cache, controller and pool regressions passed;
eight unrelated Moondream cases were excluded because the local environment
lacks that processor dependency.

The local harness retained >=4.5 GiB system headroom and a 1 GiB post-load
paging-growth cutoff, with other inference unloaded. An earlier adaptive test
was stopped by that paging cutoff; reducing configured context/cache allowed
both comparisons to finish. Keep normal server memory guards enabled.

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
