# Hy4 Flash (oQh) text inference

`omlx hy4-flash` converts [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash/tree/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959) to a group-scaled oQh artifact (8-bit by default, or 4/5/6-bit to fit smaller Macs) and runs single-sequence text generation natively on Apple Silicon. It is a standalone command. It is not registered with the server, and it does not support vision, speculative decoding, continuous batching or prefix caching.

## Format

- **Experts:** all 82 routed banks (41 layers × gate/up and down) are stored as signed 256-point Walsh–Hadamard-rotated, group-128 affine weights, **8-bit** by default. `convert --bits 4|5|6` writes smaller banks with the same rotation and groups; the bit width is recorded per bank in `config.json` and the manifest, and the runtime reads it from there. Activations are rotated before `gather_qmm`. Quantization is deterministic RTN, not calibrated.
- **Other tensors:** the remaining 1,350 text tensors are copied byte-for-byte, in their source dtypes.
- **Optional 8-bit trunk (`convert --trunk-bits 8`):** also stores the 419 dense trunk linears as stock MLX group-64 affine RTN (`nn.QuantizedLinear` layout: `weight`, `scales`, `biases`). These are the attention projections (`q_a`, `q_b`, `kv_a`, `linear_gate`, `o_proj`), the shared, dense and latent MLP projections, and the LM head. Batch-1 decode reads all of them on every token. `kv_b_proj` (raw weight in the absorbed MLA path), the router, the sparse indexer, Engram, iHC and the embeddings stay byte-exact. The module list is recorded in `config.json` (`trunk_quantization`) and the manifest; the loader quantizes exactly those modules before loading. The LM head keeps FP32 activations. Without the flag the trunk stays BF16 and the output is unchanged.
- **Engram tables:** the four trained BF16 tables (layers 1, 11, 21 and 31) stay external and unchanged. Each run authenticates them against a small private receipt. Only the requested 192-byte rows are read.
- **Size:** the core is 155.75 GiB in 141 shards at 8 bits and 86.57 GiB at 4 bits (5-bit about 104 GiB, 6-bit about 121 GiB). `--trunk-bits 8` saves another 5.5 GB (150.60 / 81.41 GiB). The first Engram table is 29.5 GiB; the three late tables total 88.5 GiB. They work from any local volume, but row reads are fastest on the internal SSD; other volumes get a warning.

## Convert, stage and verify

```sh
REV=d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959
hf download tencent/Hy4-flash --revision "$REV" --local-dir "$SRC"
python -m omlx.cli hy4-flash convert --source "$SRC" --output "$OUT" --chunk-mib 32 --trunk-bits 8  # add --bits 4 for 128 GB Macs
python -m omlx.cli hy4-flash inspect --model "$OUT" --verify-core
umask 077
python -m omlx.cli hy4-flash stage-tables --model "$OUT" --source "$SRC" --ssd-dir "$SSD"
python -m omlx.cli hy4-flash verify-tables --model "$OUT" --source "$SRC" --ssd-dir "$SSD" --receipt "$RECEIPT"
```

`$OUT` must not exist yet. `stage-tables` is optional: it copies the three late tables to `$SSD` on the internal SSD. If you skip it, pass `--ssd-dir "$SRC"`. `inspect` validates the asset hashes, core shard sizes, quantization metadata and external table locators. Add `--verify-core` to hash the core payloads. Keep `$SRC` and `$SSD` immutable after verification. `$RECEIPT` must be a new path in a private directory. Table files must be plain files: use `--local-dir`, not symlinked Hugging Face cache snapshots.

## Generate

```sh
MLX_ENABLE_TF32=0 python -m omlx.cli hy4-flash generate \
  --model "$OUT" --source "$SRC" --ssd-dir "$SSD" --receipt "$RECEIPT" \
  --prompt 'Write a Python function that merges sorted iterables lazily.' \
  --max-tokens 256 --prefill-chunk 36 \
  --first-engram pread --prepare-engrams --precast-lm-head --json
```

- **Memory:** all expert banks are loaded resident: about 156 GiB at 8 bits, or about 87 GiB at 4 bits (about 151 / 81 GiB with `--trunk-bits 8`), plus KV and scratch. The weights must fit in directly free memory: with too little, macOS pages GPU buffers and decode drops below 1 tok/s. Use `--first-engram pread` with 4-bit on 128 GB Macs. Run one model at a time on a machine with enough free unified memory. The command does not change wired-memory limits or swap.
- **`--first-engram`:** `pread` (the default) never maps the first table; it reads rows on demand like the late tables. Explicit `mlock` adds about 29.5 GiB of wired memory on top of core weights, KV, scratch and OS headroom. The CLI validates the table receipt, identities and headers before loading the core, and does not fall back if wiring fails.
- **`--prepare-engrams`:** starts each token's Engram row reads before the decoder layers.
- **`--precast-lm-head`:** precasts the BF16 output projection to FP32 once (+0.69 GiB). With an 8-bit trunk, the head already uses FP32 activations and the flag adds no resident memory.
- **Output:** `--json` reports prefill time, decode steps and decode tok/s separately, along with token IDs, MLX peak memory and per-table Engram counters. `MLX_ENABLE_TF32=0` keeps the FP32 router and indexer math reproducible.

## Historical measurements (MLX 0.32.4)

The following are single-stream **greedy decode** measurements from an earlier local profiling harness, not the CLI command above or a matched kernel A/B. The harness used its own greedy loop and a process-local MLX wired-memory limit. Neither profile nor the separate CLI result below is pinned to this candidate revision. All four Engram tables used positional reads; none was wired by the provider.

| Artifact (M5 Ultra, 256 GiB) | Coding / prose, 128 decode steps | Coding, 512 decode steps |
|---|---:|---:|
| 4-bit experts, BF16 trunk | 26.86 / 26.83 tok/s | 26.72 tok/s |
| 4-bit experts, packed 8-bit trunk | 30.87 / 30.61 tok/s | 30.11 tok/s |
| 4-bit experts, packed 8-bit trunk (instrumented repeat) | 30.17 / 30.17 tok/s | 29.97 tok/s |

The public 4-bit model repository currently hosts the packed 8-bit trunk variant. The instrumented repeat took 5.055 / 4.646 / 17.400 s for prefill plus decode, versus 5.063 / 5.067 / 19.370 s for the BF16 trunk (load excluded). Slower prefill nearly erased the short coding gain. Against an oQh8 reference continuation, teacher-forced top-1 agreement stayed at 96.90% for coding and fell from 95.35% to 93.80% for prose. That is not free-run equality. Boundary samples found no open or mapped expert shard files in the instrumented run; they do not prove continuous residency or zero expert I/O.

A separate historical CLI smoke used a different, 36-token prompt with the packed 4-bit artifact: 0.641 s prefill, 255 decode steps at 30.74 tok/s, 23.03 s setup. It is not part of the profile comparison.

## Long context

Indexed layers switch to sparse recall once a sequence passes 2,048 positions (`compress_ratio` 4 × `group_topk` 512). This path is correct on real weights: on a 3,875-token prompt, the model returned a passphrase placed at the very start, which the sliding window cannot reach. It is not optimized yet. Prefill of that prompt took about 258 s, and per-token decode is far below the short-context rate. `--context-limit` defaults to 4096.

## Tests

```sh
python -m pytest -q tests/test_hy4_flash_*.py
```

The suite checks the full public geometry without loading weights, including exact MLX RTN packing and resident expert numerics at 4, 5, 6 and 8 bits. It compares iHC, MoE, attention and Engram outputs against separately written NumPy references; one Engram/FP8 case compares against PyTorch when it is installed. It also covers full-versus-incremental decoding, corrupted-artifact refusal, and real mmap/`mlock`/`pread` behaviour on tiny tables.

[Reference architecture (transformers)](https://github.com/huggingface/transformers/blob/c2ec850510a37978ea83df226589623d5f2be776/src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py)
