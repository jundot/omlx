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

`$OUT` must not exist yet. `stage-tables` is optional: it copies the three late tables to `$SSD` on the internal SSD. If you skip it, pass `--ssd-dir "$SRC"`. Keep `$SRC` and `$SSD` immutable after verification. `$RECEIPT` must be a new path in a private directory. Table files must be plain files: use `--local-dir`, not symlinked Hugging Face cache snapshots.

## Generate

```sh
MLX_ENABLE_TF32=0 python -m omlx.cli hy4-flash generate \
  --model "$OUT" --source "$SRC" --ssd-dir "$SSD" --receipt "$RECEIPT" \
  --prompt 'Write a Python function that merges sorted iterables lazily.' \
  --max-tokens 256 --prefill-chunk 36 \
  --first-engram pread --prepare-engrams --precast-lm-head --json
```

- **Memory:** all expert banks are loaded resident: about 156 GiB at 8 bits, or about 87 GiB at 4 bits (about 151 / 81 GiB with `--trunk-bits 8`), plus KV and scratch. The weights must fit in directly free memory: with too little, macOS pages GPU buffers and decode drops below 1 tok/s. Use `--first-engram pread` with 4-bit on 128 GB Macs. Run one model at a time on a machine with enough free unified memory. The command does not change wired-memory limits or swap.
- **`--first-engram`:** `mlock` (the default) maps and wires the first table and fails if it cannot. `pread` never maps it and reads rows on demand like the late tables. This saves 29.5 GiB of RAM at negligible I/O cost: under 1 MB per table per 128 decoded tokens.
- **`--prepare-engrams`:** starts each token's Engram row reads before the decoder layers.
- **`--precast-lm-head`:** keeps the FP32 output projection resident (+0.69 GiB) instead of casting it on every token. Logits are unchanged.
- **Output:** `--json` reports prefill time, decode steps and decode tok/s separately, along with token IDs, MLX peak memory and per-table Engram counters. `MLX_ENABLE_TF32=0` keeps the FP32 router and indexer math reproducible.

## Measured (M5 Ultra, 256 GiB, MLX 0.32.4)

Greedy decoding with the command above (`--first-engram pread`). The prompts are a 62-token coding request and a 74-token prose request.

| | Load | Prefill | Decode, 128 steps | Decode, 512 steps |
|---|---:|---:|---:|---:|
| Coding | 44 s | 0.35–1.5 s | 25.4–25.8 tok/s | 25.7–25.8 tok/s |
| Prose | | 0.44 s | 25.7–26.0 tok/s | |

- **Repeatability:** token IDs and FP32 logits are bit-identical across runs and match the `--first-engram mlock` reference.
- **Resources:** process RSS peaked at 167.6 GB, with no swap.

**4-bit (`--bits 4`)**, same prompts and flags, MLX peak 95.2 GB, no swap:

| | Decode | Prefill, 2K / 8K tokens | Top-1 agreement with 8-bit (teacher-forced) |
|---|---:|---:|---:|
| M5 Ultra, 256 GiB | 26.7–26.8 tok/s | 537 / 467 tok/s | 95.3–96.9% |
| M1 Ultra, 128 GiB | 13.8–14.5 tok/s | 126 / 115 tok/s | 94.6–96.1% |

On the M5 Ultra, 4-bit decodes only about 4% faster than 8-bit: the per-token cost there is mostly not expert-weight bandwidth. Its main benefit is that the model fits 128 GB machines.

**8-bit trunk (`--trunk-bits 8`)**, M5 Ultra, same prompts. A decode step is GPU-bound, and with a BF16 trunk about 11.8 GB of trunk weights were read per token next to the routed experts. Top-1 agreement is teacher-forced against the BF16-trunk 8-bit reference; no swap:

| | Decode, 128 steps (code / prose) | Decode, 512 steps | Prefill, 2K / 8K tokens | Top-1 agreement | MLX peak |
|---|---:|---:|---:|---:|---:|
| 8-bit experts, BF16 trunk | 25.4–26.0 tok/s | 25.7–25.8 tok/s | | 100% (reference) | 168.7 GB |
| 8-bit experts, 8-bit trunk | 29.3 / 29.4 tok/s | 28.9 tok/s | 339 / 308 tok/s | 97.7 / 96.9% | 163.2 GB |
| 4-bit experts, BF16 trunk | 26.8 / 26.7 tok/s | 26.7 tok/s | 537 / 467 tok/s | 96.9 / 95.3% | 95.2 GB |
| 4-bit experts, 8-bit trunk | 31.7 / 31.2 tok/s | 31.0 tok/s | 528 / 461 tok/s | 96.9 / 93.8% | 88.9 GB |

The `generate` command above on the 4-bit/8-bit-trunk artifact decoded 255 steps at 30.7 tok/s (MLX peak 87.6 GB). Lower trunk widths were tried in memory and rejected: 4-bit attention dropped prose agreement to 86%.

## Long context

Indexed layers switch to sparse recall once a sequence passes 2,048 positions (`compress_ratio` 4 × `group_topk` 512). This path is correct on real weights: on a 3,875-token prompt, the model returned a passphrase placed at the very start, which the sliding window cannot reach. It is not optimized yet. Prefill of that prompt took about 258 s, and per-token decode is far below the short-context rate. `--context-limit` defaults to 4096.

## Tests

```sh
python -m pytest -q tests/test_hy4_flash_*.py
```

The suite checks the full public geometry without loading weights, including exact MLX RTN packing and resident expert numerics at 4, 5, 6 and 8 bits. It compares iHC, MoE, attention and Engram outputs against separately written NumPy references; one Engram/FP8 case compares against PyTorch when it is installed. It also covers full-versus-incremental decoding, corrupted-artifact refusal, and real mmap/`mlock`/`pread` behaviour on tiny tables.

[Reference architecture (transformers)](https://github.com/huggingface/transformers/blob/c2ec850510a37978ea83df226589623d5f2be776/src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py)
