# DeepSeek-V4 ANE/CPU/GPU Prefill (Experimental)

This source-build experiment extends the Qwen ANE prompt-processing runtime
to DeepSeek-V4-Flash. For prompt chunks that exactly match the configured
fixed shape, three dense projections run as a hybrid split: an INT8 channel
prefix on both ANEs, an optional FP16 CPU middle, and an affine-q8 GPU suffix
requantized from the mxfp8 checkpoint weights. The CPU branch uses the same
performance-aware shared-resource scheduler as Qwen3.5.

Accelerated per layer:

- The shared-expert gate/up pair (50% of channels on ANE; the rest and the
  down projection stay on GPU). CPU sharing is deliberately disabled here.
- The attention `wq_b` query projection (50% on ANE, 12.5% on CPU).
- On sparse-attention layers, the indexer `wq_b` is stacked into the same
  procedure as the attention `wq_b`, since both consume the same
  `q_residual` input. The combined projection uses the same 12.5% CPU share
  and adds no extra dispatches.

Routed experts, `wo_a`/`wo_b`, the attention core, decode, and DSpark
verification keep the existing GPU path. `wo_b` was measured as a net loss
in-model and is excluded; `wo_a` needs a grouped GPU suffix and remains a
follow-up. The bank ladder and fixed-shape eligibility are shared with the
Qwen implementation. Unlike the Qwen path, which runs its GPU suffix on NAX
qmm kernels since the split tuner landed, the DeepSeek hybrid still skips
itself on NAX GPUs (the M5 family): its much smaller per-operation work is
untested against the tensor units, and `OMLX_QWEN35_ANE_PREFILL=1` forces
the path on for benchmarking there.

## Settings

```json
{
  "deepseek_ane_prefill_enabled": true,
  "deepseek_ane_prefill_sequence_length": 4096,
  "deepseek_ane_prefill_cpu_enabled": true,
  "deepseek_ane_prefill_cpu_fraction": 0.125,
  "deepseek_ane_prefill_cpu_threads": 12,
  "deepseek_ane_prefill_cpu_shared_resource": true
}
```

The controls are exposed in the web per-model settings editor for detected
DeepSeek-V4 models.

CPU sharing is applied only to plain and stacked `wq_b`. The 12.5% share and
12-worker setting came from repeated 4,096-token shape probes on the reference
M3 Ultra. If the native shared-resource scheduling API is unavailable, the
runtime keeps ANE/GPU offload active but disables the CPU middle instead of
silently using a slower scheduling policy.

The 4,096-token shape is the measured optimum: at 2,048 the per-operation
host synchronization cost of the hybrid primitive exceeds the savings,
because DeepSeek-V4 projections are five to ten times smaller than the Qwen
MLP slices the runtime was built around. Enabling the feature also realigns
the paged cache block size to the fixed ANE shape (pooling-cache models
clamp prefill chunks to block boundaries), which rebuilds the model's SSD
cache once and doubles the spacing of boundary snapshots.

## Measured results

### CPU-sharing microbench

M3 Ultra, 4,096 BF16 input rows, affine-q8 GPU suffix, median of 15 timed
iterations after three warmups. The repeated query results below compare the
same 50% dual-ANE split with and without a 12.5% CPU middle:

| Projection | ANE/GPU | ANE/CPU/GPU | CPU change | Decision |
|---|---:|---:|---:|---|
| Shared gate/up | 4.64 ms | 5.25 ms | **-13.1%** | Keep CPU off |
| Attention `wq_b` | 8.90 ms | 8.34 ms | **+6.2%** | Enable CPU |
| Stacked attention/indexer `wq_b` | 10.99 ms | 9.98 ms | **+9.2%** | Enable CPU |
| `wo_b` | 14.23 ms | ~14.57 ms | **-2.4%** | Keep offload off |

The stacked-query result was repeated three times with shared scheduling; its
12.5% CPU median ranged from 9.44 to 10.37 ms versus 10.97 to 11.05 ms without
CPU sharing. Smaller 10% splits helped less, while 15% became CPU-bound.

Run the safe synthetic sweep with:

```bash
python benchmarks/dsv4_ane_shape_bench.py --sequence-length 4096 \
  --ane-fractions 0.5 --cpu-fractions 0,0.1,0.125,0.15 --cpu-threads 12
```

The full checkpoint has been observed above 100 GiB and does not safely fit a
96 GiB machine. Full-model validation is therefore explicitly opt-in and the
benchmark refuses to load weights without the acknowledgement flag:

```bash
python benchmarks/dsv4_ane_model_bench.py /path/to/model \
  --allow-large-model
```

On a 96 GiB host, use the shape benchmark and unit tests; reserve the opt-in
full-model run for a larger-memory system.

### End-to-end ANE/GPU baseline

M3 Ultra, built-in throughput benchmark (`code_python` context, TG=128,
greedy), fresh server and cleared SSD cache per configuration.
`DeepSeek-V4-Flash-0731-oQ2.5e-mtp`:

| MTP | Prompt | pp ANE off | pp ANE on | pp change | tg ANE off | tg ANE on |
|:---:|---:|---:|---:|---:|---:|---:|
| Off | 4k | 621.9 | 637.7 | **+2.5%** | 26.2 | 26.0 |
| Off | 16k | 600.4 | 638.3 | **+6.3%** | 25.0 | 25.1 |
| Off | 32k | 583.3 | 627.1 | **+7.5%** | 24.1 | 24.2 |
| On | 4k | 604.2 | 633.2 | **+4.8%** | 40.7 | 39.0 |
| On | 16k | 583.3 | 621.6 | **+6.6%** | 46.4 | 44.9 |
| On | 32k | 567.0 | 608.4 | **+7.3%** | 42.1 | 42.0 |

`DeepSeek-V4-Flash-0731` (original fp8 precision):

| MTP | Prompt | pp ANE off | pp ANE on | pp change | tg ANE off | tg ANE on |
|:---:|---:|---:|---:|---:|---:|---:|
| Off | 4k | 636.0 | 650.7 | **+2.3%** | 25.1 | 24.6 |
| Off | 16k | 613.4 | 634.8 | **+3.5%** | 23.9 | 23.7 |
| Off | 32k | 595.8 | 621.7 | **+4.3%** | 23.3 | 23.0 |
| On | 4k | 617.4 | 631.5 | **+2.3%** | 35.0 | 38.3 |
| On | 16k | 595.7 | 617.1 | **+3.6%** | 41.8 | 44.4 |
| On | 32k | 578.6 | 603.9 | **+4.4%** | 42.4 | 44.5 |

A nominal 4k request prefills 4,095 tokens, one short of the ANE shape, so
its gain comes from the block realignment collapsing prefill into a single
chunk rather than from ANE execution; the ANE contributes from 16k up.
DSpark decode throughput swings a few tokens per second run to run with
the generated content, and repeated runs showed no systematic decode
regression; non-speculative decode stays within about one percent. The
smaller fp8 gains are not a coverage difference: both checkpoints engage
the identical 86 procedures with numerically identical accelerated
tensors, and the mxfp4 expert path simply streams about 1.6x the weight
bytes, competing with the ANE for unified-memory bandwidth.

Costs on the oQ2.5e checkpoint before CPU sharing: model memory 100.60 GB to
101.29 GB (+0.7 GB), eager load 3.95 s to 8.7 s (+4.8 s for the procedure
banks). The selected FP16 query slices add roughly another 0.4 GiB, reinforcing
why the full-model test is opt-in on memory-constrained hosts.
Enabling the feature realigns the paged cache block to 4,096, which
rebuilds the model's SSD cache once; a 4,096-block store/restore round
trip reproduced the cold output bitwise. A long-context spot check
(needle retrieval at seven depths plus multi-fact aggregation) scored
10/10 on both the GPU and ANE paths.

Like the Qwen variant this is an approximate path, not bit-exact inference:
outputs are deterministic run-to-run, and matched the GPU output bitwise on
some prompts while diverging in chain-of-thought wording on others. Both
ANEs stayed pinned through two split procedure banks; the compiler rejected
the monolithic bank for this model and the load ladder recovered
automatically.
