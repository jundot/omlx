# DeepSeek-V4 ANE/CPU/GPU Prefill (Experimental)

This source-build experiment extends the Qwen ANE prompt-processing runtime
to DeepSeek-V4-Flash. Exact prompt chunks, plus configured profitable short
tails zero-padded to the compiled shape, run six dense projection groups as a
hybrid split: an INT8 channel
prefix on both ANEs, an optional FP16 CPU middle, and an affine-q8 GPU suffix
requantized from the mxfp8 checkpoint weights. The CPU branch uses the same
performance-aware shared-resource scheduler as Qwen3.5.

Accelerated per layer:

- The shared-expert gate/up pair (50% of channels on ANE) and its down
  projection (62.5% after alignment). CPU sharing is deliberately disabled
  for both.
- Every attention projection that directly consumes the residual stream is
  stacked into one dispatch: `wq_a`, `wkv`, both compressor projections when
  present, and the sparse indexer's compressor and weight projections when
  present. Local and ratio-128 layers use a 50% ANE share; the wider ratio-4
  sparse stack uses 56.25% after alignment. The remainder stays on the GPU,
  with no CPU middle.
- The attention `wq_b` query projection (50% on ANE, 12.5% on CPU).
- On sparse-attention layers, the indexer `wq_b` is stacked into the same
  procedure as the attention `wq_b`, since both consume the same
  `q_residual` input. The combined projection uses the same 12.5% CPU share
  and adds no extra dispatches.
- The eight-group attention-output `wo_a` projection (50% of each group's
  rows on ANE). A native grouped affine-q8 suffix preserves the compact
  per-group weights and overlaps GPU work with both ANE instances.

Routed experts, attention-output `wo_b`, the attention core, and decode keep
their existing paths. Embedded DSpark is supported natively: target prefill can
use ANE, while the armed 2–6-row verification window remains on its bundled
decode-consistent QMV, attention, ring, and top-k kernels and is never
ANE-padded. `wo_b` was measured as a net loss
in-model and is excluded. The bank ladder and fixed-shape eligibility are shared with the
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
  "deepseek_ane_prefill_tail_padding_min_tokens": 0,
  "deepseek_ane_prefill_down_enabled": true,
  "deepseek_ane_prefill_down_fraction": 0.65,
  "deepseek_ane_prefill_wo_a_enabled": true,
  "deepseek_ane_prefill_wo_a_fraction": 0.5,
  "deepseek_ane_prefill_cpu_enabled": true,
  "deepseek_ane_prefill_cpu_fraction": 0.125,
  "deepseek_ane_prefill_cpu_threads": 12,
  "deepseek_ane_prefill_cpu_shared_resource": true
}
```

The controls are exposed in the web per-model settings editor for detected
DeepSeek-V4 models. The projection-specific fields are also persisted when a
tuner recommendation is applied, allowing shared-down and grouped `wo_a` to
be retained, resized, or disabled without switching off the other projections.

The bounded tuner now includes the production shared-down shape and compact
eight-group `wo_a` shape in its synthetic search. Synthetic timing still
cannot reproduce contention with the full model. When full-model verification
is explicitly enabled, the tuner compares each projection's profiled output
completion time—including input packing and queue barriers—with its isolated
GPU baseline. It runs one profile-refined candidate, rebalances query CPU
sharing and surviving ANE splits, and disables a projection that no longer
clears the 1% threshold. The final recommendation is still selected by direct
end-to-end prompt throughput; if neither accelerated candidate clears 1%, the
tuner returns GPU-only.

When ANE prefill is active, engine startup also raises the scheduler's
effective prefill step to at least the compiled ANE sequence length. This is
required in addition to aligning cache-block boundaries: otherwise a smaller
prefill step can split every nominal 4,096-token block before it reaches the
fixed-shape ANE procedures. With the default configuration, serve logs should
report `effective_step=4096` and `chunk_tokens=4096` for each full chunk.

`deepseek_ane_prefill_tail_padding_min_tokens` ports Qwen's short-tail policy.
A value from 2 through 4,095 makes a single-prompt projection block at or above
that threshold zero-pad its token axis to 4,096, execute the compiled hybrid,
and slice the padded rows immediately. This applies independently to shared
gate/up and down, attention-input stacks, query projections, and grouped
`wo_a`; attention and cache updates never see padded tokens. Zero keeps the
feature disabled. The default remains zero until a DeepSeek full-model
crossover is measured; setting an overly low threshold can waste substantial
work on small non-DSpark calls, although one-token decode and armed DSpark
verification are always excluded.

The per-request `ane_full_tiles` trace counter is not a reliable activation
signal for DeepSeek yet: it reads a benchmark field currently populated only
by the Qwen settings path and can remain zero while ANE execution is active.
Use the serve log's `chunk_tokens` value to verify full-tile dispatch.

CPU sharing is applied only to plain and stacked `wq_b`. The 12.5% share and
12-worker setting came from repeated 4,096-token shape probes on the reference
M3 Ultra. If the native shared-resource scheduling API is unavailable, the
runtime keeps ANE/GPU offload active but disables the CPU middle instead of
silently using a slower scheduling policy.

CPU sharing is also disabled automatically when the configured fixed shape is
below 4,096 tokens. A current-revision 2,048-token recheck measured plain
`wq_b` at 4.77 ms with ANE/GPU and 5.41 ms with the 12.5% CPU middle; stacked
`wq_b` was 5.81 ms and 5.90 ms respectively. The smaller tile remains
available for ANE/GPU experiments, but it no longer silently enables a CPU
configuration that the shape probe found unprofitable.

When CPU sharing is active, pre-load admission reserves its eager FP16 query
rows from checkpoint geometry, including the wider stacked indexer slices.
Unreadable DeepSeek geometry fails closed with a conservative extra
model-sized reservation rather than admitting a near-limit checkpoint using
only its packed-weight estimate.

The 4,096-token shape is the measured full-model optimum. Individual ANE/GPU
projections can still beat GPU-only at 2,048, but the extra host synchronization
and smaller prompt chunks erase too much of that layer-level saving across the
model; adding CPU sharing makes it worse. DeepSeek-V4 projections are five to
ten times smaller than the Qwen MLP slices the runtime was built around.
Enabling the feature also realigns the paged cache block size to the fixed ANE
shape (pooling-cache models clamp prefill chunks to block boundaries), which
rebuilds the model's SSD cache once and doubles the spacing of boundary
snapshots.

## Measured results

### Attention-input stack microbench

M3 Ultra, 4,096 BF16 input rows, median of 15 timed iterations after three
warmups. These exact-shape synthetic results motivated the additional stacked
dispatch implemented after the end-to-end baseline below:

| Layer variant | Stacked output width | GPU mxfp8 | ANE/GPU | Speedup |
|---|---:|---:|---:|---:|
| Local | 1,536 | 3.427 ms | 2.287 ms | **1.50x** |
| Ratio-128 compressed | 2,560 | 5.482 ms | 3.322 ms | **1.65x** |
| Ratio-4 sparse | 4,160 | 8.617 ms | 4.621 ms | **1.86x** |

The first two use a 50% requested ANE fraction. The ratio-4 result uses 60%
requested, which becomes 56.25% after the per-instance 128-row alignment. A
full-model run of this newer stack has not yet been performed; the later
end-to-end tables describe the earlier 86-procedure path only.

Run only these shapes with:

```bash
python benchmarks/dsv4_ane_shape_bench.py --sequence-length 4096 \
  --shapes attention_input_local,attention_input_ratio128,attention_input_ratio4 \
  --ane-fractions 0.5,0.6 --cpu-fractions 0
```

### Shared down and grouped `wo_a` microbench

M3 Ultra, 4,096 BF16 rows, production hybrid suffixes:

| Projection | GPU mxfp8 | ANE/GPU | Speedup | Split |
|---|---:|---:|---:|---:|
| Shared-expert `down_proj` | 4.476 ms | 3.177 ms | **1.41x** | 62.5% ANE |
| Grouped attention `wo_a` | 17.232 ms | 12.205 ms | **1.41x** | 50% ANE |

The grouped suffix stores only the real per-group K-wide rows. It does not
construct a sparse block-diagonal matrix, which would multiply suffix memory
and GPU work. Across 43 layers these isolated timings imply approximately
272 ms less projection time per full tile; host synchronization and whole-model
memory contention can reduce the realized PP gain.

Run the two production shapes with:

```bash
python benchmarks/dsv4_ane_shape_bench.py --sequence-length 4096 \
  --shapes shared_down --ane-fractions 0.65 --cpu-fractions 0
python benchmarks/dsv4_ane_shape_bench.py --sequence-length 4096 --wo-a
```

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

A clean-worktree recheck on the current revision produced the same decisions.
Across four fresh processes, plain `wq_b` improved by about 1.5% at the
aggregate median with a 12.5% CPU middle (a small, noisy win), while stacked
`wq_b` improved by about 5.5%. The 15% share regressed both. Shared gate/up
again slowed with any CPU middle, and remains ANE/GPU-only.

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

### End-to-end ANE/GPU baseline (before attention-input stacking)

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
smaller fp8 gains are not a coverage difference: both checkpoints engaged
the identical 86 procedures used by that baseline with numerically identical accelerated
tensors, and the mxfp4 expert path simply streams about 1.6x the weight
bytes, competing with the ANE for unified-memory bandwidth.

The attention-input, shared-down, and grouped-`wo_a` work each add one
procedure per eligible layer (129 additional procedures for the reference
topology), taking the configured path from the measured 86-procedure baseline
to 215 procedures. Their
end-to-end throughput, load-time, and memory deltas remain to be measured on
a host that can safely load the checkpoint; the synthetic speedups above
must not be read as an equivalent whole-model PP gain.

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
