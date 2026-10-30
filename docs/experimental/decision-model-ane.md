# Clef-Flash Local Quantization and ANE/GPU Serving (Experimental)

Clef-Flash can run locally through `POST /v1/systemone`. Quantizing its
backbone reduces the memory needed to load it. Experimental ANE/GPU inference
can then move eligible projection work off the GPU.

`clef-flash-oQ4` is a **locally generated checkpoint name**, not an upstream
Cloudflare release. It comes from quantizing
[`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) with oMLX's
existing oQ writer. The original model and its trained decision head are the
source; no decision-model training was performed for this conversion.

## What quantization changes

oQ4 uses mixed precision rather than making every tensor 4-bit. The writer
allocates extra precision using measured layer sensitivity. It copies the joint
head and its configuration unchanged beside the quantized backbone shards.
The conversion below also retains the vision tower and processor assets.

One local conversion produced 5.846 GiB of safetensors and a reported backbone
plan of 4.70 bits per weight. A server estimated the resulting model at 6.14 GiB
and reported 6.24 GiB on load, versus an 18.64 GiB estimate for the original.
Sizes depend on the checkpoint, quantizer version and options.

ANE shares unified memory with the CPU and GPU. This hybrid backend retains GPU
weights and adds ANE programs, weight slices and I/O buffers. ANE does not
provide a separate memory pool that makes an oversized checkpoint fit.

## Create the local checkpoint

Use an oMLX environment with the repository's pinned dependencies. For native
inference, build with `make install`, or rebuild an existing installation with
`make kernels`. See [custom kernels](../custom-kernels.md).

Download the source:

```bash
hf download Cloudflare/clef-flash --local-dir ~/.omlx/models/clef-flash
```

Convert it into a separate directory:

```bash
python - <<'PY'
from pathlib import Path
from omlx.oq import quantize_oq_streaming

models = Path.home() / ".omlx/models"
quantize_oq_streaming(
    str(models / "clef-flash"),
    str(models / "clef-flash-oQ4"),
    oq_level=4,
    group_size=64,
)
PY
```

The destination must not already exist. The source is retained. Conversion can
also be started through **Models → oQ Quantization**.

When the original exceeds the live calibration budget, oQ automatically builds
a temporary quantized proxy for sensitivity measurement. It can therefore
quantize the model without first loading the entire full-precision backbone.
The observed conversion used 128 calibration samples of 256 tokens and a
temporary proxy of approximately 5.5 GiB. It used ordinary oQ4, not oQ4e.

See [oQ quantization](../oQ_Quantization.md) for enhanced quantization and
memory requirements. Calibration needs additional memory and temporary disk
space beyond the final checkpoint.

## Verify GPU serving first

```bash
omlx serve --model-dir ~/.omlx/models --host 127.0.0.1 --port 8000
```

Use the new directory name explicitly in requests. The original model's ID
still selects its original full-precision checkpoint.

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -d '{
    "model": "clef-flash-oQ4",
    "truncate": false,
    "state": "Invoice status is overdue. Total is 1250 dollars. Please refund the duplicate payment today.",
    "questions": {
      "status": {
        "type": "choice",
        "instructions": "What is the invoice status?",
        "criteria": {"paid": "Paid", "overdue": "Overdue", "draft": "Draft"}
      },
      "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for a refund?"
      },
      "size": {
        "type": "score",
        "instructions": "Which amount range contains the invoice total?",
        "criteria": ["Below 1000", "1000 to 2000", "Above 2000"]
      }
    }
  }'
```

Set `OMLX_API_KEY` to the configured inference key. Omit the authorization header
if authentication is disabled. The smoke request should select `overdue`, give
the refund proposition a probability above 0.5, and put most score probability
on level 1. The response has zero generated output tokens.

The current Clef implementation limits the complete prompt to 16,384 tokens.
With `truncate: false`, oversized inputs produce an error rather than dropping
state content. Images require a retained vision tower; video is not currently
supported by the oMLX Clef implementation.

## What ANE offloads

The first measured configuration moves 25% of eligible MLP gate/up output
channels to a single ANE. The GPU computes the remaining channels and the rest
of the model, including the decision head. This reduces GPU projection work;
it does not make the complete decision run independently of the GPU.

Use the single-ANE configuration on M3 Pro. Do not copy the M3 Ultra dual-ANE
instance-pinning configuration onto another topology.

ANE controls use model-neutral `ane_prefill_*` names. The model ID selects
which record applies: changing `models["clef-flash-oQ4"]` does not change a
Qwen model's record. `ane_prefill_cpu_enabled` controls CPU sharing within
this hybrid prefill path.

For example, the per-model settings file has this structure:

```json
{
  "version": 1,
  "models": {
    "clef-flash-oQ4": {
      "ane_prefill_enabled": true,
      "ane_prefill_sequence_length": 1024,
      "ane_prefill_fraction": 0.25,
      "ane_prefill_max_layers": 32,
      "ane_prefill_dual_ane": false,
      "ane_prefill_gdn": false,
      "ane_prefill_cpu_enabled": false,
      "ane_prefill_fused_down": false,
      "ane_prefill_tail_padding_min_tokens": 0
    }
  }
}
```

Existing `qwen35_ane_prefill_*`, `clef_flash_ane_*` and `clef_flash_cpu_*` keys
are accepted on input and normalized to the neutral names. Explicit neutral
keys win, including `false` and zero. If an older record contains both legacy
forms, explicit Clef values take precedence over Qwen values. Saved settings,
profiles and current API responses use neutral keys; update Python scripts to
use the new attributes shown below. This requires the modified source build;
an older Homebrew release does not support the new names.

These settings now reach `DecisionEngine` through the model pool. Banks compile
on the MLX executor when the model loads and are released when it unloads. The
pool reserves additional ANE memory before loading. Before each inference step,
the engine includes driver-wired weight bytes in its live headroom check; if
they no longer fit, it releases the banks and uses GPU until the next reload.

The initial measurements below predate this integration and enabled the backend
directly. Use the following tests to check the integrated path.

## Reproduce the ANE test

Unload other models first, or stop the running server, to avoid loading two
copies of the checkpoint. From the checkout:

```bash
uv run python -m scripts.benchmark_decision_ane ~/.omlx/models/clef-flash-oQ4
```

The script loads the model through `DecisionEngine`, first with ANE disabled,
then with the single-ANE settings above. It checks synthetic answers for all
three question types, reports body time, prompt throughput and speedup against
GPU in Markdown tables, and checks
native operation counters. It fails if the long ANE case compiled layers but
never dispatched them. It writes no benchmark files and prints no user paths.

### Troubleshoot `uv` and stale native kernels

If the ANE case fails with `RuntimeError: There is no Stream(gpu, 0) in current
thread.`, check whether the native kernels were built using a different Python
environment. One reproduced failure loaded Python's MLX from `.venv` while the
kernels still linked to Homebrew's `libmlx.dylib`. Both libraries reported MLX
0.32.4, but their stream state was separate. Imports and `ane_available()` passed;
the first long request using ANE failed during evaluation.

Rebuild the kernels after syncing the environment, using that environment's
Python for the build:

```bash
uv sync
uv run --no-sync make kernels PYTHON=.venv/bin/python
uv run --no-sync python -m scripts.benchmark_decision_ane ~/.omlx/models/clef-flash-oQ4
```

This requires the full Xcode Metal toolchain. `--no-sync` keeps the prepared
environment unchanged during the build and benchmark; it cannot repair stale
binaries by itself. Rebuild again after replacing MLX or changing environments.
To inspect the quantized kernel's linked libraries on macOS:

```bash
otool -L omlx/custom_kernels/quant_matmul/_ext*.so
```

The Torch message `Redirects are currently not supported in MacOs` is a warning
and was not the cause of this failure.

Verification on 2026-10-28: rebuilding all four native packages reduced the
loaded MLX library count from two to one. The original `uv run python -m
scripts.benchmark_decision_ane ~/.omlx/models/clef-flash-oQ4` command then passed
all smoke checks, with 96 ANE MLP operations in the long case. Its warm long-case
medians were 8.352 s on GPU and 8.392 s with ANE: this run confirms execution and
the environment repair, but shows no latency improvement.

### Output format and interpretation

The benchmark prints progress messages, then a performance table for each prompt
length using the columns in the [Qwen ANE guide](qwen35_ane_prefill.md).
The following reformats the recorded 2026-10-28 run; it is not a new measurement.

Short prompt (361 tokens):

| Path | Median body time | Prompt throughput | Versus GPU |
|---|---:|---:|---:|
| GPU | 1.5980 s | 225.9 tok/s | 1.000x |
| Single ANE + GPU | 1.6460 s | 219.3 tok/s | 0.971x |

Long prompt (1,963 tokens):

| Path | Median body time | Prompt throughput | Versus GPU |
|---|---:|---:|---:|
| GPU | 8.3520 s | 235.0 tok/s | 1.000x |
| Single ANE + GPU | 8.3920 s | 233.9 tok/s | 0.995x |

- **Median body time:** synchronized `DecisionEngine.systemone()` wall time,
  including the backbone, decision head and executor scheduling. It excludes
  model load, eager ANE compilation, request encoding and HTTP. With the default
  three repetitions, the first is discarded and the median of the last two is
  reported. `--repeats 7` gives six measured repetitions after the initial warmup.
  This is the complete decision inference body, not an isolated transformer-only
  timer; matching column names does not make its absolute times comparable to
  the different Qwen model/workload.
- **Prompt throughput:** complete encoded input tokens divided by median body
  time. Tokens include schema and prompt framing; no output tokens are generated.
- **Versus GPU:** GPU median divided by the path's median for the same prompt.
  Above `1.000x` means faster; below it means slower. The GPU baseline is `1.000x`.

Memory and dispatch counters follow in a secondary diagnostics table:

| Mode | Case | Tokens | Median body time (s) | Process GiB | ANE MLP operations |
|---|---|---:|---:|---:|---:|
| GPU | short | 361 | 1.5980 | 7.29 | 0 |
| GPU | long | 1963 | 8.3520 | 8.13 | 0 |
| Single ANE + GPU | short | 361 | 1.6460 | 7.88 | 0 |
| Single ANE + GPU | long | 1963 | 8.3920 | 8.62 | 96 |

- **Process GiB:** process-footprint snapshot after the final repetition for
  that case. It is not peak memory and omits some ANE driver-wired allocations.
- **ANE MLP operations:** native evaluations across all repetitions, including
  the first inference warmup. The 96 calls are 32 layers across three long
  requests, not 96 decisions or a GPU-utilization measurement.

This run recorded ANE dispatch but no latency gain; long-case observed process
memory was 0.49 GiB higher. The short case made no ANE calls. Repeat with
interleaved ordering and representative traffic before treating small timing
differences as stable gains or regressions.

### Capture and plot a run

The [plotting helper](../../scripts/plot_decision_benchmark.py) produces separate
panels for median body time, prompt throughput and speedup versus GPU, with a
reference line at `1.000x`. It reads the diagnostics rows, derives the performance
metrics, and ignores surrounding messages and summary tables. Older six-column
benchmark captures remain supported. It rejects missing/duplicate cases,
non-finite measurements, non-positive body times, negative memory/call counts,
and comparisons with different token counts. Memory remains in the Markdown
diagnostics rather than the performance charts.

From the checkout:

```bash
mkdir -p build/decision-benchmark
uv run python -m scripts.benchmark_decision_ane ~/.omlx/models/clef-flash-oQ4 \
  > build/decision-benchmark/results.md

uv run --no-project --with matplotlib python -m scripts.plot_decision_benchmark \
  build/decision-benchmark/results.md
```

The plotting command uses Matplotlib in a temporary environment without changing
project dependencies. It produces `results.png` and `results.svg` beside the
input. All three generated files stay under the Git-ignored `build/` directory.
Redirecting benchmark output captures a new run; generating charts from an
existing Markdown file does not rerun inference. The plots contain measurements
and labels, not model filesystem paths or machine identifiers.

### Verify HTTP serving

For the complete FastAPI serving path, run the optional real-checkpoint test:

```bash
OMLX_TEST_CLEF_MODEL=~/.omlx/models/clef-flash-oQ4 \
  python -m pytest -q tests/integration/test_decision_ane_model.py -m slow
```

This test uses isolated temporary server settings, checks `/v1/systemone` and
`/api/status`, verifies actual ANE execution after excluding load-time warmup,
and checks model release on shutdown. It does not change the running server's
model settings.

## Enable ANE in serving

Use the modified checkout or a build containing these changes. An existing
Homebrew release does not pick up source edits automatically.

Stop the server before changing its persisted settings outside the admin UI.
For the default base directory, apply the measured single-ANE configuration:

```bash
python - <<'PY'
from pathlib import Path
from omlx.model_settings import ModelSettingsManager

manager = ModelSettingsManager(Path.home() / ".omlx")
model_id = "clef-flash-oQ4"
settings = manager.get_settings(model_id)
settings.ane_prefill_enabled = True
settings.ane_prefill_sequence_length = 1024
settings.ane_prefill_fraction = 0.25
settings.ane_prefill_max_layers = 32
settings.ane_prefill_dual_ane = False
settings.ane_prefill_gdn = False
settings.ane_prefill_cpu_enabled = False
settings.ane_prefill_fused_down = False
settings.ane_prefill_tail_padding_min_tokens = 0
manager.set_settings(model_id, settings)
PY

python -m omlx.cli serve --model-dir ~/.omlx/models --host 127.0.0.1 --port 8000
```

For another base directory, pass the same directory to `ModelSettingsManager`
and the server's `--base-path` option. Named profiles can override model
settings; configure the profile actually used by the request.

Send the decision request shown earlier, then inspect status:

```bash
curl http://127.0.0.1:8000/api/status \
  -H "Authorization: Bearer $OMLX_API_KEY"
```

The `ane_prefill.models` entry should report configured MLP layers, no dual-ANE
layers, and `shed: false`. A model released for memory pressure reports
`configured: false` and `shed: true`. Configuration counters show compiled
layers, not proof that every request uses ANE; the native-counter tests above
verify dispatch. Requests shorter than the tile still run on GPU.

The ANE split tuner now measures `encode()` and `systemone()` for decision
engines instead of calling text generation. When tuning a BF16 checkpoint,
disable optional CPU sharing unless a compatible FP16 clone has been prepared.

Eligible projections use affine q4/q5/q6/q8 weights with group size 64 or 128.
The current minimum tile is 1,024 tokens. Short requests and insufficient tails
stay on GPU unless supported intermediate tail padding is enabled. Do not add
artificial prompt tokens to force an ANE tile.

Existing fairness controls still apply: competing chat decodes can shrink
decision chunks below the ANE tile, reducing offload coverage. The native
operation counter is the authority for whether a particular workload used ANE.
This integration does not make decision inference GPU-independent or establish
how much GPU capacity another concurrent workload gains.

If GDN acceleration is subsequently enabled, preserve the existing z-only
precision boundary: recurrent `qkv` and state updates stay off the approximate
ANE path. See [Qwen ANE/GPU prefill](qwen35_ane_prefill.md).
