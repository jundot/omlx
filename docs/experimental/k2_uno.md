# K2 Uno

Uno uses conditional adapter proposals and target verification for dense K2 models.
It is off by default. Enable **Uno** in model settings and select the matching local adapter.
The loader checks the base identity, adapter settings, and tensor layout.

| Base | Adapter | Context limit |
| --- | --- | --- |
| IFM/K2-Horizon-0.9B | IFM/K2-Horizon-0.9B-Uno | 131,072 tokens |
| IFM/K2-Horizon-7B | IFM/K2-Horizon-7B-Uno | 262,144 tokens |

These are the released dense pairs tested by this implementation.
The loader accepts compatible layouts without a model-size whitelist.
The limits include prompt and output tokens. These measurements do not test the full context limits.
Uno does not support MoVA or gated attention.

## Limits

Uno runs while one request is active. When requests overlap, the scheduler uses ordinary continuous batching.
It discards unconsumed proposals before requests merge. Each request keeps its committed tokens and KV cache.
Uno resumes when one request remains. This policy uses one model and the standard scheduler.
Uno supports temperature, top-p, top-k, stop strings, native K2 tools, and SSD prefix reuse.
It rejects thinking budgets, structured output, distributed serving, and nondefault penalties, min-p, or XTC probability.
Set `repetition_penalty` to `1.0`. The `reasoning_effort` template setting remains available.

ANE prefill is optional. ANE and compiled execution can change arithmetic and outputs.
Their prefix caches use separate namespaces when arithmetic differs.
A fixed seed does not guarantee identical outputs between Uno and ordinary decoding.

## Reproduce the comparison

Use the same Mac, base checkpoint, sampling settings, and prompts for both modes.
Start a fresh server process for each mode. Load the adapter only when Uno is enabled.
Use a separate, empty cache directory for each process.
Keep other inference workloads idle.

```sh
export OMLX_API_KEY='<local server key>'
python benchmarks/uno_compare.py --model K2 --label uno-off \
  --server-pid '<server PID>' --trials 3 --max-tokens 512 --output uno-off.json
```

Restart the server with Uno enabled. Repeat the command with `--label uno-on` and `--output uno-on.json`.
Supply the new server PID. Alternate the mode order across repeated runs.
Trial 0 warms each workload. Use later trials for comparison.

The script records response duration, first streamed text, server generation rate, and sampled process memory.
It also sends eight simultaneous requests and eight staggered requests.
Use `--concurrency` to select 1, 2, 4, or 8 requests. Memory uses macOS physical footprint samples every 50 milliseconds.
Generation rate comes from API usage. Aggregate rate divides all completion tokens by the request group's duration.
The output budget applies to reasoning and answer tokens. Inspect finish reasons before treating a response as a completed task.
