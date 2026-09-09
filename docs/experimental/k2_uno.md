# K2 Uno

Uno uses conditional adapter proposals and target verification for dense K2 models.
It is off by default. Enable **Uno** in model settings. Select the matching local adapter.
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
Load the adapter only when Uno is enabled.

1. Keep other inference workloads idle.
2. Start a fresh server process with Uno disabled. Use a separate, empty cache directory.
3. Run the comparison command with that server's PID.

   ```sh
   export OMLX_API_KEY='<local server key>'
   python benchmarks/uno_compare.py --model K2 --label uno-off \
     --server-pid '<server PID>' --trials 3 --max-tokens 512 --output uno-off.json
   ```

4. Restart the server with Uno enabled and the matching adapter loaded. Use another empty cache directory.
5. Repeat the command with `--label uno-on` and `--output uno-on.json`. Supply the new server PID.
6. Alternate the mode order across repeated runs.
7. Use trials after trial 0 for comparison. Trial 0 warms each workload.

Trials reuse prompts, so long-context results can include prefix-cache hits. The JSON records those hits.

The script records response duration, first streamed text, server generation rate, and sampled process memory.
It also sends eight simultaneous requests and eight staggered requests.
Use `--concurrency` to select 1, 2, 4, or 8 requests. Memory uses macOS physical footprint samples every 50 milliseconds.
Local runs with `--server-pid` also record host CPU/GPU utilization and thermal pressure for each workload.
Generation rate comes from API usage. Aggregate rate divides all completion tokens by the request group's duration.
The output budget applies to reasoning and answer tokens. Inspect finish reasons before treating a response as a completed task.
