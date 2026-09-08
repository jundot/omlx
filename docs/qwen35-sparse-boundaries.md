# Qwen3.8 sparse cache boundaries

This branch contains an opt-in experiment for Qwen3.8 models using embedded
Gated DeltaNet state. It keeps the normal 2048-token bulk prefill forwards
while retaining a 512-token cache block and one final 512-aligned recurrent
checkpoint near the prompt tail. The default serving policy is unchanged.

Two public tools cover separate questions:

- [bench_qwen35_cache.py](../benchmarks/bench_qwen35_cache.py) measures cold
  and warm request performance with deterministic synthetic prompts.
- [bench_qwen35_restore.py](../benchmarks/bench_qwen35_restore.py) creates
  fixed continuation fixtures and checks restored-prefix correctness.

## Activation and scope

From this checkout, use a source virtual environment and a separate settings
base so the experiment does not change the normal server configuration:

```bash
PYTHONPATH=. \
OMLX_QWEN35_SPARSE_BOUNDARIES=1 \
OMLX_GDN_SNAPSHOT_STORAGE=embedded \
/path/to/venv/bin/python -m omlx.cli serve \
  --base-path /tmp/omlx-sparse-bench \
  --model-dir /path/to/models \
  --port 8845 \
  --paged-ssd-cache-dir /tmp/omlx-sparse-cache \
  --arrays-cache-block-size 512
```

The sparse policy engages only when all of these conditions hold:

- `OMLX_QWEN35_SPARSE_BOUNDARIES=1` was present at scheduler startup;
- `--arrays-cache-block-size 512` selected an explicit 512-token block;
- GDN snapshots use embedded storage, rather than split SSD sidecars;
- the loaded model identifies as Qwen3.5/3.8; and
- `make_cache()` returns only top-level `ArraysCache` and `KVCache` entries,
  with both types present.

If any condition fails, the sparse policy stays inactive. An explicit
512-token block still retains the established every-512-boundary behaviour.
Without the environment gate, serving defaults remain unchanged. A missing
block override, `None`, or `0` keeps automatic ArraysCache block sizing.

The sparse path captures absolute 2048-token recurrent checkpoints and the
final 512-aligned position reachable during prefill. Prefill deliberately
leaves the final prompt token for generation kickoff, so an exactly
512-aligned prompt normally has its final prefill checkpoint 512 tokens
earlier. Restores inside a sparse region walk back to the newest captured
recurrent checkpoint and prefill the remaining suffix. They do not claim the
largest ordinary 512-token block below the shared prefix.

Use an isolated port and cache directory. The tools refuse production port
8843 unless `--allow-8843` is supplied deliberately.

## Performance benchmark

`bench_qwen35_cache.py` is the campaign harness. It accepts `--arm`,
`--cases`, `--runs`, `--warmup`, `--output`, and `--serve-log`; it does not
accept restore-probe options such as `--phase`, `--fixtures`, or
`--create-only`.

Validate a performance plan without HTTP or GPU work:

```bash
/path/to/venv/bin/python benchmarks/bench_qwen35_cache.py \
  --base-url http://127.0.0.1:8845 \
  --arm sparse512 \
  --cases context16k-retrieval context57k-retrieval \
  --runs 2 \
  --warmup 0 \
  --output /tmp/qwen35-cache-performance.jsonl \
  --validate-only
```

Remove `--validate-only` to run the plan against the isolated server. The
default append-only output is `results/qwen-cache-bench.jsonl`. Each row
records prompt hashes, usage counts, cached tokens, output, TTFT, end-to-end
latency, generation timing, and optional server-log diagnostics.

Compare arms in separate server runs with identical model settings, prompt
seed, cases, concurrency, and cache cleanliness. For an ordinary 512-block
control, omit `OMLX_QWEN35_SPARSE_BOUNDARIES`; keep embedded GDN and the same
`--arrays-cache-block-size 512`. For a no-cache reference, start a fresh server
with caching disabled:

```bash
/path/to/venv/bin/omlx serve --port 8845 --no-cache
```

Reusing an existing cache directory can turn a planned cold row into a warm
hit.

## Restore QA

The restore probe owns fixture creation and cached/reference phases. Its
defaults are `results/qwen35-restore-fixtures.json` and
`results/qwen35-restore-probe.jsonl`.

Create deterministic fixtures locally:

```bash
/path/to/venv/bin/python benchmarks/bench_qwen35_restore.py \
  --base-url http://127.0.0.1:8845 \
  --create-only \
  --fixtures /tmp/qwen35-restore-fixtures.json
```

Run the cached phase against a clean isolated sparse-boundary server:

```bash
/path/to/venv/bin/python benchmarks/bench_qwen35_restore.py \
  --base-url http://127.0.0.1:8845 \
  --phase cached \
  --fixtures /tmp/qwen35-restore-fixtures.json \
  --output /tmp/qwen35-restore-cached.jsonl \
  --serve-log /tmp/omlx-sparse-server.log
```

Then run the byte-identical fixtures against a fresh no-cache server:

```bash
/path/to/venv/bin/python benchmarks/bench_qwen35_restore.py \
  --base-url http://127.0.0.1:8845 \
  --phase reference \
  --fixtures /tmp/qwen35-restore-fixtures.json \
  --output /tmp/qwen35-restore-reference.jsonl \
  --serve-log /tmp/omlx-reference-server.log
```

The cached phase primes the deterministic 16K retrieval prompt, then checks
three continuations. Expected reuse is the newest captured recurrent
checkpoint shared with each fixture:

| Fixture cut | Expected answer | Expected cached tokens |
|---:|---|---:|
| 3500 | `cobalt-heron-7913` | 2048 |
| 5800 | `amber-otter-4821` | 4096 |
| 8300 | `amber-otter-4821` | 8192 |

The reference phase requires zero cached tokens. A wrong fact, cache-count
mismatch, malformed stream, missing stream terminator, or server error makes
the probe exit nonzero. `--vision` adds an independent generated 64x64 red
square check.

## Recorded evidence

The following single-session observations come from the isolated campaign
JSONL `2026-09-07-qwen27b-http.jsonl`. Corresponding rows have identical
prompt hashes within each context, were stream-valid, and returned
`amber-otter-4821`. The ordinary and sparse rows are recorded under the arm
labels `cache512-e1` and `sparse512-f1`, respectively.

| Arm | Prompt tokens | Cold TTFT | Warm TTFT | Warm cached tokens |
|---|---:|---:|---:|---:|
| ordinary 512 | 16,061 | 36.980 s | 3.021 s | 15,872 |
| sparse 512 | 16,061 | 35.338 s | 2.243 s | 15,872 |
| ordinary 512 | 57,072 | 177.370 s | 8.462 s | 56,832 |
| sparse 512 | 57,072 | 161.975 s | 5.420 s | 56,832 |

These are targeted observations, rather than a production recommendation or
a universal block-size result. Repeat the paired arms in one thermal session
before making a performance claim.

## Provenance

The cache-size plumbing follows the block-size override approach in upstream
PR 3439. Boundary timing follows the instrumentation lineage of PR 3391. The
experiment measures the final-boundary opportunity discussed in issue 3070;
these references do not imply that the changes are present in every oMLX
release.

## Three-sample default-path comparison

A subsequent comparison used the same MLX 0.32.2 source backend in both
arms, MTP depth 8, temperature zero, non-thinking requests, and three
measured samples per case after warmup. It compared the automatic 2048-token
cache blocks with sparse 512-token blocks. These are not measurements against
the installed app's older bundled MLX runtime.

| Workload | Default median total time | Sparse median total time | Reduction |
|---|---:|---:|---:|
| 16K, 400-token code continuation | 16.84 s | 12.76 s | 24.2% |
| 57K, 400-token code continuation | 29.40 s | 22.08 s | 24.9% |
| Matched 12-turn synthetic loop, total | 51.07 s | 32.56 s | 36.2% |

All compared prompts matched. Every prompt and output in the 12-turn loop
matched between arms. Long-form MTP outputs were not byte-identical across
every run, including default-path repetitions, so these results do not
establish a universal quality or decode-throughput improvement. The main
benefit is less cache and prompt-processing work.

The cache-only candidate passed 645 focused tests. Real restored-prefix
probes used exactly 2048/4096/8192 cached tokens and returned the same facts
as cache-disabled references. Long-context retrieval and vision probes also
passed.
