# Native benchmark

`scripts/bench.py` measures the in-process engine without HTTP overhead. Its
default behavior remains a single trial with model defaults.

To benchmark the same persisted per-model configuration used by an oMLX server,
pass the server base directory:

```bash
python scripts/bench.py ~/.omlx/models/Qwen3.8-27B-oQ4e-mtp \
  --settings-base ~/.omlx \
  --pp 1024 4096 8192 --gen 128 --batch 2 4 8
```

Use `--repeats N` to report the median of N single-request trials. Each trial
generates a fresh UUID-prefixed, corpus-equivalent prompt. This matters for
speculative decoding: reusing one prompt measures timing precision but hides
continuation-dependent acceptance variance.

Select an existing bundled corpus with `--context-profile`, for example:

```bash
python scripts/bench.py MODEL --repeats 3 --context-profile novel_en
```

Available profiles are shown by `scripts/bench.py --help`. Prefill length,
generation length, and metric accounting are unchanged by these controls.
