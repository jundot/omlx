#!/bin/bash
set -e
D=bench/results/fasel/l4a
env OMLX_EXPERT_STREAMING_DUAL_TIER_ORDER=small-first \
    OMLX_EXPERT_STREAMING_MEMTRACE=$D/memtrace_hf25_smallfirst.jsonl \
    .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 48 --prompt-len 8k --single-request --gate-tokens \
  --min-free-gb 16 --cold-tier 3 --hot-fraction 0.25 --out-dir $D --out $D/run3_hf25_smallfirst.json
.venv/bin/python $D/_agg.py $D/memtrace_hf25_smallfirst.jsonl > $D/agg_hf25_smallfirst.txt
echo B3_DONE