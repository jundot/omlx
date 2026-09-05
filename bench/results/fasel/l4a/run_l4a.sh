#!/bin/bash
set -e
D=bench/results/fasel/l4a
mkdir -p $D
echo "=== L4A run1: 8k HOBBIT hf=0.25 (reference) ==="
env OMLX_EXPERT_STREAMING_MEMTRACE=$D/memtrace_hf25.jsonl .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 48 --prompt-len 8k --single-request --gate-tokens \
  --min-free-gb 16 --cold-tier 3 --hot-fraction 0.25 --out-dir $D --out $D/run1_hf25.json
echo "=== L4A run2: 8k HOBBIT hf=0.1 (ratio scaling check) ==="
env OMLX_EXPERT_STREAMING_MEMTRACE=$D/memtrace_hf10.jsonl .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 48 --prompt-len 8k --single-request --gate-tokens \
  --min-free-gb 16 --cold-tier 3 --hot-fraction 0.1 --out-dir $D --out $D/run2_hf10.json
echo "=== L4A aggregation ==="
.venv/bin/python $D/_agg.py $D/memtrace_hf25.jsonl > $D/agg_hf25.txt
.venv/bin/python $D/_agg.py $D/memtrace_hf10.jsonl > $D/agg_hf10.txt
cat $D/agg_hf25.txt
echo L4A_DONE