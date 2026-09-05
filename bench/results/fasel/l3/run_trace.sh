#!/bin/bash
set -e
D=bench/results/fasel/l3
mkdir -p $D
rm -f $D/trace.jsonl
env OMLX_EXPERT_STREAMING_TRACE=$D/trace.jsonl .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens \
  --min-free-gb 16 --out-dir $D --out $D/run.json
.venv/bin/python bench/lrc_analysis.py --trace $D/trace.jsonl \
  --cache-sizes 8,16,32,64,128,256 --group-sizes 16,32,64 --out $D/lrc.json
echo TRACE_DONE