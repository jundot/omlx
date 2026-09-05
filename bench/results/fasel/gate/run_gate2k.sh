#!/bin/bash
set -e
D=bench/results/fasel/gate
mkdir -p $D
.venv/bin/python -m bench.bench_expert_streaming --model qwen --budget 0 --decode 48 --prompt-len 2k \
  --single-request --gate-tokens --min-free-gb 16 --out-dir $D --out $D/arm2k.json
cp $D/qwen_0.0g_output.json $D/fresh_2k_tokens.json
echo GATE_2K_DONE