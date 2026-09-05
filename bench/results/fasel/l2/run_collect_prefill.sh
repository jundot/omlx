#!/bin/bash
set -e
# workdir is supplied by the caller (bash tool workdir).
D=bench/results/fasel/l2
PROF=$D/profiles/prefill.json
rm -f $PROF
env OMLX_EXPERT_STREAMING_PIN_PROFILE=$PROF .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 48 --prompt-len 8k --single-request --gate-tokens \
  --min-free-gb 16 --pins --pin-gib 0.25 \
  --out-dir $D --out $D/collect_prefill.json
echo COLLECT_PREFILL_DONE