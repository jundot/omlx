#!/bin/bash
set -e
# workdir is supplied by the caller (bash tool workdir).
D=bench/results/fasel/l2
PROF=$D/profiles/decode.json
rm -f $PROF
env OMLX_EXPERT_STREAMING_PIN_PROFILE=$PROF .venv/bin/python -m bench.bench_expert_streaming \
  --model qwen --budget 0 --decode 200 --prompt-len 2k --single-request --gate-tokens \
  --min-free-gb 16 --pins --pin-gib 0.25 \
  --out-dir $D --out $D/collect_decode.json
echo COLLECT_DECODE_DONE