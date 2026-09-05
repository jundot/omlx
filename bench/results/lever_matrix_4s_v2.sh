#!/bin/bash
set -e
D=bench/results/lever_matrix/4s_v2
mkdir -p $D
COMMON='--model qwen-jang --budget 0 --decode 96 --prompt-len short --min-free-gb 12'
run_arm() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M:%S) ==="
  .venv/bin/python bench/bench_expert_streaming.py $COMMON "$@" --out $D/$name.json 2>&1 | grep -E "^decode" || true
}
# warmup (discarded - page cache + window state)
run_arm warmup
# adjacent pairs: base -> lever, x3 cycles; drift cancels in the pair ratio
for i in 1 2 3; do
  run_arm base_$i
  run_arm tk85_$i --topk 0.85
  run_arm base2_$i
  run_arm tk85p20_$i --topk 0.85 --cache-prior 2.0
done
echo MATRIX_DONE
