#!/bin/bash
set -e
D=bench/results/lever_matrix
mkdir -p $D/4m
COMMON='--model qwen-jang4m --budget 0 --decode 96 --prompt-len short --min-free-gb 12'
run_arm() {
  local name=$1; shift
  local envs=() cli=()
  for a in "$@"; do
    case "$a" in
      *=*) envs+=("$a") ;;
      *) cli+=("$a") ;;
    esac
  done
  echo "=== 4m arm $name $(date +%H:%M:%S) ==="
  env "${envs[@]}" .venv/bin/python bench/bench_expert_streaming.py $COMMON "${cli[@]}" --out $D/4m/$name.json 2>&1 | grep -E "^decode|TTFT" || true
}
run_arm base1
run_arm tk85_1 --topk 0.85
run_arm base2
run_arm tk85p20_1 --topk 0.85 --cache-prior 2.0
run_arm base3
run_arm tk85_2 --topk 0.85
run_arm tk85p20_2 --topk 0.85 --cache-prior 2.0
run_arm base4
run_arm tk85_3 --topk 0.85
run_arm tk85p20_3 --topk 0.85 --cache-prior 2.0
echo MATRIX_DONE
