#!/bin/bash
set -e
D=bench/results/slotbank
mkdir -p $D
COMMON='--model qwen-jang --budget 0 --decode 96 --prompt-len short --min-free-gb 12'
run_arm() {
  local name=$1; shift
  local envs=() cli=()
  for a in "$@"; do
    case "$a" in
      *=*) envs+=("$a") ;;
      *) cli+=("$a") ;;
    esac
  done
  echo "=== slotbank arm $name $(date +%H:%M:%S) ==="
  env "${envs[@]}" .venv/bin/python bench/bench_expert_streaming.py $COMMON "${cli[@]}" --out $D/$name.json 2>&1 | grep -E "^decode|TTFT|pinned|slot" || true
}
# interleaved A (off) / B (on, 16 slots) x3
run_arm a1
run_arm b1 OMLX_EXPERT_STREAMING_SLOT_BANK=1 OMLX_EXPERT_STREAMING_SLOT_BANK_SLOTS=16
run_arm a2
run_arm b2 OMLX_EXPERT_STREAMING_SLOT_BANK=1 OMLX_EXPERT_STREAMING_SLOT_BANK_SLOTS=16
run_arm a3
run_arm b3 OMLX_EXPERT_STREAMING_SLOT_BANK=1 OMLX_EXPERT_STREAMING_SLOT_BANK_SLOTS=16
echo MATRIX_DONE
