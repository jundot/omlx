#!/bin/bash
set -e
D=bench/results/lrc/matrix
P_DEC=$D/profiles/decode_4s.json
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
  echo "=== matrix arm $name $(date +%H:%M:%S) ==="
  env "${envs[@]}" .venv/bin/python bench/bench_expert_streaming.py $COMMON "${cli[@]}" --out $D/arms/$name.json 2>&1 | grep -E "^decode|TTFT|pinned" || true
}
run_arm a1
run_arm c1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode OMLX_EXPERT_STREAMING_PIN_KEEP=512 --pins --pin-gib 1.5
run_arm a2
run_arm c2 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode OMLX_EXPERT_STREAMING_PIN_KEEP=512 --pins --pin-gib 1.5
run_arm a3
run_arm c3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode OMLX_EXPERT_STREAMING_PIN_KEEP=512 --pins --pin-gib 1.5
echo MATRIX_DONE
