#!/bin/bash
set -e
D=bench/results/fasel/l2
P_DEC=$D/profiles/decode.json
COMMON='--model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --min-free-gb 16'
run_arm() {
  local name=$1; shift
  local envs=() cli=()
  for a in "$@"; do
    case "$a" in
      *=*) envs+=("$a") ;;
      *) cli+=("$a") ;;
    esac
  done
  echo "=== AC arm $name $(date +%H:%M:%S) ==="
  env OMLX_EXPERT_STREAMING_PROFILE=1 "${envs[@]}" .venv/bin/python -m bench.bench_expert_streaming $COMMON "${cli[@]}" --out-dir $D --out $D/arms/$name.json
}
run_arm p_a1
run_arm p_c1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
run_arm p_a2
run_arm p_c2 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
run_arm p_a3
run_arm p_c3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
echo AC_DONE