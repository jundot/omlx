#!/bin/bash
set -e
D=bench/results/fasel/l2
P_DEC=$D/profiles/decode.json
P_PRE=$D/profiles/prefill.json
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
  echo "=== L2 arm $name $(date +%H:%M:%S) ==="
  env "${envs[@]}" .venv/bin/python -m bench.bench_expert_streaming $COMMON "${cli[@]}" --out-dir $D --out $D/arms/$name.json
}
# Interleaved A/C/E x3 (same-window decision arms), then B/D sizing.
run_arm a1
run_arm c1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
run_arm e1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_PRE OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=prefill --pins --pin-gib 0.5
run_arm a2
run_arm c2 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
run_arm e2 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_PRE OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=prefill --pins --pin-gib 0.5
run_arm a3
run_arm c3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.5
run_arm e3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_PRE OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=prefill --pins --pin-gib 0.5
# B: decode profile, 256 MiB (sizing point)
run_arm b1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 0.25
# D: decode profile, 1.25 GiB (sizing point)
run_arm d1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC OMLX_EXPERT_STREAMING_PIN_SYNC=1 OMLX_EXPERT_STREAMING_PIN_REGIME=decode --pins --pin-gib 1.25
echo MATRIX_DONE