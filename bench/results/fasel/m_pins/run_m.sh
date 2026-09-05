#!/bin/bash
set -e
D=bench/results/fasel/m_pins
mkdir -p $D
P_DEC=bench/results/fasel/l2/profiles/decode.json
COMMON='--model qwen --budget 0 --decode 48 --prompt-len 2k --single-request --gate-tokens --min-free-gb 16 --knob pins_enabled --knob pin_budget_gib'
run_arm() {
  local name=$1; shift
  local envs=() cli=()
  for a in "$@"; do
    case "$a" in
      *=*) envs+=("$a") ;;
      *) cli+=("$a") ;;
    esac
  done
  echo "=== M-pins arm $name $(date +%H:%M:%S) ==="
  env OMLX_EXPERT_STREAMING_PROFILE=1 "${envs[@]}" .venv/bin/python -m bench.bench_expert_streaming $COMMON "${cli[@]}" --out-dir $D --out $D/$name.json
}
# Interleaved A (no pins) vs C (decode profile, 512 MiB, sync) x3.
run_arm m_a1
run_arm m_c1 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC --pins --pin-gib 0.5 --pin-regime decode
run_arm m_a2
run_arm m_c2 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC --pins --pin-gib 0.5 --pin-regime decode
run_arm m_a3
run_arm m_c3 OMLX_EXPERT_STREAMING_PIN_PROFILE=$P_DEC --pins --pin-gib 0.5 --pin-regime decode
echo M_PINS_DONE